"""Evaluation orchestration: dataset → model → predictions → metrics.

Two flavours of the same pipeline:

- :func:`evaluate`        — sequential, sync; kept for the previous
                            CLI flow and unit tests.
- :func:`evaluate_async`  — concurrent fan-out via ``asyncio.gather`` and
                            an ``asyncio.Semaphore``. Use this when calling
                            a real LLM provider; it turns a 1000-example
                            run from O(N · latency) into O(N · latency / C).

Both return the same :class:`EvaluationReport`; the only differences are
ordering of progress prints (async prints in completion order) and total
wall-clock time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.cost import estimate_cost
from src.metrics import compute_all
from src.model import ModelOutput, ModelProvider
from src.utils import Config, Example

log = logging.getLogger(__name__)


# Final decision tier. We deliberately keep ``REVIEW`` out of the classification
# label space — it's a routing decision on top of the model's output, not a
# category the model ever returns. All metric computations still use the 5
# configured categories so precision/recall/F1 don't drift when the threshold
# changes.
REVIEW_VERDICT = "REVIEW"


@dataclass(frozen=True)
class Prediction:
    """A single example + the model's response, ready for serialisation.

    ``predicted_verdict`` is what the model said (ALLOW/BLOCK); ``final_verdict``
    is what downstream consumers should act on — it equals ``predicted_verdict``
    when confidence ≥ threshold and ``REVIEW`` otherwise.
    """

    text: str
    expected_label: str
    expected_verdict: str
    predicted_category: str
    predicted_verdict: str
    final_verdict: str
    confidence: float
    reason: str
    parse_ok: bool
    correct: bool
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class EvaluationReport:
    config_summary: dict[str, Any]
    predictions: list[Prediction]
    metrics: dict[str, Any]
    summary: dict[str, Any]


def _expected_verdict(label: str, block_labels: list[str]) -> str:
    return "BLOCK" if label in block_labels else "ALLOW"


def resolve_final_verdict(model_verdict: str, confidence: float, threshold: float) -> str:
    """Apply the confidence threshold — below it, defer to a human.

    A threshold of 0.0 turns the REVIEW tier off entirely (useful for unit
    tests and anyone who wants the old binary behaviour).
    """
    if threshold > 0.0 and confidence < threshold:
        return REVIEW_VERDICT
    return model_verdict


def _make_prediction(ex: Example, out: ModelOutput, config: Config) -> Prediction:
    expected_v = _expected_verdict(ex.label, config.block_labels)
    final_v = resolve_final_verdict(out.verdict, float(out.confidence), config.confidence_threshold)
    return Prediction(
        text=ex.text,
        expected_label=ex.label,
        expected_verdict=expected_v,
        predicted_category=out.category,
        predicted_verdict=out.verdict,
        final_verdict=final_v,
        confidence=float(out.confidence),
        reason=out.reason,
        parse_ok=out.parse_ok,
        correct=out.category == ex.label,
        prompt_tokens=int(out.prompt_tokens),
        completion_tokens=int(out.completion_tokens),
    )


def _print_progress(i: int, total: int, ex: Example, pred: Prediction) -> None:
    mark = "✓" if pred.correct else "✗"
    print(
        f"[{i:>3}/{total}] {mark} expected={ex.label:<22} "
        f"got={pred.predicted_category:<22} "
        f"verdict={pred.predicted_verdict} "
        f"conf={pred.confidence:.2f}",
        flush=True,
    )


def _build_report(
    examples: list[Example],
    predictions: list[Prediction],
    duration: float,
    config: Config,
) -> EvaluationReport:
    total = len(examples)
    parse_failures = sum(1 for p in predictions if not p.parse_ok)

    y_true = [p.expected_label for p in predictions]
    y_pred = [p.predicted_category for p in predictions]
    metrics = compute_all(
        y_true,
        y_pred,
        labels=list(config.labels),
        metric_names=config.metrics,
        averages=config.metric_averages,  # type: ignore[arg-type]
    )

    # Verdict-level (binary) accuracy is the most operationally meaningful
    # number for moderation: how often does the system block what should be
    # blocked? We compute it here rather than coercing the metric machinery
    # into running over a different label space.
    verdict_correct = sum(1 for p in predictions if p.predicted_verdict == p.expected_verdict)
    block_recall = _block_recall(predictions)
    review_count = sum(1 for p in predictions if p.final_verdict == REVIEW_VERDICT)

    total_prompt_tokens = sum(p.prompt_tokens for p in predictions)
    total_completion_tokens = sum(p.completion_tokens for p in predictions)
    estimated_cost = estimate_cost(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        model_name=config.model_name,
    )
    # Per-1k-messages projection: handy for "what would moderating a million
    # messages cost?" — multiply by 1000.
    cost_per_1k = (estimated_cost / total * 1000) if estimated_cost is not None and total else None

    summary = {
        "n_examples": total,
        "duration_seconds": round(duration, 3),
        "parse_failures": parse_failures,
        "category_correct": sum(1 for p in predictions if p.correct),
        "verdict_correct": verdict_correct,
        "verdict_accuracy": verdict_correct / total if total else 0.0,
        "block_recall": block_recall,
        "review_count": review_count,
        "review_rate": review_count / total if total else 0.0,
        "confidence_threshold": config.confidence_threshold,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "estimated_cost_usd": estimated_cost,
        "cost_per_1k_messages_usd": cost_per_1k,
    }

    config_summary = {
        "model_type": config.model_type,
        "model_name": config.model_name,
        "temperature": config.temperature,
        "concurrency": config.concurrency,
        "confidence_threshold": config.confidence_threshold,
        "labels": list(config.labels),
        "block_labels": list(config.block_labels),
        "metrics": list(config.metrics),
        "averages": list(config.metric_averages),
    }

    return EvaluationReport(
        config_summary=config_summary,
        predictions=predictions,
        metrics=metrics,
        summary=summary,
    )


def evaluate(
    examples: list[Example],
    *,
    provider: ModelProvider,
    system_prompt: str,
    config: Config,
) -> EvaluationReport:
    """Run the full evaluation pipeline **sequentially** (sync).

    Kept for the existing CLI flow, unit tests, and any caller that prefers
    a simple, deterministic ordering. For real LLM providers, prefer
    :func:`evaluate_async`.
    """
    if not examples:
        raise ValueError("dataset is empty")

    predictions: list[Prediction] = []
    started = time.monotonic()

    total = len(examples)
    for i, ex in enumerate(examples, start=1):
        out = provider.call(system_prompt, ex.text)
        pred = _make_prediction(ex, out, config)
        predictions.append(pred)
        if config.verbose:
            _print_progress(i, total, ex, pred)

    duration = time.monotonic() - started
    return _build_report(examples, predictions, duration, config)


async def evaluate_async(
    examples: list[Example],
    *,
    provider: ModelProvider,
    system_prompt: str,
    config: Config,
    concurrency: int | None = None,
) -> EvaluationReport:
    """Run the pipeline with up to ``concurrency`` requests in flight.

    A semaphore caps in-flight requests so we don't trigger the provider's
    rate limiter; tasks complete out of order but are stitched back into
    input order before metrics are computed, so the report matches what
    :func:`evaluate` would have produced — only faster.

    ``concurrency`` falls back to ``config.concurrency`` (default 8).
    """
    if not examples:
        raise ValueError("dataset is empty")

    cap = concurrency if concurrency is not None else config.concurrency
    cap = max(1, int(cap))
    sem = asyncio.Semaphore(cap)

    total = len(examples)
    results: list[Prediction | None] = [None] * total
    completed = 0
    completed_lock = asyncio.Lock()

    async def _process(i: int, ex: Example) -> None:
        nonlocal completed
        async with sem:
            out = await provider.acall(system_prompt, ex.text)
        pred = _make_prediction(ex, out, config)
        results[i] = pred
        async with completed_lock:
            completed += 1
            if config.verbose:
                _print_progress(completed, total, ex, pred)

    started = time.monotonic()
    await asyncio.gather(*[_process(i, ex) for i, ex in enumerate(examples)])
    duration = time.monotonic() - started

    # All slots are populated by the time gather returns.
    predictions: list[Prediction] = [p for p in results if p is not None]
    if len(predictions) != total:  # pragma: no cover - defensive
        raise RuntimeError("evaluate_async: lost a result; this should not happen")
    return _build_report(examples, predictions, duration, config)


def _block_recall(predictions: list[Prediction]) -> float:
    """How many should-be-blocked messages did we actually block?"""
    blocks_expected = [p for p in predictions if p.expected_verdict == "BLOCK"]
    if not blocks_expected:
        return 1.0
    blocked = sum(1 for p in blocks_expected if p.predicted_verdict == "BLOCK")
    return blocked / len(blocks_expected)


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    """Plain dict for JSON serialisation. Keeps the report dataclass internal."""
    return {
        "config": report.config_summary,
        "summary": report.summary,
        "metrics": report.metrics,
        "predictions": [asdict(p) for p in report.predictions],
    }

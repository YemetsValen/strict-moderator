"""Evaluation orchestration: dataset → model → predictions → metrics."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.metrics import compute_all
from src.model import ModelProvider, call_model
from src.utils import Config, Example

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """A single example + the model's response, ready for serialisation."""

    text: str
    expected_label: str
    expected_verdict: str
    predicted_category: str
    predicted_verdict: str
    confidence: float
    reason: str
    parse_ok: bool
    correct: bool


@dataclass(frozen=True)
class EvaluationReport:
    config_summary: dict[str, Any]
    predictions: list[Prediction]
    metrics: dict[str, Any]
    summary: dict[str, Any]


def _expected_verdict(label: str, block_labels: list[str]) -> str:
    return "BLOCK" if label in block_labels else "ALLOW"


def evaluate(
    examples: list[Example],
    *,
    provider: ModelProvider,
    system_prompt: str,
    config: Config,
) -> EvaluationReport:
    """Run the full evaluation pipeline.

    Side-effects:
      - prints a per-example progress line if ``config.verbose`` is True
      - logs warnings for parse failures and unknown categories
    """
    if not examples:
        raise ValueError("dataset is empty")

    predictions: list[Prediction] = []
    parse_failures = 0
    started = time.monotonic()

    total = len(examples)
    for i, ex in enumerate(examples, start=1):
        result = call_model(provider, system_prompt, ex.text)
        if not result["parse_ok"]:
            parse_failures += 1
        expected_v = _expected_verdict(ex.label, config.block_labels)
        pred = Prediction(
            text=ex.text,
            expected_label=ex.label,
            expected_verdict=expected_v,
            predicted_category=str(result["category"]),
            predicted_verdict=str(result["verdict"]),
            confidence=float(result["confidence"]),
            reason=str(result["reason"]),
            parse_ok=bool(result["parse_ok"]),
            correct=result["category"] == ex.label,
        )
        predictions.append(pred)
        if config.verbose:
            mark = "✓" if pred.correct else "✗"
            print(
                f"[{i:>3}/{total}] {mark} expected={ex.label:<22} "
                f"got={pred.predicted_category:<22} "
                f"verdict={pred.predicted_verdict} "
                f"conf={pred.confidence:.2f}",
                flush=True,
            )

    duration = time.monotonic() - started

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

    summary = {
        "n_examples": total,
        "duration_seconds": round(duration, 3),
        "parse_failures": parse_failures,
        "category_correct": sum(1 for p in predictions if p.correct),
        "verdict_correct": verdict_correct,
        "verdict_accuracy": verdict_correct / total,
        "block_recall": block_recall,
    }

    config_summary = {
        "model_type": config.model_type,
        "model_name": config.model_name,
        "temperature": config.temperature,
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

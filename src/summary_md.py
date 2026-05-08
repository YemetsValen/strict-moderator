"""Render an ``EvaluationReport`` as a human-readable ``summary.md``.

The markdown report sits alongside ``reports/last_run.json`` (by default
``reports/summary.md``) and is regenerated on every CLI run. Unlike the JSON,
which is designed for programmatic consumption, the markdown is tuned for
product / management review: headline metrics, cost, and — most importantly —
concrete examples the model got wrong or flagged as low-confidence, so the
team can eyeball *why* without opening a 100 MB JSON.

The generator caps the "Errors" and "Low-confidence" sections at
:data:`MAX_EXAMPLES_PER_SECTION` rows so the file stays readable on a 100k-row
dataset; the full list remains available in the JSON report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.cost import format_cost
from src.evaluator import REVIEW_VERDICT, EvaluationReport, Prediction

MAX_EXAMPLES_PER_SECTION = 50
# Messages on matchmaking / dating platforms can be paragraphs long; truncate
# the snippet rather than bloat the markdown.
MAX_TEXT_SNIPPET_CHARS = 240


@dataclass(frozen=True)
class _SummarySections:
    errors: list[Prediction]
    reviews: list[Prediction]


def _collect_sections(predictions: list[Prediction]) -> _SummarySections:
    errors = [p for p in predictions if not p.correct]
    reviews = [p for p in predictions if p.final_verdict == REVIEW_VERDICT]
    return _SummarySections(errors=errors, reviews=reviews)


def _truncate(text: str, limit: int = MAX_TEXT_SNIPPET_CHARS) -> str:
    cleaned = text.replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_metrics_block(report: EvaluationReport) -> list[str]:
    lines: list[str] = []
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for name, value in report.metrics.items():
        if isinstance(value, float):
            lines.append(f"| {name} | {value:.3f} |")
            continue
        per_class: dict[str, float] = value.get("per_class", {})  # type: ignore[assignment]
        for label, score in per_class.items():
            lines.append(f"| {name}.{label} | {score:.3f} |")
        for avg_name, avg_score in value.items():
            if avg_name == "per_class":
                continue
            lines.append(f"| {name}.{avg_name} | {avg_score:.3f} |")
    lines.append("")
    return lines


def _render_examples_table(rows: list[Prediction]) -> list[str]:
    lines: list[str] = [
        "| # | Expected | Predicted | Verdict | Confidence | Text | Reason |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for idx, p in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {p.expected_label} | {p.predicted_category} | "
            f"{p.final_verdict} | {p.confidence:.2f} | "
            f"{_truncate(p.text)} | {_truncate(p.reason)} |"
        )
    return lines


def render_summary_md(
    report: EvaluationReport,
    *,
    dataset_path: str | Path,
    now: datetime | None = None,
    max_examples: int = MAX_EXAMPLES_PER_SECTION,
) -> str:
    """Build the markdown body. ``write_summary_md`` is the disk-writing wrapper."""
    when = (now or datetime.now(tz=UTC)).strftime("%Y-%m-%d %H:%M:%S UTC")
    cfg = report.config_summary
    s = report.summary
    sections = _collect_sections(report.predictions)

    lines: list[str] = []
    lines.append("# Moderation run — summary")
    lines.append("")
    lines.append(f"- **When:** {when}")
    lines.append(f"- **Model:** `{cfg['model_type']}` / `{cfg['model_name']}`")
    lines.append(f"- **Dataset:** `{dataset_path}` ({s['n_examples']} examples)")
    lines.append(f"- **Duration:** {s['duration_seconds']}s @ concurrency={cfg['concurrency']}")
    lines.append(f"- **Confidence threshold:** {cfg['confidence_threshold']:.2f}")
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Category accuracy:** {s['category_correct']}/{s['n_examples']}")
    lines.append(f"- **Verdict accuracy:** {_fmt_pct(s['verdict_accuracy'])}")
    lines.append(f"- **Block recall:** {_fmt_pct(s['block_recall'])}")
    lines.append(
        f"- **Review queue:** {s['review_count']} of {s['n_examples']} "
        f"({_fmt_pct(s['review_rate'])}) sent to human moderation"
    )
    lines.append(f"- **Parse failures:** {s['parse_failures']}")
    lines.append("")

    lines.append("## Cost")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Prompt tokens | {s['total_prompt_tokens']} |")
    lines.append(f"| Completion tokens | {s['total_completion_tokens']} |")
    lines.append(f"| Total tokens | {s['total_tokens']} |")
    lines.append(f"| Estimated cost | {format_cost(s['estimated_cost_usd'])} |")
    lines.append(f"| Cost per 1 000 messages | {format_cost(s['cost_per_1k_messages_usd'])} |")
    cost_per_1m = (
        s["cost_per_1k_messages_usd"] * 1000
        if s["cost_per_1k_messages_usd"] is not None
        else None
    )
    lines.append(f"| Cost per 1 000 000 messages | {format_cost(cost_per_1m)} |")
    lines.append("")

    lines.extend(_render_metrics_block(report))

    lines.append("## Errors")
    lines.append("")
    if not sections.errors:
        lines.append("_No classification errors on this dataset._")
    else:
        head = sections.errors[:max_examples]
        lines.extend(_render_examples_table(head))
        if len(sections.errors) > max_examples:
            lines.append("")
            lines.append(
                f"_Showing first {max_examples} of {len(sections.errors)} errors; "
                f"see `last_run.json` for the full list._"
            )
    lines.append("")

    lines.append("## Low-confidence (REVIEW queue)")
    lines.append("")
    lines.append(
        "Messages where `confidence < confidence_threshold` "
        f"({cfg['confidence_threshold']:.2f}); the client should route these to a human."
    )
    lines.append("")
    if not sections.reviews:
        lines.append("_No messages below the confidence threshold._")
    else:
        head = sections.reviews[:max_examples]
        lines.extend(_render_examples_table(head))
        if len(sections.reviews) > max_examples:
            lines.append("")
            lines.append(
                f"_Showing first {max_examples} of {len(sections.reviews)} "
                "low-confidence items; see `last_run.json` for the full list._"
            )
    lines.append("")

    return "\n".join(lines)


def write_summary_md(
    path: Path | str,
    report: EvaluationReport,
    *,
    dataset_path: str | Path,
    now: datetime | None = None,
    max_examples: int = MAX_EXAMPLES_PER_SECTION,
) -> Path:
    """Render and write the markdown summary; return the path for logging."""
    body = render_summary_md(
        report,
        dataset_path=dataset_path,
        now=now,
        max_examples=max_examples,
    )
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body, encoding="utf-8")
    return dest

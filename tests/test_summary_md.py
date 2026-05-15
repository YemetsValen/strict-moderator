"""Tests for the markdown summary generator."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.evaluator import evaluate_async
from src.model import ModelOutput, ModelProvider
from src.summary_md import MAX_EXAMPLES_PER_SECTION, render_summary_md, write_summary_md
from src.utils import Config, Example

REPO_ROOT = Path(__file__).resolve().parent.parent


def _clone(cfg: Config, **overrides: object) -> Config:
    from dataclasses import replace

    return replace(cfg, **overrides)


class _ScriptedProvider(ModelProvider):
    """Plays a pre-canned list of outputs in order. Wraps around on exhaustion."""

    def __init__(self, outputs: list[ModelOutput]) -> None:
        self._outputs = outputs
        self._i = 0

    def call(self, system_prompt: str, text: str) -> ModelOutput:
        out = self._outputs[self._i % len(self._outputs)]
        self._i += 1
        return out

    async def acall(self, system_prompt: str, text: str) -> ModelOutput:
        return self.call(system_prompt, text)


def _run(examples: list[Example], outputs: list[ModelOutput], *, threshold: float = 0.5):
    cfg = Config.from_file(REPO_ROOT / "config.yaml")
    cfg = _clone(cfg, verbose=False, confidence_threshold=threshold)
    provider = _ScriptedProvider(outputs)
    return (
        asyncio.run(evaluate_async(examples, provider=provider, system_prompt="s", config=cfg)),
        cfg,
    )


def _out(category: str, confidence: float, verdict: str = "ALLOW") -> ModelOutput:
    return ModelOutput(
        verdict=verdict,
        category=category,
        confidence=confidence,
        reason=f"because {category}",
        raw="",
        parse_ok=True,
        prompt_tokens=10,
        completion_tokens=5,
    )


def test_summary_contains_headline_and_cost() -> None:
    examples = [Example(text="hi", label="ok")] * 3
    report, _cfg = _run(examples, [_out("ok", 0.9)])
    md = render_summary_md(report, dataset_path="dataset.jsonl")

    assert "# Moderation run — summary" in md
    assert "## Headline" in md
    assert "## Cost" in md
    assert "## Metrics" in md
    # Per-million projection must be present for business use.
    assert "1 000 000 messages" in md


def test_summary_lists_errors_section() -> None:
    # One correct + one wrong so we can verify both paths.
    examples = [
        Example(text="hello friend", label="ok"),
        Example(text="a totally wrong one", label="spam"),
    ]
    report, _cfg = _run(examples, [_out("ok", 0.9), _out("ok", 0.9)])
    md = render_summary_md(report, dataset_path="dataset.jsonl")

    assert "## Errors" in md
    # The wrong prediction's text should appear verbatim in the errors table.
    assert "a totally wrong one" in md
    # And the correct one should NOT be listed as an error.
    errors_section = md.split("## Errors", 1)[1].split("## Low-confidence", 1)[0]
    assert "hello friend" not in errors_section


def test_summary_lists_review_section_when_low_confidence() -> None:
    examples = [Example(text="maybe spam", label="ok")] * 2
    report, _cfg = _run(examples, [_out("ok", 0.2)], threshold=0.5)
    md = render_summary_md(report, dataset_path="dataset.jsonl")

    assert "## Low-confidence (REVIEW queue)" in md
    # Review table should include the text and REVIEW as the verdict column.
    review_section = md.split("## Low-confidence", 1)[1]
    assert "maybe spam" in review_section
    assert "REVIEW" in review_section


def test_summary_caps_extremely_long_sections() -> None:
    n = MAX_EXAMPLES_PER_SECTION + 20
    examples = [Example(text=f"msg {i}", label="spam") for i in range(n)]
    # All predicted as "ok" → all errors.
    report, _cfg = _run(examples, [_out("ok", 0.9)])
    md = render_summary_md(report, dataset_path="dataset.jsonl")

    # Must mention the truncation so readers know to consult the JSON.
    assert "Showing first" in md
    # Must not blow past the cap even when thousands of errors exist.
    errors_section = md.split("## Errors", 1)[1].split("## Low-confidence", 1)[0]
    body_rows = [
        ln
        for ln in errors_section.splitlines()
        if ln.startswith("| ") and not ln.startswith("|---") and "Expected" not in ln
    ]
    assert len(body_rows) <= MAX_EXAMPLES_PER_SECTION


def test_summary_escapes_pipes_in_message_text() -> None:
    """A ``|`` inside user text must not break the markdown table."""
    examples = [Example(text="hello | world | extra", label="spam")]
    report, _cfg = _run(examples, [_out("ok", 0.9)])
    md = render_summary_md(report, dataset_path="dataset.jsonl")

    # Expect the pipe to be escaped somewhere in the table rows.
    assert "hello \\| world \\| extra" in md


def test_write_summary_md_writes_to_disk(tmp_path: Path) -> None:
    examples = [Example(text="hi", label="ok")]
    report, _cfg = _run(examples, [_out("ok", 0.9)])
    dest = tmp_path / "reports" / "summary.md"

    out = write_summary_md(dest, report, dataset_path="dataset.jsonl")
    assert out == dest
    assert dest.exists()
    content = dest.read_text(encoding="utf-8")
    assert content.startswith("# Moderation run — summary")

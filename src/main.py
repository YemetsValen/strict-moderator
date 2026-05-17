"""Command-line entry point: ``python -m src.main --config config.yaml``.

By default the CLI now runs the **async** pipeline so multi-example runs
finish quickly even against rate-limited providers. Pass ``--sync`` to fall
back to the historical sequential path (handy for debugging or when an
async client misbehaves).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from src.cost import format_cost
from src.evaluator import EvaluationReport, evaluate, evaluate_async, report_to_dict
from src.model import build_provider
from src.summary_md import write_summary_md
from src.utils import Config, configure_logging, load_jsonl, write_json

log = logging.getLogger("moderator")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="moderator",
        description="Run the moderation evaluation defined by config.yaml.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override config.dataset_path for this run only.",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["openai", "anthropic", "mock"],
        help="Override config.model_type for this run only.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override config.output_path for this run only.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Max in-flight requests when using async mode. "
            "Defaults to config.concurrency (typically 8)."
        ),
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run sequentially (skip the async fan-out). Mostly for debugging.",
    )
    return parser.parse_args(argv)


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if not (args.dataset or args.model_type or args.output or args.concurrency is not None):
        return config
    return Config(
        dataset_path=Path(args.dataset) if args.dataset else config.dataset_path,
        output_path=Path(args.output) if args.output else config.output_path,
        prompt_path=config.prompt_path,
        model_type=args.model_type or config.model_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        request_timeout_seconds=config.request_timeout_seconds,
        base_url=config.base_url,
        concurrency=args.concurrency if args.concurrency is not None else config.concurrency,
        confidence_threshold=config.confidence_threshold,
        labels=list(config.labels),
        block_labels=list(config.block_labels),
        metrics=list(config.metrics),
        metric_averages=list(config.metric_averages),
        log_level=config.log_level,
        verbose=config.verbose,
    )


def run(config: Config, *, use_async: bool = True) -> EvaluationReport:
    examples = load_jsonl(config.dataset_path)
    log.info("loaded %d example(s) from %s", len(examples), config.dataset_path)
    system_prompt = config.prompt_path.read_text(encoding="utf-8")
    provider = build_provider(config)
    log.info(
        "model_type=%s model_name=%s mode=%s concurrency=%d",
        config.model_type,
        config.model_name,
        "async" if use_async else "sync",
        config.concurrency,
    )
    if use_async:
        return asyncio.run(
            evaluate_async(
                examples,
                provider=provider,
                system_prompt=system_prompt,
                config=config,
            )
        )
    return evaluate(examples, provider=provider, system_prompt=system_prompt, config=config)


def _print_summary(report: EvaluationReport) -> None:
    s = report.summary
    print()
    print("=" * 60)
    print(f"  Examples            : {s['n_examples']}")
    print(f"  Category accuracy   : {s['category_correct']}/{s['n_examples']}")
    print(f"  Verdict accuracy    : {s['verdict_accuracy']:.3f}")
    print(f"  Block recall        : {s['block_recall']:.3f}")
    print(f"  Review queue        : {s['review_count']} ({s['review_rate'] * 100:.1f}%)")
    print(f"  Parse failures      : {s['parse_failures']}")
    print(f"  Duration (sec)      : {s['duration_seconds']}")
    print(f"  Prompt tokens       : {s['total_prompt_tokens']}")
    print(f"  Completion tokens   : {s['total_completion_tokens']}")
    print(f"  Total tokens        : {s['total_tokens']}")
    print(f"  Estimated cost      : {format_cost(s['estimated_cost_usd'])}")
    print(f"  Cost / 1k messages  : {format_cost(s['cost_per_1k_messages_usd'])}")
    print("-" * 60)
    for name, value in report.metrics.items():
        if isinstance(value, float):
            print(f"  {name:<20}: {value:.3f}")
            continue
        per_class: dict[str, float] = value.get("per_class", {})  # type: ignore[assignment]
        averages: dict[str, Any] = {k: v for k, v in value.items() if k != "per_class"}
        print(f"  {name}:")
        for label, score in per_class.items():
            print(f"      {label:<24}: {score:.3f}")
        for avg, score in averages.items():
            print(f"      [{avg}]{' ' * (24 - len(avg) - 2)}: {score:.3f}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Config.from_file(args.config)
    config = _apply_overrides(config, args)
    configure_logging(config.log_level)

    report = run(config, use_async=not args.sync)
    _print_summary(report)
    write_json(config.output_path, report_to_dict(report))
    log.info("report written to %s", config.output_path)

    # Per-run ``summary.md`` lives next to the JSON so the two files stay in
    # sync with the same timestamp and can be opened side-by-side.
    summary_path = config.output_path.with_name("summary.md")
    write_summary_md(summary_path, report, dataset_path=config.dataset_path)
    log.info("summary written to %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

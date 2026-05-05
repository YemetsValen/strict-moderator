"""Command-line entry point: ``python -m src.main --config config.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.evaluator import EvaluationReport, evaluate, report_to_dict
from src.model import build_provider
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
    return parser.parse_args(argv)


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if not (args.dataset or args.model_type or args.output):
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
        labels=list(config.labels),
        block_labels=list(config.block_labels),
        metrics=list(config.metrics),
        metric_averages=list(config.metric_averages),
        log_level=config.log_level,
        verbose=config.verbose,
    )


def run(config: Config) -> EvaluationReport:
    examples = load_jsonl(config.dataset_path)
    log.info("loaded %d example(s) from %s", len(examples), config.dataset_path)
    system_prompt = config.prompt_path.read_text(encoding="utf-8")
    provider = build_provider(config)
    log.info("model_type=%s model_name=%s", config.model_type, config.model_name)
    return evaluate(examples, provider=provider, system_prompt=system_prompt, config=config)


def _print_summary(report: EvaluationReport) -> None:
    s = report.summary
    print()
    print("=" * 60)
    print(f"  Examples            : {s['n_examples']}")
    print(f"  Category accuracy   : {s['category_correct']}/{s['n_examples']}")
    print(f"  Verdict accuracy    : {s['verdict_accuracy']:.3f}")
    print(f"  Block recall        : {s['block_recall']:.3f}")
    print(f"  Parse failures      : {s['parse_failures']}")
    print(f"  Duration (sec)      : {s['duration_seconds']}")
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

    report = run(config)
    _print_summary(report)
    write_json(config.output_path, report_to_dict(report))
    log.info("report written to %s", config.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

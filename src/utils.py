"""Shared helpers: config loading, JSONL I/O, robust JSON-from-LLM parsing."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Typed view over config.yaml.

    Frozen so a stray mutation doesn't silently change the run halfway through.
    """

    dataset_path: Path
    output_path: Path
    prompt_path: Path

    model_type: str
    model_name: str
    temperature: float
    max_tokens: int
    request_timeout_seconds: float

    labels: list[str]
    block_labels: list[str]

    metrics: list[str]
    metric_averages: list[str]

    log_level: str
    verbose: bool

    @classmethod
    def from_file(cls, path: Path | str) -> Config:
        with Path(path).open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        try:
            return cls(
                dataset_path=Path(raw["dataset_path"]),
                output_path=Path(raw["output_path"]),
                prompt_path=Path(raw["prompt_path"]),
                model_type=str(raw.get("model_type", "mock")),
                model_name=str(raw.get("model_name", "gpt-4o-mini")),
                temperature=float(raw.get("temperature", 0.0)),
                max_tokens=int(raw.get("max_tokens", 200)),
                request_timeout_seconds=float(raw.get("request_timeout_seconds", 30)),
                labels=list(raw["labels"]),
                block_labels=list(raw["block_labels"]),
                metrics=list(raw.get("metrics", ["accuracy"])),
                metric_averages=list(raw.get("metric_averages", ["macro"])),
                log_level=str(raw.get("log_level", "INFO")),
                verbose=bool(raw.get("verbose", True)),
            )
        except KeyError as e:
            raise ValueError(f"config.yaml is missing required key: {e}") from e


# ---------- dataset I/O ----------
@dataclass(frozen=True)
class Example:
    text: str
    label: str


def load_jsonl(path: Path | str) -> list[Example]:
    """Read a JSONL dataset. Skips blank lines, raises on malformed rows."""
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {e}") from e
            if "text" not in obj or "label" not in obj:
                raise ValueError(f"{path}:{lineno} is missing 'text' or 'label': {obj!r}")
            examples.append(Example(text=str(obj["text"]), label=str(obj["label"])))
    return examples


def write_json(path: Path | str, payload: Any) -> None:
    """Serialize `payload` to ``path`` (creating parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


# ---------- robust LLM JSON parsing ----------
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of a JSON object from a noisy model reply.

    Strategy:
      1. Try ``json.loads`` on the whole reply (happy path — our prompt
         tells the model to output exactly one JSON object).
      2. Strip common markdown fences and retry.
      3. Search for the first ``{...}`` substring and try to parse it.

    Returns ``None`` if all attempts fail; callers fall back to a default.
    """
    if not text:
        return None

    candidates: Iterable[str] = (text, text.strip())
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    fenced = _strip_code_fences(text)
    if fenced != text:
        try:
            obj = json.loads(fenced)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        # ```json\n{...}\n```  →  {...}
        stripped = stripped.lstrip("`")
        # drop the optional language tag on the first line
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
    return stripped.strip()


# ---------- logging ----------
def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

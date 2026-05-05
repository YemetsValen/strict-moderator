"""Robust JSON parsing and JSONL loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import extract_first_json_object, load_jsonl


def test_clean_json_object() -> None:
    obj = extract_first_json_object('{"verdict":"ALLOW","category":"ok","confidence":0.9}')
    assert obj == {"verdict": "ALLOW", "category": "ok", "confidence": 0.9}


def test_strips_markdown_code_fences() -> None:
    raw = '```json\n{"verdict":"BLOCK","category":"spam","confidence":0.8}\n```'
    obj = extract_first_json_object(raw)
    assert obj is not None
    assert obj["verdict"] == "BLOCK"


def test_recovers_from_extra_prose() -> None:
    raw = (
        'Sure! Here is your answer:\n{"verdict":"BLOCK","category":"spam","confidence":0.7}\nthanks'
    )
    obj = extract_first_json_object(raw)
    assert obj is not None
    assert obj["category"] == "spam"


def test_returns_none_on_garbage() -> None:
    assert extract_first_json_object("(no JSON here)") is None
    assert extract_first_json_object("") is None


def test_load_jsonl_rejects_missing_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"text":"hi"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(p)


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text(
        '{"text":"a","label":"ok"}\n\n  \n{"text":"b","label":"spam"}\n',
        encoding="utf-8",
    )
    rows = load_jsonl(p)
    assert [r.label for r in rows] == ["ok", "spam"]

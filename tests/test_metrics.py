"""Metric correctness — checked against hand-calculated expectations."""

from __future__ import annotations

import math

from src.metrics import accuracy, average, compute_all, f1, precision, recall

LABELS = ["a", "b", "c"]


def test_accuracy_simple() -> None:
    y_true = ["a", "a", "b", "c"]
    y_pred = ["a", "b", "b", "c"]
    assert accuracy(y_true, y_pred, LABELS) == 0.75


def test_accuracy_handles_empty_dataset() -> None:
    assert accuracy([], [], LABELS) == 0.0


def test_precision_recall_f1_per_class() -> None:
    # Confusion (truth in rows, pred in cols):
    #         a  b  c
    #     a   2  1  0
    #     b   0  1  1
    #     c   1  0  2
    y_true = ["a", "a", "a", "b", "b", "c", "c", "c"]
    y_pred = ["a", "a", "b", "b", "c", "a", "c", "c"]

    p = precision(y_true, y_pred, LABELS)
    r = recall(y_true, y_pred, LABELS)
    f = f1(y_true, y_pred, LABELS)

    # a: tp=2, fp=1 (one c→a), fn=1 (a→b)
    assert math.isclose(p["a"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(r["a"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(f["a"], 2 / 3, rel_tol=1e-6)
    # b: tp=1, fp=1 (a→b), fn=1 (b→c)
    assert math.isclose(p["b"], 0.5, rel_tol=1e-6)
    assert math.isclose(r["b"], 0.5, rel_tol=1e-6)
    assert math.isclose(f["b"], 0.5, rel_tol=1e-6)
    # c: tp=2, fp=1 (b→c), fn=1 (c→a)
    assert math.isclose(p["c"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(r["c"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(f["c"], 2 / 3, rel_tol=1e-6)


def test_macro_average() -> None:
    per_class = {"a": 1.0, "b": 0.5, "c": 0.0}
    assert math.isclose(average(per_class, method="macro"), 0.5, rel_tol=1e-6)


def test_weighted_average_uses_supports() -> None:
    per_class = {"a": 1.0, "b": 0.0}
    supports = {"a": 9, "b": 1}
    # 9*1.0 + 1*0.0 = 9 / 10 = 0.9
    assert math.isclose(average(per_class, method="weighted", supports=supports), 0.9, rel_tol=1e-6)


def test_micro_collapses_in_multiclass() -> None:
    # In single-label multi-class evaluation, micro-F1 == accuracy.
    y_true = ["a", "b", "c", "a"]
    y_pred = ["a", "b", "c", "b"]
    micro = average(
        dict.fromkeys(LABELS, 0),  # noqa: E741 - per_class shape only, values unused for micro
        method="micro",
        micro_inputs=(y_true, y_pred, LABELS),
    )
    assert math.isclose(micro, 0.75, rel_tol=1e-6)


def test_compute_all_registers_every_metric() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    out = compute_all(
        y_true,
        y_pred,
        labels=LABELS,
        metric_names=["accuracy", "precision", "recall", "f1"],
        averages=["macro", "micro"],
    )
    assert out["accuracy"] == 0.75
    for name in ("precision", "recall", "f1"):
        bundle = out[name]
        assert isinstance(bundle, dict)
        assert "per_class" in bundle
        assert "macro" in bundle
        assert "micro" in bundle

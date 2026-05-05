"""Classification metrics — implemented from scratch (no sklearn).

Exposed metrics:

- :func:`accuracy`  → single float (correct / total)
- :func:`precision`, :func:`recall`, :func:`f1`  → ``dict[label, float]``

All three per-class metrics also support averaging via :func:`average`,
which understands ``"macro"``, ``"micro"`` and ``"weighted"``.

Adding a new metric:

1. Implement a function with signature
   ``(y_true, y_pred, labels) -> float | dict[str, float]``.
2. Register it in :data:`METRIC_REGISTRY`.
3. List its name under ``metrics:`` in ``config.yaml`` — done.

The orchestration layer never special-cases metric names, so the registry
is the single seam where flexibility comes from.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Literal

Average = Literal["macro", "micro", "weighted"]
PerClass = dict[str, float]
MetricFn = Callable[[list[str], list[str], list[str]], "PerClass | float"]


# ---------- low-level counts ----------
def _validate(y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}"
        )
    if not labels:
        raise ValueError("labels list cannot be empty")


def _confusion_counts(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, dict[str, int]]:
    """Return ``{label: {tp, fp, fn, support}}`` for each label.

    ``support`` is the number of true examples for that class — needed for
    weighted averages.
    """
    _validate(y_true, y_pred, labels)
    counts = {lab: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lab in labels}
    for t, p in zip(y_true, y_pred, strict=False):
        if t in counts:
            counts[t]["support"] += 1
        if t == p and t in counts:
            counts[t]["tp"] += 1
        else:
            if p in counts:
                counts[p]["fp"] += 1
            if t in counts:
                counts[t]["fn"] += 1
    return counts


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


# ---------- metrics ----------
def accuracy(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    _validate(y_true, y_pred, labels)
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p)
    return correct / len(y_true)


def precision(y_true: list[str], y_pred: list[str], labels: list[str]) -> PerClass:
    counts = _confusion_counts(y_true, y_pred, labels)
    return {lab: _safe_div(c["tp"], c["tp"] + c["fp"]) for lab, c in counts.items()}


def recall(y_true: list[str], y_pred: list[str], labels: list[str]) -> PerClass:
    counts = _confusion_counts(y_true, y_pred, labels)
    return {lab: _safe_div(c["tp"], c["tp"] + c["fn"]) for lab, c in counts.items()}


def f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> PerClass:
    p = precision(y_true, y_pred, labels)
    r = recall(y_true, y_pred, labels)
    return {lab: _safe_div(2 * p[lab] * r[lab], p[lab] + r[lab]) for lab in labels}


# ---------- averaging ----------
def average(
    per_class: Mapping[str, float],
    *,
    method: Average,
    supports: Mapping[str, int] | None = None,
    micro_inputs: tuple[list[str], list[str], list[str]] | None = None,
) -> float:
    """Aggregate per-class scores into one number.

    - ``macro``    — unweighted mean across classes
    - ``weighted`` — mean weighted by class support (``supports`` required)
    - ``micro``    — recompute from global TP/FP/FN counts (``micro_inputs``
                     required); for accuracy/precision/recall/f1 this collapses
                     to the same number as accuracy in the multi-class case.
    """
    if not per_class:
        return 0.0
    if method == "macro":
        return sum(per_class.values()) / len(per_class)
    if method == "weighted":
        if not supports:
            raise ValueError("weighted average requires class supports")
        total = sum(supports.values())
        if total == 0:
            return 0.0
        return sum(per_class[lab] * supports.get(lab, 0) for lab in per_class) / total
    if method == "micro":
        if not micro_inputs:
            raise ValueError("micro average requires (y_true, y_pred, labels)")
        y_true, y_pred, labels = micro_inputs
        counts = _confusion_counts(y_true, y_pred, labels)
        tp = sum(c["tp"] for c in counts.values())
        fp = sum(c["fp"] for c in counts.values())
        fn = sum(c["fn"] for c in counts.values())
        # micro-F1 == micro-precision == micro-recall in the multi-class
        # single-label case, but we still pick the formula that matches the
        # metric the caller asked for.
        if not (fp or fn):
            return _safe_div(tp, tp + fp + fn) if (tp + fp + fn) else 0.0
        # Default: micro-F1 (covers precision/recall too in multi-class).
        denom = 2 * tp + fp + fn
        return _safe_div(2 * tp, denom)
    raise ValueError(f"unknown average method: {method!r}")


# ---------- registry ----------
METRIC_REGISTRY: dict[str, MetricFn] = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}


def compute_all(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    metric_names: Iterable[str],
    averages: Iterable[Average],
) -> dict[str, dict[str, float] | float]:
    """Run every metric named in ``metric_names`` and aggregate per request.

    Output shape::

        {
          "accuracy": 0.83,
          "precision": {"per_class": {...}, "macro": 0.7, "micro": 0.83},
          "recall":    {"per_class": {...}, "macro": 0.6, "micro": 0.83},
          "f1":        {"per_class": {...}, "macro": 0.65, "micro": 0.83},
        }

    Unknown metric names raise — silent skips would hide config typos.
    """
    out: dict[str, dict[str, float] | float] = {}
    counts = _confusion_counts(y_true, y_pred, labels)
    supports = {lab: c["support"] for lab, c in counts.items()}

    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"unknown metric {name!r}; registered: {sorted(METRIC_REGISTRY)}")
        result = METRIC_REGISTRY[name](y_true, y_pred, labels)
        if isinstance(result, float):
            out[name] = result
            continue
        bundle: dict[str, float] = {"per_class": result}  # type: ignore[dict-item]
        for avg in averages:
            bundle[avg] = average(
                result,
                method=avg,
                supports=supports,
                micro_inputs=(y_true, y_pred, labels),
            )
        out[name] = bundle  # type: ignore[assignment]
    return out

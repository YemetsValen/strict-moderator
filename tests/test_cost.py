"""Unit tests for src.cost — pricing math + unknown-model handling."""

from __future__ import annotations

import pytest

from src.cost import PRICES, ModelPrice, estimate_cost, format_cost


def test_estimate_cost_uses_input_and_output_rates() -> None:
    cost = estimate_cost(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        model_name="gpt-4o-mini",
    )
    price = PRICES["gpt-4o-mini"]
    assert cost == pytest.approx(price.input_per_million + price.output_per_million)


def test_estimate_cost_scales_linearly_below_one_million() -> None:
    cost = estimate_cost(
        prompt_tokens=10_000,
        completion_tokens=2_000,
        model_name="gpt-4o-mini",
    )
    expected = (
        10_000 / 1_000_000 * PRICES["gpt-4o-mini"].input_per_million
        + 2_000 / 1_000_000 * PRICES["gpt-4o-mini"].output_per_million
    )
    assert cost == pytest.approx(expected)


def test_estimate_cost_returns_zero_for_zero_tokens() -> None:
    assert estimate_cost(prompt_tokens=0, completion_tokens=0, model_name="gpt-4o") == 0


def test_estimate_cost_returns_none_for_unknown_model() -> None:
    """A fresh model name shouldn't crash a CI run — it should just omit the dollar figure."""
    assert (
        estimate_cost(prompt_tokens=1, completion_tokens=1, model_name="unknown-llm-2099") is None
    )


def test_format_cost_renders_unknown_as_na() -> None:
    assert format_cost(None) == "n/a"


def test_format_cost_keeps_precision_for_tiny_amounts() -> None:
    """Mock-provider runs can have sub-cent costs; users still want to see the magnitude."""
    s = format_cost(0.000142)
    assert s.startswith("$")
    assert "0.000142" in s


def test_format_cost_strips_trailing_zeros() -> None:
    assert format_cost(0.5) == "$0.5"
    assert format_cost(1) == "$1"


def test_format_cost_zero() -> None:
    assert format_cost(0) == "$0.00"


def test_anthropic_sonnet_pricing_is_three_dollars_per_million_input() -> None:
    """Anchor test: pricing table updates should be deliberate, not silent."""
    assert PRICES["claude-sonnet-4-5"] == ModelPrice(3.00, 15.00)
    assert PRICES["claude-3-5-haiku-latest"] == ModelPrice(0.80, 4.00)

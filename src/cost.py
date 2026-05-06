"""Cost estimation from token counts.

The price table maps **model_name** to (input, output) USD per 1M tokens.
Prices are listed openly (not pulled from an API) so readers can sanity-check
the math; update the table when providers change pricing.

Usage
-----
    >>> from src.cost import estimate_cost, format_cost
    >>> estimate_cost(prompt_tokens=12_500, completion_tokens=4_200,
    ...               model_name="gpt-4o-mini")
    0.00439...

Unknown models return ``None`` rather than raising — a fresh model name from
config shouldn't break a CI run; the report will simply omit the dollar
figure and surface the raw token totals instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1,000,000 tokens, separately for prompt vs completion."""

    input_per_million: float
    output_per_million: float


# Prices in USD per 1,000,000 tokens. Sourced from each provider's public
# pricing page; rounded to the cent. Edit / add rows here when you start
# using a new model.
PRICES: dict[str, ModelPrice] = {
    # --- OpenAI ---
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4.1": ModelPrice(2.00, 8.00),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40),
    "o3-mini": ModelPrice(1.10, 4.40),
    # --- Anthropic ---
    # Both `*-latest` aliases and the dated names point at the same price
    # tier; we list the most common ones explicitly.
    "claude-3-5-haiku-latest": ModelPrice(0.80, 4.00),
    "claude-3-5-sonnet-latest": ModelPrice(3.00, 15.00),
    "claude-3-7-sonnet-latest": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-0": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-5": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-opus-4-1": ModelPrice(15.00, 75.00),
    # --- DeepSeek (OpenAI-compatible API via base_url) ---
    "deepseek-chat": ModelPrice(0.27, 1.10),
    "deepseek-reasoner": ModelPrice(0.55, 2.19),
}


def estimate_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str,
) -> float | None:
    """USD cost of a single call. ``None`` if the model isn't in :data:`PRICES`."""
    price = PRICES.get(model_name)
    if price is None:
        return None
    return (
        prompt_tokens / 1_000_000 * price.input_per_million
        + completion_tokens / 1_000_000 * price.output_per_million
    )


def format_cost(usd: float | None) -> str:
    """Format ``usd`` with enough precision to be useful for cheap models.

    For tiny figures (sub-cent), we still want the user to see the order of
    magnitude — ``$0.000142`` rather than ``$0.00``.
    """
    if usd is None:
        return "n/a"
    if usd == 0:
        return "$0.00"
    # Six decimals keeps mock-provider runs honest ($0.000000) while still
    # rendering large bills cleanly ($1234.567890 → trim trailing zeros).
    return f"${usd:.6f}".rstrip("0").rstrip(".") or "$0.00"

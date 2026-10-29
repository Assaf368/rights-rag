"""What a model call costs, from the token counts the API returns."""

# Sonnet 5.5 and Haiku 5.5 prices, $ per million tokens (input, output).
# Haiku's price applies to prompts of 100K tokens or fewer, as all ours are.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def cost(usage: dict) -> float:
    """usage: a saved Answer or a call record with model and token counts."""
    price_in, price_out = PRICES.get(usage["model"], (0, 0))
    return (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out) / 1e6

from pathlib import Path
from typing import Any

import yaml
import numpy as np


def load_prices(prices_path: str | Path = "prices.yaml") -> dict[str, dict]:
    with open(prices_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def compute_operational(rows: list[dict], prices: dict[str, dict]) -> dict[str, float]:
    """Compute latency percentiles and estimated cost from a list of result rows.

    rows: list of dicts with keys latency_ms, prompt_tokens, completion_tokens, model_label.
    prices: dict keyed by model_label with input_per_1m / output_per_1m fields.
    """
    if not rows:
        return {}

    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    # Estimate cost per 1,000 calls from average token counts × published prices
    model_label = rows[0].get("model_label", "")
    price = prices.get(model_label, {})
    in_price = price.get("input_per_1m", 0.0)
    out_price = price.get("output_per_1m", 0.0)

    avg_prompt_tokens = float(np.mean([r.get("prompt_tokens", 0) for r in rows]))
    avg_completion_tokens = float(np.mean([r.get("completion_tokens", 0) for r in rows]))

    cost_per_1k = (
        (avg_prompt_tokens * in_price / 1_000_000)
        + (avg_completion_tokens * out_price / 1_000_000)
    ) * 1000

    return {
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "cost_per_1k_usd": cost_per_1k,
    }

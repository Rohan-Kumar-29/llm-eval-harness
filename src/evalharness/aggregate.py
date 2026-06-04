import json
from pathlib import Path

import pandas as pd

from evalharness.config import Config
from evalharness.dataset import load_dataset
from evalharness.scorers.deterministic import compute_metrics
from evalharness.scorers.operational import compute_operational, load_prices

RESULTS_DIR = Path("results")


def _safe_load_json(val) -> dict | None:
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except Exception:
        return None


def build_summary(config: Config, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Join deterministic + operational metrics into one summary row per (model_label, prompt_file)."""
    prices = load_prices("prices.yaml")
    rows = []

    for (model_label, prompt_file), grp in raw_df.groupby(["model_label", "prompt_file"]):
        records = grp.to_dict("records")

        # Ensure parsed_json is a dict not a string
        for r in records:
            r["parsed_json"] = _safe_load_json(r.get("parsed_json"))
            if isinstance(r.get("gold"), str):
                r["gold"] = _safe_load_json(r["gold"])

        det = compute_metrics(records)
        ops = compute_operational(records, prices)

        row = {
            "model_label": model_label,
            "prompt_file": Path(prompt_file).name,
            "n": len(records),
            **det,
            **ops,
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "summary.parquet"
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_df.to_parquet(out_path, index=False)
    return summary_df


def load_summary() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "summary.parquet")


def load_raw() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "raw_runs.parquet")

from typing import Any, Optional


_FIELDS = ["order_id", "issue_type", "location", "sentiment", "due_date"]


def _normalise(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip().lower()


def score_record(pred: Optional[dict], gold: dict) -> dict[str, Any]:
    """Score one prediction against one gold record.

    Returns per-field match flags plus overall exact_match and is_valid.
    """
    if pred is None:
        return {
            "exact_match": False,
            "is_valid": False,
            **{f"match_{f}": False for f in _FIELDS},
        }

    field_matches = {}
    for f in _FIELDS:
        g = _normalise(gold.get(f))
        p = _normalise(pred.get(f))
        field_matches[f"match_{f}"] = (g == p)

    exact_match = all(field_matches.values())
    return {"exact_match": exact_match, "is_valid": True, **field_matches}


def compute_metrics(rows: list[dict]) -> dict[str, float]:
    """Aggregate per-row scores into dataset-level metrics.

    Each row must have keys: parsed_json, gold, is_valid.
    Returns: json_validity, exact_match_rate, per-field F1/precision/recall,
             and macro_f1.
    """
    n = len(rows)
    if n == 0:
        return {}

    json_validity = sum(1 for r in rows if r.get("is_valid")) / n

    record_scores = [score_record(r.get("parsed_json"), r["gold"]) for r in rows]

    exact_match_rate = sum(1 for s in record_scores if s["exact_match"]) / n

    metrics: dict[str, float] = {
        "json_validity": json_validity,
        "exact_match_rate": exact_match_rate,
        "n": float(n),
    }

    f1_scores = []
    for field in _FIELDS:
        key = f"match_{field}"
        gold_present = [r["gold"].get(field) is not None for r in rows]
        pred_present = [r.get("parsed_json") is not None and r["parsed_json"].get(field) is not None for r in rows]
        matches = [s[key] for s in record_scores]

        # A prediction is a true positive only when it is present and matches gold.
        # A present-but-wrong prediction is BOTH a false positive (wrong value emitted)
        # and a false negative (correct gold value missed) — this is the standard
        # treatment and avoids inflating precision on confidently-wrong guesses.
        tp = sum(1 for m, pp, gp in zip(matches, pred_present, gold_present) if m and pp and gp)
        fp = sum(1 for m, pp in zip(matches, pred_present) if pp and not m)
        fn = sum(1 for m, gp in zip(matches, gold_present) if gp and not m)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics[f"{field}_precision"] = precision
        metrics[f"{field}_recall"] = recall
        metrics[f"{field}_f1"] = f1
        f1_scores.append(f1)

    metrics["macro_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return metrics

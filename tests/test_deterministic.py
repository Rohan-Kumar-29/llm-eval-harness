import pytest
from evalharness.scorers.deterministic import score_record, compute_metrics


GOLD = {
    "order_id": "12345",
    "issue_type": "refund",
    "location": "Pune",
    "sentiment": "negative",
    "due_date": None,
}


def test_perfect_match():
    pred = {"order_id": "12345", "issue_type": "refund", "location": "Pune", "sentiment": "negative", "due_date": None}
    result = score_record(pred, GOLD)
    assert result["exact_match"] is True
    assert all(result[f"match_{f}"] for f in ["order_id", "issue_type", "location", "sentiment", "due_date"])


def test_one_field_wrong():
    pred = {"order_id": "12345", "issue_type": "delivery_delay", "location": "Pune", "sentiment": "negative", "due_date": None}
    result = score_record(pred, GOLD)
    assert result["exact_match"] is False
    assert result["match_issue_type"] is False
    assert result["match_order_id"] is True


def test_null_prediction_returns_all_false():
    result = score_record(None, GOLD)
    assert result["exact_match"] is False
    assert result["is_valid"] is False


def test_case_insensitive_match():
    pred = {"order_id": "12345", "issue_type": "REFUND", "location": "pune", "sentiment": "negative", "due_date": None}
    result = score_record(pred, GOLD)
    assert result["match_issue_type"] is True
    assert result["match_location"] is True


def test_compute_metrics_all_correct():
    rows = [
        {"parsed_json": {"order_id": "1", "issue_type": "refund", "location": None, "sentiment": "positive", "due_date": "2026-03-01"},
         "gold": {"order_id": "1", "issue_type": "refund", "location": None, "sentiment": "positive", "due_date": "2026-03-01"},
         "is_valid": True},
        {"parsed_json": {"order_id": "2", "issue_type": "account", "location": "Delhi", "sentiment": "neutral", "due_date": "2026-04-01"},
         "gold": {"order_id": "2", "issue_type": "account", "location": "Delhi", "sentiment": "neutral", "due_date": "2026-04-01"},
         "is_valid": True},
    ]
    metrics = compute_metrics(rows)
    assert metrics["exact_match_rate"] == 1.0
    assert metrics["json_validity"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_compute_metrics_all_invalid():
    rows = [
        {"parsed_json": None, "gold": GOLD, "is_valid": False},
        {"parsed_json": None, "gold": GOLD, "is_valid": False},
    ]
    metrics = compute_metrics(rows)
    assert metrics["json_validity"] == 0.0
    assert metrics["exact_match_rate"] == 0.0


def test_wrong_but_present_counts_as_fp_and_fn():
    """A present-but-wrong value must hurt BOTH precision and recall.

    issue_type is wrong on every row (predicted 'account', gold 'refund'):
    each row is a false positive (wrong value emitted) and a false negative
    (correct value missed), so precision = recall = f1 = 0 for that field.
    """
    rows = [
        {"parsed_json": {"order_id": "1", "issue_type": "account", "location": "Pune", "sentiment": "negative", "due_date": "2026-01-01"},
         "gold": {"order_id": "1", "issue_type": "refund", "location": "Pune", "sentiment": "negative", "due_date": "2026-01-01"},
         "is_valid": True},
        {"parsed_json": {"order_id": "2", "issue_type": "account", "location": "Delhi", "sentiment": "neutral", "due_date": "2026-02-01"},
         "gold": {"order_id": "2", "issue_type": "refund", "location": "Delhi", "sentiment": "neutral", "due_date": "2026-02-01"},
         "is_valid": True},
    ]
    metrics = compute_metrics(rows)
    assert metrics["issue_type_precision"] == 0.0
    assert metrics["issue_type_recall"] == 0.0
    assert metrics["issue_type_f1"] == 0.0
    # the other four fields are perfect
    assert metrics["order_id_f1"] == 1.0


def test_compute_metrics_empty():
    assert compute_metrics([]) == {}


def test_compute_metrics_partial():
    rows = [
        {"parsed_json": {"order_id": "1", "issue_type": "refund", "location": None, "sentiment": "positive", "due_date": None},
         "gold": {"order_id": "1", "issue_type": "refund", "location": None, "sentiment": "positive", "due_date": None},
         "is_valid": True},
        {"parsed_json": None, "gold": GOLD, "is_valid": False},
    ]
    metrics = compute_metrics(rows)
    assert metrics["json_validity"] == 0.5
    assert metrics["n"] == 2.0

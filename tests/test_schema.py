import pytest
from evalharness.schema import IssueType, Sentiment, TicketExtraction, parse_model_output


def test_valid_json_parses():
    raw = '{"order_id": "12345", "issue_type": "refund", "location": "Pune", "sentiment": "negative", "due_date": null}'
    obj, valid = parse_model_output(raw)
    assert valid is True
    assert obj.order_id == "12345"
    assert obj.issue_type == IssueType.refund
    assert obj.sentiment == Sentiment.negative
    assert obj.due_date is None


def test_fenced_json_strips_and_parses():
    raw = '```json\n{"order_id": null, "issue_type": "account", "location": null, "sentiment": "neutral", "due_date": null}\n```'
    obj, valid = parse_model_output(raw)
    assert valid is True
    assert obj.issue_type == IssueType.account


def test_fence_without_language_tag():
    raw = '```\n{"order_id": "999", "issue_type": "defective", "location": null, "sentiment": "negative", "due_date": null}\n```'
    obj, valid = parse_model_output(raw)
    assert valid is True
    assert obj.order_id == "999"


def test_garbage_text_returns_invalid():
    obj, valid = parse_model_output("Sorry, I cannot help with that.")
    assert valid is False
    assert obj is None


def test_empty_string_returns_invalid():
    obj, valid = parse_model_output("")
    assert valid is False


def test_invalid_enum_value_returns_invalid():
    raw = '{"order_id": null, "issue_type": "unknown_type", "location": null, "sentiment": "neutral", "due_date": null}'
    obj, valid = parse_model_output(raw)
    assert valid is False


def test_missing_required_field_returns_invalid():
    # issue_type and sentiment are required
    raw = '{"order_id": "123", "location": null, "due_date": null}'
    obj, valid = parse_model_output(raw)
    assert valid is False


def test_all_nulls_valid():
    raw = '{"order_id": null, "issue_type": "other", "location": null, "sentiment": "positive", "due_date": null}'
    obj, valid = parse_model_output(raw)
    assert valid is True
    assert obj.issue_type == IssueType.other
    assert obj.sentiment == Sentiment.positive


def test_pydantic_model_direct():
    obj = TicketExtraction(issue_type="delivery_delay", sentiment="negative")
    assert obj.issue_type == IssueType.delivery_delay
    assert obj.order_id is None

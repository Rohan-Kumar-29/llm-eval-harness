import json
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class IssueType(str, Enum):
    delivery_delay = "delivery_delay"
    refund = "refund"
    defective = "defective"
    account = "account"
    other = "other"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class TicketExtraction(BaseModel):
    order_id: Optional[str] = None
    issue_type: IssueType
    location: Optional[str] = None
    sentiment: Sentiment
    due_date: Optional[str] = None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_model_output(raw_text: str) -> tuple[Optional[TicketExtraction], bool]:
    """Strip markdown fences, parse JSON, validate against TicketExtraction.

    Returns (parsed_object, is_valid). is_valid=False on any parse/validation failure.
    """
    if not raw_text or not raw_text.strip():
        return None, False

    text = raw_text.strip()

    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, False

    try:
        obj = TicketExtraction.model_validate(data)
        return obj, True
    except Exception:
        return None, False

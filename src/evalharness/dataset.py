import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Example:
    id: str
    input: str
    gold: dict[str, Any]


def load_dataset(path: str | Path, max_examples: Optional[int] = None) -> list[Example]:
    """Load JSONL dataset into typed Example objects."""
    examples: list[Example] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(Example(
                id=record["id"],
                input=record["input"],
                gold=record["gold"],
            ))
            if max_examples and len(examples) >= max_examples:
                break
    return examples

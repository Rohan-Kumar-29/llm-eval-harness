import asyncio
import json
import re
from typing import Optional

from dotenv import load_dotenv
from scipy import stats

from evalharness.clients import generate

load_dotenv()

_RUBRIC = """You are an expert evaluator for a structured information extraction task.

You will be given:
- INPUT: the original customer support ticket
- GOLD: the correct extraction (ground truth)
- PREDICTION: what the model extracted

Score the PREDICTION on a scale of 1 to 5:
5 - Perfect: all fields match the gold exactly
4 - Good: one minor field is wrong or missing but core fields (issue_type, sentiment) are correct
3 - Acceptable: two fields wrong/missing, or one core field wrong
2 - Poor: multiple fields wrong or core fields incorrect
1 - Fail: completely wrong or unparseable output

Return ONLY a JSON object with exactly these keys:
{
  "score": <integer 1-5>,
  "justification": "<one sentence explaining the score>"
}

INPUT: {input}
GOLD: {gold}
PREDICTION: {prediction}
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_judge_response(text: str) -> Optional[dict]:
    if not text:
        return None
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
        if "score" in data and isinstance(data["score"], int):
            return data
    except Exception:
        pass
    return None


async def judge_single(
    judge_model_id: str,
    input_text: str,
    gold: dict,
    prediction: Optional[dict],
) -> dict:
    """Ask the judge model to score one prediction. Returns score + justification."""
    prompt = _RUBRIC.format(
        input=input_text,
        gold=json.dumps(gold, ensure_ascii=False),
        prediction=json.dumps(prediction, ensure_ascii=False) if prediction else "null",
    )
    result = await generate(
        model_id=judge_model_id,
        system_prompt=prompt,
        user_input="Please evaluate the prediction above.",
        temperature=0.0,
    )
    parsed = _parse_judge_response(result.text)
    if parsed:
        return {"judge_score": parsed["score"], "judge_justification": parsed.get("justification", ""), "judge_error": None}
    return {"judge_score": None, "judge_justification": None, "judge_error": result.error or "unparseable judge response"}


async def run_judge(
    judge_model_id: str,
    rows: list[dict],
    sample_size: int,
    concurrency: int = 4,
) -> list[dict]:
    """Run judge on a sample of rows. Returns list of {example_id, judge_score, ...}."""
    sample = rows[:sample_size]
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(row: dict) -> dict:
        async with semaphore:
            result = await judge_single(
                judge_model_id=judge_model_id,
                input_text=row["input"],
                gold=row["gold"],
                prediction=row.get("parsed_json"),
            )
        return {"example_id": row["example_id"], "model_label": row["model_label"], **result}

    tasks = [_bounded(row) for row in sample]
    return await asyncio.gather(*tasks)


def compute_judge_reliability(judge_scores: list[dict], human_quality: list[dict]) -> dict:
    """Compare judge scores against human scores for overlapping example IDs.

    Returns mean_absolute_error and spearman_correlation.
    """
    human_map = {h["id"]: h["human_score"] for h in human_quality}

    pairs = [
        (j["judge_score"], human_map[j["example_id"]])
        for j in judge_scores
        if j.get("judge_score") is not None and j["example_id"] in human_map
    ]

    if len(pairs) < 2:
        return {"judge_mae": None, "judge_spearman": None, "judge_reliability_n": len(pairs)}

    judge_vals = [p[0] for p in pairs]
    human_vals = [p[1] for p in pairs]

    mae = sum(abs(j - h) for j, h in pairs) / len(pairs)
    spearman, _ = stats.spearmanr(judge_vals, human_vals)

    return {
        "judge_mae": round(mae, 3),
        "judge_spearman": round(float(spearman), 3),
        "judge_reliability_n": len(pairs),
    }

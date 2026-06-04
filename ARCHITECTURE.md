# Architecture

How the harness is structured and why. For usage, see the [README](README.md).

## Data flow

```
config.yaml ──► dataset.py ──► runner.py ──► clients.py ──► litellm ──► providers
                                   │                                  (Gemini / Groq / OpenRouter)
                                   ▼
                         results/raw_runs.parquet
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
       deterministic.py     operational.py           judge.py
       (field F1, exact     (latency p50/p95,    (LLM-as-judge +
        match, validity)     cost per 1k)         reliability vs human)
              │                    │                     │
              └────────────────────┼─────────────────────┘
                                   ▼
                            aggregate.py
                                   │
                         results/summary.parquet
                                   │
                              report.py
                          ┌────────┴─────────┐
                          ▼                  ▼
                results/sample/*.png      REPORT.md
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Loads and validates `config.yaml` into a typed `Config` (Pydantic). Single source of truth for models, prompts, and run settings. |
| `schema.py` | `TicketExtraction` Pydantic model with `IssueType`/`Sentiment` enums. `parse_model_output()` strips code fences, parses JSON, and validates — returning `(obj, is_valid)`. |
| `dataset.py` | Reads `seed_dataset.jsonl` line-by-line into typed `Example` objects. |
| `clients.py` | The model gateway. `generate()` wraps `litellm.acompletion()`, captures wall-clock latency and token usage, retries only transient errors (rate-limit/timeout/5xx) with exponential backoff, and fails clearly when an API key is missing. Hard failures are returned as a result with `error` set so one bad call never crashes the run. |
| `cache.py` | `diskcache`-backed response cache keyed by `sha256(model + prompt + input + temperature)`. Re-runs and report regeneration cost zero API calls. |
| `runner.py` | Async orchestration of the model × prompt × example matrix, bounded by an `asyncio.Semaphore`. Writes `results/raw_runs.parquet`. |
| `scorers/deterministic.py` | Per-field precision/recall/F1, macro-F1, exact-match rate, and JSON-validity rate. A present-but-wrong field counts as both a false positive and a false negative. |
| `scorers/operational.py` | Latency p50/p95 and estimated cost per 1,000 calls from token counts × `prices.yaml`. |
| `scorers/judge.py` | LLM-as-judge: rubric-based 1–5 scoring by a model kept separate from the one under test. `compute_judge_reliability()` reports MAE and Spearman correlation against human labels; the judge sample prioritises human-labelled examples so reliability can actually be measured. |
| `aggregate.py` | Joins deterministic + operational metrics into one row per (model, prompt). Writes `results/summary.parquet`. |
| `report.py` | Generates four charts into `results/sample/` and writes `REPORT.md` with the summary table, charts, an auto recommendation, and the judge-reliability section. |
| `cli.py` | Entry point behind `python -m evalharness` with `run` / `report` / `all` subcommands. |

## Key design choices

| Choice | Reasoning |
|---|---|
| **litellm as the gateway** | One interface for every provider; swapping a model is a one-line `config.yaml` edit, no code change. |
| **Temperature 0.0** | Deterministic outputs for fair, reproducible comparison. |
| **diskcache on responses** | Re-runs are free and interrupted runs resume from cache. |
| **Separate judge model** | A model judging its own output inflates scores. |
| **Pydantic output schema** | Catches partial/malformed JSON that `json.loads` alone would accept. |
| **asyncio + Semaphore** | Concurrent calls that still respect free-tier rate limits. |
| **Retry only transient errors** | Bad keys / malformed requests fail fast instead of burning quota on four doomed attempts. |

## Tests

| Test file | Covers |
|---|---|
| `tests/test_schema.py` | Output parsing/validation: fenced JSON, garbage, invalid enums, missing fields. |
| `tests/test_deterministic.py` | F1/precision/recall, exact match, case-insensitivity, and the wrong-but-present FP+FN rule. |
| `tests/test_runner_cache.py` | Cache key determinism and that a cache hit short-circuits the API call. |

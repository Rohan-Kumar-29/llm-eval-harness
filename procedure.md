# Project Build Procedure & File Registry

This file tracks every file created in the project, its purpose, and which build phase it belongs to.
Updated after each phase.

---

## Phase 1 — Scaffold (`commit: 3c8a937`)
*Goal: Create the full repo skeleton with all folders, config files, and empty module stubs.*

| File | Purpose |
|---|---|
| `requirements.txt` | Lists all Python packages this project depends on (litellm, pydantic, pandas, etc.) |
| `.gitignore` | Tells git which files/folders to never commit (.env, .venv, cache, parquet outputs) |
| `.env.example` | Template showing which API keys are needed — users copy this to `.env` and fill in their keys |
| `LICENSE` | MIT open-source license for the project |
| `config.yaml` | Single source of truth for the experiment — which models to run, how many examples, concurrency, etc. |
| `prices.yaml` | Published $/1M token pricing per model — used to compute estimated cost per 1,000 API calls |
| `Makefile` | Shortcut commands: `make setup`, `make run`, `make test`, `make report`, etc. |
| `setup.py` | Makes the `src/evalharness` package importable as `python -m evalharness` |
| `src/evalharness/__init__.py` | Marks the folder as a Python package (required by Python) |
| `src/evalharness/__main__.py` | Entry point so `python -m evalharness` works — calls `cli.main()` |
| `src/evalharness/cli.py` | Command-line interface: `run`, `report`, `all` subcommands — **stub, filled in Phase 8** |
| `src/evalharness/config.py` | Loads and validates `config.yaml` using Pydantic — ✅ implemented in Phase 3 |
| `src/evalharness/schema.py` | Pydantic model defining what a valid ticket extraction looks like — ✅ implemented in Phase 3 |
| `src/evalharness/dataset.py` | Loads `seed_dataset.jsonl` into a list of Example objects — ✅ implemented in Phase 3 |
| `src/evalharness/clients.py` | Wraps litellm to call any LLM with retry/backoff — ✅ implemented in Phase 4 |
| `src/evalharness/cache.py` | Disk-based response cache so re-runs don't cost API calls — ✅ implemented in Phase 4 |
| `src/evalharness/runner.py` | Async engine that runs all model×prompt×example combinations — ✅ implemented in Phase 5 |
| `src/evalharness/aggregate.py` | Joins all scorer outputs into one summary table — ✅ implemented in Phase 7 |
| `src/evalharness/report.py` | Generates 4 charts and writes REPORT.md — ✅ implemented in Phase 7 |
| `src/evalharness/scorers/__init__.py` | Marks scorers as a sub-package |
| `src/evalharness/scorers/deterministic.py` | Computes field-level F1, exact match, JSON validity — ✅ implemented in Phase 6 |
| `src/evalharness/scorers/judge.py` | LLM-as-judge scorer with rubric + reliability check — ✅ implemented in Phase 6 |
| `src/evalharness/scorers/operational.py` | Computes latency p50/p95 and estimated cost — ✅ implemented in Phase 6 |
| `tests/test_schema.py` | Unit tests for schema parsing and validation — ✅ implemented in Phase 3 |
| `tests/test_deterministic.py` | Unit tests for F1/precision/recall scorer — ✅ implemented in Phase 6 |
| `tests/test_runner_cache.py` | Unit tests verifying cache short-circuits API calls — ✅ implemented in Phase 5 |
| `dashboard/app.py` | Optional Streamlit dashboard for exploring results — **stub, filled in Phase 10** |
| `prompts/extract_v1.txt` | First prompt template for the extraction task — ✅ implemented in Phase 2 |
| `prompts/extract_v2.txt` | Second prompt variant to compare against v1 — ✅ implemented in Phase 2 |
| `results/sample/.gitkeep` | Keeps the `results/sample/` folder tracked by git (empty folders aren't tracked otherwise) |

---

## Phase 2 — Data (`commit: 31a518f`)
*Goal: Author the curated dataset and prompt templates.*

| File | Purpose |
|---|---|
| `data/seed_dataset.jsonl` | 80 hand-curated customer-support ticket examples with gold labels (order_id, issue_type, sentiment, etc.) |
| `data/human_quality.jsonl` | 10 examples with human 1–5 quality scores — used to verify LLM judge reliability |
| `prompts/extract_v1.txt` | Prompt v1: direct instruction style for field extraction |
| `prompts/extract_v2.txt` | Prompt v2: chain-of-thought / few-shot style — compared against v1 |

---

## Phase 3 — Schema + Dataset + Config (`commit: 6b87f46`)
*Goal: Core data models and loaders that all other modules depend on.*

| File | Purpose | Status |
|---|---|---|
| `src/evalharness/schema.py` | Pydantic `TicketExtraction` model with `IssueType` and `Sentiment` enums; `parse_model_output()` strips fences and validates | ✅ Implemented |
| `src/evalharness/dataset.py` | `load_dataset()` reads JSONL line-by-line and returns typed `Example` dataclass objects | ✅ Implemented |
| `src/evalharness/config.py` | `load_config()` reads `config.yaml`, validates all fields with Pydantic, returns typed `Config` object | ✅ Implemented |
| `tests/test_schema.py` | 9 tests: valid JSON, fenced JSON, fence without tag, garbage input, empty string, invalid enum, missing required field, all-nulls, direct model | ✅ 9/9 passing |

---

## Phase 4 — Clients + Cache (`commit: cb146e3`)
*Goal: The model gateway and caching layer — all API calls go through here.*

| File | Purpose | Status |
|---|---|---|
| `src/evalharness/clients.py` | `async generate()` wraps litellm.acompletion(), captures wall-clock latency and token usage, tenacity retry with exponential backoff (max 4 attempts), clear error on missing API key | ✅ Implemented |
| `src/evalharness/cache.py` | `get_cached()` / `set_cached()` using diskcache under `.cache/`, keyed by sha256(model+prompt+input+temperature) | ✅ Implemented |

---

## Phase 5 — Runner (`commit: 5466c21`)
*Goal: The async orchestration engine that drives the full benchmark matrix.*

| File | Purpose | Status |
|---|---|---|
| `src/evalharness/runner.py` | Builds model×prompt×example×repeat matrix, async with `Semaphore(concurrency)`, tqdm progress bar, rich summary table, saves `results/raw_runs.parquet`. Smoke mode: 5 examples × 1 model × 1 prompt | ✅ Implemented |
| `tests/test_runner_cache.py` | 5 tests: cache key determinism, key differs by model/temperature, roundtrip set/get, cache miss returns None | ✅ 5/5 passing |

---

## Phase 6 — Scorers (`commit: b2e8ef0`)
*Goal: All three scoring dimensions.*

| File | Purpose | Status |
|---|---|---|
| `src/evalharness/scorers/deterministic.py` | `score_record()` per-field exact match; `compute_metrics()` aggregates json_validity, exact_match_rate, per-field P/R/F1, macro_f1 across all rows | ✅ Implemented |
| `src/evalharness/scorers/operational.py` | `compute_operational()` computes latency p50/p95 and estimated cost per 1,000 calls from avg token counts × prices.yaml published rates | ✅ Implemented |
| `src/evalharness/scorers/judge.py` | `judge_single()` / `run_judge()` sends (input, gold, prediction) to judge model with rubric, returns 1–5 score + justification. `compute_judge_reliability()` gives MAE + Spearman vs human scores | ✅ Implemented |
| `tests/test_deterministic.py` | 8 tests: perfect match, one wrong field, null pred, case-insensitive match, all correct metrics, all invalid, empty input, partial | ✅ 8/8 passing |

---

## Phase 7 — Aggregate + Report (`commit: 8a21ed7`)
*Goal: Turn raw results into a summary table, charts, and a written recommendation.*

| File | Purpose | Status |
|---|---|---|
| `src/evalharness/aggregate.py` | `build_summary()` joins deterministic + operational metrics into one row per (model_label, prompt_file), writes `results/summary.parquet` | ✅ Implemented |
| `src/evalharness/report.py` | Generates 4 charts: F1 vs cost scatter, latency p50/p95 bar, JSON validity bar, F1 by prompt grouped bar. `write_report()` writes `REPORT.md` with table, charts, auto recommendation, judge reliability section | ✅ Implemented |
| `REPORT.md` | Auto-generated output — created at runtime by `make report` | ✅ Template ready |

---

## Phase 8 — CLI + Makefile (`commit: pending`)
*Goal: Wire everything together behind a clean command-line interface.*

| File | Purpose |
|---|---|
| `src/evalharness/cli.py` | `run` / `report` / `all` subcommands with `--smoke` flag using argparse |

---

## Phase 9 — README (`commit: pending`)
*Goal: Professional documentation that makes the repo recruiter-ready.*

| File | Purpose |
|---|---|
| `README.md` | Full project docs: problem statement, architecture diagram, quickstart, results table, design choices, limitations |

---

## Phase 10 — Optional Extras (`commit: pending`)
*Goal: Dashboard and CI polish.*

| File | Purpose |
|---|---|
| `dashboard/app.py` | Streamlit app: sortable results table, charts, per-example drill-down |

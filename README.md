# LLM Evaluation & Benchmarking Harness

A reproducible command-line harness that benchmarks multiple LLMs on a **structured entity/slot extraction** task across **accuracy, latency, and cost**. Uses both deterministic metrics (field-level F1, exact match) and an **LLM-as-judge**, then produces charts and a data-backed recommendation report — all for ₹0 on free API tiers.

---

## Architecture

```
config.yaml
    │
    ▼
dataset.py ──► runner.py (async, semaphore, cache)
                    │
                    ▼  (model × prompt × example matrix)
              clients.py ──► litellm ──► Gemini / Groq / OpenRouter
                    │
                    ▼
            results/raw_runs.parquet
                    │
          ┌─────────┼──────────────┐
          ▼         ▼              ▼
  deterministic  operational    judge.py
    scorer        scorer       (LLM-as-judge)
          │         │              │
          └─────────┴──────────────┘
                    │
              aggregate.py
                    │
                    ▼
         results/summary.parquet
                    │
               report.py
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    results/*.png          REPORT.md
```

---

## Key Features

- **Multi-provider via litellm** — Gemini, Groq, OpenRouter, and local Ollama all through one interface; swap models with a one-line `config.yaml` edit
- **Deterministic scoring** — per-field precision, recall, and F1 across 5 extraction fields; exact-match rate; JSON validity rate
- **LLM-as-judge** — rubric-based 1–5 quality scores with one-line justifications; judge model is always separate from the model under test
- **Judge reliability check** — Spearman correlation and MAE between judge scores and human-authored quality labels
- **Operational metrics** — wall-clock latency p50/p95 per model; estimated cost per 1,000 calls from published pricing
- **Disk caching** — `diskcache` keyed by sha256(model + prompt + input + temperature); re-runs and report regeneration cost zero API calls
- **Reproducible config** — all experiment parameters in `config.yaml`; temperature pinned to 0.0 for determinism
- **Smoke mode** — `--smoke` flag runs 5 examples × 1 model for a 30-second sanity check

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- Free API keys from [Google AI Studio](https://aistudio.google.com) (Gemini), [Groq Console](https://console.groq.com), and [OpenRouter](https://openrouter.ai)

### 2. Setup

```bash
git clone https://github.com/Rohan-Kumar-29/llm-eval-harness.git
cd llm-eval-harness

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate        # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Configure API keys

```bash
cp .env.example .env
# Open .env and fill in your keys:
# GEMINI_API_KEY=...
# GROQ_API_KEY=...
# OPENROUTER_API_KEY=...
```

### 4. Run

```bash
# Quick sanity check (5 examples × 1 model, ~30 seconds)
python -m evalharness run --smoke

# Full benchmark (80 examples × 3 models × 2 prompts)
python -m evalharness run

# Generate charts + REPORT.md
python -m evalharness report

# Run everything in one command
python -m evalharness all

# Run unit tests
pytest tests/ -v
```

---

## Results

Results from a full run on 80 curated customer-support ticket examples across 3 models and 2 prompt variants.

### Summary Table

| Model | Prompt | n | Macro F1 | JSON Validity | Exact Match | Latency p50 (ms) | Latency p95 (ms) | Cost/1k (USD) |
|---|---|---|---|---|---|---|---|---|
| **llama-70b** | extract_v1.txt | 80 | **0.780** | 100.0% | 65.0% | 533 | 815 | $0.1588 |
| **llama-70b** | extract_v2.txt | 80 | **0.778** | 100.0% | 63.8% | 599 | 1029 | $0.2027 |
| gemini-flash | extract_v1.txt | 80 | 0.205 | 12.5% | 7.5% | 10562 | 10985 | $0.0122 |
| gemini-flash | extract_v2.txt | 80 | 0.085 | 6.2% | 3.8% | 10455 | 11051 | $0.0074 |
| llama-8b-free | extract_v1.txt | 80 | 0.000 | 0.0% | 0.0% | 9134 | 10178 | $0.00 |
| llama-8b-free | extract_v2.txt | 80 | 0.000 | 0.0% | 0.0% | 9106 | 9506 | $0.00 |

### Recommendation

**`llama-70b` with `extract_v1.txt`** is the clear winner:
- Highest macro F1 (**0.780**) and exact match rate (**65%**) among reliable models
- 100% JSON validity — never returns unparseable output
- Fastest latency: **533ms median**, well under 1 second
- Only model that worked reliably on free tiers in this run

> **Note on gemini-flash and llama-8b-free:** Both models hit free-tier rate limits during the run (70 and 80 errors respectively out of 80 calls). Their low scores reflect rate-limiting, not model quality — Gemini 2.5 Flash is a capable model on a paid tier. Re-run with reduced `concurrency` or at off-peak times for fair comparison.

### Prompt Variant Comparison

`extract_v1.txt` (direct instruction) and `extract_v2.txt` (chain-of-thought) performed nearly identically for llama-70b (F1: 0.780 vs 0.778), suggesting the direct prompt is sufficient and more token-efficient for this task.

---

## How Evaluation Works

### Deterministic Metrics
Each model output is parsed as JSON and validated against the `TicketExtraction` Pydantic schema. For each of the 5 fields (`order_id`, `issue_type`, `location`, `sentiment`, `due_date`):
- **Precision** — of predicted non-null values, fraction that matched gold
- **Recall** — of gold non-null values, fraction the model found
- **F1** — harmonic mean of precision and recall
- **Macro F1** — average F1 across all 5 fields — the primary accuracy signal
- **Exact match rate** — fraction of records where all 5 fields matched exactly

### Operational Metrics
- **Latency p50/p95** — measured with `time.perf_counter()` wall-clock per API call
- **Cost per 1,000 calls** — computed from `avg_prompt_tokens × input_price + avg_completion_tokens × output_price` using published pay-as-you-go rates from `prices.yaml`

### LLM-as-Judge
A separate judge model (Gemini Flash) scores each prediction on a 1–5 rubric:
- **5** — perfect, all fields match gold
- **4** — one minor field wrong, core fields correct
- **3** — two fields wrong or one core field wrong
- **2** — multiple fields wrong
- **1** — completely wrong or unparseable

Judge reliability is measured as Spearman correlation and MAE against 10 human-authored quality scores in `data/human_quality.jsonl`. The judge model is always kept separate from the model under test.

---

## Design Choices & Trade-offs

| Choice | Reasoning |
|---|---|
| **litellm as gateway** | One interface for all providers — swap models with a config change, no code changes |
| **Temperature = 0.0** | Deterministic outputs for fair, reproducible comparison |
| **diskcache for responses** | Re-runs are free; interrupted runs resume from cache |
| **Separate judge model** | Using the model under test as its own judge inflates scores |
| **Pydantic for output schema** | Strict validation catches partial/malformed JSON that `json.loads` alone would accept |
| **asyncio + Semaphore** | Concurrent calls respect free-tier rate limits without blocking |

---

## Known Limitations

- **Small seed dataset** — 80 examples in a single domain (customer support tickets). Results may not generalise to other domains or ticket types.
- **Free-tier rate limits** — Gemini and OpenRouter free models were heavily rate-limited in this run. Numbers for these models reflect availability, not true model quality.
- **Single domain** — All examples are customer-support tickets. A model that performs well here may not perform well on other extraction tasks.
- **Judge bias** — The LLM judge (Gemini Flash) may have stylistic preferences that differ from human annotators. Judge reliability check requires more human-scored examples for statistical significance.
- **Gold label quality** — Gold labels are hand-authored for this project. Edge cases and ambiguous tickets may have debatable correct answers.

---

## Roadmap

- [ ] Ingest a larger public dataset (e.g. HuggingFace `datasets`) for more statistically robust results
- [ ] Add CI smoke test via GitHub Actions on every push
- [ ] Streamlit dashboard for interactive per-example drill-down
- [ ] Support for structured output APIs (Gemini/OpenAI JSON mode) to reduce invalid JSON rates
- [ ] Multi-domain evaluation (e.g. medical notes, legal clauses)

---

## Cost Note

All development runs used **free API tiers** (Google AI Studio, Groq, OpenRouter free models). The total development cost was **₹0**. Cost figures in the results table use published pay-as-you-go pricing for comparison purposes only.

---

## Project Structure

```
llm-eval-harness/
├── config.yaml              # Experiment config — models, prompts, run settings
├── prices.yaml              # Published $/1M token pricing per model
├── data/
│   ├── seed_dataset.jsonl   # 80 curated labeled examples
│   └── human_quality.jsonl  # 10 human quality scores for judge calibration
├── prompts/
│   ├── extract_v1.txt       # Direct instruction prompt
│   └── extract_v2.txt       # Chain-of-thought prompt variant
├── src/evalharness/
│   ├── cli.py               # Entrypoint: run / report / all
│   ├── config.py            # Config loader + Pydantic validation
│   ├── schema.py            # TicketExtraction Pydantic model
│   ├── dataset.py           # JSONL dataset loader
│   ├── clients.py           # litellm gateway with retry/backoff
│   ├── cache.py             # diskcache response caching
│   ├── runner.py            # Async benchmark orchestration
│   ├── aggregate.py         # Summary table builder
│   ├── report.py            # Chart generation + REPORT.md writer
│   └── scorers/
│       ├── deterministic.py # Field F1, exact match, JSON validity
│       ├── operational.py   # Latency percentiles, cost estimation
│       └── judge.py         # LLM-as-judge + reliability check
├── tests/
│   ├── test_schema.py
│   ├── test_deterministic.py
│   └── test_runner_cache.py
└── dashboard/
    └── app.py               # (Optional) Streamlit results explorer
```

---

*Built by [Rohan Kumar](https://github.com/Rohan-Kumar-29) · MIT License*

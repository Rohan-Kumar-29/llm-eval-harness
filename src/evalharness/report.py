from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
SAMPLE_DIR = RESULTS_DIR / "sample"


def _savefig(name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


# ── Chart 1: Accuracy (macro_f1) vs Cost per 1k calls ────────────────────────

def chart_f1_vs_cost(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in df.iterrows():
        ax.scatter(row["cost_per_1k_usd"], row["macro_f1"], s=120, zorder=3)
        ax.annotate(
            f"{row['model_label']}\n({row['prompt_file']})",
            xy=(row["cost_per_1k_usd"], row["macro_f1"]),
            xytext=(6, 4), textcoords="offset points", fontsize=8,
        )
    ax.set_xlabel("Estimated Cost per 1,000 calls (USD)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Accuracy (Macro F1) vs Cost")
    ax.grid(True, linestyle="--", alpha=0.5)
    return _savefig("f1_vs_cost.png")


# ── Chart 2: Latency p50 / p95 per model ─────────────────────────────────────

def chart_latency(df: pd.DataFrame) -> Path:
    labels = [f"{r['model_label']}\n{r['prompt_file']}" for _, r in df.iterrows()]
    x = range(len(labels))
    p50 = df["latency_p50_ms"].tolist()
    p95 = df["latency_p95_ms"].tolist()

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 5))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], p50, width, label="p50")
    bars2 = ax.bar([i + width / 2 for i in x], p95, width, label="p95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency p50 / p95 per Model × Prompt")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return _savefig("latency.png")


# ── Chart 3: JSON validity rate per model ────────────────────────────────────

def chart_json_validity(df: pd.DataFrame) -> Path:
    labels = [f"{r['model_label']}\n{r['prompt_file']}" for _, r in df.iterrows()]
    values = (df["json_validity"] * 100).tolist()

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 5))
    bars = ax.bar(labels, values, color="steelblue")
    ax.set_ylim(0, 110)
    ax.set_ylabel("JSON Validity (%)")
    ax.set_title("JSON Validity Rate per Model × Prompt")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return _savefig("json_validity.png")


# ── Chart 4: macro_f1 by prompt variant (grouped bar) ────────────────────────

def chart_f1_by_prompt(df: pd.DataFrame) -> Path:
    pivot = df.pivot_table(index="model_label", columns="prompt_file", values="macro_f1")
    ax = pivot.plot(kind="bar", figsize=(max(7, len(pivot) * 1.5), 5), rot=15)
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 by Model and Prompt Variant")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Prompt", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return _savefig("f1_by_prompt.png")


# ── Auto-generated recommendation ────────────────────────────────────────────

def _recommend(df: pd.DataFrame) -> str:
    """Pick best macro_f1 among models with json_validity >= 0.5 (reliable output)."""
    reliable = df[df["json_validity"] >= 0.5]
    if reliable.empty:
        reliable = df  # fall back to all if none are reliable

    best = reliable.loc[reliable["macro_f1"].idxmax()]
    worst_cost = df.loc[df["cost_per_1k_usd"].idxmax()]

    lines = [
        f"**Recommended model: `{best['model_label']}` with prompt `{best['prompt_file']}`**",
        "",
        f"- Macro F1: **{best['macro_f1']:.3f}**",
        f"- Estimated cost per 1,000 calls: **${best['cost_per_1k_usd']:.4f}**",
        f"- Latency p50 / p95: **{best['latency_p50_ms']:.0f} ms / {best['latency_p95_ms']:.0f} ms**",
        f"- JSON validity: **{best['json_validity'] * 100:.1f}%**",
        "",
        f"This model achieves the highest extraction accuracy among models with reliable JSON output "
        f"(json_validity >= 50%). "
        f"The most expensive option (`{worst_cost['model_label']}`) costs "
        f"${worst_cost['cost_per_1k_usd']:.4f}/1k calls "
        f"{'with similar accuracy' if abs(worst_cost['macro_f1'] - best['macro_f1']) < 0.05 else 'without a proportional accuracy gain'}.",
    ]
    return "\n".join(lines)


# ── REPORT.md writer ─────────────────────────────────────────────────────────

def write_report(
    summary_df: pd.DataFrame,
    judge_reliability: dict | None = None,
    judge_scores_df: pd.DataFrame | None = None,
) -> Path:
    """Generate REPORT.md with summary table, charts, recommendation, and judge reliability."""

    chart_paths = {
        "f1_vs_cost": chart_f1_vs_cost(summary_df),
        "latency": chart_latency(summary_df),
        "json_validity": chart_json_validity(summary_df),
        "f1_by_prompt": chart_f1_by_prompt(summary_df),
    }

    recommendation = _recommend(summary_df)

    # Format summary table
    display_cols = [
        "model_label", "prompt_file", "n",
        "macro_f1", "json_validity", "exact_match_rate",
        "latency_p50_ms", "latency_p95_ms", "cost_per_1k_usd",
    ]
    available_cols = [c for c in display_cols if c in summary_df.columns]
    table_df = summary_df[available_cols].copy()

    # Round for readability
    for col in ["macro_f1", "json_validity", "exact_match_rate"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(3)
    for col in ["latency_p50_ms", "latency_p95_ms"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(1)
    if "cost_per_1k_usd" in table_df.columns:
        table_df["cost_per_1k_usd"] = table_df["cost_per_1k_usd"].round(5)

    md_table = table_df.to_markdown(index=False)

    # Judge reliability section
    if judge_reliability:
        mae = judge_reliability.get("judge_mae")
        spearman = judge_reliability.get("judge_spearman")
        n_rel = judge_reliability.get("judge_reliability_n", 0)
        judge_section = (
            f"## Judge Reliability\n\n"
            f"The LLM judge was calibrated against {n_rel} human-scored examples.\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Mean Absolute Error (vs human) | {mae if mae is not None else 'N/A'} |\n"
            f"| Spearman Correlation (vs human) | {spearman if spearman is not None else 'N/A'} |\n\n"
            f"> A Spearman correlation > 0.7 indicates strong judge–human agreement.\n"
        )
    else:
        judge_section = "## Judge Reliability\n\n*Judge reliability check not run (no human quality data available).*\n"

    report_md = f"""# LLM Evaluation Report

> Auto-generated by `evalharness report`. Do not edit manually — re-run `make report` to refresh.

---

## Summary Table

{md_table}

---

## Recommendation

{recommendation}

---

## Charts

### Accuracy (Macro F1) vs Estimated Cost
![F1 vs Cost](results/f1_vs_cost.png)

### Latency p50 / p95 per Model
![Latency](results/latency.png)

### JSON Validity Rate
![JSON Validity](results/json_validity.png)

### Macro F1 by Prompt Variant
![F1 by Prompt](results/f1_by_prompt.png)

---

{judge_section}

---

## Metrics Explained

| Metric | Description |
|---|---|
| `macro_f1` | Average F1 across all 5 extraction fields — the primary accuracy signal |
| `json_validity` | Fraction of responses that parsed as valid JSON matching the schema |
| `exact_match_rate` | Fraction of records where every field matched the gold label exactly |
| `latency_p50_ms` | Median response time in milliseconds |
| `latency_p95_ms` | 95th-percentile response time — reflects tail latency |
| `cost_per_1k_usd` | Estimated USD cost for 1,000 calls using published pay-as-you-go pricing |

---

## Cost Note

All development runs used **free API tiers** (Google AI Studio, Groq, OpenRouter).
Cost figures above use published pay-as-you-go pricing for comparison only — actual dev cost was ₹0.
"""

    report_path = Path("REPORT.md")
    report_path.write_text(report_md, encoding="utf-8")
    return report_path

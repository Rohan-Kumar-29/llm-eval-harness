import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm.asyncio import tqdm as atqdm
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

from evalharness.cache import get_cached, set_cached
from evalharness.clients import generate
from evalharness.config import Config
from evalharness.dataset import Example, load_dataset
from evalharness.schema import parse_model_output

load_dotenv()

RESULTS_DIR = Path("results")
console = Console()


def _build_prompt_text(prompt_template: str, user_input: str) -> str:
    """Substitute {input} placeholder in prompt template."""
    return prompt_template.replace("{input}", user_input)


def _load_prompt(prompt_path: str) -> str:
    return Path(prompt_path).read_text(encoding="utf-8")


async def _run_single(
    run_id: str,
    model_id: str,
    model_label: str,
    prompt_file: str,
    prompt_template: str,
    example: Example,
    temperature: float,
    use_cache: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run one (model, prompt, example) cell and return a result row."""
    system_prompt = _build_prompt_text(prompt_template, example.input)

    # Check cache first
    if use_cache:
        cached = get_cached(model_id, system_prompt, example.input, temperature)
        if cached is not None:
            return cached

    async with semaphore:
        result = await generate(
            model_id=model_id,
            system_prompt=system_prompt,
            user_input=example.input,
            temperature=temperature,
        )

    parsed_obj, is_valid = parse_model_output(result.text)
    parsed_json = parsed_obj.model_dump() if parsed_obj else None

    row = {
        "run_id": run_id,
        "model_label": model_label,
        "model_id": model_id,
        "prompt_file": prompt_file,
        "example_id": example.id,
        "input": example.input,
        "gold": example.gold,
        "output_text": result.text,
        "parsed_json": parsed_json,
        "is_valid": is_valid,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if use_cache and not result.error:
        set_cached(model_id, system_prompt, example.input, temperature, row)

    return row


async def run_benchmark(config: Config, smoke: bool = False) -> pd.DataFrame:
    """Run the full benchmark matrix and return a DataFrame of all results."""
    RESULTS_DIR.mkdir(exist_ok=True)

    examples = load_dataset(config.dataset_path, max_examples=config.run.max_examples)

    if smoke or config.run.smoke_test:
        examples = examples[:5]
        models = config.models[:1]
        prompts = config.prompts[:1]
        console.print("[yellow]Smoke mode: 5 examples × 1 model × 1 prompt[/yellow]")
    else:
        models = config.models
        prompts = config.prompts

    prompt_templates = {p: _load_prompt(p) for p in prompts}
    semaphore = asyncio.Semaphore(config.run.concurrency)
    run_id = str(uuid.uuid4())[:8]

    tasks = []
    for repeat in range(config.run.repeats):
        for model in models:
            for prompt_file in prompts:
                for example in examples:
                    tasks.append(_run_single(
                        run_id=run_id,
                        model_id=model.id,
                        model_label=model.label,
                        prompt_file=prompt_file,
                        prompt_template=prompt_templates[prompt_file],
                        example=example,
                        temperature=config.run.temperature,
                        use_cache=config.run.cache,
                        semaphore=semaphore,
                    ))

    total = len(tasks)
    console.print(f"[bold]Running {total} calls[/bold] "
                  f"({len(models)} models × {len(prompts)} prompts × {len(examples)} examples × {config.run.repeats} repeat(s))")

    rows = []
    for coro in atqdm(asyncio.as_completed(tasks), total=total, desc="Benchmarking"):
        row = await coro
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save raw results
    out_path = RESULTS_DIR / "raw_runs.parquet"
    df.to_parquet(out_path, index=False)
    console.print(f"[green]Saved {len(df)} rows → {out_path}[/green]")

    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    """Print a quick rich summary table after the run."""
    table = Table(title="Run Summary", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Prompt")
    table.add_column("N", justify="right")
    table.add_column("Valid JSON %", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")

    for (model_label, prompt_file), grp in df.groupby(["model_label", "prompt_file"]):
        n = len(grp)
        valid_pct = f"{grp['is_valid'].mean() * 100:.1f}%"
        errors = int(grp["error"].notna().sum())
        avg_lat = f"{grp['latency_ms'].mean():.0f}"
        table.add_row(model_label, Path(prompt_file).name, str(n), valid_pct, str(errors), avg_lat)

    console.print(table)

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()


def _run(args: argparse.Namespace) -> None:
    from evalharness.config import load_config
    from evalharness.runner import run_benchmark

    config = load_config("config.yaml")
    smoke = args.smoke or config.run.smoke_test

    if smoke:
        console.print("[yellow]-- smoke mode: 5 examples × 1 model × 1 prompt --[/yellow]")

    asyncio.run(run_benchmark(config, smoke=smoke))


def _report(args: argparse.Namespace) -> None:
    from evalharness.aggregate import build_summary, load_raw
    from evalharness.config import load_config
    from evalharness.dataset import load_dataset
    from evalharness.report import write_report
    from evalharness.scorers.judge import compute_judge_reliability, run_judge

    config = load_config("config.yaml")

    raw_path = Path("results/raw_runs.parquet")
    if not raw_path.exists():
        console.print("[red]results/raw_runs.parquet not found — run `python -m evalharness run` first.[/red]")
        sys.exit(1)

    console.print("[bold]Building summary table...[/bold]")
    raw_df = load_raw()
    summary_df = build_summary(config, raw_df)

    # Run LLM judge on a sample
    judge_reliability = None
    try:
        console.print(f"[bold]Running LLM judge (sample_size={config.run.judge_sample_size})...[/bold]")
        rows = raw_df.to_dict("records")
        for r in rows:
            if isinstance(r.get("gold"), str):
                r["gold"] = json.loads(r["gold"])
            if isinstance(r.get("parsed_json"), str):
                r["parsed_json"] = json.loads(r["parsed_json"])

        judge_scores = asyncio.run(run_judge(
            judge_model_id=config.judge_model.id,
            rows=rows,
            sample_size=config.run.judge_sample_size,
            concurrency=config.run.concurrency,
        ))

        human_quality_path = Path(config.human_quality_path)
        if human_quality_path.exists():
            import json as _json
            human_quality = [
                _json.loads(line)
                for line in human_quality_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            judge_reliability = compute_judge_reliability(judge_scores, human_quality)
            console.print(f"[green]Judge reliability: MAE={judge_reliability.get('judge_mae')}, "
                          f"Spearman={judge_reliability.get('judge_spearman')} "
                          f"(n={judge_reliability.get('judge_reliability_n')})[/green]")
    except Exception as exc:
        console.print(f"[yellow]Judge step skipped: {exc}[/yellow]")

    console.print("[bold]Generating charts and REPORT.md...[/bold]")
    report_path = write_report(summary_df, judge_reliability=judge_reliability)
    console.print(f"[green]Report written → {report_path}[/green]")


def _all(args: argparse.Namespace) -> None:
    _run(args)
    _report(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evalharness",
        description="LLM Evaluation & Benchmarking Harness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run the benchmark matrix and save raw results")
    p_run.add_argument("--smoke", action="store_true", help="Smoke test: 5 examples × 1 model × 1 prompt")
    p_run.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p_run.set_defaults(func=_run)

    # report
    p_report = sub.add_parser("report", help="Build charts and REPORT.md from existing results")
    p_report.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p_report.set_defaults(func=_report)

    # all
    p_all = sub.add_parser("all", help="Run benchmark then generate report")
    p_all.add_argument("--smoke", action="store_true", help="Smoke test mode")
    p_all.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p_all.set_defaults(func=_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

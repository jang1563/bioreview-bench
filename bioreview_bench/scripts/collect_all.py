"""Unified multi-source collection pipeline orchestrator.

Runs all enabled collectors sequentially and writes to separate per-source JSONL files.
Use this for full Phase 1 data collection (~5,000 articles).

Usage:
    # dry-run all sources (cost $0)
    uv run python -m bioreview_bench.scripts.collect_all --dry-run

    # full run with targets
    uv run python -m bioreview_bench.scripts.collect_all \\
        --elife 1500 --plos 1200 --f1000 1000 --nature 500

    # skip specific sources
    uv run python -m bioreview_bench.scripts.collect_all \\
        --elife 0 --plos 1200 --f1000 1000 --nature 0
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parents[2]


def _run_collector(
    module: str,
    max_articles: int,
    output: Path,
    extra_args: list[str],
    dry_run: bool,
    model: str,
) -> int:
    """Run a collector module as a subprocess and return exit code."""
    cmd = [
        sys.executable, "-m", module,
        "--max-articles", str(max_articles),
        "--output", str(output),
        "--model", model,
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(extra_args)

    console.print(f"\n[bold cyan]Running {module}[/bold cyan]")
    console.print(f"  command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


@click.command()
@click.option(
    "--elife",
    default=1500,
    show_default=True,
    help="Max eLife articles (0 = skip)",
)
@click.option(
    "--plos",
    default=1200,
    show_default=True,
    help="Max PLOS articles (0 = skip)",
)
@click.option(
    "--f1000",
    default=1000,
    show_default=True,
    help="Max F1000Research articles (0 = skip)",
)
@click.option(
    "--nature",
    default=500,
    show_default=True,
    help="Max Nature articles (0 = skip)",
)
@click.option(
    "--output-dir",
    default=None,
    help="Output directory for JSONL files (default: data/processed/)",
)
@click.option(
    "--model",
    default="claude-haiku-4-5-20251001",
    show_default=True,
    help="Anthropic model ID for concern extraction",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run all collectors in dry-run mode ($0 cost)",
)
def main(
    elife: int,
    plos: int,
    f1000: int,
    nature: int,
    output_dir: str | None,
    model: str,
    dry_run: bool,
) -> None:
    """Run all bioreview-bench data collectors sequentially.

    Produces one JSONL file per source in the output directory.
    Each collector can be run independently if only a subset is needed.
    """
    out_dir = Path(output_dir) if output_dir else ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]bioreview-bench Phase 1 Collection Orchestrator[/bold]")
    console.print(f"  eLife  : {elife or 'skip'}")
    console.print(f"  PLOS   : {plos or 'skip'}")
    console.print(f"  F1000  : {f1000 or 'skip'}")
    console.print(f"  Nature : {nature or 'skip'}")
    console.print(f"  dry-run: {dry_run}")
    console.print(f"  output : {out_dir}")

    results: dict[str, str] = {}

    # eLife
    if elife > 0:
        rc = _run_collector(
            module="bioreview_bench.scripts.collect_elife",
            max_articles=elife,
            output=out_dir / "elife_v1.jsonl",
            extra_args=[
                "--subjects", "genetics-genomics",
                "--subjects", "cell-biology",
                "--subjects", "neuroscience",
                "--subjects", "biochemistry",
                "--subjects", "developmental-biology",
                "--start-date", "2018-01-01",
            ],
            dry_run=dry_run,
            model=model,
        )
        results["elife"] = "ok" if rc == 0 else f"FAILED (rc={rc})"

    # PLOS
    if plos > 0:
        rc = _run_collector(
            module="bioreview_bench.scripts.collect_plos",
            max_articles=plos,
            output=out_dir / "plos_v1.jsonl",
            extra_args=["--start-date", "2019-01-01"],
            dry_run=dry_run,
            model=model,
        )
        results["plos"] = "ok" if rc == 0 else f"FAILED (rc={rc})"

    # F1000Research
    if f1000 > 0:
        rc = _run_collector(
            module="bioreview_bench.scripts.collect_f1000",
            max_articles=f1000,
            output=out_dir / "f1000_v1.jsonl",
            extra_args=["--start-date", "2013-01-01"],
            dry_run=dry_run,
            model=model,
        )
        results["f1000"] = "ok" if rc == 0 else f"FAILED (rc={rc})"

    # Nature
    if nature > 0:
        rc = _run_collector(
            module="bioreview_bench.scripts.collect_nature",
            max_articles=nature,
            output=out_dir / "nature_v1.jsonl",
            extra_args=["--start-date", "2022-01-01"],
            dry_run=dry_run,
            model=model,
        )
        results["nature"] = "ok" if rc == 0 else f"FAILED (rc={rc})"

    # Summary
    console.print()
    table = Table(title="Collection Results")
    table.add_column("Source")
    table.add_column("Status")
    for source, status in results.items():
        color = "green" if status == "ok" else "red"
        table.add_row(source, f"[{color}]{status}")
    console.print(table)

    if any("FAILED" in s for s in results.values()):
        console.print("\n[red]Some collectors failed. Check output above.[/red]")
        sys.exit(1)
    else:
        console.print("\n[green]All collectors completed successfully.[/green]")
        console.print(
            "\nNext step: run [cyan]python scripts/rebuild_splits.py[/cyan] "
            "to create multi-source train/val/test splits."
        )


if __name__ == "__main__":
    main()

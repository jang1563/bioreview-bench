"""Benchmark runner CLI (minimal v0.1 scaffold).

This command exists to keep the published entrypoint stable while
the full evaluation harness is being implemented.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option(
    "--predictions",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to predictions JSON file in leaderboard submission format",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate input format only, skip scoring",
)
def main(predictions: Path | None, dry_run: bool) -> None:
    """bioreview-bench evaluation runner (current: schema validation scaffold)."""
    if predictions is None:
        console.print("[red]Error:[/red] --predictions path is required.")
        sys.exit(2)

    try:
        payload = json.loads(predictions.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error:[/red] JSON parsing failed: {exc}")
        sys.exit(2)

    required = {"tool_name", "tool_version", "predictions"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        console.print(f"[red]Error:[/red] Missing required fields: {', '.join(missing)}")
        sys.exit(2)

    n_articles = len(payload.get("predictions", {}))
    console.print(f"[green]Input validation passed[/green] - articles={n_articles}, dry_run={dry_run}")
    console.print(
        "[yellow]Note:[/yellow] Full metric computation is not yet implemented in v0.1. "
        "Will be extended per EVALUATION_PROTOCOL.md."
    )

    # Keep behavior deterministic for CI callers.
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Generate split statistics and check docs against the canonical v3 splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from bioreview_bench.stats import check_documentation, render_markdown_summary, summarize_splits

console = Console()


@click.command()
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=Path("data/splits/v3"),
    show_default=True,
    help="Directory containing train.jsonl, val.jsonl, and test.jsonl.",
)
@click.option(
    "--output-json",
    type=click.Path(path_type=Path),
    default=Path("data/stats/v3_summary.json"),
    show_default=True,
    help="Where to write the JSON summary.",
)
@click.option(
    "--output-md",
    type=click.Path(path_type=Path),
    default=Path("data/stats/v3_summary.md"),
    show_default=True,
    help="Where to write the markdown summary.",
)
@click.option(
    "--check-docs/--no-check-docs",
    default=False,
    show_default=True,
    help="Validate README.md and DATASHEET.md against the generated summary.",
)
def main(
    splits_dir: Path,
    output_json: Path,
    output_md: Path,
    check_docs: bool,
) -> None:
    """Generate summary stats from the canonical split files."""
    summary = summarize_splits(splits_dir)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown_summary(summary), encoding="utf-8")

    console.print(f"Wrote JSON summary: {output_json}")
    console.print(f"Wrote markdown summary: {output_md}")

    if not check_docs:
        return

    errors = check_documentation(summary, [Path("README.md"), Path("DATASHEET.md")])
    if errors:
        console.print("[red]Documentation drift detected:[/red]")
        for err in errors:
            console.print(f"  - {err}")
        sys.exit(1)

    console.print("[green]README.md and DATASHEET.md match the split stats.[/green]")


if __name__ == "__main__":
    main()

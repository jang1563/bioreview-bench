"""CLI for running the baseline LLM reviewer.

Usage::

    uv run bioreview-baseline --split val --model claude-haiku-4-5-20251001
    uv run bioreview-baseline --split val --model gpt-4o-mini --provider openai -n 10
    uv run bioreview-baseline --split val --dry-run  # cost estimate only
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()

_DEFAULT_SPLITS_DIR = Path("data/splits/v3")


def _safe_model_name(model: str) -> str:
    """Convert model ID to filesystem-safe string."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model)[:50]


@click.command()
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="val",
    show_default=True,
    help="Dataset split to run on.",
)
@click.option(
    "--model", "-m",
    default="claude-haiku-4-5-20251001",
    show_default=True,
    help="Model identifier.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai"]),
    default="anthropic",
    show_default=True,
    help="LLM provider.",
)
@click.option(
    "--max-articles", "-n",
    type=int,
    default=None,
    help="Limit number of articles to process.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSONL path (default: tool_outputs/{model}_{split}.jsonl).",
)
@click.option(
    "--concurrency",
    type=int,
    default=5,
    show_default=True,
    help="Max concurrent API calls.",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    show_default=True,
    help="Resume from existing output file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Estimate cost without making API calls.",
)
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Splits directory (default: data/splits/v3).",
)
@click.option(
    "--max-input-chars",
    type=int,
    default=80_000,
    show_default=True,
    help="Max chars for paper input (truncated if exceeded).",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    show_default=True,
    help="LLM sampling temperature.",
)
def main(
    split: str,
    model: str,
    provider: str,
    max_articles: int | None,
    output: Path | None,
    concurrency: int,
    resume: bool,
    dry_run: bool,
    splits_dir: Path | None,
    max_input_chars: int,
    temperature: float,
) -> None:
    """Run the baseline LLM reviewer on a dataset split."""
    splits_dir = splits_dir or _DEFAULT_SPLITS_DIR
    split_path = splits_dir / f"{split}.jsonl"

    if not split_path.exists():
        console.print(f"[red]Error:[/red] Split file not found: {split_path}")
        sys.exit(1)

    # Load articles
    console.print(f"Loading {split} split from {split_path} ...")
    articles = _load_articles(split_path)
    console.print(f"  {len(articles)} articles loaded")

    # Filter to usable articles (those with at least 1 concern in GT)
    usable = [a for a in articles if len(a.get("concerns", [])) > 0]
    console.print(f"  {len(usable)} usable articles (with GT concerns)")

    if max_articles:
        usable = usable[:max_articles]
        console.print(f"  Limited to {len(usable)} articles")

    # Output path
    if output is None:
        output = Path("tool_outputs") / f"{_safe_model_name(model)}_{split}.jsonl"

    # Cost estimate
    from bioreview_bench.baseline.runner import estimate_cost

    cost = estimate_cost(usable, model, provider, max_input_chars)
    console.print(f"\n[bold]Cost estimate:[/bold]")
    console.print(f"  Articles:      {cost['n_articles']}")
    console.print(f"  Input tokens:  ~{cost['est_input_tokens']:,}")
    console.print(f"  Output tokens: ~{cost['est_output_tokens']:,}")
    console.print(f"  Est. cost:     ${cost['est_cost_usd']:.2f}")

    if dry_run:
        console.print("\n[yellow]Dry run[/yellow] — no API calls made.")
        return

    # Resume
    resume_ids: set[str] = set()
    if resume:
        from bioreview_bench.baseline.runner import load_existing_ids

        resume_ids = load_existing_ids(output)
        if resume_ids:
            console.print(f"\n  Resuming: {len(resume_ids)} articles already processed")

    # Run
    from bioreview_bench.baseline.runner import run_baseline
    from bioreview_bench.baseline.reviewer import BaselineReviewer

    reviewer = BaselineReviewer(
        model=model,
        provider=provider,
        max_input_chars=max_input_chars,
        temperature=temperature,
    )

    console.print(f"\nRunning baseline reviewer ({model}) ...")
    console.print(f"  Output: {output}")
    console.print(f"  Concurrency: {concurrency}\n")

    stats = run_baseline(
        reviewer=reviewer,
        articles=usable,
        output_path=output,
        concurrency=concurrency,
        resume_ids=resume_ids,
    )

    # Report
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Processed: {stats['processed']}")
    console.print(f"  Skipped:   {stats['skipped']}")
    console.print(f"  Failed:    {stats['failed']}")
    console.print(f"  Concerns:  {stats['total_concerns']}")
    console.print(f"  Output:    {output}")


def _load_articles(path: Path) -> list[dict]:
    """Load articles from a JSONL split file."""
    articles = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return articles


if __name__ == "__main__":
    main()

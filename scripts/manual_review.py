"""Legacy display-only explorer for extracted review concerns.

This script is retained for inspecting complete source-review stimuli and for
auditing historical CSV files. Its former interactive workflow is deprecated
and must not be used as evidence of independent validation. Use
``bioreview-validation-pack`` for new two-rater, system-label-blinded audits.

Usage:
    # Display 20 articles from val split (read-only)
    python scripts/manual_review.py --n 20 --split val

    # New annotations (replacement workflow)
    bioreview-validation-pack create --help

    # View a specific article
    python scripts/manual_review.py --article-id elife:84798

    # Compute agreement stats from a completed CSV
    python scripts/manual_review.py --review-csv data/validation/my_review.csv --stats

    # Filter to journal-format articles only
    python scripts/manual_review.py --n 20 --format-filter journal
"""

from __future__ import annotations

import csv
import random
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bioreview_bench.io import load_jsonl
from bioreview_bench.project_defaults import DEFAULT_BENCHMARK_SPLITS_DIR
from bioreview_bench.validate.agreement import compute_label_agreement

ROOT = Path(__file__).resolve().parents[1]
console = Console()

REVIEW_COLS = [
    "concern_id", "concern_text", "llm_category", "llm_stance",
    "llm_confidence", "human_category", "human_stance", "notes",
]

VALID_CATEGORIES = [
    "design_flaw", "statistical_methodology", "missing_experiment",
    "figure_issue", "prior_art_novelty", "writing_clarity",
    "reagent_method_specificity", "interpretation", "other",
]

VALID_STANCES = ["conceded", "rebutted", "partial", "unclear", "no_response"]


def load_entries(split_path: Path) -> list[dict]:
    return load_jsonl(split_path)


def display_article(entry: dict, *, show_model_labels: bool = True) -> None:
    """Display complete source-review context and extracted concerns."""
    concerns = entry.get("concerns", [])

    header = (
        f"[bold]{entry.get('id', '?')}[/bold]  "
        f"fmt={entry.get('review_format', '?')}  "
        f"date={entry.get('published_date', '?')}\n"
        f"{entry.get('title', '')[:100]}"
    )
    console.print(Panel(header, border_style="cyan"))
    console.print(f"  subjects: {', '.join(entry.get('subjects', [])[:3])}")
    console.print(
        f"  concerns: {len(concerns)} | "
        f"has_author_response: {entry.get('has_author_response')}"
    )
    console.print()

    source_review = entry.get("decision_letter_raw", "")
    if source_review:
        console.print("[bold]Complete source review / decision text[/bold]")
        console.print(source_review)
        console.print()

    if not concerns:
        console.print("  [dim]No concerns[/dim]\n")
        return

    for i, c in enumerate(concerns, 1):
        stance = c.get("author_stance", "?")
        conf = c.get("resolution_confidence", 0)
        cat = c.get("category", "?")
        sev = c.get("severity", "?")

        stance_style = {
            "conceded": "green", "rebutted": "red",
            "partial": "yellow", "unclear": "dim",
            "no_response": "bright_black",
        }.get(stance, "white")

        if show_model_labels:
            console.print(
                f"  [bold cyan]C{i}[/bold cyan] "
                f"[{stance_style}]{stance}[/{stance_style}] "
                f"conf={conf:.2f}  [{cat}] [{sev}]"
            )
        else:
            console.print(f"  [bold cyan]C{i}[/bold cyan]")
        console.print(f"    {c.get('concern_text', '')}")

        resp = c.get("author_response_text", "")
        if resp:
            console.print(f"    [dim]Author response: {resp}[/dim]")
        console.print()


def prompt_required_choice(label: str, choices: list[str]) -> str:
    """Prompt until the annotator explicitly enters one allowed label."""
    while True:
        value = input(f"  {label}: ").strip()
        if value in choices:
            return value
        console.print(
            f"[yellow]  Enter an explicit label; blank responses are not "
            f"accepted. Valid values: {', '.join(choices)}[/yellow]"
        )


def interactive_review(entry: dict, out_rows: list[dict]) -> None:
    """Interactive labeling mode — prompts for human label on each concern."""
    display_article(entry, show_model_labels=False)
    concerns = entry.get("concerns", [])
    if not concerns:
        return

    console.print(
        "[bold yellow]Rate each concern explicitly. Blank labels are not accepted, "
        "and model labels are not used as defaults.[/bold yellow]"
    )
    console.print(f"  Categories: {', '.join(VALID_CATEGORIES)}")
    console.print(f"  Stances: {', '.join(VALID_STANCES)}")
    console.print()

    for i, c in enumerate(concerns, 1):
        console.print(f"[bold cyan]Concern {i}/{len(concerns)}[/bold cyan]")
        console.print(f"  Text: {c.get('concern_text', '')}")

        try:
            human_cat = prompt_required_choice("Category", VALID_CATEGORIES)
            human_stance = prompt_required_choice("Stance", VALID_STANCES)
            notes = input("  Notes: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Interrupted.[/yellow]")
            break

        out_rows.append({
            "concern_id": c.get("concern_id", f"{entry['id']}:C{i}"),
            "concern_text": c.get("concern_text", ""),
            "llm_category": c.get("category", ""),
            "llm_stance": c.get("author_stance", ""),
            "llm_confidence": c.get("resolution_confidence", ""),
            "human_category": human_cat,
            "human_stance": human_stance,
            "notes": notes,
        })
    console.print()


def compute_agreement(rows: list[dict]) -> None:
    """Compute LLM vs human agreement statistics from validated CSV rows."""
    incomplete = [
        row for row in rows
        if not row.get("human_category") or not row.get("human_stance")
    ]
    if incomplete:
        console.print(
            f"[red]Cannot compute agreement: {len(incomplete)} rows lack an "
            "explicit human category or stance.[/red]"
        )
        return

    summary = compute_label_agreement(rows)
    if summary.n_rows == 0:
        console.print("[red]No review data found.[/red]")
        return

    console.print(
        f"\n[bold]=== Legacy Agreement Audit (n={summary.n_rows}) ===[/bold]"
    )
    console.print(
        "[yellow]These statistics do not establish independent validation "
        "without provenance from a label-blinded annotation protocol.[/yellow]"
    )
    console.print(
        f"  Category agreement: {summary.category_agreement * 100:.1f}%"
    )
    console.print(
        f"  Stance agreement:   {summary.stance_agreement * 100:.1f}%"
    )
    console.print(f"  Cohen's kappa (stance): {summary.kappa:.3f}")
    console.print(f"  -> {summary.quality_label} agreement")

    exact_prefill_pattern = all(
        row.get("llm_category") == row.get("human_category")
        and row.get("llm_stance") == row.get("human_stance")
        for row in rows
    )
    if exact_prefill_pattern:
        console.print(
            "[bold red]  WARNING: every human label is identical to its model "
            "label. This pattern is compatible with legacy prefill behavior and "
            "must not be cited as independent validation without annotation "
            "provenance.[/bold red]"
        )

    category_table = Table(title="Per-category agreement")
    category_table.add_column("Category")
    category_table.add_column("Rows", justify="right")
    category_table.add_column("Category", justify="right")
    category_table.add_column("Stance", justify="right")
    for bucket in summary.per_category:
        category_table.add_row(
            bucket.category,
            str(bucket.n_rows),
            f"{bucket.category_agreement * 100:.1f}%",
            f"{bucket.stance_agreement * 100:.1f}%",
        )
    console.print()
    console.print(category_table)

    console.print("\n  [bold]Stance disagreements:[/bold]")
    for row in summary.stance_disagreements:
        console.print(
            f"    LLM:{row['llm_stance']} -> Human:{row['human_stance']}  "
            f"{row['concern_text'][:80]}"
        )


@click.command()
@click.option("--split", default="val",
              type=click.Choice(["train", "val", "test"]),
              help="Data split to sample from")
@click.option("--n", default=20, show_default=True, help="Number of articles to sample")
@click.option("--article-id", default=None, help="View a specific article ID")
@click.option("--seed", default=42, show_default=True)
@click.option("--interactive", "-i", is_flag=True, default=False,
              help="Deprecated; use bioreview-validation-pack for new annotations")
@click.option("--review-csv", default=None,
              help="Path to existing review CSV (use with --stats)")
@click.option("--stats", is_flag=True, default=False,
              help="Compute agreement stats from existing CSV only")
@click.option("--input-jsonl", default=None,
              help="Review entries from this JSONL file instead of sampling a split")
@click.option("--splits-dir", default=None,
              help=(
                  "Directory containing split JSONL files "
                  f"(default: {DEFAULT_BENCHMARK_SPLITS_DIR})"
              ))
@click.option("--format-filter",
              type=click.Choice(["journal", "reviewed_preprint", "all"]),
              default="all", help="Filter by review_format")
def main(
    split: str,
    n: int,
    article_id: str | None,
    seed: int,
    interactive: bool,
    review_csv: str | None,
    stats: bool,
    input_jsonl: str | None,
    splits_dir: str | None,
    format_filter: str,
) -> None:
    """Inspect full review stimuli or audit a historical annotation CSV."""
    if interactive:
        raise click.UsageError(
            "The legacy interactive workflow is disabled because it cannot "
            "support independent validation. Use bioreview-validation-pack "
            "to create a two-rater, system-label-blinded audit."
        )

    # Stats-only mode
    if stats and review_csv:
        csv_path = Path(review_csv)
        if not csv_path.exists():
            console.print(f"[red]CSV not found: {csv_path}[/red]")
            sys.exit(1)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        compute_agreement(rows)
        return

    # Load split data
    if input_jsonl:
        split_path = Path(input_jsonl)
    else:
        resolved_splits_dir = (
            Path(splits_dir)
            if splits_dir
            else ROOT / DEFAULT_BENCHMARK_SPLITS_DIR
        )
        split_path = resolved_splits_dir / f"{split}.jsonl"

    if not split_path.exists():
        console.print(f"[red]{split_path} not found.[/red]")
        sys.exit(1)

    entries = load_entries(split_path)
    if format_filter != "all":
        entries = [e for e in entries if e.get("review_format") == format_filter]
    usable = [e for e in entries if e.get("concerns")]

    if article_id:
        sample = [e for e in entries if e.get("id") == article_id]
        if not sample:
            console.print(f"[red]Article {article_id} not found[/red]")
            sys.exit(1)
    else:
        rng = random.Random(seed)
        sample = rng.sample(usable, min(n, len(usable)))

    console.print("[bold cyan]bioreview-bench Manual Review[/bold cyan]")
    console.print(f"  split={split}  format={format_filter}  n={len(sample)}")
    console.print()

    out_rows: list[dict] = []

    for entry in sample:
        if interactive:
            interactive_review(entry, out_rows)
        else:
            display_article(entry)

    # Save interactive results
    if interactive and out_rows:
        out_dir = ROOT / "data" / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = out_dir / f"manual_review_{run_stamp}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_COLS)
            writer.writeheader()
            writer.writerows(out_rows)
        console.print(f"[green]Saved {len(out_rows)} rows -> {out_path}[/green]")
        compute_agreement(out_rows)

    # Summary table for display mode
    if not interactive:
        table = Table(title=f"Sample Summary ({len(sample)} articles)")
        table.add_column("ID")
        table.add_column("Format")
        table.add_column("Date")
        table.add_column("Concerns")
        table.add_column("Conceded%")
        table.add_column("NoResp%")

        for e in sample[:30]:
            c = e.get("concerns", [])
            n_c = len(c)
            conceded_n = sum(
                1 for x in c if x.get("author_stance") == "conceded"
            )
            noresp_n = sum(
                1 for x in c if x.get("author_stance") == "no_response"
            )
            conceded_pct = f"{conceded_n / max(n_c, 1) * 100:.0f}%"
            noresp_pct = f"{noresp_n / max(n_c, 1) * 100:.0f}%"
            table.add_row(
                e.get("id", "?")[-15:],
                e.get("review_format", "?"),
                str(e.get("published_date", "?"))[:7],
                str(n_c),
                conceded_pct,
                noresp_pct,
            )
        console.print(table)


if __name__ == "__main__":
    main()

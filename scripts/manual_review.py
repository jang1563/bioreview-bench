"""Manual validation sampler — Phase 2d annotation quality check.

Samples articles from val/test splits and displays their concerns in the
terminal for human quality assessment. Results are saved to CSV for
Cohen's kappa computation.

Usage:
    # Display 20 articles from val split (read-only)
    python scripts/manual_review.py --n 20 --split val

    # Interactive labeling mode (enter human labels)
    python scripts/manual_review.py --n 20 --split val --interactive

    # View a specific article
    python scripts/manual_review.py --article-id elife:84798

    # Compute agreement stats from a completed CSV
    python scripts/manual_review.py --review-csv data/validation/my_review.csv --stats

    # Filter to journal-format articles only
    python scripts/manual_review.py --n 20 --format-filter journal
"""

from __future__ import annotations

import csv
import json
import random
import sys
from datetime import date
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
    entries = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def display_article(entry: dict) -> None:
    """Display a single article's concerns in the terminal."""
    concerns = entry.get("concerns", [])

    header = (
        f"[bold]{entry.get('id', '?')}[/bold]  "
        f"fmt={entry.get('review_format', '?')}  "
        f"date={entry.get('published_date', '?')}\n"
        f"{entry.get('title', '')[:100]}"
    )
    console.print(Panel(header, border_style="cyan"))
    console.print(f"  subjects: {', '.join(entry.get('subjects', [])[:3])}")
    console.print(f"  concerns: {len(concerns)} | has_author_response: {entry.get('has_author_response')}")
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

        console.print(
            f"  [bold cyan]C{i}[/bold cyan] "
            f"[{stance_style}]{stance}[/{stance_style}] "
            f"conf={conf:.2f}  [{cat}] [{sev}]"
        )
        console.print(f"    {c.get('concern_text', '')[:200]}")

        resp = c.get("author_response_text", "")
        if resp:
            snippet = resp[:150].replace("\n", " ")
            console.print(f"    [dim]Author response: {snippet}...[/dim]")
        console.print()


def interactive_review(entry: dict, out_rows: list[dict]) -> None:
    """Interactive labeling mode — prompts for human label on each concern."""
    display_article(entry)
    concerns = entry.get("concerns", [])
    if not concerns:
        return

    console.print("[bold yellow]Rate each concern (press Enter to keep LLM label):[/bold yellow]")
    console.print(f"  Categories: {', '.join(VALID_CATEGORIES)}")
    console.print(f"  Stances: {', '.join(VALID_STANCES)}")
    console.print()

    for i, c in enumerate(concerns, 1):
        console.print(f"[bold cyan]Concern {i}/{len(concerns)}[/bold cyan]")
        console.print(f"  LLM: category={c.get('category')}  stance={c.get('author_stance')}")
        console.print(f"  Text: {c.get('concern_text', '')[:200]}")

        try:
            human_cat = input(f"  Category [{c.get('category')}]: ").strip()
            human_stance = input(f"  Stance [{c.get('author_stance')}]: ").strip()
            notes = input(f"  Notes: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Interrupted.[/yellow]")
            break

        out_rows.append({
            "concern_id": c.get("concern_id", f"{entry['id']}:C{i}"),
            "concern_text": c.get("concern_text", "")[:300],
            "llm_category": c.get("category", ""),
            "llm_stance": c.get("author_stance", ""),
            "llm_confidence": c.get("resolution_confidence", ""),
            "human_category": human_cat or c.get("category", ""),
            "human_stance": human_stance or c.get("author_stance", ""),
            "notes": notes,
        })
    console.print()


def compute_agreement(rows: list[dict]) -> None:
    """Compute LLM vs human agreement statistics from validated CSV rows."""
    if not rows:
        console.print("[red]No review data found.[/red]")
        return

    cat_agree = sum(1 for r in rows if r["llm_category"] == r["human_category"])
    stance_agree = sum(1 for r in rows if r["llm_stance"] == r["human_stance"])
    n = len(rows)

    console.print(f"\n[bold]=== Agreement Statistics (n={n}) ===[/bold]")
    console.print(f"  Category agreement: {cat_agree}/{n} = {cat_agree/n*100:.1f}%")
    console.print(f"  Stance agreement:   {stance_agree}/{n} = {stance_agree/n*100:.1f}%")

    from collections import Counter
    llm_dist = Counter(r["llm_stance"] for r in rows)
    human_dist = Counter(r["human_stance"] for r in rows)

    p_agree = stance_agree / n
    p_chance = sum(
        (llm_dist.get(s, 0) / n) * (human_dist.get(s, 0) / n)
        for s in VALID_STANCES
    )
    kappa = (p_agree - p_chance) / (1 - p_chance) if p_chance < 1 else 0.0
    console.print(f"  Cohen's kappa (stance): {kappa:.3f}")

    quality_label = (
        "Poor" if kappa < 0.2 else
        "Fair" if kappa < 0.4 else
        "Moderate" if kappa < 0.6 else
        "Substantial" if kappa < 0.8 else
        "Almost perfect"
    )
    console.print(f"  -> {quality_label} agreement")

    if kappa >= 0.6:
        console.print("[green]  OK: Phase 2 quality threshold met (kappa >= 0.6)[/green]")
    else:
        console.print("[yellow]  WARNING: Below threshold — review extraction prompts[/yellow]")

    console.print("\n  [bold]Stance disagreements:[/bold]")
    for r in rows:
        if r["llm_stance"] != r["human_stance"]:
            console.print(
                f"    LLM:{r['llm_stance']} -> Human:{r['human_stance']}  "
                f"{r['concern_text'][:80]}"
            )


@click.command()
@click.option("--split", default="val",
              type=click.Choice(["train", "val", "test"]),
              help="Data split to sample from")
@click.option("--n", default=20, show_default=True, help="Number of articles to sample")
@click.option("--article-id", default=None, help="View a specific article ID")
@click.option("--seed", default=42, show_default=True)
@click.option("--interactive", "-i", is_flag=True, default=False,
              help="Interactive labeling mode (enter human labels)")
@click.option("--review-csv", default=None,
              help="Path to existing review CSV (use with --stats)")
@click.option("--stats", is_flag=True, default=False,
              help="Compute agreement stats from existing CSV only")
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
    format_filter: str,
) -> None:
    """Manual annotation quality checker for bioreview-bench concerns."""
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
    splits_dir = ROOT / "data" / "splits"
    split_path = splits_dir / f"{split}.jsonl"
    if not split_path.exists():
        console.print(f"[red]{split_path} not found. Run create_splits.py first.[/red]")
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

    console.print(f"[bold cyan]bioreview-bench Manual Review[/bold cyan]")
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
        out_path = out_dir / f"manual_review_{date.today()}.csv"
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
            conceded_pct = f"{sum(1 for x in c if x.get('author_stance') == 'conceded') / max(n_c, 1) * 100:.0f}%"
            noresp_pct = f"{sum(1 for x in c if x.get('author_stance') == 'no_response') / max(n_c, 1) * 100:.0f}%"
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

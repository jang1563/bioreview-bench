"""PeerJ open peer review collection pipeline CLI.

Usage:
    # dry-run (metadata discovery only, $0 cost)
    uv run python -m bioreview_bench.scripts.collect_peerj --max-articles 10 --dry-run

    # full run
    uv run python -m bioreview_bench.scripts.collect_peerj --max-articles 1000
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..collect.peerj import PeerJCollector
from ..collect.postprocess import (
    finalize_manifest,
    load_known_ids_with_log,
    load_or_create_manifest,
    make_progress_bar,
    normalize_date,
    print_collection_summary,
    write_entry,
)
from ..models.entry import OpenPeerReviewEntry
from ..parse.concern_extractor import ConcernExtractor
from ..parse.jats import ParsedReview

console = Console()
ROOT = Path(__file__).resolve().parents[2]


async def _run(
    start_date: str,
    end_date: str | None,
    max_articles: int,
    output: Path,
    manifest_path: Path,
    model: str,
    dry_run: bool,
    no_extract: bool = False,
    append: bool = False,
    known_ids: set[str] | None = None,
) -> dict:
    manifest = load_or_create_manifest(
        manifest_path, model,
        manifest_id="em-peerj-v1.0",
        parsing_rule_hash="sha256:peerj-html-v1",
    )
    extractor = (
        ConcernExtractor(model=model, manifest_id=manifest.manifest_id)
        if not dry_run and not no_extract
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_fetched": 0,
        "skipped": 0,
        "no_review": 0,
        "ok": 0,
        "total_concerns": 0,
        "figure_concerns": 0,
    }

    summary_table = Table(title="PeerJ Collection Summary")
    summary_table.add_column("Article ID", style="cyan")
    summary_table.add_column("Decision")
    summary_table.add_column("Reviewers")
    summary_table.add_column("Concerns")
    summary_table.add_column("Status", style="green")

    with (
        output.open("a" if (append or dry_run) else "w", encoding="utf-8") as fout,
        make_progress_bar(console) as progress,
    ):
        task = progress.add_task(
            f"Collecting PeerJ ({'dry-run' if dry_run else 'full'})...",
            total=max_articles,
        )

        async with PeerJCollector() as collector:
            async for meta, review_data in collector.iter_articles(
                start_date=start_date,
                end_date=end_date,
                max_articles=max_articles,
                dry_run=dry_run,
            ):
                stats["total_fetched"] += 1
                article_id = f"peerj:{meta.article_id}"

                if known_ids and article_id in known_ids:
                    stats["skipped"] += 1
                    progress.advance(task)
                    continue

                if dry_run:
                    summary_table.add_row(article_id, "-", "-", "-", "[yellow]dry-run")
                    progress.advance(task)
                    continue

                if review_data is None:
                    stats["no_review"] += 1
                    progress.advance(task)
                    continue

                review_texts, editorial_decision, has_author_response = review_data

                if not review_texts:
                    stats["no_review"] += 1
                    progress.advance(task)
                    continue

                # Build ParsedReview objects for concern extraction
                parsed_reviews = [
                    ParsedReview(
                        reviewer_num=i + 1,
                        review_text=text,
                        author_response_text="",
                    )
                    for i, text in enumerate(review_texts)
                ]

                # Concatenate all reviewer texts as decision_letter_raw
                decision_letter_raw = "\n\n---\n\n".join(
                    f"Reviewer {r.reviewer_num}:\n\n{r.review_text}"
                    for r in parsed_reviews
                )

                # Concern extraction
                all_concerns = []
                if extractor:
                    for review in parsed_reviews:
                        try:
                            concerns = extractor.process_review(
                                review,
                                article_doi=meta.doi,
                                article_source="peerj",
                            )
                            all_concerns.extend(concerns)
                        except Exception as e:
                            console.print(
                                f"[red]Extraction error {article_id} R{review.reviewer_num}: {e}"
                            )

                fig_count = sum(1 for c in all_concerns if c.requires_figure_reading)
                stats["total_concerns"] += len(all_concerns)
                stats["figure_concerns"] += fig_count
                stats["ok"] += 1

                pub_date = normalize_date(meta.published)

                entry = OpenPeerReviewEntry(
                    id=article_id,
                    source="peerj",
                    doi=meta.doi,
                    title=meta.title,
                    abstract=meta.abstract,
                    subjects=meta.subjects,
                    editorial_decision=editorial_decision,
                    published_date=pub_date,
                    review_format="journal",
                    has_author_response=has_author_response,
                    paper_text_sections={},
                    structured_references=[],
                    decision_letter_raw=decision_letter_raw,
                    author_response_raw="",  # PDF not downloaded
                    concerns=all_concerns,
                    extraction_manifest_id=manifest.manifest_id,
                )

                write_entry(fout, entry, progress, task)

                summary_table.add_row(
                    article_id,
                    editorial_decision,
                    str(len(review_texts)),
                    f"{len(all_concerns) - fig_count} (+{fig_count} fig)",
                    "[green]ok",
                )

    print_collection_summary(console, summary_table, stats, output, dry_run)
    finalize_manifest(manifest, manifest_path, stats["ok"])

    # Alias for update_pipeline compatibility (expects "xml_ok")
    stats["xml_ok"] = stats["ok"]
    return stats


@click.command()
@click.option(
    "--start-date",
    default="2013-01-01",
    show_default=True,
    help="Collection start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    default=None,
    help="Collection end date (YYYY-MM-DD). None = no limit.",
)
@click.option(
    "--max-articles",
    "-n",
    default=10,
    show_default=True,
    help="Maximum number of articles to attempt",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output JSONL file path (default: data/processed/peerj_v1.jsonl)",
)
@click.option(
    "--manifest",
    default=None,
    help="ExtractionManifest JSON path",
)
@click.option(
    "--model",
    default=lambda: os.getenv("BIOREVIEW_MODEL_ID", "claude-haiku-4-5-20251001"),
    show_default=True,
    help="Anthropic model ID",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Metadata discovery only, no HTML fetch or LLM ($0 cost)",
)
@click.option(
    "--no-extract",
    is_flag=True,
    default=False,
    help="Collect & parse HTML but skip LLM concern extraction ($0 API cost). "
    "Entries saved with concerns=[]. Use scripts/backfill_concerns.py to extract later.",
)
@click.option(
    "--append",
    is_flag=True,
    default=False,
    help="Append to existing output file, skipping already-collected articles",
)
def main(
    start_date: str,
    end_date: str | None,
    max_articles: int,
    output: str | None,
    manifest: str | None,
    model: str,
    dry_run: bool,
    no_extract: bool,
    append: bool,
) -> None:
    """PeerJ open peer review collection pipeline."""
    output_path = Path(output) if output else ROOT / "data" / "processed" / "peerj_v1.jsonl"
    manifest_path = (
        Path(manifest) if manifest else ROOT / "data" / "manifests" / "em-peerj-v1.0.json"
    )

    known_ids = load_known_ids_with_log(output_path, append, console)

    console.print("[bold cyan]bioreview-bench PeerJ Collector[/bold cyan]")
    console.print(f"  start    : {start_date}")
    console.print(f"  end      : {end_date or '(none)'}")
    console.print(f"  max      : {max_articles}")
    console.print(f"  dry-run  : {dry_run}")
    console.print(f"  no-extract: {no_extract}")
    console.print(f"  output   : {output_path}")
    console.print()

    try:
        asyncio.run(
            _run(
                start_date=start_date,
                end_date=end_date,
                max_articles=max_articles,
                output=output_path,
                manifest_path=manifest_path,
                model=model,
                dry_run=dry_run,
                no_extract=no_extract,
                append=append,
                known_ids=known_ids or None,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}")
        raise


if __name__ == "__main__":
    main()

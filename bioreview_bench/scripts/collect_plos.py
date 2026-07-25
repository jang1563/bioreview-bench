"""PLOS Open Access collection pipeline CLI.

Usage:
    # dry-run (collection + parsing only, no LLM, $0 cost)
    uv run python -m bioreview_bench.scripts.collect_plos --max-articles 10 --dry-run

    # full run
    uv run python -m bioreview_bench.scripts.collect_plos --max-articles 1200
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..collect.plos import PLOSCollector
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
from ..parse.jats import JATSParser

console = Console()
ROOT = Path(__file__).resolve().parents[2]


async def _run(
    journals: list[str],
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
    manifest = load_or_create_manifest(manifest_path, model, manifest_id="em-plos-v1.0")
    parser = JATSParser()
    extractor = (
        ConcernExtractor(model=model, manifest_id=manifest.manifest_id)
        if not dry_run and not no_extract
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_fetched": 0,
        "skipped": 0,
        "xml_ok": 0,
        "xml_fail": 0,
        "no_review": 0,
        "total_concerns": 0,
        "figure_concerns": 0,
    }

    summary_table = Table(title="PLOS Collection Summary")
    summary_table.add_column("Article ID", style="cyan")
    summary_table.add_column("Journal")
    summary_table.add_column("Reviews")
    summary_table.add_column("Concerns")
    summary_table.add_column("Status", style="green")

    # dry_run never writes output — open in append mode to avoid truncating existing data
    with (
        output.open("a" if (append or dry_run) else "w", encoding="utf-8") as fout,
        make_progress_bar(console) as progress,
    ):
        task = progress.add_task(
            f"Collecting PLOS ({'dry-run' if dry_run else 'full'})...",
            total=max_articles,
        )

        async with PLOSCollector() as collector:
            async for meta, xml_bytes in collector.iter_articles(
                journals=journals or None,
                start_date=start_date,
                end_date=end_date,
                max_articles=max_articles,
                dry_run=dry_run,
            ):
                stats["total_fetched"] += 1
                article_id = f"plos:{meta.article_id}"

                # Skip already-collected articles (incremental mode)
                if known_ids and article_id in known_ids:
                    stats["skipped"] += 1
                    progress.advance(task)
                    continue

                status = "ok"

                if xml_bytes is None:
                    if dry_run:
                        progress.advance(task)
                        continue
                    stats["xml_fail"] += 1
                    summary_table.add_row(
                        article_id, meta.journal[:20], "-", "-", "[yellow]no_xml"
                    )
                    progress.advance(task)
                    continue

                try:
                    parsed = parser.parse(xml_bytes, article_id=meta.article_id)
                    stats["xml_ok"] += 1
                except ValueError as e:
                    console.print(f"[red]Parse error {article_id}: {e}")
                    stats["xml_fail"] += 1
                    progress.advance(task)
                    continue

                if not parsed.reviews:
                    stats["no_review"] += 1
                    status = "no_review"

                all_concerns = []
                if not dry_run and extractor and parsed.reviews:
                    for review in parsed.reviews:
                        try:
                            concerns = extractor.process_review(
                                review,
                                article_doi=parsed.doi,
                                article_source="plos",
                            )
                            all_concerns.extend(concerns)
                        except Exception as e:
                            console.print(
                                f"[red]Extraction error {article_id} R{review.reviewer_num}: {e}"
                            )

                fig_count = sum(1 for c in all_concerns if c.requires_figure_reading)
                stats["total_concerns"] += len(all_concerns)
                stats["figure_concerns"] += fig_count

                pub_date = normalize_date(parsed.published_date, meta.published)

                entry = OpenPeerReviewEntry(
                    id=article_id,
                    source="plos",
                    doi=parsed.doi or meta.doi,
                    title=parsed.title or meta.title,
                    abstract=parsed.abstract or meta.abstract,
                    subjects=parsed.subjects or meta.subjects,
                    editorial_decision=parsed.editorial_decision,
                    published_date=pub_date,
                    review_format="journal",
                    has_author_response=bool(parsed.author_response_raw.strip()),
                    paper_text_sections=parsed.sections,
                    structured_references=parsed.references,
                    decision_letter_raw=parsed.decision_letter_raw,
                    author_response_raw=parsed.author_response_raw,
                    concerns=all_concerns,
                    extraction_manifest_id=manifest.manifest_id,
                )

                write_entry(fout, entry, progress, task)

                summary_table.add_row(
                    article_id,
                    meta.journal[:20],
                    str(len(parsed.reviews)),
                    f"{len(all_concerns) - fig_count} (+{fig_count} fig)",
                    f"[green]{status}",
                )

    print_collection_summary(console, summary_table, stats, output, dry_run)
    finalize_manifest(manifest, manifest_path, stats["xml_ok"])

    return stats


@click.command()
@click.option(
    "--journals",
    "-j",
    multiple=True,
    default=[],
    help="PLOS journal keys (default: all 5 with peer review). "
    "Choices: PLoSBiology PLoSGenetics PLoSCompBiol PLoSPathogens PLoSMedicine",
)
@click.option(
    "--start-date",
    default="2019-01-01",
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
    help="Maximum number of articles to collect",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output JSONL file path (default: data/processed/plos_v1.jsonl)",
)
@click.option(
    "--manifest",
    default=None,
    help="ExtractionManifest JSON path (default: data/manifests/em-plos-v1.0.json)",
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
    help="Collection + parsing only, no LLM ($0 cost)",
)
@click.option(
    "--no-extract",
    is_flag=True,
    default=False,
    help="Collect & parse XML but skip LLM concern extraction ($0 API cost). "
    "Entries saved with concerns=[]. Use scripts/backfill_concerns.py to extract later.",
)
@click.option(
    "--append",
    is_flag=True,
    default=False,
    help="Append to existing output file, skipping already-collected article IDs",
)
def main(
    journals: tuple[str, ...],
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
    """PLOS article collection pipeline."""
    output_path = Path(output) if output else ROOT / "data" / "processed" / "plos_v1.jsonl"
    manifest_path = (
        Path(manifest) if manifest else ROOT / "data" / "manifests" / "em-plos-v1.0.json"
    )

    known_ids = load_known_ids_with_log(output_path, append, console)

    console.print("[bold cyan]bioreview-bench PLOS Collector[/bold cyan]")
    console.print(f"  journals : {list(journals) or '(all 5)'}")
    console.print(f"  start    : {start_date}")
    console.print(f"  end      : {end_date or '(none)'}")
    console.print(f"  max      : {max_articles}")
    console.print(f"  dry-run  : {dry_run}")
    console.print(f"  no-extract: {no_extract}")
    console.print(f"  append   : {append}")
    console.print(f"  output   : {output_path}")
    console.print()

    try:
        asyncio.run(
            _run(
                journals=list(journals),
                start_date=start_date,
                end_date=end_date,
                max_articles=max_articles,
                output=output_path,
                manifest_path=manifest_path,
                model=model,
                dry_run=dry_run,
                no_extract=no_extract,
                append=append,
                known_ids=known_ids,
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

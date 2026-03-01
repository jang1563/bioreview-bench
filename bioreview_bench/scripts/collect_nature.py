"""Nature portfolio transparent peer review collection pipeline CLI.

Fetches article metadata via CrossRef, then scrapes the article HTML page
to discover the peer review PDF URL, downloads it, and parses it with
NaturePDFParser.  No JATS XML section extraction is available for Nature.

Usage:
    # dry-run (metadata only, no PDF download or LLM)
    uv run python -m bioreview_bench.scripts.collect_nature --max-articles 10 --dry-run

    # full run
    uv run python -m bioreview_bench.scripts.collect_nature --max-articles 500
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from ..collect.nature import NatureCollector
from ..models.entry import OpenPeerReviewEntry
from ..models.manifest import ExtractionManifest
from ..parse.concern_extractor import ConcernExtractor
from ..parse.pdf import NaturePDFParser

console = Console()
ROOT = Path(__file__).resolve().parents[2]


def _load_or_create_manifest(manifest_path: Path, model: str) -> ExtractionManifest:
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        return ExtractionManifest(**data)

    import hashlib

    from ..parse.concern_extractor import CONCERN_EXTRACTION_SYSTEM, RESOLUTION_SYSTEM

    manifest = ExtractionManifest(
        manifest_id="em-nature-v1.0",
        model_id=model,
        model_date=datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
        prompt_hash="sha256:"
        + hashlib.sha256(
            (CONCERN_EXTRACTION_SYSTEM + RESOLUTION_SYSTEM).encode()
        ).hexdigest()[:16],
        parsing_rule_hash="sha256:placeholder",
        cost_per_article_usd=0.002,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


async def _run(
    journals: list[str],
    start_date: str,
    end_date: str | None,
    max_articles: int,
    output: Path,
    manifest_path: Path,
    model: str,
    dry_run: bool,
) -> None:
    manifest = _load_or_create_manifest(manifest_path, model)
    pdf_parser = NaturePDFParser()
    extractor = (
        ConcernExtractor(model=model, manifest_id=manifest.manifest_id)
        if not dry_run
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_fetched": 0,
        "pdf_ok": 0,
        "pdf_fail": 0,
        "no_review": 0,
        "total_concerns": 0,
        "figure_concerns": 0,
    }

    summary_table = Table(title="Nature Collection Summary")
    summary_table.add_column("Article ID", style="cyan")
    summary_table.add_column("Journal")
    summary_table.add_column("Decision letter")
    summary_table.add_column("Concerns")
    summary_table.add_column("Status", style="green")

    with (
        output.open("w", encoding="utf-8") as fout,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task(
            f"Collecting Nature ({'dry-run' if dry_run else 'full'})...",
            total=max_articles,
        )

        async with NatureCollector() as collector:
            metas = await collector.list_reviewed_articles(
                start_date=start_date,
                end_date=end_date,
                max_articles=max_articles,
                journals=journals or None,
            )

            for meta in metas:
                stats["total_fetched"] += 1
                article_id = f"nature:{meta.article_id}"
                status = "ok"

                if dry_run:
                    progress.advance(task)
                    continue

                # Scrape article page → download PDF
                pdf_bytes = await collector.fetch_peer_review_pdf(meta.doi)

                if pdf_bytes is None:
                    stats["pdf_fail"] += 1
                    summary_table.add_row(
                        article_id, meta.journal[:20], "-", "-", "[yellow]no_pdf"
                    )
                    progress.advance(task)
                    continue

                # Parse PDF into decision_letter_raw + author_response_raw
                parsed_pdf = pdf_parser.parse(pdf_bytes)
                decision_letter_raw = parsed_pdf["decision_letter_raw"]
                author_response_raw = parsed_pdf["author_response_raw"]

                if not decision_letter_raw:
                    stats["pdf_fail"] += 1
                    summary_table.add_row(
                        article_id, meta.journal[:20], "0", "-", "[yellow]empty_pdf"
                    )
                    progress.advance(task)
                    continue

                stats["pdf_ok"] += 1

                # For Nature, we have no structured per-reviewer ParsedReview objects
                # (PDF text is not structured enough to reliably split by reviewer).
                # Concern extraction runs on the full decision_letter_raw as one "review".
                all_concerns = []
                if not dry_run and extractor and decision_letter_raw:
                    # Import ParsedReview for concern extractor compatibility
                    from ..parse.jats import ParsedReview

                    synthetic_review = ParsedReview(
                        reviewer_num=0,  # 0 = combined / unsplit
                        review_text=decision_letter_raw,
                    )
                    try:
                        concerns = extractor.process_review(
                            synthetic_review,
                            article_doi=meta.doi,
                            article_source="nature",
                        )
                        all_concerns.extend(concerns)
                    except Exception as e:
                        console.print(f"[red]Extraction error {article_id}: {e}")

                if not all_concerns:
                    stats["no_review"] += 1
                    status = "no_concerns"

                fig_count = sum(1 for c in all_concerns if c.requires_figure_reading)
                stats["total_concerns"] += len(all_concerns)
                stats["figure_concerns"] += fig_count

                raw_date = meta.published or "2022-01-01"
                pub_date = raw_date.split("T")[0] if "T" in raw_date else raw_date
                if len(pub_date) == 4:
                    pub_date = f"{pub_date}-01-01"

                entry = OpenPeerReviewEntry(
                    id=article_id,
                    source="nature",
                    doi=meta.doi,
                    title=meta.title,
                    abstract=meta.abstract,
                    subjects=meta.subjects,
                    editorial_decision="unknown",
                    published_date=pub_date,
                    review_format="journal",
                    has_author_response=bool(author_response_raw.strip()),
                    paper_text_sections={},  # No JATS XML for Nature
                    structured_references=[],
                    decision_letter_raw=decision_letter_raw,
                    author_response_raw=author_response_raw,
                    concerns=all_concerns,
                    extraction_manifest_id=manifest.manifest_id,
                )

                fout.write(entry.model_dump_json() + "\n")
                fout.flush()

                summary_table.add_row(
                    article_id,
                    meta.journal[:20],
                    f"{len(decision_letter_raw)} chars",
                    f"{len(all_concerns) - fig_count} (+{fig_count} fig)",
                    f"[green]{status}",
                )
                progress.advance(task)

    console.print()
    console.print(summary_table)
    console.print()
    console.print("[bold]Collection complete[/bold]")
    console.print(f"  Total articles: {stats['total_fetched']}")
    if not dry_run:
        console.print(f"  PDF success:    {stats['pdf_ok']}")
        console.print(f"  PDF failed:     {stats['pdf_fail']}")
        console.print(f"  No review:      {stats['no_review']}")
        console.print(f"  Total concerns: {stats['total_concerns']}")
        console.print(f"  Figure concerns:{stats['figure_concerns']}")
    console.print(f"  Output: {output}")

    manifest.n_articles_processed = stats["pdf_ok"]
    manifest_path.write_text(manifest.model_dump_json(indent=2))


@click.command()
@click.option(
    "--journals",
    "-j",
    multiple=True,
    default=[],
    help="Nature journal names (default: all 6 with peer review). "
    "Choices: 'Nature' 'Nature Communications' 'Nature Methods' "
    "'Nature Genetics' 'Nature Cell Biology' 'Communications Biology'",
)
@click.option(
    "--start-date",
    default="2022-01-01",
    show_default=True,
    help="Collection start date. Default 2022 = when transparent peer review started.",
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
    help="Output JSONL file path (default: data/processed/nature_v1.jsonl)",
)
@click.option(
    "--manifest",
    default=None,
    help="ExtractionManifest JSON path (default: data/manifests/em-nature-v1.0.json)",
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
    help="Metadata discovery only, no PDF download or LLM ($0 cost)",
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
) -> None:
    """Nature portfolio peer review PDF collection pipeline."""
    output_path = (
        Path(output) if output else ROOT / "data" / "processed" / "nature_v1.jsonl"
    )
    manifest_path = (
        Path(manifest)
        if manifest
        else ROOT / "data" / "manifests" / "em-nature-v1.0.json"
    )

    console.print("[bold cyan]bioreview-bench Nature Collector[/bold cyan]")
    console.print(f"  journals : {list(journals) or '(all 6)'}")
    console.print(f"  start    : {start_date}")
    console.print(f"  end      : {end_date or '(none)'}")
    console.print(f"  max      : {max_articles}")
    console.print(f"  dry-run  : {dry_run}")
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

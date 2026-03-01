"""eLife collection pipeline CLI.

Usage:
    # dry-run (collection + parsing only, no LLM, $0 cost)
    uv run bioreview-collect --max-articles 10 --dry-run

    # full run (includes LLM concern extraction)
    uv run bioreview-collect --max-articles 500 --subjects genetics-genomics cell-biology
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from ..collect.elife import ELifeCollector
from ..collect.postprocess import infer_review_format
from ..models.entry import OpenPeerReviewEntry
from ..models.manifest import ExtractionManifest
from ..parse.jats import JATSParser
from ..parse.concern_extractor import ConcernExtractor

console = Console()
ROOT = Path(__file__).resolve().parents[2]  # peer-review-benchmark/


def _load_or_create_manifest(manifest_path: Path, model: str) -> ExtractionManifest:
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        return ExtractionManifest(**data)

    import hashlib
    from ..parse.concern_extractor import CONCERN_EXTRACTION_SYSTEM, RESOLUTION_SYSTEM

    manifest = ExtractionManifest(
        manifest_id=f"em-v1.0",
        model_id=model,
        model_date=datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
        prompt_hash="sha256:" + hashlib.sha256(
            (CONCERN_EXTRACTION_SYSTEM + RESOLUTION_SYSTEM).encode()
        ).hexdigest()[:16],
        parsing_rule_hash="sha256:placeholder",
        cost_per_article_usd=0.002,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        manifest.model_dump_json(indent=2)
    )
    return manifest


def _entry_to_jsonl_line(entry: OpenPeerReviewEntry) -> str:
    return entry.model_dump_json()


async def _run(
    subjects: list[str],
    start_date: str,
    end_date: str | None,
    order: str,
    max_articles: int,
    output: Path,
    manifest_path: Path,
    model: str,
    dry_run: bool,
    append: bool = False,
    known_ids: set[str] | None = None,
) -> dict:
    manifest = _load_or_create_manifest(manifest_path, model)
    parser = JATSParser()
    extractor = ConcernExtractor(model=model, manifest_id=manifest.manifest_id) if not dry_run else None

    output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_fetched": 0,
        "skipped": 0,
        "xml_ok": 0,
        "xml_fail": 0,
        "total_concerns": 0,
        "figure_concerns": 0,
        "no_review": 0,
    }

    summary_table = Table(title="Collection Summary")
    summary_table.add_column("Article ID", style="cyan")
    summary_table.add_column("DOI")
    summary_table.add_column("Reviews")
    summary_table.add_column("Concerns")
    summary_table.add_column("Status", style="green")

    with (
        output.open("a" if append else "w", encoding="utf-8") as fout,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task(
            f"Collecting ({'dry-run' if dry_run else 'full'})...",
            total=max_articles,
        )

        async with ELifeCollector() as collector:
            async for meta, xml_bytes in collector.iter_articles(
                subjects=subjects or None,
                start_date=start_date,
                end_date=end_date,
                order=order,
                max_articles=max_articles,
            ):
                stats["total_fetched"] += 1
                article_id = f"elife:{meta.article_id}"

                # Skip already-collected articles (incremental mode)
                if known_ids and article_id in known_ids:
                    stats["skipped"] += 1
                    progress.advance(task)
                    continue

                status = "ok"

                if xml_bytes is None:
                    stats["xml_fail"] += 1
                    status = "no_xml"
                    progress.advance(task)
                    summary_table.add_row(article_id, meta.doi, "-", "-", "[yellow]no_xml")
                    continue

                # JATS parsing
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

                # LLM concern extraction (only when not dry_run)
                all_concerns = []
                if not dry_run and extractor and parsed.reviews:
                    for review in parsed.reviews:
                        try:
                            concerns = extractor.process_review(
                                review,
                                article_doi=parsed.doi,
                                article_source="elife",
                            )
                            all_concerns.extend(concerns)
                        except Exception as e:
                            console.print(f"[red]Extraction error {article_id} R{review.reviewer_num}: {e}")

                fig_count = sum(1 for c in all_concerns if c.requires_figure_reading)
                stats["total_concerns"] += len(all_concerns)
                stats["figure_concerns"] += fig_count

                # Build and save OpenPeerReviewEntry
                raw_date = parsed.published_date or meta.published or "2020-01-01"
                # ISO datetime → date conversion (e.g. "2024-06-01T00:00:00Z" → "2024-06-01")
                pub_date = raw_date.split("T")[0] if "T" in raw_date else raw_date
                # year-only → complete date (e.g. "2024" → "2024-01-01")
                if len(pub_date) == 4:
                    pub_date = f"{pub_date}-01-01"

                # Detect review_format using shared postprocess logic
                fmt = infer_review_format({
                    "decision_letter_raw": parsed.decision_letter_raw,
                    "published_date": pub_date,
                })

                entry = OpenPeerReviewEntry(
                    id=article_id,
                    source="elife",
                    doi=parsed.doi or meta.doi,
                    title=parsed.title or meta.title,
                    abstract=parsed.abstract,
                    subjects=parsed.subjects or meta.subjects,
                    editorial_decision=parsed.editorial_decision,
                    published_date=pub_date,
                    review_format=fmt,
                    has_author_response=bool(parsed.author_response_raw.strip()),
                    paper_text_sections=parsed.sections,
                    structured_references=parsed.references,
                    decision_letter_raw=parsed.decision_letter_raw,
                    author_response_raw=parsed.author_response_raw,
                    concerns=all_concerns,
                    extraction_manifest_id=manifest.manifest_id,
                )

                fout.write(_entry_to_jsonl_line(entry) + "\n")
                fout.flush()

                summary_table.add_row(
                    article_id,
                    (parsed.doi or meta.doi)[:40],
                    str(len(parsed.reviews)),
                    f"{len(all_concerns) - fig_count} (+{fig_count} fig)",
                    f"[green]{status}",
                )
                progress.advance(task)

    # Print final summary
    console.print()
    console.print(summary_table)
    console.print()
    console.print(f"[bold]Collection complete[/bold]")
    console.print(f"  Total articles: {stats['total_fetched']}")
    if stats["skipped"]:
        console.print(f"  Skipped (known): {stats['skipped']}")
    console.print(f"  XML success: {stats['xml_ok']}")
    console.print(f"  XML failed: {stats['xml_fail']}")
    console.print(f"  No review: {stats['no_review']}")
    if not dry_run:
        console.print(f"  Total concerns: {stats['total_concerns']}")
        console.print(f"  Figure concerns: {stats['figure_concerns']}")
    console.print(f"  Output: {output}")

    # Update manifest (increment, not overwrite)
    manifest.n_articles_processed = (manifest.n_articles_processed or 0) + stats["xml_ok"]
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    return stats


@click.command()
@click.option(
    "--subjects", "-s",
    multiple=True,
    default=["genetics-genomics", "cell-biology", "neuroscience"],
    show_default=True,
    help="eLife subject areas to collect (multiple allowed)",
)
@click.option(
    "--start-date",
    default="2018-01-01",
    show_default=True,
    help="Collection start date (YYYY-MM-DD), filtered by published_date",
)
@click.option(
    "--end-date",
    default=None,
    help="Collection end date (YYYY-MM-DD), use for old-format collection (e.g. --end-date 2022-12-31)",
)
@click.option(
    "--order",
    default="desc",
    type=click.Choice(["desc", "asc"]),
    show_default=True,
    help="Sort order: desc=newest first (default), asc=oldest first (efficient for old-format collection)",
)
@click.option(
    "--max-articles", "-n",
    default=10,
    show_default=True,
    help="Maximum number of articles to collect",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output JSONL file path (default: data/processed/elife_v1.jsonl)",
)
@click.option(
    "--manifest",
    default=None,
    help="ExtractionManifest JSON path (default: data/manifests/em-v1.0.json)",
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
    help="Run collection + parsing only, no LLM ($0 cost, for pipeline validation)",
)
def main(
    subjects: tuple[str, ...],
    start_date: str,
    end_date: str | None,
    order: str,
    max_articles: int,
    output: str | None,
    manifest: str | None,
    model: str,
    dry_run: bool,
) -> None:
    """eLife article collection pipeline."""
    output_path = Path(output) if output else ROOT / "data" / "processed" / "elife_v1.jsonl"
    manifest_path = Path(manifest) if manifest else ROOT / "data" / "manifests" / "em-v1.0.json"

    console.print(f"[bold cyan]bioreview-bench eLife Collector[/bold cyan]")
    console.print(f"  subjects : {list(subjects)}")
    console.print(f"  start    : {start_date}")
    console.print(f"  end      : {end_date or '(none)'}")
    console.print(f"  order    : {order}")
    console.print(f"  max      : {max_articles}")
    console.print(f"  dry-run  : {dry_run}")
    console.print(f"  output   : {output_path}")
    console.print()

    try:
        asyncio.run(
            _run(
                subjects=list(subjects),
                start_date=start_date,
                end_date=end_date,
                order=order,
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
        sys.exit(1)


if __name__ == "__main__":
    main()

"""F1000Research / Wellcome Open Research / Gates Open Research collection pipeline CLI.

Usage:
    # dry-run
    uv run python -m bioreview_bench.scripts.collect_f1000 --max-articles 10 --dry-run

    # full run
    uv run python -m bioreview_bench.scripts.collect_f1000 --max-articles 1000
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

from ..collect.f1000 import F1000Collector
from ..models.entry import OpenPeerReviewEntry
from ..models.manifest import ExtractionManifest
from ..parse.concern_extractor import ConcernExtractor
from ..parse.jats import JATSParser

console = Console()
ROOT = Path(__file__).resolve().parents[2]


def _load_or_create_manifest(manifest_path: Path, model: str) -> ExtractionManifest:
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        return ExtractionManifest(**data)

    import hashlib

    from ..parse.concern_extractor import CONCERN_EXTRACTION_SYSTEM, RESOLUTION_SYSTEM

    manifest = ExtractionManifest(
        manifest_id="em-f1000-v1.0",
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
    parser = JATSParser()
    extractor = (
        ConcernExtractor(model=model, manifest_id=manifest.manifest_id)
        if not dry_run
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_fetched": 0,
        "xml_ok": 0,
        "xml_fail": 0,
        "no_review": 0,
        "total_concerns": 0,
        "figure_concerns": 0,
    }

    summary_table = Table(title="F1000 Collection Summary")
    summary_table.add_column("Article ID", style="cyan")
    summary_table.add_column("Journal")
    summary_table.add_column("Reviews")
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
            f"Collecting F1000 ({'dry-run' if dry_run else 'full'})...",
            total=max_articles,
        )

        async with F1000Collector() as collector:
            async for meta, xml_bytes in collector.iter_articles(
                journals=journals or None,
                start_date=start_date,
                end_date=end_date,
                max_articles=max_articles,
                dry_run=dry_run,
            ):
                stats["total_fetched"] += 1
                article_id = f"f1000:{meta.article_id}"
                status = "ok"

                if xml_bytes is None:
                    if dry_run:
                        # In dry-run, yield meta even without XML
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
                                article_source="f1000",
                            )
                            all_concerns.extend(concerns)
                        except Exception as e:
                            console.print(
                                f"[red]Extraction error {article_id} R{review.reviewer_num}: {e}"
                            )

                fig_count = sum(1 for c in all_concerns if c.requires_figure_reading)
                stats["total_concerns"] += len(all_concerns)
                stats["figure_concerns"] += fig_count

                raw_date = parsed.published_date or meta.published or "2020-01-01"
                pub_date = raw_date.split("T")[0] if "T" in raw_date else raw_date
                if len(pub_date) == 4:
                    pub_date = f"{pub_date}-01-01"

                entry = OpenPeerReviewEntry(
                    id=article_id,
                    source="f1000",
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

                fout.write(entry.model_dump_json() + "\n")
                fout.flush()

                summary_table.add_row(
                    article_id,
                    meta.journal[:20],
                    str(len(parsed.reviews)),
                    f"{len(all_concerns) - fig_count} (+{fig_count} fig)",
                    f"[green]{status}",
                )
                progress.advance(task)

    console.print()
    console.print(summary_table)
    console.print()
    console.print("[bold]Collection complete[/bold]")
    console.print(f"  Total articles: {stats['total_fetched']}")
    console.print(f"  XML success:    {stats['xml_ok']}")
    console.print(f"  XML failed:     {stats['xml_fail']}")
    console.print(f"  No review:      {stats['no_review']}")
    if not dry_run:
        console.print(f"  Total concerns: {stats['total_concerns']}")
        console.print(f"  Figure concerns:{stats['figure_concerns']}")
    console.print(f"  Output: {output}")

    manifest.n_articles_processed = stats["xml_ok"]
    manifest_path.write_text(manifest.model_dump_json(indent=2))


@click.command()
@click.option(
    "--journals",
    "-j",
    multiple=True,
    default=[],
    help="F1000-family journal names (default: all three). "
    "Choices: 'F1000Research' 'Wellcome Open Research' 'Gates Open Research'",
)
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
    help="Maximum number of articles to collect",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output JSONL file path (default: data/processed/f1000_v1.jsonl)",
)
@click.option(
    "--manifest",
    default=None,
    help="ExtractionManifest JSON path (default: data/manifests/em-f1000-v1.0.json)",
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
    help="Metadata discovery only, no XML download or LLM ($0 cost)",
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
    """F1000Research / Wellcome Open Research / Gates Open Research collection pipeline."""
    output_path = Path(output) if output else ROOT / "data" / "processed" / "f1000_v1.jsonl"
    manifest_path = (
        Path(manifest) if manifest else ROOT / "data" / "manifests" / "em-f1000-v1.0.json"
    )

    console.print("[bold cyan]bioreview-bench F1000 Collector[/bold cyan]")
    console.print(f"  journals : {list(journals) or '(all 3)'}")
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

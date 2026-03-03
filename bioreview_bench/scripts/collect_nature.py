"""Nature portfolio transparent peer review collection pipeline CLI.

Fetches article metadata via CrossRef, then scrapes the article HTML page
to discover the peer review PDF URL, downloads it, and parses it with
NaturePDFParser.  Full-text body sections are retrieved from Europe PMC
JATS XML (DOI → PMCID → fullTextXML → JATSParser).

Usage:
    # dry-run (metadata only, no PDF download or LLM)
    uv run python -m bioreview_bench.scripts.collect_nature --max-articles 10 --dry-run

    # full run
    uv run python -m bioreview_bench.scripts.collect_nature --max-articles 500
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

# Peer-review content validation: the downloaded PDF should mention reviewers/referees.
# If none of these appear in the first 3000 chars, we likely downloaded a supplementary file.
_PEER_REVIEW_CONTENT_RE = re.compile(
    r"reviewer|referee|review\s+comment|peer\s+review|revis(?:ion|ed)",
    re.IGNORECASE,
)

import click
from rich.console import Console
from rich.table import Table

from ..collect.nature import NatureCollector
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
from ..parse.pdf import NaturePDFParser

console = Console()
ROOT = Path(__file__).resolve().parents[2]


# Minimum character length to distinguish a real reviewer block from an editor preamble.
# Blocks shorter than this threshold in the DL are treated as "tiny DL" format.
_TRIVIAL_BLOCK_LEN = 1500


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
    manifest = load_or_create_manifest(manifest_path, model, manifest_id="em-nature-v1.0")
    pdf_parser = NaturePDFParser()
    extractor = (
        ConcernExtractor(model=model, manifest_id=manifest.manifest_id)
        if not dry_run and not no_extract
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    jats_parser = JATSParser()

    _known = known_ids or set()

    stats = {
        "total_fetched": 0,
        "skipped": 0,
        "pdf_ok": 0,
        "pdf_fail": 0,
        "no_review": 0,
        "total_concerns": 0,
        "figure_concerns": 0,
        "epmc_ok": 0,
        "epmc_fail": 0,
        "epmc_not_found": 0,
    }

    summary_table = Table(title="Nature Collection Summary")
    summary_table.add_column("Article ID", style="cyan")
    summary_table.add_column("Journal")
    summary_table.add_column("Sections")
    summary_table.add_column("Decision letter")
    summary_table.add_column("Concerns")
    summary_table.add_column("Status", style="green")

    # dry_run never writes output — open in append mode to avoid truncating existing data
    file_mode = "a" if (append or dry_run) else "w"
    with (
        output.open(file_mode, encoding="utf-8") as fout,
        make_progress_bar(console) as progress,
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

                if article_id in _known:
                    stats["skipped"] += 1
                    progress.advance(task)
                    continue

                status = "ok"

                if dry_run:
                    progress.advance(task)
                    continue

                # --no-extract: PDF collection only, skip LLM concern extraction
                # (concerns saved as empty list; use scripts/backfill_concerns.py to extract later)

                # Scrape article page → download PDF
                pdf_bytes = await collector.fetch_peer_review_pdf(meta.doi)

                if pdf_bytes is None:
                    stats["pdf_fail"] += 1
                    summary_table.add_row(
                        article_id, meta.journal[:20], "-", "-", "-", "[yellow]no_pdf"
                    )
                    progress.advance(task)
                    continue

                # Parse PDF into decision_letter_raw + author_response_raw
                try:
                    parsed_pdf = pdf_parser.parse(pdf_bytes)
                except Exception as e:
                    console.print(f"[red]PDF parse error {article_id}: {e}")
                    stats["pdf_fail"] += 1
                    summary_table.add_row(
                        article_id, meta.journal[:20], "-", "-", "-", "[yellow]parse_fail"
                    )
                    progress.advance(task)
                    continue
                decision_letter_raw = parsed_pdf["decision_letter_raw"]
                author_response_raw = parsed_pdf["author_response_raw"]

                if not decision_letter_raw:
                    stats["pdf_fail"] += 1
                    summary_table.add_row(
                        article_id, meta.journal[:20], "-", "0", "-", "[yellow]empty_pdf"
                    )
                    progress.advance(task)
                    continue

                # Content validation: reject PDFs that don't mention reviewers/referees.
                # This catches supplementary data files that slip past URL-level filtering.
                check_text = (decision_letter_raw + " " + author_response_raw)[:3000]
                if not _PEER_REVIEW_CONTENT_RE.search(check_text):
                    console.print(f"[yellow]Not a peer review PDF (no reviewer keywords): {article_id}")
                    stats["pdf_fail"] += 1
                    summary_table.add_row(
                        article_id, meta.journal[:20], "-", "-", "-", "[yellow]not_review_pdf"
                    )
                    progress.advance(task)
                    continue

                stats["pdf_ok"] += 1

                # Fetch full-text body sections via Europe PMC JATS XML
                paper_sections: dict[str, str] = {}
                try:
                    pmcid = await collector.lookup_pmcid(meta.doi)
                    if pmcid:
                        xml_bytes = await collector.fetch_epmc_xml(pmcid)
                        if xml_bytes:
                            parsed_xml = jats_parser.parse(xml_bytes, article_id=article_id)
                            paper_sections = parsed_xml.sections
                            stats["epmc_ok"] += 1
                        else:
                            stats["epmc_fail"] += 1
                    else:
                        stats["epmc_not_found"] += 1
                except Exception as e:
                    console.print(f"[yellow]EPMC/JATS error {article_id}: {e}")
                    stats["epmc_fail"] += 1

                # Extract per-reviewer concern blocks.
                # review_texts is a list[str] split by reviewer header.
                review_texts: list[str] = parsed_pdf.get("review_texts", [])

                # "Tiny DL" format (Nature flagship, NatMeth):
                # Editor decision (~200–1500 chars) is in decision_letter_raw; actual
                # reviewer comments are interleaved in author_response_raw.
                # Detect: all DL-derived blocks are trivially short (<= _TRIVIAL chars).
                if author_response_raw and (
                    not review_texts
                    or max((len(b) for b in review_texts), default=0) <= _TRIVIAL_BLOCK_LEN
                ):
                    # Split AR by first-round reviewer headers using interleaved detection
                    _, _, ar_blocks = pdf_parser._split_interleaved(author_response_raw, 0)
                    if ar_blocks and max(len(b) for b in ar_blocks) > _TRIVIAL_BLOCK_LEN:
                        review_texts = ar_blocks
                        console.print(
                            f"[dim]{article_id}: tiny-DL format — "
                            f"using {len(ar_blocks)} reviewer blocks from AR[/dim]"
                        )

                if not review_texts and decision_letter_raw:
                    review_texts = [decision_letter_raw]

                all_concerns = []
                if not dry_run and extractor and review_texts:
                    from ..parse.jats import ParsedReview

                    for r_idx, r_text in enumerate(review_texts, start=1):
                        synthetic_review = ParsedReview(
                            reviewer_num=r_idx,
                            review_text=r_text,
                            author_response_text=author_response_raw,
                        )
                        try:
                            concerns = extractor.process_review(
                                synthetic_review,
                                article_doi=meta.doi,
                                article_source="nature",
                            )
                            all_concerns.extend(concerns)
                        except Exception as e:
                            console.print(
                                f"[red]Extraction error {article_id} R{r_idx}: {e}"
                            )

                if not all_concerns:
                    if no_extract:
                        # concerns were intentionally skipped; don't count as "no review found"
                        status = "ok"
                    else:
                        stats["no_review"] += 1
                        status = "no_concerns"

                fig_count = sum(1 for c in all_concerns if c.requires_figure_reading)
                stats["total_concerns"] += len(all_concerns)
                stats["figure_concerns"] += fig_count

                pub_date = normalize_date(meta.published, fallback="2022-01-01")

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
                    paper_text_sections=paper_sections,
                    structured_references=[],
                    decision_letter_raw=decision_letter_raw,
                    author_response_raw=author_response_raw,
                    concerns=all_concerns,
                    extraction_manifest_id=manifest.manifest_id,
                )

                write_entry(fout, entry, progress, task)

                sec_info = f"{len(paper_sections)} keys" if paper_sections else "[dim]-"
                summary_table.add_row(
                    article_id,
                    meta.journal[:20],
                    sec_info,
                    f"{len(decision_letter_raw)} chars",
                    f"{len(all_concerns) - fig_count} (+{fig_count} fig)",
                    f"[green]{status}",
                )

    print_collection_summary(console, summary_table, stats, output, dry_run)
    finalize_manifest(manifest, manifest_path, stats["pdf_ok"])

    # Alias for update_pipeline compatibility (expects "xml_ok")
    stats["xml_ok"] = stats["pdf_ok"]
    stats["xml_fail"] = stats["pdf_fail"]
    return stats


@click.command()
@click.option(
    "--journals",
    "-j",
    multiple=True,
    default=[],
    help="Nature journal names (default: 7 mandatory-TPR journals). "
    "Mandatory TPR: 'Nature Communications' 'Communications Biology' "
    "'Communications Chemistry' 'Communications Earth and Environment' "
    "'Communications Physics' 'Communications Materials' 'Communications Medicine' "
    "'Communications Psychology' 'Communications Engineering'. "
    "Opt-in TPR (~50% hit rate): 'Nature' 'Nature Methods' 'Nature Genetics' 'Nature Cell Biology'",
)
@click.option(
    "--start-date",
    default="2022-01-01",
    show_default=True,
    help="Collection start date. Per-journal TPR start dates are enforced automatically "
    "(e.g. NatComm uses 2022-11-01 regardless). Default 2022 = conservative minimum.",
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
@click.option(
    "--no-extract",
    is_flag=True,
    default=False,
    help="Download & parse PDFs but skip LLM concern extraction ($0 API cost). "
    "Entries are saved with concerns=[]. Use scripts/backfill_concerns.py to extract later.",
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
    """Nature portfolio peer review PDF collection pipeline."""
    output_path = (
        Path(output) if output else ROOT / "data" / "processed" / "nature_v1.jsonl"
    )
    manifest_path = (
        Path(manifest)
        if manifest
        else ROOT / "data" / "manifests" / "em-nature-v1.0.json"
    )

    known_ids = load_known_ids_with_log(output_path, append, console)

    console.print("[bold cyan]bioreview-bench Nature Collector[/bold cyan]")
    console.print(f"  journals : {list(journals) or '(all 7 mandatory-TPR)'}")
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

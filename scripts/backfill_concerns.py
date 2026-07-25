"""Backfill concern extraction for entries collected with --no-extract.

Reads a JSONL file, finds entries with concerns=[] that have non-empty
decision_letter_raw, runs LLM concern extraction, and writes updated entries.

Usage:
    # Nature articles (in-place, atomic overwrite)
    uv run python scripts/backfill_concerns.py \
        --input-file data/processed/nature_v1.jsonl \
        --model claude-haiku-4-5-20251001

    # All sources
    for src in elife_v1.1 plos_v1 f1000_v1 peerj_v1 nature_v1; do
        uv run python scripts/backfill_concerns.py \\
            --input-file data/processed/${src}.jsonl
    done
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreview_bench.collect.postprocess import load_or_create_manifest, finalize_manifest
from bioreview_bench.parse.concern_extractor import ConcernExtractor, split_into_reviewer_blocks
from bioreview_bench.parse.jats import ParsedReview

console = Console()


def extract_concerns_for_entry(
    entry: dict,
    extractor: ConcernExtractor,
) -> list[dict]:
    """Extract concerns from a single entry's stored raw text."""
    decision_letter_raw = entry.get("decision_letter_raw", "")
    author_response_raw = entry.get("author_response_raw", "")
    article_doi = entry.get("doi", "")
    article_source = entry.get("source", "elife")

    # Prefer pre-split review_texts when available (avoids regex splitting issues)
    review_texts = entry.get("review_texts", [])
    review_blocks = review_texts if review_texts else split_into_reviewer_blocks(decision_letter_raw)
    if not review_blocks:
        return []

    all_concerns = []
    for r_idx, block_text in enumerate(review_blocks, start=1):
        review = ParsedReview(
            reviewer_num=r_idx,
            review_text=block_text,
            author_response_text=author_response_raw,
        )
        try:
            concerns = extractor.process_review(
                review,
                article_doi=article_doi,
                article_source=article_source,
            )
            all_concerns.extend(concerns)
        except Exception as e:
            console.print(f"[red]Extraction error {entry.get('id', '?')} R{r_idx}: {e}")

    return [c.model_dump() for c in all_concerns]


@click.command()
@click.option(
    "--input-file",
    "-i",
    required=True,
    help="Input JSONL file path",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output JSONL file path (default: overwrite input file)",
)
@click.option(
    "--model",
    default=lambda: os.getenv("BIOREVIEW_MODEL_ID", "claude-haiku-4-5-20251001"),
    show_default=True,
    help="Anthropic model ID for concern extraction",
)
@click.option(
    "--manifest",
    default=None,
    help="ExtractionManifest JSON path (default: auto-detect from input filename)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-extract even for entries that already have concerns",
)
def main(
    input_file: str,
    output: str | None,
    model: str,
    manifest: str | None,
    force: bool,
) -> None:
    """Backfill concern extraction for entries collected with --no-extract."""
    input_path = Path(input_file)
    output_path = Path(output) if output else input_path

    # Auto-detect manifest from input filename stem
    if manifest is None:
        stem = input_path.stem  # e.g. "nature_v1", "elife_v1.1"
        source = stem.split("_")[0]  # e.g. "nature", "elife"
        manifest_map = {
            "nature": "em-nature-v1.0",
            "elife": "em-v1.0",
            "plos": "em-plos-v1.0",
            "peerj": "em-peerj-v1.0",
            "f1000": "em-f1000-v1.0",
        }
        manifest_id = manifest_map.get(source, f"em-{source}-v1.0")
        manifest_path = ROOT / "data" / "manifests" / f"{manifest_id}.json"
    else:
        manifest_path = Path(manifest)
        manifest_id = manifest_path.stem

    console.print(f"[bold cyan]bioreview-bench Concern Backfill[/bold cyan]")
    console.print(f"  input    : {input_path}")
    console.print(f"  output   : {output_path}")
    console.print(f"  model    : {model}")
    console.print(f"  manifest : {manifest_path}")
    console.print(f"  force    : {force}")
    console.print()

    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}")
        sys.exit(1)

    # Load all entries
    entries = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    console.print(f"Loaded {len(entries)} entries from {input_path}")

    # Identify which entries need concern extraction
    needs_extraction = [
        e for e in entries
        if (force or not e.get("concerns"))
        and e.get("decision_letter_raw", "").strip()
    ]
    console.print(f"  {len(needs_extraction)} entries need concern extraction")
    console.print(f"  {len(entries) - len(needs_extraction)} entries already have concerns (skip)")
    console.print()

    skipped_count = len(entries) - len(needs_extraction)

    if not needs_extraction:
        console.print("[green]Nothing to do.")
        return

    em = load_or_create_manifest(manifest_path, model, manifest_id=manifest_id)
    extractor = ConcernExtractor(model=model, manifest_id=em.manifest_id)

    # Index entries by id for fast update
    entry_map = {e["id"]: e for e in entries}

    stats = {"processed": 0, "total_concerns": 0, "skipped": skipped_count}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting concerns...", total=len(needs_extraction))

        for entry in needs_extraction:
            concerns = extract_concerns_for_entry(entry, extractor)
            entry_map[entry["id"]]["concerns"] = concerns
            stats["processed"] += 1
            stats["total_concerns"] += len(concerns)
            progress.advance(task)

    # Atomic write: write to a temp file in the same directory, then rename.
    # This prevents data loss if an error occurs mid-write (especially when
    # output_path == input_path, i.e., in-place update).
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=output_path.parent, suffix=".tmp", prefix=output_path.name
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry_map[entry["id"]], ensure_ascii=False) + "\n")
        Path(tmp_name).replace(output_path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    finalize_manifest(em, manifest_path, stats["processed"])

    console.print()
    console.print(f"[green]Done.[/green]")
    console.print(f"  Processed  : {stats['processed']} entries")
    console.print(f"  Skipped    : {stats['skipped']} entries (already had concerns)")
    console.print(f"  Concerns   : {stats['total_concerns']} total")
    console.print(f"  Output     : {output_path}")


if __name__ == "__main__":
    main()

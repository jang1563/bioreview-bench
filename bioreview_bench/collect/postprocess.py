"""Shared postprocessing and collection utility functions.

Canonical implementation of infer_review_format(), subject cleaning,
date normalization, manifest loading, and JSONL deduplication.

Used by the update pipeline and all per-source collection scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.entry import _ARTICLE_TYPE_LABELS

# Keywords indicating the eLife Reviewed Preprint format (post-2023)
_RP_KEYWORDS: list[str] = [
    "elife assessment",
    "reviewed preprint",
    "public review",
    "referee report",
]

# Keywords indicating the traditional journal format (pre-2022)
_JOURNAL_KEYWORDS: list[str] = [
    "major concerns",
    "minor concerns",
    "essential revisions",
    "we have read your manuscript",
    "after consultation with the reviewers",
    "reviewer 1:",
    "reviewer #1",
]


def infer_review_format(entry: dict) -> str:
    """Infer review format from decision_letter_raw content.

    Returns one of: "reviewed_preprint", "journal", "unknown".

    Detection priority:
    1. Keyword match in decision_letter_raw (most reliable)
    2. Year-based heuristic as fallback
    """
    letter = (entry.get("decision_letter_raw") or "").lower()
    if not letter:
        return "unknown"

    for kw in _RP_KEYWORDS:
        if kw in letter:
            return "reviewed_preprint"

    for kw in _JOURNAL_KEYWORDS:
        if kw in letter:
            return "journal"

    # Heuristic based on publication year
    pub_date = str(entry.get("published_date", ""))
    year = pub_date[:4] if pub_date else ""
    if year and year.isdigit():
        if int(year) <= 2022:
            return "journal"
        elif int(year) >= 2024:
            return "reviewed_preprint"

    return "unknown"


def clean_subjects(subjects: list[str]) -> list[str]:
    """Remove article-type labels from subjects list.

    Uses _ARTICLE_TYPE_LABELS from models/entry.py as the single source of truth.
    """
    return [s for s in subjects if s not in _ARTICLE_TYPE_LABELS]


def postprocess_entry(entry: dict) -> dict:
    """Apply all postprocessing to a single entry dict (in-place + return).

    Operations:
    1. Clean subjects (remove article-type labels)
    2. Infer review_format
    3. Set has_author_response
    4. Ensure schema_version is "1.1"
    """
    entry["subjects"] = clean_subjects(entry.get("subjects", []))
    entry["review_format"] = infer_review_format(entry)
    entry["has_author_response"] = bool((entry.get("author_response_raw") or "").strip())
    entry["schema_version"] = "1.1"
    return entry


# ── Collection utilities ──────────────────────────────────────────


def normalize_date(*candidates: str | None, fallback: str = "2020-01-01") -> str:
    """Normalize a published date to YYYY-MM-DD format.

    Picks the first non-empty candidate, strips ISO datetime suffixes,
    and pads year-only values.
    """
    raw = fallback
    for c in candidates:
        if c:
            raw = c
            break
    date_str = raw.split("T")[0] if "T" in raw else raw
    if len(date_str) == 4:
        date_str = f"{date_str}-01-01"
    return date_str


def load_known_ids(jsonl_path: Path) -> set[str]:
    """Load article IDs from an existing JSONL file for deduplication."""
    import json

    ids: set[str] = set()
    if not jsonl_path.exists():
        return ids
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                pass
    return ids


def load_known_ids_with_log(
    output_path: Path, append: bool, console: Any,
) -> set[str]:
    """Load known article IDs for dedup when appending, logging the count."""
    if not append:
        return set()
    known = load_known_ids(output_path)
    if known:
        console.print(f"  [dim]Loaded {len(known)} existing IDs for dedup[/dim]")
    return known


def make_progress_bar(console: Any) -> Any:
    """Return a standard Rich Progress bar for collection scripts."""
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def write_entry(fout: Any, entry: Any, progress: Any, task: Any) -> None:
    """Write a single JSONL entry, flush, and advance the progress bar."""
    fout.write(entry.model_dump_json() + "\n")
    fout.flush()
    progress.advance(task)


# ── Summary statistics ────────────────────────────────────────────

_STAT_LABELS: list[tuple[str, str]] = [
    ("total_fetched", "Total articles"),
    ("skipped", "Skipped (known)"),
    ("xml_ok", "XML success"),
    ("xml_fail", "XML failed"),
    ("pdf_ok", "PDF success"),
    ("pdf_fail", "PDF failed"),
    ("ok", "OK (with reviews)"),
    ("no_review", "No review"),
    ("epmc_ok", "EPMC sections"),
    ("epmc_not_found", "EPMC not found"),
    ("epmc_fail", "EPMC failed"),
    ("total_concerns", "Total concerns"),
    ("figure_concerns", "Figure concerns"),
]
_DRY_RUN_SKIP = {"total_concerns", "figure_concerns"}
_NONZERO_ONLY = {"skipped"}


def print_collection_summary(
    console: Any,
    summary_table: Any,
    stats: dict[str, int],
    output: Path,
    dry_run: bool,
) -> None:
    """Print the summary table and post-collection statistics."""
    console.print()
    console.print(summary_table)
    console.print()
    console.print("[bold]Collection complete[/bold]")
    for key, label in _STAT_LABELS:
        if key not in stats:
            continue
        if key in _DRY_RUN_SKIP and dry_run:
            continue
        if key in _NONZERO_ONLY and stats[key] == 0:
            continue
        console.print(f"  {label + ':':20s}{stats[key]}")
    console.print(f"  Output: {output}")


def finalize_manifest(manifest: Any, manifest_path: Path, n_new: int) -> None:
    """Increment manifest article count and write to disk."""
    manifest.n_articles_processed = (manifest.n_articles_processed or 0) + n_new
    manifest_path.write_text(manifest.model_dump_json(indent=2))


def load_or_create_manifest(
    manifest_path: Path,
    model: str,
    manifest_id: str = "em-v1.0",
    cost_per_article_usd: float = 0.009,
    parsing_rule_hash: str = "sha256:placeholder",
) -> "ExtractionManifest":  # noqa: F821
    """Load existing manifest or create a new one with the given parameters."""
    import datetime as _dt
    import hashlib
    import json
    from datetime import datetime

    from ..models.manifest import ExtractionManifest
    from ..parse.concern_extractor import CONCERN_EXTRACTION_SYSTEM, RESOLUTION_SYSTEM

    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        return ExtractionManifest(**data)

    manifest = ExtractionManifest(
        manifest_id=manifest_id,
        model_id=model,
        model_date=datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
        prompt_hash="sha256:"
        + hashlib.sha256(
            (CONCERN_EXTRACTION_SYSTEM + RESOLUTION_SYSTEM).encode()
        ).hexdigest()[:16],
        parsing_rule_hash=parsing_rule_hash,
        cost_per_article_usd=cost_per_article_usd,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest

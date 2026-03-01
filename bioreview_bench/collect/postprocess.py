"""Shared postprocessing functions for collected articles.

Canonical implementation of infer_review_format() and subject cleaning.
Used by both the update pipeline and the legacy scripts/postprocess_v1.py.

This eliminates the duplication between collect_elife.py (lines 172-180)
and scripts/postprocess_v1.py (lines 50-73).
"""

from __future__ import annotations

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

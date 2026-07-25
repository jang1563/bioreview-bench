"""HuggingFace dataset config transforms.

Each transform function converts raw JSONL entry dicts into a specific
HuggingFace config format.  No external dependencies beyond stdlib.

Configs
-------
- default:         All fields preserved (date → str).
- benchmark:       Task-relevant fields only; train/val get simplified concerns,
                   test has concerns=[].  Uses descriptive field names (``source``,
                   ``paper_text_sections``).  The evaluation harness
                   ``to_task_input()`` uses shorter aliases (``journal``,
                   ``sections``, ``references``); mapping is documented in
                   TASK_DEFINITION.md.
- concerns_flat:   One row per concern with article-level context.
- elife/plos/f1000: Source-filtered subsets of ``default``.
"""

from __future__ import annotations

from typing import Literal


# Fields kept in simplified concern for benchmark train/val
_BENCHMARK_CONCERN_KEYS: tuple[str, ...] = (
    "concern_text",
    "category",
    "severity",
)

# Article-level fields included in concerns_flat rows
_FLAT_ARTICLE_KEYS: tuple[str, ...] = (
    "doi",
    "source",
    "title",
    "abstract",
    "published_date",
    "editorial_decision",
    "paper_text_sections",
)

# Concern fields included in concerns_flat (excludes internal trace IDs,
# ``resolution`` (1:1 alias of ``author_stance``), and ``source`` (already
# present from article-level fields via _FLAT_ARTICLE_KEYS))
_FLAT_CONCERN_KEYS: tuple[str, ...] = (
    "concern_id",
    "reviewer_num",
    "concern_text",
    "category",
    "severity",
    "author_response_text",
    "author_stance",
    "evidence_of_change",
    "resolution_confidence",
    "was_valid",
    "raised_by_multiple",
    "requires_figure_reading",
    "article_doi",
)


def _as_str_date(value: object) -> str:
    """Return ISO date string; passthrough if already str."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _concerns_as_dicts(entry: dict) -> list[dict]:
    """Return the concerns list as plain dicts (safe for .get())."""
    raw = entry.get("concerns") or []
    if raw and hasattr(raw[0], "model_dump"):
        return [c.model_dump() for c in raw]
    return raw


# ────────────────────────────────────────────────────────────────────
# Config: default
# ────────────────────────────────────────────────────────────────────

def transform_default(entries: list[dict]) -> list[dict]:
    """Full data — all fields preserved, dates stringified."""
    out: list[dict] = []
    for e in entries:
        row = dict(e)  # shallow copy
        row["published_date"] = _as_str_date(row.get("published_date"))
        row["concerns"] = _concerns_as_dicts(row)
        out.append(row)
    return out


# ────────────────────────────────────────────────────────────────────
# Config: benchmark
# ────────────────────────────────────────────────────────────────────

def transform_benchmark(
    entries: list[dict],
    split: Literal["train", "validation", "test"],
) -> list[dict]:
    """Benchmark task format — no leakage fields.

    train/val: include simplified concerns (text + category + severity).
    test:      concerns=[] to prevent label leakage.

    Excluded fields (leakage or internal):
        decision_letter_raw, author_response_raw, review_format,
        has_author_response, editorial_decision, revision_round,
        extraction_manifest_id, schema_version, and all concern fields
        beyond text/category/severity.
    """
    out: list[dict] = []
    for e in entries:
        row = {
            "article_id": e["id"],
            "doi": e["doi"],
            "source": e["source"],
            "title": e["title"],
            "abstract": e.get("abstract", ""),
            "subjects": e.get("subjects", []),
            "published_date": _as_str_date(e.get("published_date")),
            "paper_text_sections": e.get("paper_text_sections", {}),
            "structured_references": e.get("structured_references", []),
        }

        if split == "test":
            row["concerns"] = []
        else:
            concerns = _concerns_as_dicts(e)
            row["concerns"] = [
                {k: c.get(k, "") for k in _BENCHMARK_CONCERN_KEYS}
                for c in concerns
            ]

        out.append(row)
    return out


# ────────────────────────────────────────────────────────────────────
# Config: concerns_flat
# ────────────────────────────────────────────────────────────────────

def transform_concerns_flat(entries: list[dict]) -> list[dict]:
    """Flatten: one row per concern with article context.

    Articles with zero concerns produce zero rows.
    """
    out: list[dict] = []
    for e in entries:
        article_fields = {"article_id": e["id"]}
        for k in _FLAT_ARTICLE_KEYS:
            v = e.get(k)
            if k == "published_date":
                v = _as_str_date(v)
            article_fields[k] = v

        for c in _concerns_as_dicts(e):
            row = dict(article_fields)
            for k in _FLAT_CONCERN_KEYS:
                row[k] = c.get(k)
            out.append(row)

    return out


# ────────────────────────────────────────────────────────────────────
# Config: source subsets (elife / plos / f1000)
# ────────────────────────────────────────────────────────────────────

def transform_source_subset(
    entries: list[dict],
    source: str,
) -> list[dict]:
    """Filter by source, then apply default transform."""
    filtered = [e for e in entries if e.get("source") == source]
    return transform_default(filtered)

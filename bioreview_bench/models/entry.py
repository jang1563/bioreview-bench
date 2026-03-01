"""OpenPeerReviewEntry — integrated schema for article + review + concerns."""

from __future__ import annotations

from datetime import date
from typing import Literal
from pydantic import BaseModel, Field

from .concern import ReviewerConcern

# Article-type labels to remove from the subjects field
_ARTICLE_TYPE_LABELS: frozenset[str] = frozenset({
    "Research Article", "Research Advance", "Insight", "Correction",
    "Short Report", "Review Article", "Tools and Resources",
    "Feature Article", "Editorial", "Registered Report",
    "Replication Study", "Scientific Correspondence", "Cell Press Exclusive",
})


class OpenPeerReviewEntry(BaseModel):
    """Complete open peer review data for a single article."""

    # ── Identifiers ───────────────────────────────────────
    id: str = Field(description="e.g. 'elife:84798'")
    source: Literal["elife", "plos", "nature", "peerj", "f1000"]
    doi: str

    # ── Article metadata ──────────────────────────────────
    title: str
    abstract: str
    subjects: list[str] = Field(default_factory=list)
    editorial_decision: Literal[
        "accept", "major_revision", "minor_revision", "reject", "unknown"
    ] = "unknown"
    revision_round: int = Field(default=1, ge=1)
    published_date: date

    # ── Peer review format ────────────────────────────────
    review_format: Literal["journal", "reviewed_preprint", "unknown"] = Field(
        default="unknown",
        description=(
            "journal: traditional decision letter + point-by-point author response (≤2022). "
            "reviewed_preprint: eLife Assessment + free-form response (2023+)."
        ),
    )
    has_author_response: bool = Field(
        default=False,
        description="Whether author response text is present.",
    )

    # ── Article body (task input) ─────────────────────────
    paper_text_sections: dict[str, str] = Field(
        default_factory=dict,
        description="section name → text, e.g. {'introduction': '...', 'methods': '...'}"
    )
    structured_references: list[dict] = Field(default_factory=list)

    # ── Raw review text ───────────────────────────────────
    decision_letter_raw: str = ""
    author_response_raw: str = ""

    # ── Extracted concerns ────────────────────────────────
    concerns: list[ReviewerConcern] = Field(default_factory=list)

    # ── Reproducibility (CRITICAL_REVIEW B2) ─────────────
    extraction_manifest_id: str = Field(
        default="",
        description="ExtractionManifest ID used to extract this entry"
    )

    schema_version: str = "1.1"

    @property
    def clean_subjects(self) -> list[str]:
        """Scientific subject list with article-type labels removed."""
        return [s for s in self.subjects if s not in _ARTICLE_TYPE_LABELS]

    @property
    def requires_figure_reading_count(self) -> int:
        return sum(1 for c in self.concerns if c.requires_figure_reading)

    @property
    def scorable_concerns(self) -> list[ReviewerConcern]:
        """Concerns used for base metric computation (figure_issue excluded)."""
        return [c for c in self.concerns if not c.requires_figure_reading]

    @property
    def is_usable(self) -> bool:
        """Whether this entry can be used for benchmark evaluation.

        Requires at least one concern to be a valid benchmark sample.
        """
        return len(self.concerns) > 0

    def to_task_input(self) -> dict:
        """Input format for AI tools (compliant with TASK_DEFINITION.md).

        Excludes leakage fields such as decision_letter, author_response.
        """
        return {
            "article_id": self.id,
            "doi": self.doi,
            "journal": self.source,
            "title": self.title,
            "abstract": self.abstract,
            "sections": self.paper_text_sections,
            "references": self.structured_references,
        }

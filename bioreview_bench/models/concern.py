"""ReviewerConcern schema — two-stage silver label system."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class ConcernCategory(str, Enum):
    DESIGN_FLAW = "design_flaw"
    STATISTICAL_METHODOLOGY = "statistical_methodology"
    MISSING_EXPERIMENT = "missing_experiment"
    FIGURE_ISSUE = "figure_issue"
    PRIOR_ART_NOVELTY = "prior_art_novelty"
    WRITING_CLARITY = "writing_clarity"
    REAGENT_METHOD_SPECIFICITY = "reagent_method_specificity"
    INTERPRETATION = "interpretation"
    OTHER = "other"


class AuthorStance(str, Enum):
    """Silver label — addresses CRITICAL_REVIEW B1.

    Stance inferred from the author response. This is an outcome-anchored
    silver label, not an objective ground truth.
    """
    CONCEDED = "conceded"       # Explicit agreement + mention of revision
    REBUTTED = "rebutted"       # Clear counter-argument provided
    PARTIAL = "partial"         # Partial agreement + partial rebuttal, or agreed but no changes
    UNCLEAR = "unclear"         # Ambiguous response
    NO_RESPONSE = "no_response" # No response to this specific concern


class Resolution(str, Enum):
    """Summary alias for AuthorStance (backward compatibility)."""
    CONCEDED = "conceded"
    REBUTTED = "rebutted"
    PARTIAL = "partial"
    UNCLEAR = "unclear"
    NO_RESPONSE = "no_response"


class ReviewerConcern(BaseModel):
    """A single reviewer concern with author response outcome."""

    # ── Identifiers ───────────────────────────────────────
    concern_id: str = Field(
        description="Unique ID, e.g. 'elife:84798:R1C3'"
    )
    reviewer_num: int = Field(ge=1, description="Reviewer number (1-indexed)")

    # ── Concern content ───────────────────────────────────
    concern_text: str = Field(min_length=10, max_length=2000)
    category: ConcernCategory
    severity: Literal["major", "minor", "optional"]
    author_response_text: str | None = None

    # ── Two-stage silver label (CRITICAL_REVIEW B1) ───────
    author_stance: AuthorStance
    evidence_of_change: bool | None = Field(
        default=None,
        description="Evidence of manuscript revision or additional experiments. None=unclear"
    )
    resolution_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classifier confidence 0.0-1.0"
    )

    # Derived fields (auto-computed)
    resolution: Resolution = Field(
        description="Summary alias based on author_stance (backward compatibility)"
    )
    was_valid: bool | None = Field(
        default=None,
        description="conceded/partial AND evidence_of_change is not False"
    )

    # ── Quality flags ─────────────────────────────────────
    raised_by_multiple: bool = False
    requires_figure_reading: bool = Field(
        default=False,
        description="Concern requires viewing a figure to assess. Excluded from v1.0 base metrics."
    )

    # ── Reproducibility (CRITICAL_REVIEW B2) ─────────────
    extraction_trace_id: str = Field(description="Extraction trace UUID")
    extraction_manifest_id: str = Field(description="Which manifest was used for extraction")

    # ── Source ────────────────────────────────────────────
    source: str = Field(description="e.g. 'elife', 'plos'")
    article_doi: str

    @model_validator(mode="before")
    @classmethod
    def derive_resolution_and_validity(cls, data: dict) -> dict:
        """Auto-compute resolution and was_valid from author_stance."""
        stance_raw = data.get("author_stance")
        evidence = data.get("evidence_of_change")

        if stance_raw is not None:
            # resolution maps directly to author_stance
            if "resolution" not in data or data.get("resolution") is None:
                data["resolution"] = stance_raw

            # was_valid: True when conceded/partial AND evidence_of_change is not False
            stance_str = (
                stance_raw.value
                if isinstance(stance_raw, AuthorStance)
                else str(stance_raw)
            )
            if stance_str in ("conceded", "partial"):
                data["was_valid"] = evidence is not False
            else:
                data["was_valid"] = False

        return data

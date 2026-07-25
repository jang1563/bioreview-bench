"""Schema validation for OpenPeerReviewEntry records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from bioreview_bench.models.concern import AuthorStance, Resolution
from bioreview_bench.models.entry import OpenPeerReviewEntry


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation finding for one entry."""

    field: str
    message: str
    severity: Literal["error", "warning"] = "error"


@dataclass
class ValidationResult:
    """Aggregated validation outcome for a single OpenPeerReviewEntry."""

    entry_id: str
    issues: list[ValidationIssue] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """True when there are no error-level issues."""
        return all(i.severity != "error" for i in self.issues)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable one-liner describing the result."""
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"[{status}] {self.entry_id}: "
            f"{self.n_errors} error(s), {self.n_warnings} warning(s)"
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

# Pattern: <source>:<article_id>  e.g. "elife:84798" or "plos:10.1371/abc"
_ID_PATTERN = re.compile(r"^[a-z0-9_]+:.+$")


class SchemaValidator:
    """Validate OpenPeerReviewEntry objects against the benchmark schema."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(self, entry: OpenPeerReviewEntry) -> ValidationResult:
        """Validate a single parsed OpenPeerReviewEntry.

        Args:
            entry: A fully-constructed OpenPeerReviewEntry instance.

        Returns:
            A ValidationResult collecting all detected issues.
        """
        result = ValidationResult(entry_id=entry.id)
        self._check_identifiers(entry, result)
        self._check_article_metadata(entry, result)
        self._check_review_data(entry, result)
        self._check_concerns(entry, result)
        self._check_reproducibility(entry, result)
        return result

    def validate_batch(
        self, entries: list[OpenPeerReviewEntry]
    ) -> list[ValidationResult]:
        """Validate a list of entries and return one ValidationResult per entry.

        Args:
            entries: List of OpenPeerReviewEntry instances.

        Returns:
            List of ValidationResult objects in the same order as *entries*.
        """
        return [self.validate(e) for e in entries]

    def validate_dict(self, data: dict) -> ValidationResult:
        """Validate a raw dictionary by first parsing it with Pydantic.

        If Pydantic parsing fails, the parse errors are returned as
        error-level issues and no further checks are performed.

        Args:
            data: Raw mapping (e.g. parsed from JSON/JSONL).

        Returns:
            ValidationResult.  entry_id is taken from data["id"] when
            available, otherwise falls back to "<unknown>".
        """
        entry_id = str(data.get("id", "<unknown>"))
        try:
            entry = OpenPeerReviewEntry.model_validate(data)
        except ValidationError as exc:
            result = ValidationResult(entry_id=entry_id)
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                result.issues.append(
                    ValidationIssue(
                        field=loc,
                        message=f"Parse error: {err['msg']}",
                        severity="error",
                    )
                )
            return result

        return self.validate(entry)

    # ------------------------------------------------------------------
    # Private check methods
    # ------------------------------------------------------------------

    def _check_identifiers(
        self, entry: OpenPeerReviewEntry, result: ValidationResult
    ) -> None:
        """Check id format (source:article_id) and DOI prefix."""
        if not _ID_PATTERN.match(entry.id):
            result.issues.append(
                ValidationIssue(
                    field="id",
                    message=(
                        f"Entry id '{entry.id}' does not match expected "
                        "'<source>:<article_id>' pattern (e.g. 'elife:84798')."
                    ),
                    severity="error",
                )
            )
        else:
            # The source prefix in the id should match entry.source
            id_source = entry.id.split(":", 1)[0]
            if id_source != entry.source:
                result.issues.append(
                    ValidationIssue(
                        field="id",
                        message=(
                            f"Source prefix in id ('{id_source}') does not "
                            f"match entry.source ('{entry.source}')."
                        ),
                        severity="error",
                    )
                )

        if not entry.doi.startswith("10."):
            result.issues.append(
                ValidationIssue(
                    field="doi",
                    message=(
                        f"DOI '{entry.doi}' does not start with '10.' — "
                        "expected a valid DOI."
                    ),
                    severity="error",
                )
            )

    def _check_article_metadata(
        self, entry: OpenPeerReviewEntry, result: ValidationResult
    ) -> None:
        """Check title, abstract length, sections, and editorial decision."""
        if not entry.title.strip():
            result.issues.append(
                ValidationIssue(
                    field="title",
                    message="Title must not be empty.",
                    severity="error",
                )
            )

        if len(entry.abstract) < 50:
            result.issues.append(
                ValidationIssue(
                    field="abstract",
                    message=(
                        f"Abstract is only {len(entry.abstract)} character(s); "
                        "expected at least 50."
                    ),
                    severity="error",
                )
            )

        if not entry.paper_text_sections:
            result.issues.append(
                ValidationIssue(
                    field="paper_text_sections",
                    message="At least one paper_text_section is required.",
                    severity="error",
                )
            )

        if entry.editorial_decision == "unknown":
            result.issues.append(
                ValidationIssue(
                    field="editorial_decision",
                    message=(
                        "editorial_decision is 'unknown'; "
                        "consider assigning a specific decision label."
                    ),
                    severity="warning",
                )
            )

    def _check_review_data(
        self, entry: OpenPeerReviewEntry, result: ValidationResult
    ) -> None:
        """Check consistency between has_author_response and raw text fields."""
        if entry.has_author_response and not entry.author_response_raw.strip():
            result.issues.append(
                ValidationIssue(
                    field="author_response_raw",
                    message=(
                        "has_author_response is True but author_response_raw "
                        "is empty."
                    ),
                    severity="warning",
                )
            )

        if not entry.decision_letter_raw.strip():
            result.issues.append(
                ValidationIssue(
                    field="decision_letter_raw",
                    message="decision_letter_raw is empty.",
                    severity="warning",
                )
            )

    def _check_concerns(
        self, entry: OpenPeerReviewEntry, result: ValidationResult
    ) -> None:
        """Check concern list integrity."""
        seen_ids: set[str] = set()

        for concern in entry.concerns:
            cid = concern.concern_id

            # Duplicate concern_id
            if cid in seen_ids:
                result.issues.append(
                    ValidationIssue(
                        field="concerns.concern_id",
                        message=f"Duplicate concern_id '{cid}'.",
                        severity="error",
                    )
                )
            seen_ids.add(cid)

            # resolution must mirror author_stance
            expected_resolution = Resolution(concern.author_stance.value)
            if concern.resolution != expected_resolution:
                result.issues.append(
                    ValidationIssue(
                        field=f"concerns[{cid}].resolution",
                        message=(
                            f"resolution '{concern.resolution.value}' does not "
                            f"match author_stance '{concern.author_stance.value}'. "
                            "They must be equal."
                        ),
                        severity="error",
                    )
                )

            # was_valid consistency
            stance = concern.author_stance
            if stance in (AuthorStance.CONCEDED, AuthorStance.PARTIAL):
                expected_valid = concern.evidence_of_change is not False
            else:
                expected_valid = False

            if concern.was_valid is not None and concern.was_valid != expected_valid:
                result.issues.append(
                    ValidationIssue(
                        field=f"concerns[{cid}].was_valid",
                        message=(
                            f"was_valid={concern.was_valid} is inconsistent with "
                            f"author_stance='{stance.value}' and "
                            f"evidence_of_change={concern.evidence_of_change}; "
                            f"expected {expected_valid}."
                        ),
                        severity="warning",
                    )
                )

            # resolution_confidence bounds (Pydantic enforces ge/le but belt-and-braces)
            if not (0.0 <= concern.resolution_confidence <= 1.0):
                result.issues.append(
                    ValidationIssue(
                        field=f"concerns[{cid}].resolution_confidence",
                        message=(
                            f"resolution_confidence={concern.resolution_confidence} "
                            "is outside [0.0, 1.0]."
                        ),
                        severity="error",
                    )
                )

            # concern_text length (Pydantic enforces, belt-and-braces)
            text_len = len(concern.concern_text)
            if text_len < 10:
                result.issues.append(
                    ValidationIssue(
                        field=f"concerns[{cid}].concern_text",
                        message=(
                            f"concern_text is only {text_len} character(s); "
                            "minimum is 10."
                        ),
                        severity="error",
                    )
                )
            elif text_len > 2000:
                result.issues.append(
                    ValidationIssue(
                        field=f"concerns[{cid}].concern_text",
                        message=(
                            f"concern_text is {text_len} character(s); "
                            "maximum is 2000."
                        ),
                        severity="error",
                    )
                )

    def _check_reproducibility(
        self, entry: OpenPeerReviewEntry, result: ValidationResult
    ) -> None:
        """Check extraction provenance fields."""
        if not entry.extraction_manifest_id.strip():
            result.issues.append(
                ValidationIssue(
                    field="extraction_manifest_id",
                    message="extraction_manifest_id is empty.",
                    severity="warning",
                )
            )

        for concern in entry.concerns:
            if not concern.extraction_trace_id.strip():
                result.issues.append(
                    ValidationIssue(
                        field=f"concerns[{concern.concern_id}].extraction_trace_id",
                        message=(
                            f"Concern '{concern.concern_id}' has an empty "
                            "extraction_trace_id."
                        ),
                        severity="warning",
                    )
                )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def validate_jsonl_file(
    path: Path, verbose: bool = False
) -> tuple[int, int, int]:
    """Validate all entries in a JSONL file.

    Each line must be a JSON object representing one OpenPeerReviewEntry.
    Blank lines are skipped silently.

    Args:
        path: Path to the ``.jsonl`` file to validate.
        verbose: When True, print a summary line for every entry to stdout.

    Returns:
        ``(n_total, n_valid, n_invalid)`` tuple.
    """
    validator = SchemaValidator()
    n_total = 0
    n_valid = 0
    n_invalid = 0

    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            n_total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                n_invalid += 1
                if verbose:
                    print(
                        f"[ERROR] Line {line_no}: JSON parse error — {exc}"
                    )
                continue

            result = validator.validate_dict(data)
            if result.is_valid:
                n_valid += 1
            else:
                n_invalid += 1

            if verbose:
                print(f"  Line {line_no}: {result.summary()}")

    return n_total, n_valid, n_invalid

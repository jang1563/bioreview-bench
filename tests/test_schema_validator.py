"""Tests for SchemaValidator against OpenPeerReviewEntry records."""

from __future__ import annotations

import pytest

from bioreview_bench.models.concern import Resolution, ReviewerConcern
from bioreview_bench.models.entry import OpenPeerReviewEntry
from bioreview_bench.validate.schema_validator import SchemaValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validator() -> SchemaValidator:
    return SchemaValidator()


def _build_entry(overrides: dict | None = None) -> OpenPeerReviewEntry:
    """Build a fully valid OpenPeerReviewEntry, applying optional overrides."""
    base: dict = {
        "id": "elife:84798",
        "source": "elife",
        "doi": "10.7554/eLife.84798",
        "title": "A study of neural circuits in Drosophila melanogaster",
        "abstract": (
            "We investigated the role of specific neural circuits in Drosophila. "
            "Our results show that these circuits control locomotion."
        ),
        "published_date": "2023-01-15",
        "paper_text_sections": {"introduction": "Background text.", "methods": "We used..."},
        "extraction_manifest_id": "em-v1.0",
        "editorial_decision": "accept",
        "concerns": [
            {
                "concern_id": "elife:84798:R1C1",
                "reviewer_num": 1,
                "concern_text": "The statistical analysis is insufficient and lacks proper controls.",
                "category": "statistical_methodology",
                "severity": "major",
                "author_stance": "conceded",
                "resolution_confidence": 0.9,
                "extraction_trace_id": "trace-abc-123",
                "extraction_manifest_id": "em-v1.0",
                "source": "elife",
                "article_doi": "10.7554/eLife.84798",
            }
        ],
    }
    if overrides:
        base.update(overrides)
    return OpenPeerReviewEntry.model_validate(base)


# ---------------------------------------------------------------------------
# Basic validity tests
# ---------------------------------------------------------------------------


def test_valid_entry_passes(sample_entry_dict):
    """A properly constructed entry with no issues validates with no errors."""
    # Augment the sample with fields the validator requires
    data = {
        **sample_entry_dict,
        "extraction_manifest_id": "em-v1.0",
        "editorial_decision": "accept",
        "decision_letter_raw": "We are pleased to accept your manuscript.",
    }
    entry = OpenPeerReviewEntry.model_validate(data)
    result = _validator().validate(entry)
    assert result.n_errors == 0


def test_empty_id_is_error():
    """An entry whose id is empty (does not match source:article_id pattern) → error."""
    entry = _build_entry()
    # Bypass Pydantic to set id to an invalid value
    object.__setattr__(entry, "id", "")
    result = _validator().validate(entry)
    assert result.n_errors >= 1
    assert any("id" in issue.field for issue in result.issues if issue.severity == "error")


def test_missing_doi_prefix():
    """A DOI that does not start with '10.' → error."""
    entry = _build_entry()
    object.__setattr__(entry, "doi", "not-a-valid-doi/12345")
    result = _validator().validate(entry)
    assert result.n_errors >= 1
    assert any("doi" in issue.field for issue in result.issues if issue.severity == "error")


def test_short_abstract_is_error():
    """An abstract shorter than 50 characters triggers an error-level issue."""
    entry = _build_entry()
    object.__setattr__(entry, "abstract", "Too short.")
    result = _validator().validate(entry)
    assert any("abstract" in issue.field for issue in result.issues if issue.severity == "error")


def test_duplicate_concern_ids():
    """Two concerns sharing the same concern_id → error."""
    concern_data = {
        "concern_id": "elife:84798:R1C1",
        "reviewer_num": 1,
        "concern_text": "The statistical analysis is insufficient and lacks proper controls.",
        "category": "statistical_methodology",
        "severity": "major",
        "author_stance": "conceded",
        "resolution_confidence": 0.9,
        "extraction_trace_id": "trace-abc-123",
        "extraction_manifest_id": "em-v1.0",
        "source": "elife",
        "article_doi": "10.7554/eLife.84798",
    }
    # Second concern has the same concern_id
    concern_data_2 = {**concern_data, "concern_text": "Another concern text that is long enough."}
    entry = OpenPeerReviewEntry.model_validate(
        {
            "id": "elife:84798",
            "source": "elife",
            "doi": "10.7554/eLife.84798",
            "title": "A study of neural circuits in Drosophila melanogaster",
            "abstract": (
                "We investigated the role of specific neural circuits in Drosophila. "
                "Our results show that these circuits control locomotion."
            ),
            "published_date": "2023-01-15",
            "paper_text_sections": {"introduction": "Background text."},
            "extraction_manifest_id": "em-v1.0",
            "editorial_decision": "accept",
            "concerns": [concern_data, concern_data_2],
        }
    )
    result = _validator().validate(entry)
    assert result.n_errors >= 1
    dup_errors = [
        i for i in result.issues
        if i.severity == "error" and "Duplicate" in i.message
    ]
    assert len(dup_errors) >= 1


def test_resolution_mismatch_is_error():
    """resolution != author_stance on a concern → error from _check_concerns."""
    entry = _build_entry()
    # The Pydantic model_validator auto-derives resolution == author_stance.
    # We mutate resolution directly after construction to create a mismatch.
    concern = entry.concerns[0]
    # concern.author_stance is CONCEDED; we force resolution to REBUTTED
    object.__setattr__(concern, "resolution", Resolution.REBUTTED)

    result = _validator().validate(entry)
    assert result.n_errors >= 1
    mismatch_errors = [
        i for i in result.issues
        if i.severity == "error" and "resolution" in i.field and "match" in i.message
    ]
    assert len(mismatch_errors) >= 1


def test_empty_manifest_id_is_warning():
    """extraction_manifest_id="" → warning from _check_reproducibility."""
    entry = _build_entry({"extraction_manifest_id": ""})
    result = _validator().validate(entry)
    warnings = [i for i in result.issues if i.severity == "warning" and "manifest" in i.field]
    assert len(warnings) >= 1


def test_unknown_decision_is_warning():
    """editorial_decision='unknown' → warning from _check_article_metadata."""
    entry = _build_entry({"editorial_decision": "unknown"})
    result = _validator().validate(entry)
    warnings = [
        i for i in result.issues
        if i.severity == "warning" and "editorial_decision" in i.field
    ]
    assert len(warnings) >= 1


def test_n_errors_n_warnings():
    """n_errors and n_warnings properties return correct counts."""
    entry = _build_entry()
    # Force two separate issues: a DOI error and an editorial_decision warning
    object.__setattr__(entry, "doi", "invalid-doi-no-prefix")
    object.__setattr__(entry, "editorial_decision", "unknown")
    result = _validator().validate(entry)

    assert result.n_errors == sum(1 for i in result.issues if i.severity == "error")
    assert result.n_warnings == sum(1 for i in result.issues if i.severity == "warning")
    # Verify they match the raw counts
    assert result.n_errors >= 1     # DOI error
    assert result.n_warnings >= 1   # unknown editorial_decision warning


def test_is_valid_no_errors():
    """is_valid is True only when there are zero error-level issues."""
    # Fully valid entry → is_valid=True
    entry = _build_entry()
    object.__setattr__(entry, "doi", "10.7554/eLife.84798")  # ensure valid DOI
    result_valid = _validator().validate(entry)
    # Strip any warnings by checking is_valid property logic
    assert result_valid.is_valid == (result_valid.n_errors == 0)

    # Entry with an error → is_valid=False
    object.__setattr__(entry, "doi", "invalid-no-prefix")
    result_invalid = _validator().validate(entry)
    assert result_invalid.n_errors >= 1
    assert result_invalid.is_valid is False

"""Tests for ReviewerConcern and OpenPeerReviewEntry models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bioreview_bench.models.concern import (
    AuthorStance,
    ConcernCategory,
    Resolution,
    ReviewerConcern,
)
from bioreview_bench.models.entry import OpenPeerReviewEntry


# ---------------------------------------------------------------------------
# ReviewerConcern tests
# ---------------------------------------------------------------------------


def test_concern_creation_minimal(sample_concern_dict):
    """Create a ReviewerConcern from the minimal valid dict and verify it parses."""
    concern = ReviewerConcern.model_validate(sample_concern_dict)

    assert concern.concern_id == "elife:84798:R1C1"
    assert concern.reviewer_num == 1
    assert concern.category == ConcernCategory.STATISTICAL_METHODOLOGY
    assert concern.severity == "major"
    assert concern.author_stance == AuthorStance.CONCEDED
    assert concern.resolution_confidence == 0.9
    # resolution is auto-derived
    assert concern.resolution is not None


def test_concern_resolution_auto_derived(sample_concern_dict):
    """resolution must equal author_stance value after model_validator runs."""
    concern = ReviewerConcern.model_validate(sample_concern_dict)
    assert concern.resolution == Resolution.CONCEDED
    assert concern.resolution.value == concern.author_stance.value


def test_concern_was_valid_conceded(sample_concern_dict):
    """was_valid is True when stance=conceded and evidence_of_change is None."""
    data = {**sample_concern_dict, "author_stance": "conceded", "evidence_of_change": None}
    concern = ReviewerConcern.model_validate(data)
    assert concern.was_valid is True


def test_concern_was_valid_conceded_with_evidence_false(sample_concern_dict):
    """was_valid is False when stance=conceded but evidence_of_change=False."""
    data = {**sample_concern_dict, "author_stance": "conceded", "evidence_of_change": False}
    concern = ReviewerConcern.model_validate(data)
    assert concern.was_valid is False


def test_concern_was_valid_partial(sample_concern_dict):
    """was_valid is True when stance=partial and evidence_of_change is None."""
    data = {**sample_concern_dict, "author_stance": "partial", "evidence_of_change": None}
    concern = ReviewerConcern.model_validate(data)
    assert concern.was_valid is True


def test_concern_was_valid_rebutted(sample_concern_dict):
    """was_valid is False when stance=rebutted."""
    data = {**sample_concern_dict, "author_stance": "rebutted"}
    concern = ReviewerConcern.model_validate(data)
    assert concern.was_valid is False


def test_concern_was_valid_no_response(sample_concern_dict):
    """was_valid is False when stance=no_response."""
    data = {**sample_concern_dict, "author_stance": "no_response"}
    concern = ReviewerConcern.model_validate(data)
    assert concern.was_valid is False


def test_concern_text_too_short(sample_concern_dict):
    """Pydantic raises ValidationError when concern_text is shorter than 10 chars."""
    data = {**sample_concern_dict, "concern_text": "Too short"}
    with pytest.raises(ValidationError) as exc_info:
        ReviewerConcern.model_validate(data)
    errors = exc_info.value.errors()
    assert any("concern_text" in str(e["loc"]) for e in errors)


def test_concern_reviewer_num_zero(sample_concern_dict):
    """Pydantic raises ValidationError when reviewer_num=0 (must be >= 1)."""
    data = {**sample_concern_dict, "reviewer_num": 0}
    with pytest.raises(ValidationError) as exc_info:
        ReviewerConcern.model_validate(data)
    errors = exc_info.value.errors()
    assert any("reviewer_num" in str(e["loc"]) for e in errors)


def test_concern_confidence_out_of_range(sample_concern_dict):
    """Pydantic raises ValidationError when resolution_confidence > 1.0."""
    data = {**sample_concern_dict, "resolution_confidence": 1.5}
    with pytest.raises(ValidationError) as exc_info:
        ReviewerConcern.model_validate(data)
    errors = exc_info.value.errors()
    assert any("resolution_confidence" in str(e["loc"]) for e in errors)


# ---------------------------------------------------------------------------
# OpenPeerReviewEntry tests
# ---------------------------------------------------------------------------


def test_entry_creation_minimal(sample_entry_dict):
    """Create an OpenPeerReviewEntry from the minimal valid dict."""
    entry = OpenPeerReviewEntry.model_validate(sample_entry_dict)

    assert entry.id == "elife:84798"
    assert entry.source == "elife"
    assert entry.doi == "10.7554/eLife.84798"
    assert entry.title == "A study of neural circuits in Drosophila"
    assert entry.schema_version == "1.1"
    assert len(entry.concerns) == 1


def test_entry_clean_subjects(sample_entry_dict):
    """clean_subjects removes known article-type labels and keeps scientific subjects."""
    data = {
        **sample_entry_dict,
        "subjects": [
            "Neuroscience",
            "Research Article",
            "Genetics",
            "Short Report",
            "Cell Biology",
        ],
    }
    entry = OpenPeerReviewEntry.model_validate(data)
    clean = entry.clean_subjects

    assert "Neuroscience" in clean
    assert "Genetics" in clean
    assert "Cell Biology" in clean
    assert "Research Article" not in clean
    assert "Short Report" not in clean


def test_entry_scorable_concerns(sample_concern_dict):
    """scorable_concerns excludes concerns where requires_figure_reading=True."""
    figure_concern = {
        **sample_concern_dict,
        "concern_id": "elife:84798:R1C2",
        "requires_figure_reading": True,
        "category": "figure_issue",
    }
    non_figure_concern = {
        **sample_concern_dict,
        "concern_id": "elife:84798:R1C1",
        "requires_figure_reading": False,
    }
    entry = OpenPeerReviewEntry.model_validate(
        {
            "id": "elife:84798",
            "source": "elife",
            "doi": "10.7554/eLife.84798",
            "title": "A study of neural circuits in Drosophila",
            "abstract": "We investigated the role of specific neural circuits " * 3,
            "published_date": "2023-01-15",
            "paper_text_sections": {"introduction": "Background..."},
            "concerns": [figure_concern, non_figure_concern],
        }
    )
    scorable = entry.scorable_concerns
    assert len(scorable) == 1
    assert scorable[0].concern_id == "elife:84798:R1C1"


def test_entry_is_usable_with_concerns(sample_entry_dict):
    """is_usable returns True when the entry has at least one concern."""
    entry = OpenPeerReviewEntry.model_validate(sample_entry_dict)
    assert entry.is_usable is True


def test_entry_is_usable_without_concerns(sample_entry_dict):
    """is_usable returns False when the concerns list is empty."""
    data = {**sample_entry_dict, "concerns": []}
    entry = OpenPeerReviewEntry.model_validate(data)
    assert entry.is_usable is False


def test_entry_requires_figure_reading_count(sample_concern_dict):
    """requires_figure_reading_count returns the count of figure-requiring concerns."""
    figure_concern_1 = {
        **sample_concern_dict,
        "concern_id": "elife:84798:R1C2",
        "requires_figure_reading": True,
        "category": "figure_issue",
    }
    figure_concern_2 = {
        **sample_concern_dict,
        "concern_id": "elife:84798:R1C3",
        "requires_figure_reading": True,
        "category": "figure_issue",
    }
    non_figure_concern = {
        **sample_concern_dict,
        "concern_id": "elife:84798:R1C1",
        "requires_figure_reading": False,
    }
    entry = OpenPeerReviewEntry.model_validate(
        {
            "id": "elife:84798",
            "source": "elife",
            "doi": "10.7554/eLife.84798",
            "title": "A study of neural circuits in Drosophila",
            "abstract": "We investigated the role of specific neural circuits " * 3,
            "published_date": "2023-01-15",
            "paper_text_sections": {"introduction": "Background..."},
            "concerns": [figure_concern_1, figure_concern_2, non_figure_concern],
        }
    )
    assert entry.requires_figure_reading_count == 2


def test_entry_to_task_input(sample_entry_dict):
    """to_task_input() includes required keys and excludes leakage fields."""
    entry = OpenPeerReviewEntry.model_validate(sample_entry_dict)
    task_input = entry.to_task_input()

    required_keys = {"article_id", "doi", "journal", "title", "abstract", "sections", "references"}
    assert set(task_input.keys()) == required_keys

    # Verify values are correct
    assert task_input["article_id"] == entry.id
    assert task_input["doi"] == entry.doi
    assert task_input["journal"] == entry.source
    assert task_input["title"] == entry.title
    assert task_input["abstract"] == entry.abstract
    assert task_input["sections"] == entry.paper_text_sections
    assert task_input["references"] == entry.structured_references

    # Leakage fields must NOT be present
    leakage_fields = {
        "decision_letter_raw",
        "author_response_raw",
        "concerns",
        "editorial_decision",
    }
    for field in leakage_fields:
        assert field not in task_input, f"Leakage field '{field}' found in task input"


def test_entry_round_trip_json(sample_entry_dict):
    """model_dump_json + model_validate_json round trip preserves the model."""
    original = OpenPeerReviewEntry.model_validate(sample_entry_dict)
    json_str = original.model_dump_json()
    restored = OpenPeerReviewEntry.model_validate_json(json_str)

    assert restored.id == original.id
    assert restored.doi == original.doi
    assert restored.source == original.source
    assert restored.title == original.title
    assert restored.published_date == original.published_date
    assert len(restored.concerns) == len(original.concerns)
    assert restored.concerns[0].concern_id == original.concerns[0].concern_id
    assert restored.concerns[0].resolution == original.concerns[0].resolution
    assert restored.concerns[0].was_valid == original.concerns[0].was_valid

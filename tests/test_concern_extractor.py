from __future__ import annotations

import pytest

from bioreview_bench.parse.concern_extractor import (
    ConcernExtractor,
    split_into_reviewer_blocks,
)
from bioreview_bench.parse.jats import ParsedReview


def test_coerce_evidence_of_change_strict_bool() -> None:
    assert ConcernExtractor._coerce_evidence_of_change("true") is True
    assert ConcernExtractor._coerce_evidence_of_change("false") is False
    assert ConcernExtractor._coerce_evidence_of_change("unknown") is None
    assert ConcernExtractor._coerce_evidence_of_change(1) is True
    assert ConcernExtractor._coerce_evidence_of_change(0) is False


def test_article_token_from_doi_elife_revision_suffix() -> None:
    token = ConcernExtractor._article_token_from_doi("10.7554/eLife.84798.3")
    assert token == "84798"


def test_process_review_builds_stable_concern_id(monkeypatch) -> None:
    extractor = ConcernExtractor(manifest_id="em-test")

    monkeypatch.setattr(
        extractor,
        "_extract_concerns_from_review",
        lambda _review_text: [
            {
                "text": "The control condition is missing for experiment 2.",
                "category": "design_flaw",
                "severity": "major",
            }
        ],
    )
    monkeypatch.setattr(
        extractor,
        "_classify_resolutions",
        lambda _concerns, _response, reviewer_num=1: [
            {"author_stance": "conceded", "evidence_of_change": "false"}
        ],
    )

    review = ParsedReview(
        reviewer_num=2,
        review_text="Dummy review",
        author_response_text="We agree but cannot add experiments in this revision.",
    )
    concerns = extractor.process_review(
        review,
        article_doi="10.7554/eLife.84798.3",
        article_source="elife",
    )

    assert len(concerns) == 1
    assert concerns[0].concern_id == "elife:84798:R2C1"
    assert concerns[0].evidence_of_change is False


# ── Multi-provider tests ─────────────────────────────────────────────────────


def test_lazy_client_init() -> None:
    """Client should be None until first LLM call."""
    ext = ConcernExtractor(model="test", provider="anthropic")
    assert ext._client is None


def test_invalid_provider_raises() -> None:
    ext = ConcernExtractor(model="test", provider="invalid")
    with pytest.raises(ValueError, match="Unsupported provider"):
        ext._get_client()


def test_provider_dispatch_openai(monkeypatch) -> None:
    """Verify OpenAI client is created for provider='openai'."""
    mock_client = object()
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda: mock_client)
    ext = ConcernExtractor(model="gpt-4o-mini", provider="openai")
    assert ext._get_client() is mock_client


# ── split_into_reviewer_blocks tests ─────────────────────────────────────────


def test_split_single_block() -> None:
    """No reviewer headers → single block."""
    result = split_into_reviewer_blocks("Some review text without headers")
    assert len(result) == 1
    assert "Some review text" in result[0]


def test_split_multiple_reviewers() -> None:
    """Standard reviewer headers split correctly."""
    text = "Reviewer #1: First review\nReviewer #2: Second review"
    result = split_into_reviewer_blocks(text)
    assert len(result) == 2


def test_split_empty_input() -> None:
    """Empty string → empty list."""
    assert split_into_reviewer_blocks("") == []
    assert split_into_reviewer_blocks("   ") == []

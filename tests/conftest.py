"""Shared pytest fixtures for the bioreview-bench test suite."""

from __future__ import annotations

import pytest
from datetime import date


@pytest.fixture
def sample_concern_dict():
    """Minimal valid ReviewerConcern dict."""
    return {
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


@pytest.fixture
def sample_entry_dict(sample_concern_dict):
    """Minimal valid OpenPeerReviewEntry dict."""
    return {
        "id": "elife:84798",
        "source": "elife",
        "doi": "10.7554/eLife.84798",
        "title": "A study of neural circuits in Drosophila",
        "abstract": "We investigated the role of specific neural circuits " * 3,
        "published_date": "2023-01-15",
        "paper_text_sections": {"introduction": "Background...", "methods": "We used..."},
        "concerns": [sample_concern_dict],
    }

"""Tests for build_ensemble_gt.py ensemble logic (Jaccard mode, no SPECTER2)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioreview_bench.evaluate.metrics import ConcernMatcher
from scripts.build_ensemble_gt import build_ensemble_concerns


def _matcher(threshold: float = 0.65, borderline: float = 0.50) -> ConcernMatcher:
    """Jaccard-mode matcher (no sentence-transformers)."""
    return ConcernMatcher(
        threshold=threshold,
        exclude_figure=False,
        use_embedding=False,
        algorithm="hungarian",
    )


def _make_concern(text: str, category: str = "other", severity: str = "major") -> dict:
    return {
        "concern_text": text,
        "category": category,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# Basic matching tests
# ---------------------------------------------------------------------------


def test_identical_concerns_all_matched():
    """When both extractors produce identical concerns, all are 'both'."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    haiku = {"concerns": [_make_concern(text)]}
    gpt = {"concerns": [_make_concern(text)]}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), threshold=0.65, borderline_threshold=0.50)

    assert len(result) == 1
    assert result[0]["ensemble_agreement"] == "both"
    assert result[0]["ensemble_match_score"] >= 0.65


def test_empty_both():
    """Both empty → empty result."""
    haiku = {"concerns": []}
    gpt = {"concerns": []}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), threshold=0.65, borderline_threshold=0.50)
    assert result == []


def test_haiku_only_all_excluded():
    """Haiku has concerns, GPT has none → all haiku excluded (max_sim=0.0 < borderline)."""
    haiku = {"concerns": [_make_concern("The statistical analysis is wrong.")]}
    gpt = {"concerns": []}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), threshold=0.65, borderline_threshold=0.50)
    # max_sim to empty set = 0.0, which is < borderline (0.50), so excluded
    assert len(result) == 0


def test_gpt_only_all_excluded():
    """GPT has concerns, Haiku has none → all gpt excluded."""
    haiku = {"concerns": []}
    gpt = {"concerns": [_make_concern("The statistical analysis is wrong.")]}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), threshold=0.65, borderline_threshold=0.50)
    assert len(result) == 0


def test_no_overlap_both_excluded():
    """Completely unrelated concerns → all excluded."""
    haiku = {"concerns": [_make_concern("The statistical analysis is completely wrong and needs revision.")]}
    gpt = {"concerns": [_make_concern("Fluorescent microscopy imaging shows abnormal cellular distribution patterns.")]}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), threshold=0.65, borderline_threshold=0.50)
    # Jaccard similarity between these should be ~0, both excluded
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Agreement type tests
# ---------------------------------------------------------------------------


def test_matched_pair_uses_haiku_text():
    """Matched pairs use Haiku text, not GPT text."""
    haiku_text = "The statistical analysis lacks proper controls and sufficient sample size validation."
    gpt_text = "Statistical analysis lacks proper controls and sufficient sample size validation."
    haiku = {"concerns": [_make_concern(haiku_text, category="statistical_methodology")]}
    gpt = {"concerns": [_make_concern(gpt_text, category="statistical_methodology")]}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), threshold=0.65, borderline_threshold=0.50)

    assert len(result) == 1
    assert result[0]["concern_text"] == haiku_text
    assert result[0]["ensemble_agreement"] == "both"


def test_category_agreement_flag():
    """category_agreed is True when both agree, False otherwise."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    haiku = {"concerns": [_make_concern(text, category="statistical_methodology")]}
    gpt_same = {"concerns": [_make_concern(text, category="statistical_methodology")]}
    gpt_diff = {"concerns": [_make_concern(text, category="design_flaw")]}

    result_same = build_ensemble_concerns(haiku, gpt_same, _matcher(), 0.65, 0.50)
    result_diff = build_ensemble_concerns(haiku, gpt_diff, _matcher(), 0.65, 0.50)

    assert result_same[0]["category_agreed"] is True
    assert result_diff[0]["category_agreed"] is False


def test_severity_conservative_major():
    """If either extractor says 'major', ensemble severity is 'major'."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    haiku = {"concerns": [_make_concern(text, severity="minor")]}
    gpt = {"concerns": [_make_concern(text, severity="major")]}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), 0.65, 0.50)

    assert result[0]["severity"] == "major"


def test_severity_both_minor():
    """If both say 'minor', ensemble severity is 'minor'."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    haiku = {"concerns": [_make_concern(text, severity="minor")]}
    gpt = {"concerns": [_make_concern(text, severity="minor")]}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), 0.65, 0.50)

    assert result[0]["severity"] == "minor"


# ---------------------------------------------------------------------------
# Multiple concerns
# ---------------------------------------------------------------------------


def test_multiple_matched_pairs():
    """Multiple concerns that all match get 'both' agreement."""
    texts = [
        "The statistical analysis lacks proper controls and sufficient sample size.",
        "The western blot normalization procedure is missing and needs correction.",
    ]
    haiku = {"concerns": [_make_concern(t) for t in texts]}
    gpt = {"concerns": [_make_concern(t) for t in texts]}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), 0.65, 0.50)

    assert len(result) == 2
    assert all(c["ensemble_agreement"] == "both" for c in result)


def test_ensemble_match_score_stored():
    """ensemble_match_score is stored and is a float."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    haiku = {"concerns": [_make_concern(text)]}
    gpt = {"concerns": [_make_concern(text)]}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), 0.65, 0.50)

    assert isinstance(result[0]["ensemble_match_score"], float)
    assert 0.0 <= result[0]["ensemble_match_score"] <= 1.0


# ---------------------------------------------------------------------------
# String concerns (non-dict format)
# ---------------------------------------------------------------------------


def test_string_concerns_handled():
    """Concerns that are plain strings (not dicts) should be handled."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    haiku = {"concerns": [text]}
    gpt = {"concerns": [text]}

    result = build_ensemble_concerns(haiku, gpt, _matcher(), 0.65, 0.50)

    assert len(result) == 1
    assert result[0]["concern_text"] == text

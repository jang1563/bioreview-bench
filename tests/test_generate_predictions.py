"""Tests for scripts/generate_predictions.py helpers (no API calls)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.generate_predictions import (
    PREDICTION_SYSTEM,
    _TRUNCATION_RE,
    _parse_json,
    _truncate_sections,
)


# ---------------------------------------------------------------------------
# Prompt checks (A1)
# ---------------------------------------------------------------------------


def test_prompt_no_fixed_count():
    """Prompt should NOT contain the old '5-15' fixed count advice."""
    assert "5-15" not in PREDICTION_SYSTEM
    assert "Aim for" not in PREDICTION_SYSTEM


def test_prompt_adaptive_count():
    """Prompt should encourage adaptive concern count."""
    assert "as many or as few" in PREDICTION_SYSTEM
    assert "Do NOT pad" in PREDICTION_SYSTEM


def test_prompt_truncation_rule():
    """Prompt should warn about truncation markers."""
    assert "truncated" in PREDICTION_SYSTEM.lower()
    assert "Do NOT raise concerns about missing or incomplete text" in PREDICTION_SYSTEM


# ---------------------------------------------------------------------------
# Truncation marker (A2)
# ---------------------------------------------------------------------------


def test_truncation_marker_added_when_truncated():
    """Sections that exceed the limit should have a truncation marker."""
    entry = {
        "title": "Test",
        "abstract": "Abstract.",
        "paper_text_sections": {
            "introduction": "x" * 5000,  # limit is 4000
        },
    }
    text = _truncate_sections(entry)
    assert "[…truncated]" in text


def test_no_truncation_marker_when_short():
    """Short sections should NOT have a truncation marker."""
    entry = {
        "title": "Test",
        "abstract": "Abstract.",
        "paper_text_sections": {
            "introduction": "Short intro.",
        },
    }
    text = _truncate_sections(entry)
    assert "[…truncated]" not in text


def test_truncation_marker_for_extra_sections():
    """Non-standard sections should also get markers when truncated."""
    entry = {
        "title": "Test",
        "abstract": "Abstract.",
        "paper_text_sections": {
            "data_availability": "y" * 3000,  # limit is 2000
        },
    }
    text = _truncate_sections(entry)
    assert "[…truncated]" in text


def test_truncated_text_length_within_limits():
    """Output section text should not exceed the limit + marker."""
    entry = {
        "title": "Test",
        "abstract": "Abstract.",
        "paper_text_sections": {
            "methods": "z" * 10000,  # limit is 5000
        },
    }
    text = _truncate_sections(entry)
    # Find the METHODS section content (between METHODS: and next section or end)
    lines = text.split("\n")
    in_methods = False
    methods_chars = 0
    for line in lines:
        if line == "METHODS:":
            in_methods = True
            continue
        if in_methods:
            if line.startswith("[…truncated]"):
                methods_chars += len(line)
                break
            methods_chars += len(line) + 1  # +1 for newline
    # Methods content should be around 5000 + marker
    assert methods_chars <= 5100


# ---------------------------------------------------------------------------
# Truncation regex filter (A2)
# ---------------------------------------------------------------------------


class TestTruncationRegex:
    """Test _TRUNCATION_RE catches artifact concerns without false positives."""

    def test_matches_truncated(self):
        assert _TRUNCATION_RE.search("The methods section appears to be truncated.")

    def test_matches_truncation(self):
        assert _TRUNCATION_RE.search("Due to truncation, the discussion is incomplete.")

    def test_matches_cut_off(self):
        assert _TRUNCATION_RE.search("The results section is cut off before conclusions.")

    def test_matches_text_incomplete(self):
        assert _TRUNCATION_RE.search("The text is incomplete and lacks final analysis.")

    def test_matches_not_shown(self):
        assert _TRUNCATION_RE.search("The remainder of the section is not shown.")

    def test_matches_stop_abruptly(self):
        assert _TRUNCATION_RE.search("The methods section appears to stop abruptly.")

    def test_no_match_normal_concern(self):
        """Normal scientific concerns should NOT be filtered."""
        assert not _TRUNCATION_RE.search(
            "The statistical analysis lacks proper controls and sufficient sample size."
        )

    def test_no_match_missing_experiment(self):
        assert not _TRUNCATION_RE.search(
            "The study does not include a negative control experiment."
        )

    def test_no_match_writing_concern(self):
        assert not _TRUNCATION_RE.search(
            "The methods section lacks sufficient detail on antibody concentrations."
        )

    def test_no_match_design_concern(self):
        assert not _TRUNCATION_RE.search(
            "The sample size is insufficient for the claimed statistical power."
        )

    def test_no_match_truncated_protein(self):
        """Real science: truncated protein construct should NOT be filtered."""
        assert not _TRUNCATION_RE.search(
            "The truncated protein construct lacks the C-terminal domain "
            "needed for proper folding validation."
        )

    def test_no_match_truncating_mutations(self):
        """Real science: truncating mutations (genetics) should NOT be filtered."""
        assert not _TRUNCATION_RE.search(
            "The study does not address whether truncating mutations in the "
            "BRCA1 gene lead to different phenotypes."
        )

    def test_no_match_truncating_series(self):
        """Real science: mathematical truncation should NOT be filtered."""
        assert not _TRUNCATION_RE.search(
            "Error bounds for truncating the infinite series at finite order "
            "are not discussed."
        )

    def test_matches_mid_sentence(self):
        """Mid-sentence ending is a truncation artifact."""
        assert _TRUNCATION_RE.search(
            "The Results section is incomplete, ending mid-sentence."
        )


# ---------------------------------------------------------------------------
# _parse_json (existing, no changes needed but verify still works)
# ---------------------------------------------------------------------------


def test_parse_json_array():
    raw = '[{"text": "concern 1", "category": "design_flaw"}]'
    result = _parse_json(raw)
    assert len(result) == 1
    assert result[0]["text"] == "concern 1"


def test_parse_json_fenced():
    raw = '```json\n[{"text": "c1"}]\n```'
    result = _parse_json(raw)
    assert len(result) == 1


def test_parse_json_with_trailing():
    raw = '[{"text": "c1"}]\n\nI hope this helps!'
    result = _parse_json(raw)
    assert len(result) == 1


def test_parse_json_empty():
    assert _parse_json("no json here") == []

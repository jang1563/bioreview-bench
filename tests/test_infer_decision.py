"""Tests for JATSParser._infer_decision() and correction title filter."""

from __future__ import annotations

import pytest

from bioreview_bench.parse.jats import JATSParser


@pytest.fixture
def parser():
    return JATSParser()


# ── _infer_decision: major revision keywords ──────────────────────────────

@pytest.mark.parametrize("text", [
    "The authors should perform major revision of the manuscript.",
    "We recommend major revisions before acceptance.",
    "Essential revision: the statistical analysis must be redone.",
    "The manuscript requires essential revisions.",
    "Substantive revision is needed for the methods section.",
    "We request substantive revisions to address the concerns.",
    "Substantial revision of the figures is required.",
    "Substantial revisions are expected.",
    "Significant revision to the discussion is needed.",
    "Significant revisions are required before publication.",
])
def test_major_revision_keywords(parser, text):
    assert parser._infer_decision(text) == "major_revision"


# ── _infer_decision: minor revision keywords ──────────────────────────────

@pytest.mark.parametrize("text", [
    "Only minor revision is needed.",
    "The paper needs minor revisions.",
    "Optional revision of figure labels.",
    "A few optional revisions are suggested.",
])
def test_minor_revision_keywords(parser, text):
    assert parser._infer_decision(text) == "minor_revision"


# ── _infer_decision: accept / reject / unknown ───────────────────────────

def test_accept(parser):
    assert parser._infer_decision("We are pleased to accept this manuscript.") == "accept"


def test_accept_with_revision_returns_unknown(parser):
    """'accept' + 'revision' in same text → should NOT be accept."""
    assert parser._infer_decision("We accept pending revision.") != "accept"


def test_reject(parser):
    assert parser._infer_decision("We regret to reject this manuscript.") == "reject"


def test_unknown_no_keywords(parser):
    assert parser._infer_decision("Thank you for your submission.") == "unknown"


def test_empty_string(parser):
    assert parser._infer_decision("") == "unknown"


# ── _infer_decision: precedence ───────────────────────────────────────────

def test_major_beats_minor(parser):
    """When both major and minor keywords present, major wins."""
    text = "The paper needs major revision of methods and minor revision of figures."
    assert parser._infer_decision(text) == "major_revision"


def test_major_beats_accept(parser):
    text = "We would accept after major revision."
    assert parser._infer_decision(text) == "major_revision"


def test_major_beats_reject(parser):
    text = "We reject without major revision."
    assert parser._infer_decision(text) == "major_revision"


# ── _infer_decision: case insensitivity ───────────────────────────────────

def test_case_insensitive(parser):
    assert parser._infer_decision("MAJOR REVISION required") == "major_revision"
    assert parser._infer_decision("Minor Revisions suggested") == "minor_revision"
    assert parser._infer_decision("ACCEPT this paper") == "accept"


# ── Correction title regex ────────────────────────────────────────────────

import re
import sys
sys.path.insert(0, "scripts")


def _load_correction_re():
    """Load the regex from rebuild_splits.py without importing the full script."""
    return re.compile(
        r"^(correction|erratum|retraction|corrigendum)[\s:]", re.IGNORECASE
    )


_CORRECTION_RE = _load_correction_re()


@pytest.mark.parametrize("title", [
    "Correction: Original study title here",
    "Correction to the article 'Neural circuits'",
    "Erratum: Missing figure data",
    "Erratum for Smith et al., 2023",
    "Retraction: Fabricated data detected",
    "Retraction notice for study XYZ",
    "Corrigendum: Updated author affiliations",
    "CORRECTION: case insensitive test",
    "Erratum - dash separator",
])
def test_correction_title_matches(title):
    assert _CORRECTION_RE.match(title), f"Should match: {title!r}"


@pytest.mark.parametrize("title", [
    "A study requiring correction of methodology",
    "Neural circuits in the erratum region",
    "Post-retraction analysis of data",
    "The corrigendum pathway in cellular biology",
    "A normal research article title",
    "",
])
def test_correction_title_does_not_match(title):
    assert not _CORRECTION_RE.match(title), f"Should NOT match: {title!r}"

"""Tests for rebuild_splits.py split functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.rebuild_splits import (
    balanced_test_split,
    frozen_split,
    get_stratum,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entries(source: str, n: int, decision: str = "accept") -> list[dict]:
    """Create n dummy entries for a given source."""
    return [
        {
            "id": f"{source}:{i}",
            "source": source,
            "editorial_decision": decision,
            "review_format": "journal",
            "concerns": [{"concern_text": f"concern {i}"}],
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# get_stratum tests
# ---------------------------------------------------------------------------


def test_get_stratum():
    entry = {
        "source": "elife",
        "editorial_decision": "accept",
        "review_format": "journal",
    }
    assert get_stratum(entry) == "elife|accept|journal"


def test_get_stratum_missing_fields():
    assert get_stratum({}) == "unknown|unknown|unknown"


# ---------------------------------------------------------------------------
# stratified_split tests
# ---------------------------------------------------------------------------


def test_stratified_split_basic():
    entries = _make_entries("elife", 50) + _make_entries("plos", 50)
    train, val, test = stratified_split(entries, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(train) + len(val) + len(test) == 100
    assert len(val) > 0
    assert len(test) > 0

    # No duplicates
    all_ids = [e["id"] for e in train + val + test]
    assert len(all_ids) == len(set(all_ids))


def test_stratified_split_reproducible():
    entries = _make_entries("elife", 100)
    split1 = stratified_split(entries, val_ratio=0.15, test_ratio=0.15, seed=42)
    split2 = stratified_split(entries, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert [e["id"] for e in split1[0]] == [e["id"] for e in split2[0]]
    assert [e["id"] for e in split1[1]] == [e["id"] for e in split2[1]]
    assert [e["id"] for e in split1[2]] == [e["id"] for e in split2[2]]


def test_stratified_split_different_seeds():
    entries = _make_entries("elife", 100)
    split1 = stratified_split(entries, val_ratio=0.15, test_ratio=0.15, seed=42)
    split2 = stratified_split(entries, val_ratio=0.15, test_ratio=0.15, seed=99)

    # Different seeds should produce different splits
    ids1 = {e["id"] for e in split1[2]}
    ids2 = {e["id"] for e in split2[2]}
    assert ids1 != ids2


# ---------------------------------------------------------------------------
# frozen_split tests
# ---------------------------------------------------------------------------


def test_frozen_split_preserves_test_ids():
    entries = _make_entries("elife", 50)
    frozen_ids = {f"elife:{i}" for i in range(10)}

    train, val, test = frozen_split(entries, frozen_ids, val_ratio=0.15, seed=42)

    test_ids = {e["id"] for e in test}
    assert test_ids == frozen_ids
    assert len(train) + len(val) + len(test) == 50


def test_frozen_split_missing_ids_ignored():
    entries = _make_entries("elife", 20)
    frozen_ids = {"elife:0", "elife:1", "nonexistent:999"}

    train, val, test = frozen_split(entries, frozen_ids, val_ratio=0.15, seed=42)

    test_ids = {e["id"] for e in test}
    assert "nonexistent:999" not in test_ids
    assert len(test) == 2


# ---------------------------------------------------------------------------
# balanced_test_split tests
# ---------------------------------------------------------------------------


def test_balanced_test_split_exact_counts():
    """Each source gets exactly the requested number in test."""
    entries = (
        _make_entries("elife", 100)
        + _make_entries("plos", 100)
        + _make_entries("f1000", 100)
    )
    per_source_test = {"elife": 15, "plos": 15, "f1000": 15}

    train, val, test = balanced_test_split(entries, per_source_test, val_ratio=0.15, seed=42)

    # Verify per-source test counts
    source_counts = {}
    for e in test:
        s = e["source"]
        source_counts[s] = source_counts.get(s, 0) + 1

    assert source_counts["elife"] == 15
    assert source_counts["plos"] == 15
    assert source_counts["f1000"] == 15
    assert len(test) == 45

    # Total preserved
    assert len(train) + len(val) + len(test) == 300


def test_balanced_test_split_no_duplicates():
    entries = _make_entries("elife", 50) + _make_entries("plos", 50)
    per_source_test = {"elife": 10, "plos": 10}

    train, val, test = balanced_test_split(entries, per_source_test, val_ratio=0.15, seed=42)

    all_ids = [e["id"] for e in train + val + test]
    assert len(all_ids) == len(set(all_ids))


def test_balanced_test_split_caps_at_available():
    """If fewer articles than requested, cap at available."""
    entries = _make_entries("elife", 5)
    per_source_test = {"elife": 100}

    train, val, test = balanced_test_split(entries, per_source_test, val_ratio=0.15, seed=42)

    assert len(test) == 5
    assert len(train) == 0
    assert len(val) == 0


def test_balanced_test_split_reproducible():
    entries = _make_entries("elife", 100) + _make_entries("plos", 100)
    per_source_test = {"elife": 20, "plos": 20}

    split1 = balanced_test_split(entries, per_source_test, val_ratio=0.15, seed=42)
    split2 = balanced_test_split(entries, per_source_test, val_ratio=0.15, seed=42)

    assert [e["id"] for e in split1[2]] == [e["id"] for e in split2[2]]


def test_balanced_test_split_missing_source_key():
    """Sources not in per_source_test get 0 test articles."""
    entries = _make_entries("elife", 50) + _make_entries("nature", 20)
    per_source_test = {"elife": 10}  # nature not listed

    train, val, test = balanced_test_split(entries, per_source_test, val_ratio=0.15, seed=42)

    test_sources = {e["source"] for e in test}
    assert "nature" not in test_sources
    assert len(test) == 10
    assert len(train) + len(val) + len(test) == 70

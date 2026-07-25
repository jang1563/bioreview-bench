"""Tests for rebuild_splits.py split functions."""

# ruff: noqa: E402, I001

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import rebuild_splits as rebuild_splits_module
from scripts.rebuild_splits import (
    balanced_test_split,
    frozen_split,
    get_stratum,
    load_frozen_ids,
    main,
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


def _write_entries(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


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


@pytest.mark.parametrize("key", ["ids", "test_ids"])
def test_load_frozen_ids_accepts_current_and_legacy_keys(
    tmp_path: Path,
    key: str,
) -> None:
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({key: ["elife:1", "plos:2"]}), encoding="utf-8")

    assert load_frozen_ids(path) == {"elife:1", "plos:2"}


def test_load_frozen_ids_accepts_matching_dual_schema(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    path.write_text(
        json.dumps(
            {
                "ids": ["elife:1", "plos:2"],
                "test_ids": ["plos:2", "elife:1"],
            }
        ),
        encoding="utf-8",
    )

    assert load_frozen_ids(path) == {"elife:1", "plos:2"}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"ids": []},
        {"ids": ["elife:1", "elife:1"]},
        {"ids": ["elife:1", 2]},
        {"ids": ["elife:1"], "test_ids": ["plos:2"]},
    ],
)
def test_load_frozen_ids_rejects_malformed_artifacts(
    tmp_path: Path,
    payload: dict,
) -> None:
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_frozen_ids(path)


def test_bare_command_cannot_overwrite_canonical_v4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rebuild_splits_module, "ROOT", tmp_path)

    result = CliRunner().invoke(main, [])

    assert result.exit_code != 0
    assert "Refusing to overwrite canonical data/splits/v4" in result.output
    assert not (tmp_path / "data" / "splits" / "v4" / "train.jsonl").exists()


def test_generic_ratio_mode_requires_noncanonical_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "processed"
    output_dir = tmp_path / "splits" / "experiment"
    _write_entries(input_dir / "elife_v1.1.jsonl", _make_entries("elife", 20))

    result = CliRunner().invoke(
        main,
        [
            "-s",
            "elife",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "train.jsonl").exists()
    frozen_payload = json.loads(
        (output_dir / "test_ids_frozen_experiment.json").read_text(encoding="utf-8")
    )
    assert set(frozen_payload) == {"ids"}


def test_frozen_cli_reads_writer_schema_and_fails_on_missing_ids(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "processed"
    output_dir = tmp_path / "splits" / "experiment"
    _write_entries(input_dir / "elife_v1.1.jsonl", _make_entries("elife", 20))
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(
        json.dumps({"ids": ["elife:0", "elife:missing"]}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "-s",
            "elife",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--frozen-test",
            str(frozen_path),
        ],
    )

    assert result.exit_code != 0
    assert "frozen test IDs are missing" in result.output
    assert not (output_dir / "train.jsonl").exists()


def test_canonical_balanced_mode_must_reproduce_release_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rebuild_splits_module, "ROOT", tmp_path)
    input_dir = tmp_path / "data" / "processed"
    for source, filename in {
        "elife": "elife_v1.1.jsonl",
        "plos": "plos_v1.jsonl",
        "f1000": "f1000_v1.jsonl",
        "nature": "nature_v1.jsonl",
        "peerj": "peerj_v1.jsonl",
    }.items():
        _write_entries(input_dir / filename, _make_entries(source, 3))

    result = CliRunner().invoke(
        main,
        [
            "--balanced-test",
            '{"elife":150,"plos":150,"f1000":150,"nature":100,"peerj":50}',
        ],
    )

    assert result.exit_code != 0
    assert "must reproduce" in result.output
    assert "5387" in result.output
    assert not (tmp_path / "data" / "splits" / "v4" / "train.jsonl").exists()


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

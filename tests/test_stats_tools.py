from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioreview_bench.stats import (
    check_documentation,
    paired_micro_f1_randomization_pvalue,
    paired_sign_flip_pvalue,
    summarize_splits,
)


def test_summarize_splits_small_fixture(tmp_path: Path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    train_entry = {
        "id": "a:1",
        "source": "alpha",
        "review_format": "journal",
        "has_author_response": True,
        "concerns": [
            {"category": "design_flaw", "severity": "major", "author_stance": "conceded"},
            {"category": "other", "severity": "minor", "author_stance": "partial"},
        ],
    }
    val_entry = {
        "id": "b:1",
        "source": "beta",
        "review_format": "reviewed_preprint",
        "has_author_response": False,
        "concerns": [
            {"category": "interpretation", "severity": "major", "author_stance": "no_response"},
        ],
    }

    (splits_dir / "train.jsonl").write_text(json.dumps(train_entry) + "\n", encoding="utf-8")
    (splits_dir / "val.jsonl").write_text(json.dumps(val_entry) + "\n", encoding="utf-8")
    (splits_dir / "test.jsonl").write_text("", encoding="utf-8")

    summary = summarize_splits(splits_dir)

    assert summary["total_articles"] == 2
    assert summary["total_concerns"] == 3
    assert summary["splits"]["train"]["articles"] == 1
    assert summary["splits"]["validation"]["concerns"] == 1
    assert summary["severity_distribution"]["major"]["count"] == 2
    assert summary["author_stance_distribution"]["no_response"]["count"] == 1


def test_repo_docs_match_generated_split_stats():
    splits_dir = Path("data/splits/v4")
    summary_path = Path("data/stats/v4_summary.json")
    frozen_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_paths = [splits_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl")]

    if all(path.exists() for path in split_paths):
        summary = summarize_splits(splits_dir)
        assert summary == frozen_summary
    else:
        summary = frozen_summary

    errors = check_documentation(summary, [Path("README.md"), Path("DATASHEET.md")])

    assert errors == []


def test_paired_sign_flip_is_exact_for_small_samples() -> None:
    assert paired_sign_flip_pvalue([3.0, 2.0, 1.0], [0.0, 0.0, 0.0]) == 0.25
    assert paired_sign_flip_pvalue([1.0, 1.0], [0.0, 0.0]) == 0.5


def test_paired_sign_flip_handles_empty_and_tied_pairs() -> None:
    assert paired_sign_flip_pvalue([], []) is None
    assert paired_sign_flip_pvalue([0.2, 0.8], [0.2, 0.8]) == 1.0


def test_paired_sign_flip_monte_carlo_is_seeded() -> None:
    left = [float(index % 7) for index in range(40)]
    right = [float((index + 2) % 5) for index in range(40)]

    first = paired_sign_flip_pvalue(
        left,
        right,
        exact_max_pairs=8,
        n_resamples=2_000,
        seed=17,
    )
    second = paired_sign_flip_pvalue(
        left,
        right,
        exact_max_pairs=8,
        n_resamples=2_000,
        seed=17,
    )

    assert first == second
    assert first is not None and 0.0 < first <= 1.0


def test_paired_sign_flip_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_sign_flip_pvalue([1.0], [])
    with pytest.raises(ValueError, match="finite"):
        paired_sign_flip_pvalue([float("nan")], [0.0])
    with pytest.raises(ValueError, match="n_resamples"):
        paired_sign_flip_pvalue(
            range(20),
            [0.0] * 20,
            exact_max_pairs=1,
            n_resamples=0,
        )


def test_paired_micro_f1_randomization_is_exact() -> None:
    counts_a = [(1, 1, 1), (1, 1, 1)]
    counts_b = [(0, 1, 1), (0, 1, 1)]

    assert paired_micro_f1_randomization_pvalue(counts_a, counts_b) == 0.5


def test_paired_micro_f1_randomization_recomputes_nonlinear_metric() -> None:
    counts_a = [(2, 4, 2), (3, 3, 3), (0, 3, 3)]
    counts_b = [(0, 4, 2), (0, 3, 5), (3, 3, 5)]

    observed = paired_micro_f1_randomization_pvalue(counts_a, counts_b)
    reversed_tools = paired_micro_f1_randomization_pvalue(counts_b, counts_a)

    assert observed == 0.75
    assert reversed_tools == 0.75


def test_paired_micro_f1_randomization_monte_carlo_is_seeded_and_nonzero() -> None:
    counts_a = [(1, 2, 1) if index % 2 else (0, 2, 3) for index in range(40)]
    counts_b = [(0, 2, 3) if index % 2 else (1, 2, 1) for index in range(40)]

    first = paired_micro_f1_randomization_pvalue(
        counts_a,
        counts_b,
        exact_max_pairs=8,
        n_resamples=2_000,
        seed=17,
    )
    second = paired_micro_f1_randomization_pvalue(
        counts_a,
        counts_b,
        exact_max_pairs=8,
        n_resamples=2_000,
        seed=17,
    )

    assert first == second
    assert first is not None and 0.0 < first <= 1.0


def test_paired_micro_f1_randomization_rejects_misaligned_counts() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_micro_f1_randomization_pvalue([(1, 1, 1)], [])
    with pytest.raises(ValueError, match="same reference count"):
        paired_micro_f1_randomization_pvalue([(1, 1, 1)], [(1, 2, 1)])
    with pytest.raises(ValueError, match="cannot exceed"):
        paired_micro_f1_randomization_pvalue([(2, 1, 2)], [(0, 1, 1)])

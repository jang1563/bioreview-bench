from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from bioreview_bench.evaluate.leaderboard import Leaderboard


def _write_result(
    path: Path,
    *,
    threshold: float,
    algorithm: str,
    model: str = "allenai/specter2_base",
) -> None:
    data = {
        "tool_name": "tool-a",
        "tool_version": "v1",
        "git_hash": "",
        "benchmark_version": "4.1",
        "extraction_manifest_id": "em-v1.0",
        "split": "test",
        "run_date": datetime(2026, 3, 10).isoformat(),
        "recall_overall": 0.5,
        "precision_overall": 0.5,
        "f1_micro": 0.5,
        "recall_major": 0.5,
        "f1_macro": 0.0,
        "soft_recall_overall": 0.0,
        "soft_precision_overall": 0.0,
        "soft_f1": 0.0,
        "ci_recall": None,
        "ci_precision": None,
        "bootstrap_n": 0,
        "per_category": {},
        "per_stance": {},
        "matching_stats": {
            "n_tool_concerns": 10,
            "n_human_concerns": 10,
            "n_matched_pairs": 5,
            "threshold": threshold,
            "configured_threshold": threshold,
            "method": "embedding",
            "embedding_model": model,
            "embedding_revision": None,
            "algorithm": algorithm,
            "figure_policy": "exclude",
        },
        "n_articles": 1,
        "n_human_concerns": 10,
        "n_tool_concerns": 10,
        "excluded_figure_concerns": 0,
        "dedup_gt": False,
        "notes": "",
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_leaderboard_footer_uses_result_matching_metadata(tmp_path: Path) -> None:
    _write_result(tmp_path / "result.json", threshold=0.65, algorithm="hungarian")

    markdown = Leaderboard(results_dir=tmp_path, split="test").to_markdown()

    assert "threshold=0.65" in markdown
    assert "cardinality-first Hungarian matching" in markdown
    assert "`allenai/specter2_base`" in markdown
    assert "unadapted" in markdown
    assert "automatic mean pooling" in markdown
    assert "provisional" in markdown
    assert "unpopulated legacy sentinel" in markdown
    assert "category-macro F1 is invalid and unreported" in markdown
    assert "Frozen Score Snapshot" in markdown
    assert "not an open public leaderboard" in markdown
    assert "v4.1.3" in markdown


def test_leaderboard_excludes_non_frozen_matching_metadata(tmp_path: Path) -> None:
    _write_result(tmp_path / "result_a.json", threshold=0.65, algorithm="hungarian")
    _write_result(tmp_path / "result_b.json", threshold=0.85, algorithm="greedy")

    leaderboard = Leaderboard(results_dir=tmp_path, split="test")
    markdown = leaderboard.to_markdown()

    assert len(leaderboard.entries) == 1
    assert leaderboard.excluded_incompatible == ["result_b.json"]
    assert "Excluded 1 result file(s)" in markdown


def test_leaderboard_excludes_custom_embedding_model(tmp_path: Path) -> None:
    _write_result(tmp_path / "frozen.json", threshold=0.65, algorithm="hungarian")
    _write_result(
        tmp_path / "custom.json",
        threshold=0.65,
        algorithm="hungarian",
        model="org/custom-review-encoder",
    )

    leaderboard = Leaderboard(results_dir=tmp_path, split="test")

    assert len(leaderboard.entries) == 1
    assert leaderboard.entries[0].result_file == "frozen.json"
    assert leaderboard.excluded_incompatible == ["custom.json"]


def test_leaderboard_excludes_dedup_runs(tmp_path: Path) -> None:
    _write_result(tmp_path / "base.json", threshold=0.65, algorithm="hungarian")
    dedup = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    dedup["dedup_gt"] = True
    dedup["f1_micro"] = 0.99
    (tmp_path / "dedup.json").write_text(json.dumps(dedup), encoding="utf-8")

    lb = Leaderboard(results_dir=tmp_path, split="test")

    assert len(lb.entries) == 1
    assert lb.entries[0].f1 == 0.5


def test_leaderboard_keeps_best_run_per_tool(tmp_path: Path) -> None:
    _write_result(tmp_path / "older.json", threshold=0.65, algorithm="hungarian")

    newer = json.loads((tmp_path / "older.json").read_text(encoding="utf-8"))
    newer["f1_micro"] = 0.7
    newer["recall_overall"] = 0.7
    newer["precision_overall"] = 0.7
    newer["run_date"] = datetime(2026, 3, 11).isoformat()
    (tmp_path / "newer.json").write_text(json.dumps(newer), encoding="utf-8")

    lb = Leaderboard(results_dir=tmp_path, split="test")

    assert len(lb.entries) == 1
    assert lb.entries[0].f1 == 0.7

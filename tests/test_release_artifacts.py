from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.rebuild_release_artifacts import build_release_manifest


def _write_result(
    path: Path,
    *,
    tool_name: str,
    tool_version: str,
    f1_micro: float,
    dedup_gt: bool = False,
) -> None:
    data = {
        "tool_name": tool_name,
        "tool_version": tool_version,
        "git_hash": "",
        "benchmark_version": "1.0",
        "extraction_manifest_id": "em-v1.0",
        "split": "test",
        "run_date": datetime(2026, 3, 10, tzinfo=UTC).isoformat(),
        "recall_overall": f1_micro,
        "precision_overall": f1_micro,
        "f1_micro": f1_micro,
        "recall_major": f1_micro,
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
            "threshold": 0.65,
            "algorithm": "hungarian",
        },
        "n_articles": 1,
        "n_human_concerns": 10,
        "n_tool_concerns": 10,
        "excluded_figure_concerns": 0,
        "dedup_gt": dedup_gt,
        "notes": "",
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_release_manifest_uses_filtered_leaderboard(tmp_path: Path) -> None:
    results_dir = tmp_path / "results_v3"
    output_dir = tmp_path / "public_results"
    results_dir.mkdir()

    _write_result(
        results_dir / "tool_a.json",
        tool_name="ToolA",
        tool_version="v1",
        f1_micro=0.7,
    )
    _write_result(
        results_dir / "tool_a_dedup.json",
        tool_name="ToolA",
        tool_version="v1",
        f1_micro=0.99,
        dedup_gt=True,
    )
    _write_result(
        results_dir / "tool_b_old.json",
        tool_name="ToolB",
        tool_version="v2",
        f1_micro=0.5,
    )
    _write_result(
        results_dir / "tool_b_new.json",
        tool_name="ToolB",
        tool_version="v2",
        f1_micro=0.8,
    )

    manifest = build_release_manifest(
        source_results_dir=results_dir,
        output_dir=output_dir,
        split="test",
    )

    included = manifest["included_results"]
    assert len(included) == 2
    assert [row["tool_name"] for row in included] == ["ToolB", "ToolA"]
    assert included[0]["f1"] == 0.8
    assert included[1]["f1"] == 0.7
    assert manifest["matching"]["thresholds"] == [0.65]
    assert manifest["matching"]["algorithms"] == ["hungarian"]

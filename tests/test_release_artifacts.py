from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from bioreview_bench.models.benchmark import CategoryMetrics
from scripts.check_release_artifacts import (
    _load_documentation_summary,
    _render_release_artifacts,
    validate_release_artifacts,
)
from scripts.rebuild_release_artifacts import build_release_manifest


def _write_result(
    path: Path,
    *,
    tool_name: str,
    tool_version: str,
    f1_micro: float,
    f1_macro: float = 0.0,
    dedup_gt: bool = False,
) -> None:
    data = {
        "tool_name": tool_name,
        "tool_version": tool_version,
        "git_hash": "",
        "benchmark_version": "4.1",
        "extraction_manifest_id": "em-v1.0",
        "split": "test",
        "run_date": datetime(2026, 3, 10, tzinfo=UTC).isoformat(),
        "recall_overall": f1_micro,
        "precision_overall": f1_micro,
        "f1_micro": f1_micro,
        "recall_major": f1_micro,
        "f1_macro": f1_macro,
        "soft_recall_overall": 0.0,
        "soft_precision_overall": 0.0,
        "soft_f1": 0.0,
        "ci_recall": None,
        "ci_precision": None,
        "bootstrap_n": 0,
        "per_category": {
            "other": {
                "recall": f1_micro,
                "precision": f1_micro,
                "f1_micro": f1_micro,
                "f1_macro": 0.0,
                "aucpr": 0.0,
                "brier_score": None,
                "n_human_concerns": 10,
                "n_matched": 5,
                "ci_recall": None,
                "ci_precision": None,
            }
        },
        "per_stance": {},
        "matching_stats": {
            "n_tool_concerns": 10,
            "n_human_concerns": 10,
            "n_matched_pairs": 5,
            "threshold": 0.65,
            "configured_threshold": 0.65,
            "method": "embedding",
            "embedding_model": "allenai/specter2_base",
            "embedding_revision": None,
            "algorithm": "hungarian",
            "figure_policy": "exclude",
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
    assert manifest["matching"]["effective_threshold"] == 0.65
    assert manifest["matching"]["algorithm"] == "hungarian"
    assert manifest["matching"]["embedding_model"] == "allenai/specter2_base"
    assert manifest["matching"]["embedding_wrapper"] == (
        "SentenceTransformer automatic mean pooling"
    )
    assert manifest["matching"]["embedding_task_adapter"] is None
    assert manifest["matching"]["threshold_validation"].startswith("provisional")
    macro_status = manifest["metrics"]["f1_macro"]
    assert macro_status["definition"].startswith("unweighted mean")
    assert macro_status["historical_release_status"] == ("invalid_unpopulated_legacy_sentinel")
    assert macro_status["historical_release_reported"] is False
    assert macro_status["legacy_sentinel_value"] == 0.0
    assert macro_status["legacy_sentinel_result_files"] == [
        "tool_b_new.json",
        "tool_a.json",
    ]
    assert macro_status["runner_aggregation_status"] == (
        "computed_for_new_runs_in_v4.1.2_and_later"
    )
    assert manifest["software_release"] == "v4.1.3"
    assert manifest["dataset_split_version"] == "v4"
    assert manifest["source_results_dir"] == "results_v3"
    assert manifest["output_dir"] == "public_results"
    assert manifest["artifacts"]["leaderboard_md"] == "public_results/leaderboard.md"
    assert manifest["artifacts"]["leaderboard_json"] == "public_results/leaderboard.json"
    assert manifest["artifacts"]["release_manifest_json"] == "public_results/release_manifest.json"
    assert included[0]["result_file"] == "tool_b_new.json"
    assert included[1]["result_file"] == "tool_a.json"


def test_build_release_manifest_marks_populated_macro_f1_as_reported(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results_v4"
    output_dir = tmp_path / "public_results"
    results_dir.mkdir()
    _write_result(
        results_dir / "tool_a.json",
        tool_name="ToolA",
        tool_version="v1",
        f1_micro=0.7,
        f1_macro=0.7,
    )

    manifest = build_release_manifest(
        source_results_dir=results_dir,
        output_dir=output_dir,
        split="test",
    )

    macro_status = manifest["metrics"]["f1_macro"]
    assert macro_status["historical_release_status"] == "reported"
    assert macro_status["historical_release_reported"] is True
    assert macro_status["legacy_sentinel_value"] is None
    assert macro_status["legacy_sentinel_result_files"] == []


def test_build_release_manifest_includes_complete_measurement_audit_pair(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results_v4"
    output_dir = tmp_path / "public_results"
    results_dir.mkdir()
    _write_result(
        results_dir / "tool_a.json",
        tool_name="ToolA",
        tool_version="v1",
        f1_micro=0.7,
    )
    (results_dir / "measurement_audit.json").write_text("{}", encoding="utf-8")
    (results_dir / "measurement_audit.md").write_text("# Audit\n", encoding="utf-8")

    manifest = build_release_manifest(
        source_results_dir=results_dir,
        output_dir=output_dir,
        split="test",
    )

    assert manifest["artifacts"]["measurement_audit_json"] == ("results_v4/measurement_audit.json")
    assert manifest["artifacts"]["measurement_audit_md"] == ("results_v4/measurement_audit.md")


def test_category_metrics_preserves_historical_nested_sentinel_but_defaults_to_none() -> None:
    required = {
        "recall": 0.5,
        "precision": 0.5,
        "f1_micro": 0.5,
        "n_human_concerns": 10,
        "n_matched": 5,
    }

    assert CategoryMetrics(**required).f1_macro is None
    assert CategoryMetrics(**required).aucpr is None
    assert CategoryMetrics(**required, f1_macro=0.0).f1_macro == 0.0
    assert CategoryMetrics(**required, aucpr=0.0).aucpr == 0.0


def test_validate_release_artifacts_passes_for_regenerated_outputs(tmp_path: Path) -> None:
    versioned_results_dir = tmp_path / "results_v4"
    root_results_dir = tmp_path / "results"
    versioned_results_dir.mkdir()

    _write_result(
        versioned_results_dir / "tool_a.json",
        tool_name="ToolA",
        tool_version="v1",
        f1_micro=0.7,
    )
    _render_release_artifacts(
        source_results_dir=versioned_results_dir,
        output_dir=versioned_results_dir,
        split="test",
    )
    _render_release_artifacts(
        source_results_dir=versioned_results_dir,
        output_dir=root_results_dir,
        split="test",
    )

    assert (
        validate_release_artifacts(
            versioned_results_dir=versioned_results_dir,
            root_results_dir=root_results_dir,
            split="test",
        )
        == []
    )


def test_load_documentation_summary_uses_frozen_stats_without_private_splits(
    tmp_path: Path,
) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    stats_json = tmp_path / "summary.json"
    expected = {"total_articles": 12, "total_concerns": 34, "splits": {}}
    stats_json.write_text(json.dumps(expected), encoding="utf-8")

    summary = _load_documentation_summary(
        splits_dir=splits_dir,
        stats_json=stats_json,
    )

    assert summary == expected


def test_validate_release_artifacts_reports_drift(tmp_path: Path) -> None:
    versioned_results_dir = tmp_path / "results_v4"
    root_results_dir = tmp_path / "results"
    versioned_results_dir.mkdir()

    _write_result(
        versioned_results_dir / "tool_a.json",
        tool_name="ToolA",
        tool_version="v1",
        f1_micro=0.7,
    )
    _render_release_artifacts(
        source_results_dir=versioned_results_dir,
        output_dir=versioned_results_dir,
        split="test",
    )
    _render_release_artifacts(
        source_results_dir=versioned_results_dir,
        output_dir=root_results_dir,
        split="test",
    )
    (root_results_dir / "leaderboard.md").write_text("drift", encoding="utf-8")

    errors = validate_release_artifacts(
        versioned_results_dir=versioned_results_dir,
        root_results_dir=root_results_dir,
        split="test",
    )

    assert errors
    assert "Artifact drift detected" in errors[0]


def test_validate_release_artifacts_rejects_malformed_manifest_pointers(
    tmp_path: Path,
) -> None:
    versioned_results_dir = tmp_path / "results_v4"
    root_results_dir = tmp_path / "results"
    versioned_results_dir.mkdir()

    _write_result(
        versioned_results_dir / "tool_a.json",
        tool_name="ToolA",
        tool_version="v1",
        f1_micro=0.7,
    )
    _render_release_artifacts(
        source_results_dir=versioned_results_dir,
        output_dir=versioned_results_dir,
        split="test",
    )
    _render_release_artifacts(
        source_results_dir=versioned_results_dir,
        output_dir=root_results_dir,
        split="test",
    )

    manifest_path = root_results_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_generated_at"] = "not-a-timestamp"
    manifest["output_dir"] = "/Users/private/path"
    manifest["artifacts"] = {
        "leaderboard_md": "paper/private-review.md",
        "leaderboard_json": "missing.json",
        "release_manifest_json": "../release_manifest.json",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_release_artifacts(
        versioned_results_dir=versioned_results_dir,
        root_results_dir=root_results_dir,
        split="test",
    )

    joined = "\n".join(errors)
    assert "invalid release_generated_at" in joined
    assert "output_dir must be" in joined
    assert "artifacts must be exact repo-relative pointers" in joined


def test_v3_pairwise_significance_artifact_is_withdrawn() -> None:
    artifact = json.loads(Path("results/v3/pairwise_significance.json").read_text(encoding="utf-8"))

    assert artifact["status"] == "withdrawn_invalid_inference"
    assert artifact["numeric_results_included"] is False
    assert artifact["replacement_method"] == (
        "article_paired_label_swap_randomization_delta_micro_f1"
    )
    assert "p_value" not in artifact

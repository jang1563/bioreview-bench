"""Rebuild official release artifacts from evaluated result JSON files.

This script turns a directory of BenchmarkResult JSON files into:

- ``leaderboard.md``
- ``leaderboard.json``
- ``release_manifest.json``

The manifest freezes which result files are part of the public release and
records the evaluation settings inferred from those files.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from bioreview_bench.evaluate.leaderboard import (
    FROZEN_MATCHER_SIGNATURE,
    Leaderboard,
    update_leaderboard,
)
from bioreview_bench.project_defaults import (
    DEFAULT_BENCHMARK_SPLIT_VERSION,
    DEFAULT_PUBLIC_RESULTS_DIR,
    DEFAULT_SOFTWARE_RELEASE_VERSION,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_F1_MACRO_DEFINITION = (
    "unweighted mean of dataset-level per-category F1 values for represented "
    "non-figure reference categories"
)


def _portable_path(path: Path, *, base: Path | None = None) -> str:
    """Return a portable repo-relative path string where possible."""
    try:
        if base is not None:
            return path.relative_to(base).as_posix()
    except ValueError:
        pass

    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _legacy_f1_macro_sentinel_files(
    *,
    source_results_dir: Path,
    result_files: list[str],
) -> list[str]:
    """Identify included results whose stored macro F1 is a legacy sentinel.

    A zero overall ``f1_macro`` accompanied by any non-zero per-category F1
    proves that the overall field was not populated from its component values.
    """
    affected: list[str] = []
    for result_file in result_files:
        path = source_results_dir / result_file
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        per_category = payload.get("per_category")
        if not isinstance(per_category, dict):
            continue
        category_f1_values = [
            metrics.get("f1_micro")
            for metrics in per_category.values()
            if isinstance(metrics, dict)
        ]
        has_nonzero_category_f1 = any(
            isinstance(value, (int, float)) and value != 0.0
            for value in category_f1_values
        )
        if payload.get("f1_macro") == 0.0 and has_nonzero_category_f1:
            affected.append(result_file)

    return affected


def build_release_manifest(
    *,
    source_results_dir: Path,
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    """Build a release manifest from the filtered public leaderboard."""
    lb = Leaderboard(results_dir=source_results_dir, split=split)

    included_results: list[dict[str, Any]] = []
    for entry in lb.entries:
        included_results.append(
            {
                "rank": entry.rank,
                "tool_name": entry.tool_name,
                "tool_version": entry.tool_version,
                "result_file": entry.result_file,
                "run_date": entry.run_date,
                "recall": entry.recall,
                "precision": entry.precision,
                "f1": entry.f1,
                "recall_major": entry.recall_major,
                "n_articles": entry.n_articles,
            }
        )

    included_result_files = [row["result_file"] for row in included_results]
    legacy_macro_sentinel_files = _legacy_f1_macro_sentinel_files(
        source_results_dir=source_results_dir,
        result_files=included_result_files,
    )

    artifact_prefix = _portable_path(output_dir)
    artifacts = {
        "leaderboard_md": f"{artifact_prefix}/leaderboard.md",
        "leaderboard_json": f"{artifact_prefix}/leaderboard.json",
        "release_manifest_json": f"{artifact_prefix}/release_manifest.json",
    }
    audit_json = source_results_dir / "measurement_audit.json"
    audit_md = source_results_dir / "measurement_audit.md"
    if audit_json.exists() != audit_md.exists():
        raise ValueError(
            "Measurement audit JSON and Markdown must either both exist or both be absent"
        )
    if audit_json.exists():
        audit_prefix = _portable_path(source_results_dir)
        artifacts.update(
            {
                "measurement_audit_json": f"{audit_prefix}/measurement_audit.json",
                "measurement_audit_md": f"{audit_prefix}/measurement_audit.md",
            }
        )

    manifest = {
        "release_generated_at": datetime.now(UTC).isoformat(),
        "software_release": DEFAULT_SOFTWARE_RELEASE_VERSION,
        "dataset_split_version": DEFAULT_BENCHMARK_SPLIT_VERSION,
        "split": split,
        "source_results_dir": _portable_path(source_results_dir),
        "output_dir": _portable_path(output_dir),
        "policy": {
            "exclude_dedup_gt": True,
            "one_result_per_tool_version": True,
            "ranking_metric": "f1_micro",
        },
        "metrics": {
            "f1_micro": {
                "status": "reported",
                "role": "score_snapshot_ranking_metric",
            },
            "f1_macro": {
                "definition": _F1_MACRO_DEFINITION,
                "historical_release_status": (
                    "invalid_unpopulated_legacy_sentinel"
                    if legacy_macro_sentinel_files
                    else "reported"
                ),
                "historical_release_reported": not legacy_macro_sentinel_files,
                "legacy_sentinel_value": (
                    0.0 if legacy_macro_sentinel_files else None
                ),
                "legacy_sentinel_result_files": legacy_macro_sentinel_files,
                "runner_aggregation_status": (
                    "computed_for_new_runs_in_v4.1.2_and_later"
                ),
            },
        },
        "matching": {
            **asdict(FROZEN_MATCHER_SIGNATURE),
            "embedding_wrapper": "SentenceTransformer automatic mean pooling",
            "embedding_task_adapter": None,
            "threshold_validation": "provisional; independent validation pending",
        },
        "artifacts": artifacts,
        "included_results": included_results,
    }
    return manifest


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_PUBLIC_RESULTS_DIR,
    show_default=True,
    help="Directory containing evaluated BenchmarkResult JSON files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_PUBLIC_RESULTS_DIR,
    show_default=True,
    help="Directory where public release artifacts are written.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="test",
    show_default=True,
    help="Benchmark split to publish.",
)
def main(results_dir: Path, output_dir: Path, split: str) -> None:
    """Rebuild leaderboard artifacts and a frozen release manifest."""
    lb = update_leaderboard(results_dir=results_dir, split=split, output_dir=output_dir)
    manifest = build_release_manifest(
        source_results_dir=results_dir,
        output_dir=output_dir,
        split=split,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    click.echo(f"Leaderboard entries: {len(lb.entries)}")
    click.echo(f"Release manifest: {manifest_path}")


if __name__ == "__main__":
    main()

"""Validate that checked-in public artifacts match regenerated outputs."""

from __future__ import annotations

import difflib
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from bioreview_bench.evaluate.leaderboard import update_leaderboard
from bioreview_bench.project_defaults import (
    DEFAULT_BENCHMARK_SPLITS_DIR,
    DEFAULT_PUBLIC_RESULTS_DIR,
    DEFAULT_STATS_JSON,
    RESULTS_ROOT,
)
from bioreview_bench.stats import check_documentation, summarize_splits
from scripts.rebuild_release_artifacts import build_release_manifest

_ARTIFACT_FILES = ("leaderboard.md", "leaderboard.json", "release_manifest.json")
_SPLIT_FILENAMES = ("train.jsonl", "val.jsonl", "test.jsonl")
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _portable_expected_dir(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _validate_manifest_contract(
    manifest_path: Path,
    *,
    expected_output_dir: Path,
    expected_source_results_dir: Path,
) -> list[str]:
    """Validate public pointers instead of normalizing malformed values away."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return [f"Unable to read release manifest contract {manifest_path}: {exc}"]
    if not isinstance(payload, dict):
        return [f"{manifest_path}: release manifest must be a JSON object"]

    errors: list[str] = []
    timestamp = payload.get("release_generated_at")
    if not isinstance(timestamp, str):
        errors.append(f"{manifest_path}: release_generated_at must be an ISO timestamp")
    else:
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            errors.append(f"{manifest_path}: invalid release_generated_at {timestamp!r}")
        else:
            if parsed_timestamp.tzinfo is None:
                errors.append(
                    f"{manifest_path}: release_generated_at must include a timezone"
                )

    output_prefix = _portable_expected_dir(expected_output_dir)
    source_prefix = _portable_expected_dir(expected_source_results_dir)
    if payload.get("output_dir") != output_prefix:
        errors.append(
            f"{manifest_path}: output_dir must be {output_prefix!r}, "
            f"got {payload.get('output_dir')!r}"
        )
    if payload.get("source_results_dir") != source_prefix:
        errors.append(
            f"{manifest_path}: source_results_dir must be {source_prefix!r}, "
            f"got {payload.get('source_results_dir')!r}"
        )

    expected_artifacts = {
        "leaderboard_md": f"{output_prefix}/leaderboard.md",
        "leaderboard_json": f"{output_prefix}/leaderboard.json",
        "release_manifest_json": f"{output_prefix}/release_manifest.json",
    }
    audit_json = expected_source_results_dir / "measurement_audit.json"
    audit_md = expected_source_results_dir / "measurement_audit.md"
    if audit_json.exists() and audit_md.exists():
        expected_artifacts.update(
            {
                "measurement_audit_json": f"{source_prefix}/measurement_audit.json",
                "measurement_audit_md": f"{source_prefix}/measurement_audit.md",
            }
        )
    if payload.get("artifacts") != expected_artifacts:
        errors.append(
            f"{manifest_path}: artifacts must be exact repo-relative pointers "
            f"{expected_artifacts!r}, got {payload.get('artifacts')!r}"
        )
    return errors


def _render_release_artifacts(
    *,
    source_results_dir: Path,
    output_dir: Path,
    split: str,
) -> None:
    update_leaderboard(results_dir=source_results_dir, split=split, output_dir=output_dir)
    manifest = build_release_manifest(
        source_results_dir=source_results_dir,
        output_dir=output_dir,
        split=split,
    )
    (output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _normalize_artifact_text(filename: str, text: str) -> str:
    if filename == "leaderboard.md":
        return re.sub(
            r"\*Last updated: .*?\. Ranked by F1\.\*",
            "*Last updated: <normalized>. Ranked by F1.*",
            text,
        )
    if filename == "release_manifest.json":
        payload = json.loads(text)
        payload["release_generated_at"] = "<normalized>"
        payload["output_dir"] = "<normalized>"
        payload["artifacts"] = {
            key: "<normalized>"
            for key in (payload.get("artifacts") or {})
        }
        return json.dumps(payload, indent=2)
    return text


def _diff_text(
    filename: str,
    expected_path: Path,
    generated_path: Path,
) -> str:
    expected = _normalize_artifact_text(
        filename,
        expected_path.read_text(encoding="utf-8"),
    ).splitlines()
    generated = _normalize_artifact_text(
        filename,
        generated_path.read_text(encoding="utf-8"),
    ).splitlines()
    return "\n".join(
        difflib.unified_diff(
            expected,
            generated,
            fromfile=str(expected_path),
            tofile=str(generated_path),
            lineterm="",
            n=2,
        )
    )


def _compare_artifact_sets(expected_dir: Path, generated_dir: Path) -> list[str]:
    errors: list[str] = []
    for filename in _ARTIFACT_FILES:
        expected_path = expected_dir / filename
        generated_path = generated_dir / filename
        if not expected_path.exists():
            errors.append(f"Missing checked-in artifact: {expected_path}")
            continue
        if not generated_path.exists():
            errors.append(f"Missing generated artifact: {generated_path}")
            continue
        expected_text = _normalize_artifact_text(
            filename,
            expected_path.read_text(encoding="utf-8"),
        )
        generated_text = _normalize_artifact_text(
            filename,
            generated_path.read_text(encoding="utf-8"),
        )
        if expected_text != generated_text:
            diff = _diff_text(filename, expected_path, generated_path)
            if len(diff) > 4000:
                diff = diff[:4000] + "\n... diff truncated ..."
            errors.append(f"Artifact drift detected for {expected_path}:\n{diff}")
    return errors


def validate_release_artifacts(
    *,
    versioned_results_dir: Path,
    root_results_dir: Path,
    split: str,
) -> list[str]:
    """Compare checked-in release artifacts with regenerated outputs."""
    if not versioned_results_dir.exists():
        return [f"Versioned results directory does not exist: {versioned_results_dir}"]
    if not root_results_dir.exists():
        return [f"Root results directory does not exist: {root_results_dir}"]

    errors: list[str] = []
    errors.extend(
        _validate_manifest_contract(
            versioned_results_dir / "release_manifest.json",
            expected_output_dir=versioned_results_dir,
            expected_source_results_dir=versioned_results_dir,
        )
    )
    errors.extend(
        _validate_manifest_contract(
            root_results_dir / "release_manifest.json",
            expected_output_dir=root_results_dir,
            expected_source_results_dir=versioned_results_dir,
        )
    )

    with tempfile.TemporaryDirectory(prefix="bioreview-release-check-") as tmp:
        tmp_root = Path(tmp)
        generated_versioned = tmp_root / versioned_results_dir.name
        generated_root = tmp_root / root_results_dir.name

        _render_release_artifacts(
            source_results_dir=versioned_results_dir,
            output_dir=generated_versioned,
            split=split,
        )
        _render_release_artifacts(
            source_results_dir=versioned_results_dir,
            output_dir=generated_root,
            split=split,
        )

        errors.extend(_compare_artifact_sets(versioned_results_dir, generated_versioned))
        errors.extend(_compare_artifact_sets(root_results_dir, generated_root))
        return errors


def _load_documentation_summary(
    *,
    splits_dir: Path,
    stats_json: Path,
) -> dict[str, Any]:
    """Load canonical stats, recomputing them when private split files exist."""
    split_paths = [splits_dir / filename for filename in _SPLIT_FILENAMES]
    frozen_summary = json.loads(stats_json.read_text(encoding="utf-8"))

    if not all(path.exists() for path in split_paths):
        return frozen_summary

    generated_summary = summarize_splits(splits_dir)
    if generated_summary != frozen_summary:
        raise ValueError(
            f"Frozen split statistics are stale: regenerate {stats_json} "
            f"from {splits_dir}"
        )
    return generated_summary


@click.command()
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_BENCHMARK_SPLITS_DIR,
    show_default=True,
    help="Canonical split directory used for documentation checks.",
)
@click.option(
    "--versioned-results-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_PUBLIC_RESULTS_DIR,
    show_default=True,
    help="Versioned public results directory to regenerate and compare.",
)
@click.option(
    "--stats-json",
    type=click.Path(path_type=Path),
    default=DEFAULT_STATS_JSON,
    show_default=True,
    help="Checked-in aggregate statistics used when private split files are absent.",
)
@click.option(
    "--root-results-dir",
    type=click.Path(path_type=Path),
    default=RESULTS_ROOT,
    show_default=True,
    help="Root results mirror directory to regenerate and compare.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="test",
    show_default=True,
    help="Benchmark split to validate in the public release artifacts.",
)
@click.option(
    "--check-docs/--no-check-docs",
    default=True,
    show_default=True,
    help="Also validate README.md and DATASHEET.md against canonical split stats.",
)
def main(
    splits_dir: Path,
    versioned_results_dir: Path,
    stats_json: Path,
    root_results_dir: Path,
    split: str,
    check_docs: bool,
) -> None:
    """Fail if checked-in release artifacts or docs drift from regenerated outputs."""
    errors: list[str] = []

    if check_docs:
        try:
            summary = _load_documentation_summary(
                splits_dir=splits_dir,
                stats_json=stats_json,
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"Unable to validate documentation statistics: {exc}")
        else:
            errors.extend(
                check_documentation(summary, [Path("README.md"), Path("DATASHEET.md")])
            )

    errors.extend(
        validate_release_artifacts(
            versioned_results_dir=versioned_results_dir,
            root_results_dir=root_results_dir,
            split=split,
        )
    )

    if errors:
        for error in errors:
            click.echo(error, err=True)
        raise SystemExit(1)

    click.echo("Release artifacts and docs are in sync.")


if __name__ == "__main__":
    main()

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from bioreview_bench.evaluate.leaderboard import Leaderboard, update_leaderboard


def build_release_manifest(
    *,
    source_results_dir: Path,
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    """Build a release manifest from the filtered public leaderboard."""
    lb = Leaderboard(results_dir=source_results_dir, split=split)

    included_results: list[dict[str, Any]] = []
    thresholds: set[float] = set()
    algorithms: set[str] = set()

    for entry in lb.entries:
        result_path = Path(entry.result_file)
        data = json.loads(result_path.read_text(encoding="utf-8"))
        matching = data.get("matching_stats") or {}
        if "threshold" in matching:
            thresholds.add(float(matching["threshold"]))
        if "algorithm" in matching:
            algorithms.add(str(matching["algorithm"]))

        included_results.append(
            {
                "rank": entry.rank,
                "tool_name": entry.tool_name,
                "tool_version": entry.tool_version,
                "result_file": str(result_path),
                "run_date": entry.run_date,
                "recall": entry.recall,
                "precision": entry.precision,
                "f1": entry.f1,
                "recall_major": entry.recall_major,
                "n_articles": entry.n_articles,
            }
        )

    manifest = {
        "release_generated_at": datetime.now(UTC).isoformat(),
        "split": split,
        "source_results_dir": str(source_results_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "policy": {
            "exclude_dedup_gt": True,
            "one_result_per_tool_version": True,
            "ranking_metric": "f1_micro",
        },
        "matching": {
            "thresholds": sorted(thresholds),
            "algorithms": sorted(algorithms),
        },
        "artifacts": {
            "leaderboard_md": str((output_dir / "leaderboard.md").resolve()),
            "leaderboard_json": str((output_dir / "leaderboard.json").resolve()),
            "release_manifest_json": str((output_dir / "release_manifest.json").resolve()),
        },
        "included_results": included_results,
    }
    return manifest


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=Path("results/v3"),
    show_default=True,
    help="Directory containing evaluated BenchmarkResult JSON files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("results"),
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

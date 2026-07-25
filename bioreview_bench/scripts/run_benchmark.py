"""Benchmark runner CLI — evaluate tool outputs against bioreview-bench ground truth.

Usage::

    uv run bioreview-run -i tool_outputs/haiku_val.jsonl --tool-name "Haiku-Baseline" \\
        --split val --bootstrap 1000 -o results/haiku_baseline_val.json \\
        --update-leaderboard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from bioreview_bench.evaluate.metrics import DEFAULT_EMBEDDING_MODEL
from bioreview_bench.project_defaults import (
    DEFAULT_BENCHMARK_SPLITS_DIR,
    DEFAULT_PUBLIC_RESULTS_DIR,
)

console = Console()

_DEFAULT_SPLITS_DIR = DEFAULT_BENCHMARK_SPLITS_DIR
_DEFAULT_RESULTS_DIR = DEFAULT_PUBLIC_RESULTS_DIR


@click.command()
@click.option(
    "--tool-output",
    "-i",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="JSONL file with tool output. Each line: {article_id, concerns: [str]}.",
)
@click.option(
    "--tool-name",
    required=True,
    help="Name of the AI tool being evaluated.",
)
@click.option(
    "--tool-version",
    default="unknown",
    show_default=True,
    help="Version string for the tool.",
)
@click.option(
    "--git-hash",
    default="",
    help="Git commit hash of the tool (optional).",
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="val",
    show_default=True,
    help="Dataset split to evaluate against.",
)
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Directory containing split JSONL files (default: {_DEFAULT_SPLITS_DIR}).",
)
@click.option(
    "--threshold",
    type=float,
    default=0.65,
    show_default=True,
    help=(
        "Similarity threshold for concern matching. Thresholds are model- and "
        "method-specific; calibrate after changing the matcher."
    ),
)
@click.option(
    "--embedding-model",
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help=(
        "Exact sentence-transformers model ID. The threshold is not "
        "automatically recalibrated for a different model."
    ),
)
@click.option(
    "--embedding-revision",
    default=None,
    help=(
        "Optional Hugging Face model revision or commit SHA. Pin this for reproducible new runs."
    ),
)
@click.option(
    "--allow-fallback",
    is_flag=True,
    default=False,
    help=(
        "Explicitly allow embedding failures to use Jaccard. Disabled by "
        "default so evaluation runs fail closed."
    ),
)
@click.option(
    "--no-embedding",
    is_flag=True,
    default=False,
    help="Use explicit Jaccard mode instead of embedding similarity.",
)
@click.option(
    "--include-figure",
    is_flag=True,
    default=False,
    help="Include figure_issue concerns in GT (excluded by default).",
)
@click.option(
    "--bootstrap",
    type=int,
    default=0,
    show_default=True,
    help="Bootstrap resamples for 95%% CI (0 = skip). Use 1000 for final results.",
)
@click.option(
    "--extraction-manifest-id",
    default="em-v1.0",
    show_default=True,
    help="ExtractionManifest ID used for ground truth.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save BenchmarkResult JSON to this file.",
)
@click.option(
    "--coverage-log",
    type=click.Path(path_type=Path),
    default=None,
    help="Save per-article coverage log (JSONL) to this file.",
)
@click.option(
    "--notes",
    default="",
    help="Free-text notes to include in the result.",
)
@click.option(
    "--update-leaderboard/--no-update-leaderboard",
    default=False,
    show_default=True,
    help="Regenerate leaderboard after evaluation.",
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Results directory for leaderboard (default: {_DEFAULT_RESULTS_DIR}).",
)
@click.option(
    "--push-hf",
    is_flag=True,
    default=False,
    help="Disabled legacy publisher option; public releases use bioreview-hf-public.",
)
@click.option(
    "--dedup-gt/--no-dedup-gt",
    default=False,
    show_default=True,
    help="Remove near-duplicate GT concerns before matching.",
)
@click.option(
    "--dedup-threshold",
    default=0.95,
    show_default=True,
    help="Cosine similarity threshold for GT dedup (requires --dedup-gt).",
)
def main(
    tool_output: Path,
    tool_name: str,
    tool_version: str,
    git_hash: str,
    split: str,
    splits_dir: Path | None,
    threshold: float,
    embedding_model: str,
    embedding_revision: str | None,
    allow_fallback: bool,
    no_embedding: bool,
    include_figure: bool,
    bootstrap: int,
    extraction_manifest_id: str,
    output: Path | None,
    coverage_log: Path | None,
    notes: str,
    update_leaderboard: bool,
    results_dir: Path | None,
    push_hf: bool,
    dedup_gt: bool,
    dedup_threshold: float,
) -> None:
    """Evaluate AI tool concern outputs against bioreview-bench ground truth."""
    from bioreview_bench.evaluate.runner import run_evaluation

    if push_hf:
        raise click.ClickException(
            "The partial --push-hf publisher is disabled because it cannot "
            "produce an atomic, rights-audited dataset release. Use the "
            "bioreview-hf-public builder, validate its strict-whitelist output, "
            "and publish it from a clean repository history."
        )

    splits_dir = splits_dir or _DEFAULT_SPLITS_DIR
    results_dir = results_dir or _DEFAULT_RESULTS_DIR

    try:
        result, cov_log = run_evaluation(
            tool_output=tool_output,
            splits_dir=splits_dir,
            split=split,
            threshold=threshold,
            exclude_figure=not include_figure,
            use_embedding=not no_embedding,
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
            allow_fallback=allow_fallback,
            bootstrap_n=bootstrap,
            tool_name=tool_name,
            tool_version=tool_version,
            git_hash=git_hash,
            extraction_manifest_id=extraction_manifest_id,
            notes=notes,
            dedup_gt=dedup_gt,
            dedup_threshold=dedup_threshold,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # Save result JSON
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"\nResult saved to: {output}")

    # Save coverage log
    if coverage_log:
        coverage_log.parent.mkdir(parents=True, exist_ok=True)
        with open(coverage_log, "w", encoding="utf-8") as fh:
            for row in cov_log:
                fh.write(json.dumps(row) + "\n")
        console.print(f"Coverage log saved to: {coverage_log}")

    # Update leaderboard
    if update_leaderboard:
        from bioreview_bench.evaluate.leaderboard import update_leaderboard as _update_lb

        lb = _update_lb(results_dir=results_dir, split=split)
        console.print(
            f"\nLeaderboard updated: {len(lb.entries)} tool(s) ranked "
            f"({results_dir / 'leaderboard.md'})"
        )


if __name__ == "__main__":
    main()

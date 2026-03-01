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

console = Console()

_DEFAULT_SPLITS_DIR = Path("data/splits/v2")
_DEFAULT_RESULTS_DIR = Path("results")


@click.command()
@click.option(
    "--tool-output", "-i",
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
    help="Similarity threshold for concern matching.",
)
@click.option(
    "--no-embedding",
    is_flag=True,
    default=False,
    help="Skip SPECTER2 embeddings and use Jaccard fallback.",
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
    "--output", "-o",
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
    help="Push leaderboard to HuggingFace Hub (requires --update-leaderboard).",
)
def main(
    tool_output: Path,
    tool_name: str,
    tool_version: str,
    git_hash: str,
    split: str,
    splits_dir: Path | None,
    threshold: float,
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
) -> None:
    """Evaluate AI tool concern outputs against bioreview-bench ground truth."""
    from bioreview_bench.evaluate.runner import run_evaluation

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
            bootstrap_n=bootstrap,
            tool_name=tool_name,
            tool_version=tool_version,
            git_hash=git_hash,
            extraction_manifest_id=extraction_manifest_id,
            notes=notes,
        )
    except FileNotFoundError as exc:
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

        if push_hf:
            _push_leaderboard_hf(results_dir, split)


def _push_leaderboard_hf(results_dir: Path, split: str) -> None:
    """Push leaderboard files to HuggingFace Hub."""
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        console.print(
            "[yellow]Warning:[/yellow] huggingface_hub not installed; "
            "skipping HF push. Install with: uv sync --extra hub"
        )
        return

    repo_id = "jang1563/bioreview-bench"
    api = HfApi()

    operations = []
    for fname in ["leaderboard.md", "leaderboard.json"]:
        fpath = results_dir / fname
        if fpath.exists():
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"results/{fname}",
                    path_or_fileobj=str(fpath),
                )
            )

    if not operations:
        console.print("[yellow]Warning:[/yellow] No leaderboard files to push.")
        return

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Update leaderboard ({split} split)",
        )
        console.print(f"[green]Pushed leaderboard to {repo_id}[/green]")
    except Exception as exc:
        console.print(f"[red]HF push failed:[/red] {exc}")


if __name__ == "__main__":
    main()

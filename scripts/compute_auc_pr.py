"""Compute AUC-PR (area under precision-recall curve) across multiple thresholds.

Sweeps matching thresholds for one or more models and computes the area under
the precision-recall curve using trapezoidal integration.

Usage::

    python scripts/compute_auc_pr.py \
      --tool-output tool_outputs/v1/haiku_test.jsonl --tool-name "Haiku-4.5" \
      --tool-output tool_outputs/gpt-4o-mini_test.jsonl --tool-name "GPT-4o-mini" \
      --split test --splits-dir data/splits/v3 \
      -o results/v3/auc_pr_comparison.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import numpy as np


THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


class _ModelSpec:
    def __init__(self, tool_output: str, tool_name: str):
        self.tool_output = Path(tool_output)
        self.tool_name = tool_name


@click.command()
@click.option(
    "--tool-output", "-i",
    multiple=True, required=True,
    help="JSONL tool output file (repeat for each model).",
)
@click.option(
    "--tool-name", "-n",
    multiple=True, required=True,
    help="Model name (must match --tool-output count).",
)
@click.option("--split", default="test", show_default=True)
@click.option("--splits-dir", type=click.Path(path_type=Path), default=Path("data/splits/v3"))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def main(
    tool_output: tuple[str, ...],
    tool_name: tuple[str, ...],
    split: str,
    splits_dir: Path,
    output: Path | None,
) -> None:
    """Compute AUC-PR for multiple models via threshold sweep."""
    if len(tool_output) != len(tool_name):
        click.echo("Error: --tool-output and --tool-name must appear the same number of times.", err=True)
        sys.exit(1)

    from bioreview_bench.evaluate.runner import run_evaluation

    models = [_ModelSpec(to, tn) for to, tn in zip(tool_output, tool_name)]

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is None:
        click.echo("Error: numpy has no trapezoid or trapz function.", err=True)
        sys.exit(1)

    all_results: dict[str, dict] = {}
    t0 = time.time()

    # Process models sequentially (SPECTER2 is not parallelism-safe)
    for model in models:
        click.echo(f"\n{'='*60}")
        click.echo(f"Model: {model.tool_name}")
        click.echo(f"{'='*60}")

        points: list[dict] = []
        for t in THRESHOLDS:
            click.echo(f"  threshold={t:.2f} ... ", nl=False)
            result, _ = run_evaluation(
                tool_output=model.tool_output,
                splits_dir=splits_dir,
                split=split,
                threshold=t,
                bootstrap_n=0,
                tool_name=model.tool_name,
            )
            points.append({
                "threshold": t,
                "recall": result.recall_overall,
                "precision": result.precision_overall,
                "f1": result.f1_micro,
            })
            click.echo(f"R={result.recall_overall:.4f}  P={result.precision_overall:.4f}  F1={result.f1_micro:.4f}")

        # Sort by recall ascending for trapezoidal integration
        sorted_pts = sorted(points, key=lambda p: p["recall"])
        recalls = np.array([p["recall"] for p in sorted_pts])
        precisions = np.array([p["precision"] for p in sorted_pts])

        auc_pr = float(trapz_fn(precisions, recalls))

        # Best F1 threshold
        best_pt = max(points, key=lambda p: p["f1"])

        all_results[model.tool_name] = {
            "points": points,
            "auc_pr": auc_pr,
            "best_f1_threshold": best_pt["threshold"],
            "best_f1": best_pt["f1"],
        }

        click.echo(f"  AUC-PR: {auc_pr:.4f}  |  Best F1: {best_pt['f1']:.4f} @ t={best_pt['threshold']:.2f}")

    elapsed = time.time() - t0
    click.echo(f"\nTotal time: {elapsed:.0f}s")

    out = {
        "thresholds": THRESHOLDS,
        "split": split,
        "splits_dir": str(splits_dir),
        "models": all_results,
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        click.echo(f"Saved to {output}")
    else:
        click.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

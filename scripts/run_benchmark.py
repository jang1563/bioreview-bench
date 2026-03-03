"""Benchmark runner for bioreview-bench (thin wrapper).

Delegates to ``bioreview_bench.evaluate.runner`` for the actual logic.
Preserves the original argparse CLI for backward compatibility.

Usage:
    # Evaluate on val split
    python scripts/run_benchmark.py \\
        --tool-output tool_outputs/my_tool_val.jsonl \\
        --tool-name "MyTool" \\
        --tool-version "1.0.0" \\
        --split val \\
        --output results/my_tool_val_result.json

    # With bootstrap CI (slower)
    python scripts/run_benchmark.py \\
        --tool-output tool_outputs/my_tool_val.jsonl \\
        --tool-name "MyTool" \\
        --tool-version "1.0.0" \\
        --split val \\
        --bootstrap 1000 \\
        --output results/my_tool_val_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# -- Project root on sys.path -------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioreview_bench.evaluate.runner import run_evaluation  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate AI tool concern outputs against bioreview-bench ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--tool-output", "-i",
        required=True,
        type=Path,
        help="JSONL file with tool output. Each line: {article_id, concerns: [str or {text}]}",
    )
    p.add_argument(
        "--tool-name",
        required=True,
        help="Name of the AI tool being evaluated (e.g. 'MyReviewTool').",
    )
    p.add_argument(
        "--tool-version",
        default="unknown",
        help="Version string for the tool (default: 'unknown').",
    )
    p.add_argument(
        "--git-hash",
        default="",
        help="Git commit hash of the tool being evaluated (optional).",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to evaluate against (default: val).",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "splits" / "v2",
        help="Directory containing split JSONL files.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Similarity threshold for concern matching (default: 0.65).",
    )
    p.add_argument(
        "--no-embedding",
        action="store_true",
        help="Skip SPECTER2 embeddings and use Jaccard fallback.",
    )
    p.add_argument(
        "--algorithm",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Matching algorithm: 'hungarian' (optimal, default) or 'greedy' (legacy).",
    )
    p.add_argument(
        "--include-figure",
        action="store_true",
        help="Include figure_issue concerns in GT (excluded by default).",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="Number of bootstrap resamples for 95%% CI (0 = skip, default: 0). "
             "Use --bootstrap 1000 for final results.",
    )
    p.add_argument(
        "--extraction-manifest-id",
        default="em-v1.0",
        help="ExtractionManifest ID used to extract ground truth (default: em-v1.0).",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save BenchmarkResult JSON to this file. If not set, only prints to stdout.",
    )
    p.add_argument(
        "--coverage-log",
        type=Path,
        default=None,
        help="Save per-article coverage log (JSONL) to this file.",
    )
    p.add_argument(
        "--notes",
        default="",
        help="Free-text notes to include in the result (e.g. prompt version).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result, coverage_log = run_evaluation(
            tool_output=args.tool_output,
            splits_dir=args.splits_dir,
            split=args.split,
            threshold=args.threshold,
            exclude_figure=not args.include_figure,
            use_embedding=not args.no_embedding,
            algorithm=args.algorithm,
            bootstrap_n=args.bootstrap,
            tool_name=args.tool_name,
            tool_version=args.tool_version,
            git_hash=args.git_hash,
            extraction_manifest_id=args.extraction_manifest_id,
            notes=args.notes,
        )
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    # Save result JSON
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        print(f"Result saved to: {args.output}")

    # Save coverage log
    if args.coverage_log:
        args.coverage_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.coverage_log, "w", encoding="utf-8") as fh:
            for row in coverage_log:
                fh.write(json.dumps(row) + "\n")
        print(f"Coverage log saved to: {args.coverage_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

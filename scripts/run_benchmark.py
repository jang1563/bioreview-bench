"""Benchmark runner for bioreview-bench.

Evaluates an AI tool's concern outputs against the ground-truth split
(train / val / test) and reports recall, precision, F1, and per-category
breakdowns with optional bootstrap confidence intervals.

Tool output format (JSONL, one article per line):
    {"article_id": "elife:84798", "concerns": ["concern text 1", "concern text 2"]}

    Alternatively, concerns may be a list of dicts:
    {"article_id": "elife:84798", "concerns": [{"text": "..."}, ...]}

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
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# -- Project root on sys.path -------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioreview_bench.evaluate.metrics import ConcernMatcher, EvalResult
from bioreview_bench.models.benchmark import (
    BenchmarkResult,
    CategoryMetrics,
    ConfidenceInterval,
    MatchingStats,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load all JSON lines from a file."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] Skipping malformed JSON at line {lineno}: {exc}", file=sys.stderr)
    return records


def _normalise_tool_concerns(raw: list) -> list[str]:
    """Accept both string lists and dict lists; return strings."""
    if not raw:
        return []
    if isinstance(raw[0], str):
        return [t for t in raw if t.strip()]
    # dict list: accept "text" or "concern_text" key
    return [
        c.get("text", c.get("concern_text", ""))
        for c in raw
        if isinstance(c, dict)
    ]


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    article_results: list[EvalResult],
    n_bootstrap: int,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, ConfidenceInterval]:
    """Compute bootstrap 95% CI for recall and precision.

    Samples articles with replacement and computes macro-average metrics
    for each resample.

    Returns:
        Dict with keys 'recall' and 'precision', each a ConfidenceInterval.
    """
    rng = random.Random(seed)
    n = len(article_results)
    if n == 0:
        empty_ci = ConfidenceInterval(lo=0.0, hi=0.0, n_bootstrap=n_bootstrap)
        return {"recall": empty_ci, "precision": empty_ci}

    recall_samples: list[float] = []
    precision_samples: list[float] = []

    for _ in range(n_bootstrap):
        sample = [rng.choice(article_results) for _ in range(n)]
        recall_samples.append(sum(r.recall for r in sample) / n)
        precision_samples.append(sum(r.precision for r in sample) / n)

    recall_samples.sort()
    precision_samples.sort()
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1

    return {
        "recall": ConfidenceInterval(
            lo=recall_samples[lo_idx],
            hi=recall_samples[hi_idx],
            n_bootstrap=n_bootstrap,
        ),
        "precision": ConfidenceInterval(
            lo=precision_samples[lo_idx],
            hi=precision_samples[hi_idx],
            n_bootstrap=n_bootstrap,
        ),
    }


# ---------------------------------------------------------------------------
# Per-article evaluation
# ---------------------------------------------------------------------------

def _evaluate_per_article(
    tool_map: dict[str, list[str]],
    gt_entries: list[dict],
    matcher: ConcernMatcher,
) -> tuple[list[EvalResult], list[dict]]:
    """Run per-article evaluation.

    Returns:
        (article_results, coverage_log)
        coverage_log: list of dicts with article-level info for diagnostics.
    """
    gt_by_id = {entry["id"]: entry for entry in gt_entries}
    article_results: list[EvalResult] = []
    coverage_log: list[dict] = []

    matched_ids: set[str] = set()

    for art_id, gt_entry in gt_by_id.items():
        tool_concerns = tool_map.get(art_id, [])
        gt_concerns = gt_entry.get("concerns", [])

        result = matcher.score_article(tool_concerns, gt_concerns)
        article_results.append(result)

        if art_id in tool_map:
            matched_ids.add(art_id)

        coverage_log.append({
            "article_id": art_id,
            "n_gt": result.n_gt_total,
            "n_tool": result.n_tool_total,
            "n_matched": result.n_matched,
            "recall": result.recall,
            "precision": result.precision,
            "f1": result.f1,
            "in_tool_output": art_id in tool_map,
        })

    # Warn about tool articles not in GT
    extra_ids = set(tool_map) - set(gt_by_id)
    if extra_ids:
        print(
            f"[warn] {len(extra_ids)} article(s) in tool output not found in split "
            f"(first few: {sorted(extra_ids)[:5]})",
            file=sys.stderr,
        )

    return article_results, coverage_log


# ---------------------------------------------------------------------------
# Aggregate to BenchmarkResult
# ---------------------------------------------------------------------------

def _aggregate(
    article_results: list[EvalResult],
    n_bootstrap: int,
    tool_name: str,
    tool_version: str,
    git_hash: str,
    split: str,
    extraction_manifest_id: str,
    n_articles: int,
    n_human_concerns: int,
    n_tool_concerns: int,
    n_figure_excluded: int,
    notes: str,
) -> BenchmarkResult:
    """Compute macro-averaged metrics and build BenchmarkResult."""
    n = len(article_results)
    if n == 0:
        return BenchmarkResult(
            tool_name=tool_name,
            tool_version=tool_version,
            git_hash=git_hash,
            extraction_manifest_id=extraction_manifest_id,
            split=split,  # type: ignore[arg-type]
            recall_overall=0.0,
            precision_overall=0.0,
            f1_micro=0.0,
            n_articles=n_articles,
            n_human_concerns=n_human_concerns,
            n_tool_concerns=n_tool_concerns,
            excluded_figure_concerns=n_figure_excluded,
            notes=notes,
        )

    recall = sum(r.recall for r in article_results) / n
    precision = sum(r.precision for r in article_results) / n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    recall_major = sum(r.recall_major for r in article_results) / n

    # Per-category aggregation
    agg_cat: dict[str, list] = defaultdict(list)
    for r in article_results:
        for cat, cm in r.per_category.items():
            agg_cat[cat].append(cm)

    from bioreview_bench.evaluate.metrics import CategoryMetrics as EvalCategoryMetrics
    per_category: dict[str, CategoryMetrics] = {}
    for cat, cms in agg_cat.items():
        cat_recall = sum(m.recall for m in cms) / len(cms)
        cat_prec = sum(m.precision for m in cms) / len(cms)
        cat_n_gt = sum(m.n_gt for m in cms)
        cat_n_matched = sum(m.n_matched for m in cms)
        f1_micro = (2 * cat_prec * cat_recall / (cat_prec + cat_recall)) if (cat_prec + cat_recall) > 0 else 0.0
        per_category[cat] = CategoryMetrics(
            recall=cat_recall,
            precision=cat_prec,
            f1_micro=f1_micro,
            n_human_concerns=cat_n_gt,
            n_matched=cat_n_matched,
        )

    # Matching stats (from first article as representative sample)
    matching_stats = MatchingStats(
        n_tool_concerns=n_tool_concerns,
        n_human_concerns=n_human_concerns,
        n_matched_pairs=sum(r.n_matched for r in article_results),
        threshold=article_results[0].threshold if article_results else 0.65,
        algorithm="bipartite",
    )

    # Bootstrap CI
    ci_recall = None
    ci_precision = None
    if n_bootstrap > 0:
        print(f"  Running {n_bootstrap} bootstrap resamples...", flush=True)
        cis = _bootstrap_ci(article_results, n_bootstrap)
        ci_recall = cis["recall"]
        ci_precision = cis["precision"]

    return BenchmarkResult(
        tool_name=tool_name,
        tool_version=tool_version,
        git_hash=git_hash,
        extraction_manifest_id=extraction_manifest_id,
        split=split,  # type: ignore[arg-type]
        recall_overall=recall,
        precision_overall=precision,
        f1_micro=f1,
        recall_major=recall_major,
        ci_recall=ci_recall,
        ci_precision=ci_precision,
        bootstrap_n=n_bootstrap,
        per_category=per_category,
        matching_stats=matching_stats,
        n_articles=n_articles,
        n_human_concerns=n_human_concerns,
        n_tool_concerns=n_tool_concerns,
        excluded_figure_concerns=n_figure_excluded,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_result(result: BenchmarkResult, coverage_log: list[dict]) -> None:
    """Print a readable summary to stdout."""
    sep = "─" * 60

    print(sep)
    print(f"  bioreview-bench Evaluation Report")
    print(sep)
    print(f"  Tool:       {result.tool_name} v{result.tool_version}")
    if result.git_hash:
        print(f"  Git hash:   {result.git_hash}")
    print(f"  Split:      {result.split}")
    print(f"  Run date:   {result.run_date.strftime('%Y-%m-%d %H:%M UTC')}")
    print(sep)

    # Coverage
    n_covered = sum(1 for row in coverage_log if row["in_tool_output"])
    n_total = len(coverage_log)
    print(f"  Coverage:   {n_covered}/{n_total} articles in tool output")
    print(f"  GT concerns:{result.n_human_concerns} ({result.excluded_figure_concerns} figure excluded)")
    print(f"  Tool output:{result.n_tool_concerns} concerns")
    print()

    # Overall metrics
    print(f"  {'Metric':<22}  {'Value':>8}  {'95% CI':>16}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*16}")

    def _ci_str(ci) -> str:
        if ci is None:
            return "      n/a     "
        return f"[{ci.lo:.3f}, {ci.hi:.3f}]"

    print(f"  {'Recall (overall)':<22}  {result.recall_overall:>8.3f}  {_ci_str(result.ci_recall):>16}")
    print(f"  {'Precision (overall)':<22}  {result.precision_overall:>8.3f}  {_ci_str(result.ci_precision):>16}")
    print(f"  {'F1 (micro)':<22}  {result.f1_micro:>8.3f}")
    print(f"  {'Recall (major only)':<22}  {result.recall_major:>8.3f}")
    print()

    # Per-category
    if result.per_category:
        print(f"  {'Category':<30}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'#GT':>5}")
        print(f"  {'─'*30}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}")
        for cat, cm in sorted(result.per_category.items()):
            print(
                f"  {cat:<30}  {cm.recall:>7.3f}  {cm.precision:>7.3f}  "
                f"{cm.f1_micro:>7.3f}  {cm.n_human_concerns:>5}"
            )
        print()

    # Matching stats
    if result.matching_stats:
        ms = result.matching_stats
        print(f"  Matching: threshold={ms.threshold:.2f}, algorithm={ms.algorithm}")
        print(f"           matched {ms.n_matched_pairs} pairs "
              f"({ms.recall:.1%} recall, {ms.precision:.1%} precision)")
        print()

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        default=_REPO_ROOT / "data" / "splits",
        help="Directory containing split JSONL files (default: data/splits/).",
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

    # -- Load ground truth ----------------------------------------------------
    gt_path = args.splits_dir / f"{args.split}.jsonl"
    if not gt_path.exists():
        print(f"[error] Ground truth split not found: {gt_path}", file=sys.stderr)
        return 1

    print(f"Loading ground truth from {gt_path} ...", flush=True)
    gt_entries = load_jsonl(gt_path)
    print(f"  {len(gt_entries)} articles in split.")

    # -- Load tool output -----------------------------------------------------
    tool_path = args.tool_output
    if not tool_path.exists():
        print(f"[error] Tool output file not found: {tool_path}", file=sys.stderr)
        return 1

    print(f"Loading tool output from {tool_path} ...", flush=True)
    tool_rows = load_jsonl(tool_path)
    print(f"  {len(tool_rows)} articles in tool output.")

    # Build tool map: article_id -> list[str]
    tool_map: dict[str, list[str]] = {}
    for row in tool_rows:
        art_id = row.get("article_id", row.get("id", ""))
        if not art_id:
            print("[warn] Tool output row missing 'article_id'; skipping.", file=sys.stderr)
            continue
        raw_concerns = row.get("concerns", [])
        tool_map[art_id] = _normalise_tool_concerns(raw_concerns)

    # -- Run evaluation -------------------------------------------------------
    matcher = ConcernMatcher(
        threshold=args.threshold,
        exclude_figure=not args.include_figure,
        use_embedding=not args.no_embedding,
    )

    print("Running per-article evaluation ...", flush=True)
    article_results, coverage_log = _evaluate_per_article(tool_map, gt_entries, matcher)
    print(f"  Evaluated {len(article_results)} articles.")

    # Aggregate counts
    n_human = sum(r.n_gt_total for r in article_results)
    n_tool = sum(r.n_tool_total for r in article_results)
    n_figure = sum(r.n_gt_figure_excluded for r in article_results)

    # -- Aggregate + build BenchmarkResult ------------------------------------
    print("Aggregating results ...", flush=True)
    result = _aggregate(
        article_results=article_results,
        n_bootstrap=args.bootstrap,
        tool_name=args.tool_name,
        tool_version=args.tool_version,
        git_hash=args.git_hash,
        split=args.split,
        extraction_manifest_id=args.extraction_manifest_id,
        n_articles=len(article_results),
        n_human_concerns=n_human,
        n_tool_concerns=n_tool,
        n_figure_excluded=n_figure,
        notes=args.notes,
    )

    # -- Display --------------------------------------------------------------
    print()
    _print_result(result, coverage_log)

    # -- Save outputs ---------------------------------------------------------
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        print(f"Result saved to: {args.output}")

    if args.coverage_log:
        args.coverage_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.coverage_log, "w", encoding="utf-8") as fh:
            for row in coverage_log:
                fh.write(json.dumps(row) + "\n")
        print(f"Coverage log saved to: {args.coverage_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

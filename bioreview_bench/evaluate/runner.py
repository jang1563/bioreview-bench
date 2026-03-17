"""Evaluation runner — core logic for evaluating tool outputs against ground truth.

Extracted from ``scripts/run_benchmark.py`` to be reusable by both the CLI
entry point (``bioreview-run``) and programmatic callers.

Usage::

    from bioreview_bench.evaluate.runner import run_evaluation

    result, coverage = run_evaluation(
        tool_output=Path("tool_outputs/haiku_val.jsonl"),
        splits_dir=Path("data/splits/v3"),
        split="val",
        tool_name="Haiku-Baseline",
    )
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ..evaluate.metrics import ConcernMatcher, EvalResult
from ..models.benchmark import (
    BenchmarkResult,
    CategoryMetrics,
    ConfidenceInterval,
    MatchingStats,
)


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════


def load_jsonl(path: Path) -> list[dict]:
    """Load all JSON lines from a file."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] Skipping malformed JSON at line {lineno}: {exc}",
                    file=sys.stderr,
                )
    return records


def load_split(splits_dir: Path, split: str) -> list[dict]:
    """Load ground-truth entries from a split JSONL file.

    Args:
        splits_dir: Directory containing {split}.jsonl files.
        split: One of "train", "val", "test".

    Returns:
        List of article dicts.

    Raises:
        FileNotFoundError: If the split file does not exist.
    """
    gt_path = splits_dir / f"{split}.jsonl"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth split not found: {gt_path}")
    return load_jsonl(gt_path)


def build_tool_map(tool_output_path: Path) -> dict[str, list[str]]:
    """Load tool output JSONL and build article_id -> concern_texts mapping.

    Accepts both string lists and dict lists for concerns:
        - ``{"article_id": "x", "concerns": ["text1", "text2"]}``
        - ``{"article_id": "x", "concerns": [{"text": "..."}, ...]}``
    """
    tool_rows = load_jsonl(tool_output_path)
    tool_map: dict[str, list[str]] = {}

    for row in tool_rows:
        art_id = row.get("article_id", row.get("id", ""))
        if not art_id:
            print(
                "[warn] Tool output row missing 'article_id'; skipping.",
                file=sys.stderr,
            )
            continue
        raw_concerns = row.get("concerns", [])
        tool_map[art_id] = _normalise_tool_concerns(raw_concerns)

    return tool_map


def _normalise_tool_concerns(raw: list) -> list[str]:
    """Accept both string lists and dict lists; return strings."""
    if not raw:
        return []
    if isinstance(raw[0], str):
        return [t for t in raw if t.strip()]
    return [
        c.get("text", c.get("concern_text", ""))
        for c in raw
        if isinstance(c, dict)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Per-article evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_articles(
    tool_map: dict[str, list[str]],
    gt_entries: list[dict],
    matcher: ConcernMatcher,
) -> tuple[list[EvalResult], list[dict]]:
    """Run per-article evaluation.

    Returns:
        (article_results, coverage_log)
    """
    gt_by_id = {entry["id"]: entry for entry in gt_entries}
    article_results: list[EvalResult] = []
    coverage_log: list[dict] = []

    for art_id, gt_entry in gt_by_id.items():
        tool_concerns = tool_map.get(art_id, [])
        gt_concerns = gt_entry.get("concerns", [])

        result = matcher.score_article(tool_concerns, gt_concerns)
        article_results.append(result)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════════════


def bootstrap_ci(
    article_results: list[EvalResult],
    n_bootstrap: int,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, ConfidenceInterval]:
    """Compute bootstrap 95% CI for recall and precision.

    Samples articles with replacement and computes micro-averaged metrics
    (count-weighted) for each resample.

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
        total_matched = sum(r.n_matched for r in sample)
        total_gt = sum(r.n_gt_total for r in sample)
        total_tool = sum(r.n_tool_total for r in sample)
        recall_samples.append(total_matched / total_gt if total_gt > 0 else 0.0)
        precision_samples.append(total_matched / total_tool if total_tool > 0 else 0.0)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════════


def aggregate_results(
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
    dedup_gt: bool = False,
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

    total_matched = sum(r.n_matched for r in article_results)
    total_gt = sum(r.n_gt_total for r in article_results)
    total_tool = sum(r.n_tool_total for r in article_results)

    # Primary overall metrics are micro-averaged (count-weighted).
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    precision = total_matched / total_tool if total_tool > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    recall_major = sum(r.recall_major for r in article_results) / n
    soft_recall = sum(r.soft_recall for r in article_results) / n
    soft_precision = sum(r.soft_precision for r in article_results) / n
    soft_f1 = (
        (2 * soft_precision * soft_recall / (soft_precision + soft_recall))
        if (soft_precision + soft_recall) > 0
        else 0.0
    )

    # Per-category aggregation
    from bioreview_bench.evaluate.metrics import CategoryMetrics as EvalCategoryMetrics  # noqa: F811
    agg_cat: dict[str, list[EvalCategoryMetrics]] = defaultdict(list)
    for r in article_results:
        for cat, cm in r.per_category.items():
            agg_cat[cat].append(cm)

    per_category: dict[str, CategoryMetrics] = {}
    for cat, cms in agg_cat.items():
        cat_recall = sum(m.recall for m in cms) / len(cms)
        cat_prec = sum(m.precision for m in cms) / len(cms)
        cat_n_gt = sum(m.n_gt for m in cms)
        cat_n_matched = sum(m.n_matched for m in cms)
        f1_micro = (
            (2 * cat_prec * cat_recall / (cat_prec + cat_recall))
            if (cat_prec + cat_recall) > 0
            else 0.0
        )
        per_category[cat] = CategoryMetrics(
            recall=cat_recall,
            precision=cat_prec,
            f1_micro=f1_micro,
            n_human_concerns=cat_n_gt,
            n_matched=cat_n_matched,
        )

    # Matching stats
    algo = article_results[0].algorithm if article_results else "hungarian"
    matching_stats = MatchingStats(
        n_tool_concerns=n_tool_concerns,
        n_human_concerns=n_human_concerns,
        n_matched_pairs=sum(r.n_matched for r in article_results),
        threshold=article_results[0].threshold if article_results else 0.65,
        algorithm=algo,
    )

    # Bootstrap CI
    ci_recall = None
    ci_precision = None
    if n_bootstrap > 0:
        print(f"  Running {n_bootstrap} bootstrap resamples...", flush=True)
        cis = bootstrap_ci(article_results, n_bootstrap)
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
        soft_recall_overall=soft_recall,
        soft_precision_overall=soft_precision,
        soft_f1=soft_f1,
        ci_recall=ci_recall,
        ci_precision=ci_precision,
        bootstrap_n=n_bootstrap,
        per_category=per_category,
        matching_stats=matching_stats,
        n_articles=n_articles,
        n_human_concerns=n_human_concerns,
        n_tool_concerns=n_tool_concerns,
        excluded_figure_concerns=n_figure_excluded,
        dedup_gt=dedup_gt,
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(result: BenchmarkResult, coverage_log: list[dict]) -> None:
    """Print a readable summary to stdout."""
    sep = "\u2500" * 60

    print(sep)
    print("  bioreview-bench Evaluation Report")
    print(sep)
    print(f"  Tool:       {result.tool_name} v{result.tool_version}")
    if result.git_hash:
        print(f"  Git hash:   {result.git_hash}")
    print(f"  Split:      {result.split}")
    print(f"  Run date:   {result.run_date.strftime('%Y-%m-%d %H:%M UTC')}")
    print(sep)

    n_covered = sum(1 for row in coverage_log if row["in_tool_output"])
    n_total = len(coverage_log)
    print(f"  Coverage:   {n_covered}/{n_total} articles in tool output")
    print(f"  GT concerns:{result.n_human_concerns} ({result.excluded_figure_concerns} figure excluded)")
    print(f"  Tool output:{result.n_tool_concerns} concerns")
    print()

    print(f"  {'Metric':<22}  {'Value':>8}  {'95% CI':>16}")
    print(f"  {'\u2500'*22}  {'\u2500'*8}  {'\u2500'*16}")

    def _ci_str(ci: ConfidenceInterval | None) -> str:
        if ci is None:
            return "      n/a     "
        return f"[{ci.lo:.3f}, {ci.hi:.3f}]"

    print(f"  {'Recall (overall)':<22}  {result.recall_overall:>8.3f}  {_ci_str(result.ci_recall):>16}")
    print(f"  {'Precision (overall)':<22}  {result.precision_overall:>8.3f}  {_ci_str(result.ci_precision):>16}")
    print(f"  {'F1 (micro)':<22}  {result.f1_micro:>8.3f}")
    print(f"  {'Recall (major only)':<22}  {result.recall_major:>8.3f}")
    if result.soft_f1 > 0:
        print(f"  {'Soft Recall':<22}  {result.soft_recall_overall:>8.3f}")
        print(f"  {'Soft Precision':<22}  {result.soft_precision_overall:>8.3f}")
        print(f"  {'Soft F1':<22}  {result.soft_f1:>8.3f}")
    print()

    if result.per_category:
        print(f"  {'Category':<30}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'#GT':>5}")
        print(f"  {'\u2500'*30}  {'\u2500'*7}  {'\u2500'*7}  {'\u2500'*7}  {'\u2500'*5}")
        for cat, cm in sorted(result.per_category.items()):
            print(
                f"  {cat:<30}  {cm.recall:>7.3f}  {cm.precision:>7.3f}  "
                f"{cm.f1_micro:>7.3f}  {cm.n_human_concerns:>5}"
            )
        print()

    if result.matching_stats:
        ms = result.matching_stats
        print(f"  Matching: threshold={ms.threshold:.2f}, algorithm={ms.algorithm}")
        print(
            f"           matched {ms.n_matched_pairs} pairs "
            f"({ms.recall:.1%} recall, {ms.precision:.1%} precision)"
        )
        print()

    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# High-level runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_evaluation(
    tool_output: Path,
    splits_dir: Path,
    split: str = "val",
    threshold: float = 0.65,
    exclude_figure: bool = True,
    use_embedding: bool = True,
    algorithm: str = "hungarian",
    bootstrap_n: int = 0,
    tool_name: str = "",
    tool_version: str = "unknown",
    git_hash: str = "",
    extraction_manifest_id: str = "em-v1.0",
    notes: str = "",
    dedup_gt: bool = False,
    dedup_threshold: float = 0.95,
) -> tuple[BenchmarkResult, list[dict]]:
    """Run full evaluation pipeline: load data, match, aggregate, report.

    Args:
        tool_output: Path to tool output JSONL file.
        splits_dir: Directory containing {split}.jsonl ground truth files.
        split: Dataset split ("train", "val", "test").
        threshold: Similarity threshold for concern matching.
        exclude_figure: Exclude figure_issue concerns from GT.
        use_embedding: Use SPECTER2 embeddings (falls back to Jaccard).
        bootstrap_n: Number of bootstrap resamples for CI (0 = skip).
        tool_name: Name of the evaluated tool.
        tool_version: Version of the evaluated tool.
        git_hash: Git commit hash of the tool.
        extraction_manifest_id: Manifest ID for ground truth extraction.
        notes: Free-text notes.
        dedup_gt: Remove near-duplicate GT concerns before matching.
        dedup_threshold: Cosine similarity threshold for GT dedup.

    Returns:
        (BenchmarkResult, coverage_log)
    """
    # Load data
    print(f"Loading ground truth from {splits_dir / f'{split}.jsonl'} ...", flush=True)
    gt_entries = load_split(splits_dir, split)
    print(f"  {len(gt_entries)} articles in split.")

    print(f"Loading tool output from {tool_output} ...", flush=True)
    tool_map = build_tool_map(tool_output)
    print(f"  {len(tool_map)} articles in tool output.")

    # Evaluate
    matcher = ConcernMatcher(
        threshold=threshold,
        exclude_figure=exclude_figure,
        use_embedding=use_embedding,
        algorithm=algorithm,  # type: ignore[arg-type]
        dedup_gt=dedup_gt,
        dedup_threshold=dedup_threshold,
    )

    print("Running per-article evaluation ...", flush=True)
    article_results, coverage_log = evaluate_articles(tool_map, gt_entries, matcher)
    print(f"  Evaluated {len(article_results)} articles.")

    # Aggregate
    n_human = sum(r.n_gt_total for r in article_results)
    n_tool = sum(r.n_tool_total for r in article_results)
    n_figure = sum(r.n_gt_figure_excluded for r in article_results)

    print("Aggregating results ...", flush=True)
    result = aggregate_results(
        article_results=article_results,
        n_bootstrap=bootstrap_n,
        tool_name=tool_name,
        tool_version=tool_version,
        git_hash=git_hash,
        split=split,
        extraction_manifest_id=extraction_manifest_id,
        n_articles=len(article_results),
        n_human_concerns=n_human,
        n_tool_concerns=n_tool,
        n_figure_excluded=n_figure,
        notes=notes,
        dedup_gt=dedup_gt,
    )

    # Print report
    print()
    print_report(result, coverage_log)

    return result, coverage_log

"""Comprehensive source-level analysis for bioreview-bench.

Produces:
  1. per_source_metrics.json   — per-source R/P/F1 + bootstrap CI (6 models × 4-5 sources)
  2. pairwise_significance.json — article-paired randomization tests for all 15 model pairs
  3. similarity_distribution.json — SPECTER2 score histograms (matched vs unmatched)
  4. concern_counts.json        — per-model concern count statistics
  5. Updated current release result JSONs with micro-averaged per-category

Also runs DOI cross-source dedup check across all splits.

Usage:
    python scripts/source_analysis.py
    python scripts/source_analysis.py --splits-dir data/splits/v4 --results-dir results/v4
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioreview_bench.evaluate.metrics import ConcernMatcher, EvalResult
from bioreview_bench.evaluate.runner import (
    aggregate_results,
    bootstrap_ci,
    build_tool_map,
    load_jsonl,
    load_split,
)
from bioreview_bench.models.benchmark import BenchmarkResult
from bioreview_bench.project_defaults import (
    DEFAULT_BENCHMARK_SPLITS_DIR,
    DEFAULT_PUBLIC_RESULTS_DIR,
)
from bioreview_bench.stats import paired_micro_f1_randomization_pvalue

# ═══════════════════════════════════════════════════════════════════════════════
# Model configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODELS: dict[str, dict] = {
    "Haiku-4.5": {
        "tool_output": "tool_outputs/haiku_test_v4.jsonl",
        "result_file": "results/v4/haiku_test_v4.json",
        "tool_version": "claude-haiku-4-5-20251001",
    },
    "GPT-4o-mini": {
        "tool_output": "tool_outputs/gpt4omini_test_v4.jsonl",
        "result_file": "results/v4/gpt4omini_test_v4.json",
        "tool_version": "gpt-4o-mini",
    },
    "Gemini-2.5-Flash": {
        "tool_output": "tool_outputs/gemini25flash_test_v4.jsonl",
        "result_file": "results/v4/gemini25flash_test_v4.json",
        "tool_version": "gemini-2.5-flash",
    },
    "BM25": {
        "tool_output": "tool_outputs/bm25_test_v4.jsonl",
        "result_file": "results/v4/bm25_test_v4.json",
        "tool_version": "bm25-specter2",
    },
    "Gemini-Flash-Lite": {
        "tool_output": "tool_outputs/gemini_flash_lite_test_v4.jsonl",
        "result_file": "results/v4/gemini_flash_lite_test_v4.json",
        "tool_version": "gemini-2.5-flash-lite",
    },
    "Llama-3.3-70B": {
        "tool_output": "tool_outputs/llama33_test_v4.jsonl",
        "result_file": "results/v4/llama33_test_v4.json",
        "tool_version": "llama-3.3-70b",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _micro_f1_from_results(results: list[EvalResult]) -> float:
    """Compute micro-averaged F1 from a list of EvalResult."""
    total_matched = sum(r.n_matched for r in results)
    total_gt = sum(r.n_gt_total for r in results)
    total_tool = sum(r.n_tool_total for r in results)
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    precision = total_matched / total_tool if total_tool > 0 else 0.0
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def _micro_metrics(results: list[EvalResult]) -> dict:
    """Compute micro-averaged R/P/F1 from EvalResult list."""
    total_matched = sum(r.n_matched for r in results)
    total_gt = sum(r.n_gt_total for r in results)
    total_tool = sum(r.n_tool_total for r in results)
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    precision = total_matched / total_tool if total_tool > 0 else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "n_articles": len(results),
        "n_gt": total_gt,
        "n_tool": total_tool,
        "n_matched": total_matched,
    }


def _bootstrap_ci_dict(results: list[EvalResult], n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """Bootstrap CI for recall and precision, returns dict."""
    cis = bootstrap_ci(results, n_bootstrap, seed=seed)
    return {
        "ci_recall": [round(cis["recall"].lo, 4), round(cis["recall"].hi, 4)],
        "ci_precision": [round(cis["precision"].lo, 4), round(cis["precision"].hi, 4)],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-source evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_model(
    model_name: str,
    tool_output_path: Path,
    gt_entries: list[dict],
    source_map: dict[str, str],
    matcher: ConcernMatcher,
) -> tuple[dict[str, list[EvalResult]], list[float], list[float]]:
    """Evaluate a single model, returning per-source EvalResults + similarity scores.

    Returns:
        (source_results, matched_scores, all_max_scores)
        - source_results: {source: [EvalResult]}
        - matched_scores: list of cosine sim for matched pairs
        - all_max_scores: list of max cosine sim per tool concern (including unmatched)
    """
    tool_map = build_tool_map(tool_output_path)
    gt_by_id = {entry["id"]: entry for entry in gt_entries}

    source_results: dict[str, list[EvalResult]] = defaultdict(list)
    matched_scores: list[float] = []
    all_max_scores: list[float] = []

    for art_id, gt_entry in gt_by_id.items():
        tool_concerns = tool_map.get(art_id, [])
        gt_concerns = gt_entry.get("concerns", [])
        source = source_map.get(art_id, "unknown")

        # Run evaluation
        result = matcher.score_article(tool_concerns, gt_concerns)
        source_results[source].append(result)

        # Collect similarity scores for histogram
        if tool_concerns and gt_concerns:
            active_gt = [c for c in gt_concerns if not c.get("requires_figure_reading", False)]
            gt_texts = [c["concern_text"] for c in active_gt]
            if gt_texts and tool_concerns:
                scores = matcher._compute_scores(tool_concerns, gt_texts)
                if scores.matrix:
                    # Record matched pair scores
                    matches = matcher._match(scores)
                    for m in matches:
                        matched_scores.append(m.score)
                    # Record max score per tool concern
                    for row in scores.matrix:
                        all_max_scores.append(max(row))

    return dict(source_results), matched_scores, all_max_scores


# ═══════════════════════════════════════════════════════════════════════════════
# Pairwise significance
# ═══════════════════════════════════════════════════════════════════════════════


def paired_micro_f1_randomization_test(
    results_a: dict[str, EvalResult],
    results_b: dict[str, EvalResult],
    n_resamples: int = 10000,
    seed: int = 42,
    exact_max_pairs: int = 16,
) -> dict:
    """Test paired delta micro-F1 by swapping complete tool results per article.

    Article IDs are checked explicitly. Each randomization exchanges tool labels
    within selected articles and recomputes both dataset-level micro-F1 values.

    Returns:
        Method metadata, delta_f1, p_value, and randomization counts.
    """
    article_ids_a = set(results_a)
    article_ids_b = set(results_b)
    if article_ids_a != article_ids_b:
        only_a = sorted(article_ids_a - article_ids_b)
        only_b = sorted(article_ids_b - article_ids_a)
        raise ValueError(
            "Paired tools must cover identical article IDs; "
            f"only_a={only_a[:3]}, only_b={only_b[:3]}"
        )

    article_ids = sorted(article_ids_a)
    ordered_a = [results_a[article_id] for article_id in article_ids]
    ordered_b = [results_b[article_id] for article_id in article_ids]
    counts_a = [(result.n_matched, result.n_gt_total, result.n_tool_total) for result in ordered_a]
    counts_b = [(result.n_matched, result.n_gt_total, result.n_tool_total) for result in ordered_b]
    p_value = paired_micro_f1_randomization_pvalue(
        counts_a,
        counts_b,
        n_resamples=n_resamples,
        seed=seed,
        exact_max_pairs=exact_max_pairs,
    )
    observed_delta = _micro_f1_from_results(ordered_a) - _micro_f1_from_results(ordered_b)
    n_informative = sum(left != right for left, right in zip(counts_a, counts_b))
    exact = n_informative <= exact_max_pairs
    randomization_draws = 2**n_informative if exact else n_resamples

    return {
        "method": "article_paired_label_swap_randomization_delta_micro_f1",
        "delta_f1": round(observed_delta, 6),
        "p_value": p_value,
        "n_articles": len(article_ids),
        "n_informative_pairs": n_informative,
        "randomization_mode": "exact" if exact else "monte_carlo",
        "randomization_draws": randomization_draws,
        "plus_one_correction": not exact,
        "seed": None if exact else seed,
        "multiplicity_adjustment": "none",
        "interpretation": "exploratory_unadjusted",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DOI dedup check
# ═══════════════════════════════════════════════════════════════════════════════


def check_doi_dedup(splits_dir: Path) -> dict:
    """Check for DOI duplicates across sources in all splits."""
    doi_source: dict[str, list[tuple[str, str]]] = defaultdict(list)  # doi → [(source, id)]

    for split_name in ("train", "val", "test"):
        split_file = splits_dir / f"{split_name}.jsonl"
        if not split_file.exists():
            continue
        entries = load_jsonl(split_file)
        for entry in entries:
            doi = entry.get("doi", "").strip().lower().rstrip(".")
            if doi:
                doi_source[doi].append((entry.get("source", "?"), entry.get("id", "?")))

    duplicates = {
        doi: entries
        for doi, entries in doi_source.items()
        if len(set(s for s, _ in entries)) > 1  # different sources
    }

    return {
        "total_unique_dois": len(doi_source),
        "cross_source_duplicates": len(duplicates),
        "duplicate_details": {
            doi: [{"source": s, "id": i} for s, i in entries] for doi, entries in duplicates.items()
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Concern counts
# ═══════════════════════════════════════════════════════════════════════════════


def compute_concern_counts(models: dict[str, dict], repo_root: Path) -> dict:
    """Compute concern count statistics from tool output files."""
    import statistics

    result = {}
    for model_name, config in models.items():
        tool_path = repo_root / config["tool_output"]
        if not tool_path.exists():
            continue
        tool_map = build_tool_map(tool_path)
        counts = [len(concerns) for concerns in tool_map.values()]
        if not counts:
            continue
        result[model_name] = {
            "mean": round(statistics.mean(counts), 2),
            "median": round(statistics.median(counts), 1),
            "stdev": round(statistics.stdev(counts), 2) if len(counts) > 1 else 0.0,
            "min": min(counts),
            "max": max(counts),
            "n_articles": len(counts),
            "total_concerns": sum(counts),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Similarity histogram
# ═══════════════════════════════════════════════════════════════════════════════


def build_histogram(scores: list[float], n_bins: int = 30) -> dict:
    """Build histogram bins for similarity score distribution."""
    if not scores:
        return {"bins": [], "counts": [], "n": 0}

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for s in scores:
        idx = min(int(s * n_bins), n_bins - 1)
        if idx < 0:
            idx = 0
        counts[idx] += 1

    return {
        "bin_edges": [round(e, 4) for e in bin_edges],
        "counts": counts,
        "n": len(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "median": round(sorted(scores)[len(scores) // 2], 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Comprehensive source-level analysis for bioreview-bench.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=_REPO_ROOT / DEFAULT_BENCHMARK_SPLITS_DIR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_REPO_ROOT / DEFAULT_PUBLIC_RESULTS_DIR,
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=1000,
        help="Bootstrap resamples for per-source CI (default: 1000)",
    )
    parser.add_argument(
        "--significance-n",
        type=int,
        default=10000,
        help=("Monte Carlo draws for article-paired label-swap randomization (default: 10000)"),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip re-evaluation, only run DOI check and concern counts.",
    )
    args = parser.parse_args(argv)

    splits_dir = args.splits_dir
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── DOI dedup check ───────────────────────────────────────────
    print("=" * 60)
    print("DOI cross-source dedup check")
    print("=" * 60)
    dedup = check_doi_dedup(splits_dir)
    print(f"  Total unique DOIs: {dedup['total_unique_dois']}")
    print(f"  Cross-source duplicates: {dedup['cross_source_duplicates']}")
    if dedup["cross_source_duplicates"] > 0:
        for doi, entries in dedup["duplicate_details"].items():
            print(f"    {doi}: {entries}")
    else:
        print("  ✓ No cross-source duplicates found.")
    print()

    # ── Concern counts (no evaluation needed) ─────────────────────
    print("=" * 60)
    print("Concern count statistics")
    print("=" * 60)
    concern_counts = compute_concern_counts(MODELS, _REPO_ROOT)
    for model_name, stats in concern_counts.items():
        print(
            f"  {model_name}: mean={stats['mean']:.1f} ± {stats['stdev']:.1f}, "
            f"median={stats['median']:.0f}, range=[{stats['min']}, {stats['max']}], "
            f"n={stats['n_articles']}"
        )
    counts_path = results_dir / "concern_counts.json"
    counts_path.write_text(json.dumps(concern_counts, indent=2), encoding="utf-8")
    print(f"  → Saved to {counts_path}")
    print()

    if args.skip_eval:
        print("Skipping evaluation (--skip-eval). Done.")
        return 0

    # ── Load ground truth ──────────────────────────────────────────
    print("Loading ground truth ...", flush=True)
    gt_entries = load_split(splits_dir, "test")
    print(f"  {len(gt_entries)} articles in test split.")

    # Build source mapping
    source_map: dict[str, str] = {}
    for entry in gt_entries:
        source_map[entry["id"]] = entry.get("source", "unknown")

    # ── Evaluate all models ────────────────────────────────────────
    matcher = ConcernMatcher(threshold=0.65, exclude_figure=True, algorithm="hungarian")

    all_model_results: dict[str, dict[str, list[EvalResult]]] = {}
    all_matched_scores: list[float] = []
    all_max_scores: list[float] = []

    for model_name, config in MODELS.items():
        tool_path = _REPO_ROOT / config["tool_output"]
        if not tool_path.exists():
            print(f"[SKIP] {model_name}: {tool_path} not found")
            continue

        print("=" * 60)
        print(f"Evaluating {model_name}")
        print("=" * 60)
        t0 = time.time()

        source_results, matched_scores, max_scores = evaluate_model(
            model_name,
            tool_path,
            gt_entries,
            source_map,
            matcher,
        )
        all_model_results[model_name] = source_results
        all_matched_scores.extend(matched_scores)
        all_max_scores.extend(max_scores)

        # Print per-source summary
        for source in sorted(source_results):
            metrics = _micro_metrics(source_results[source])
            print(
                f"  {source}: F1={metrics['f1']:.3f} R={metrics['recall']:.3f} "
                f"P={metrics['precision']:.3f} (n={metrics['n_articles']})"
            )

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        print()

    # ── Per-source metrics with bootstrap CI ───────────────────────
    print("=" * 60)
    print("Computing per-source bootstrap CIs")
    print("=" * 60)

    per_source_output: dict[str, dict] = {}
    for model_name, source_results in all_model_results.items():
        model_out: dict[str, dict] = {}
        for source, results in sorted(source_results.items()):
            metrics = _micro_metrics(results)
            if len(results) >= 5:  # Skip CI for very small N
                ci = _bootstrap_ci_dict(results, n_bootstrap=args.bootstrap_n)
                metrics.update(ci)
            else:
                metrics["ci_recall"] = None
                metrics["ci_precision"] = None
                metrics["note"] = f"n={len(results)} too small for bootstrap CI"
            model_out[source] = metrics
        per_source_output[model_name] = model_out
        print(f"  {model_name}: {len(model_out)} sources")

    ps_path = results_dir / "per_source_metrics.json"
    ps_path.write_text(json.dumps(per_source_output, indent=2), encoding="utf-8")
    print(f"  → Saved to {ps_path}")
    print()

    # ── Pairwise randomization tests across the shared non-PeerJ subset ──────
    print("=" * 60)
    print("Article-paired micro-F1 randomization tests")
    print("Exploratory unadjusted p-values; no significance stars are reported")
    print("=" * 60)

    # Build explicitly ID-keyed per-article results from non-PeerJ sources.
    common_sources = {"elife", "plos", "f1000", "nature"}
    print("  Building aligned article results ...", flush=True)
    aligned_results: dict[str, dict[str, EvalResult]] = {}
    common_article_list = [e for e in gt_entries if e.get("source") in common_sources]
    common_article_list.sort(key=lambda e: e["id"])

    for model_name, config in MODELS.items():
        tool_path = _REPO_ROOT / config["tool_output"]
        if not tool_path.exists():
            continue
        tool_map = build_tool_map(tool_path)
        model_results: dict[str, EvalResult] = {}
        for entry in common_article_list:
            art_id = entry["id"]
            if art_id in model_results:
                raise ValueError(f"Duplicate ground-truth article ID: {art_id}")
            tool_concerns = tool_map.get(art_id, [])
            gt_concerns = entry.get("concerns", [])
            article_result = matcher.score_article(tool_concerns, gt_concerns)
            model_results[art_id] = article_result
        aligned_results[model_name] = model_results

    print(f"  {len(common_article_list)} common articles, {len(aligned_results)} models")

    # Run pairwise tests
    pairwise_output: dict[str, dict] = {}
    model_names = sorted(aligned_results.keys())
    for model_a, model_b in combinations(model_names, 2):
        key = f"{model_a} vs {model_b}"
        print(f"  Testing: {key} ...", end=" ", flush=True)
        test_result = paired_micro_f1_randomization_test(
            aligned_results[model_a],
            aligned_results[model_b],
            n_resamples=args.significance_n,
        )
        pairwise_output[key] = test_result
        p_value = test_result["p_value"]
        if p_value is None:
            print(f"ΔF1={test_result['delta_f1']:+.6f}, p=NA")
            continue
        print(f"ΔF1={test_result['delta_f1']:+.6f}, p={p_value:.6g} (unadjusted)")

    sig_path = results_dir / "pairwise_significance.json"
    sig_path.write_text(json.dumps(pairwise_output, indent=2), encoding="utf-8")
    print(f"  → Saved to {sig_path}")
    print()

    # ── Similarity distribution ────────────────────────────────────
    print("=" * 60)
    print("SPECTER2 similarity score distribution")
    print("=" * 60)

    # Separate matched from best-unmatched
    unmatched_max_scores = [s for s in all_max_scores if s < 0.65]
    matched_above = [s for s in all_matched_scores if s >= 0.65]

    sim_output = {
        "matched": build_histogram(matched_above),
        "unmatched_max": build_histogram(unmatched_max_scores),
        "all_max_scores": build_histogram(all_max_scores),
        "threshold": 0.65,
    }
    print(
        f"  Matched pairs: n={sim_output['matched']['n']}, "
        f"mean={sim_output['matched'].get('mean', 0):.3f}"
    )
    print(
        f"  Unmatched max:  n={sim_output['unmatched_max']['n']}, "
        f"mean={sim_output['unmatched_max'].get('mean', 0):.3f}"
    )
    print(
        f"  All max scores: n={sim_output['all_max_scores']['n']}, "
        f"mean={sim_output['all_max_scores'].get('mean', 0):.3f}"
    )

    sim_path = results_dir / "similarity_distribution.json"
    sim_path.write_text(json.dumps(sim_output, indent=2), encoding="utf-8")
    print(f"  → Saved to {sim_path}")
    print()

    # ── Update result JSONs ────────────────────────────────────────
    print("=" * 60)
    print("Updating result JSON files (micro-averaged per-category)")
    print("=" * 60)

    for model_name, config in MODELS.items():
        result_path = _REPO_ROOT / config["result_file"]
        if not result_path.exists():
            print(f"  [SKIP] {model_name}: {result_path} not found")
            continue
        if model_name not in aligned_results:
            continue

        # Load existing result for metadata
        existing = BenchmarkResult.model_validate(
            json.loads(result_path.read_text(encoding="utf-8"))
        )

        # Aggregate with aligned results from the shared non-PeerJ subset.
        article_results = list(aligned_results[model_name].values())
        n_human = sum(r.n_gt_total for r in article_results)
        n_tool = sum(r.n_tool_total for r in article_results)
        n_figure = sum(r.n_gt_figure_excluded for r in article_results)

        updated = aggregate_results(
            article_results=article_results,
            n_bootstrap=args.bootstrap_n,
            tool_name=existing.tool_name,
            tool_version=existing.tool_version,
            git_hash=existing.git_hash,
            split="test",
            extraction_manifest_id=existing.extraction_manifest_id,
            n_articles=len(article_results),
            n_human_concerns=n_human,
            n_tool_concerns=n_tool,
            n_figure_excluded=n_figure,
            notes=existing.notes,
        )

        result_path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        print(
            f"  ✓ {model_name}: {result_path.name} updated "
            f"(F1={updated.f1_micro:.4f}, n={updated.n_articles})"
        )

    print()
    print("=" * 60)
    print("All done!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

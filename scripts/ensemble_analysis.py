"""Analyze ensemble GT quality and re-evaluate baselines.

Produces:
  1. Per-source ensemble agreement breakdown (matched, borderline, excluded)
  2. Volume comparison: Haiku-only vs Ensemble vs GPT-only
  3. Category distribution shift
  4. Re-evaluate 6 baselines against ensemble GT, compute Kendall tau

Usage:
    uv run python scripts/ensemble_analysis.py \
        --haiku-split data/splits/v3/test.jsonl \
        --ensemble-split results/v3/test_ensemble_v3.jsonl \
        --gpt-gt results/v3/cross_model_gt.jsonl \
        --output results/v3/ensemble_analysis.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioreview_bench.evaluate.metrics import ConcernMatcher, EvalResult
from bioreview_bench.evaluate.runner import build_tool_map


def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# Model configuration (same as source_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════════

MODELS: dict[str, dict] = {
    "Haiku-4.5": {
        "tool_output": "tool_outputs/haiku_test_v4.jsonl",
        "tool_version": "claude-haiku-4-5-20251001",
    },
    "GPT-4o-mini": {
        "tool_output": "tool_outputs/gpt4omini_test_v4.jsonl",
        "tool_version": "gpt-4o-mini",
    },
    "Gemini-2.5-Flash": {
        "tool_output": "tool_outputs/gemini25flash_test_v4.jsonl",
        "tool_version": "gemini-2.5-flash",
    },
    "BM25": {
        "tool_output": "tool_outputs/bm25_test_v4.jsonl",
        "tool_version": "bm25-specter2",
    },
    "Gemini-Flash-Lite": {
        "tool_output": "tool_outputs/gemini_flash_lite_test_v4.jsonl",
        "tool_version": "gemini-2.5-flash-lite",
    },
    "Llama-3.3-70B": {
        "tool_output": "tool_outputs/llama33_test_v4.jsonl",
        "tool_version": "llama-3.3-70b",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Agreement analysis
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_agreement(ensemble_entries: list[dict]) -> dict:
    """Per-source ensemble agreement breakdown."""
    source_stats: dict[str, Counter] = defaultdict(Counter)
    total_stats: Counter = Counter()

    for entry in ensemble_entries:
        source = entry.get("source", "unknown")
        method = entry.get("ensemble_method", "unknown")

        if method == "haiku_only_no_gpt":
            n = len(entry.get("concerns", []))
            source_stats[source]["haiku_only_no_gpt"] += n
            total_stats["haiku_only_no_gpt"] += n
            continue

        for c in entry.get("concerns", []):
            ag = c.get("ensemble_agreement", "unknown")
            source_stats[source][ag] += 1
            total_stats[ag] += 1

    result: dict[str, dict] = {}
    for source in sorted(source_stats):
        counts = source_stats[source]
        total = sum(counts.values())
        result[source] = {
            "total": total,
            "both": counts.get("both", 0),
            "haiku_only_borderline": counts.get("haiku_only_borderline", 0),
            "gpt_only_borderline": counts.get("gpt_only_borderline", 0),
            "haiku_only_no_gpt": counts.get("haiku_only_no_gpt", 0),
            "pct_both": round(counts.get("both", 0) / max(total, 1) * 100, 1),
        }

    total_all = sum(total_stats.values())
    result["_total"] = {
        "total": total_all,
        "both": total_stats.get("both", 0),
        "haiku_only_borderline": total_stats.get("haiku_only_borderline", 0),
        "gpt_only_borderline": total_stats.get("gpt_only_borderline", 0),
        "haiku_only_no_gpt": total_stats.get("haiku_only_no_gpt", 0),
        "pct_both": round(total_stats.get("both", 0) / max(total_all, 1) * 100, 1),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Volume comparison
# ═══════════════════════════════════════════════════════════════════════════════


def compare_volumes(
    haiku_entries: list[dict],
    ensemble_entries: list[dict],
    gpt_entries: list[dict],
) -> dict:
    """Compare concern counts across Haiku-only, Ensemble, and GPT-only GT."""
    import statistics

    def _counts(entries: list[dict]) -> list[int]:
        return [len(e.get("concerns", [])) for e in entries]

    haiku_counts = _counts(haiku_entries)
    ens_counts = _counts(ensemble_entries)
    gpt_counts = _counts(gpt_entries)

    def _stats(counts: list[int]) -> dict:
        if not counts:
            return {"mean": 0, "median": 0, "total": 0, "n": 0}
        return {
            "mean": round(statistics.mean(counts), 2),
            "median": round(statistics.median(counts), 1),
            "total": sum(counts),
            "n": len(counts),
        }

    return {
        "haiku": _stats(haiku_counts),
        "ensemble": _stats(ens_counts),
        "gpt": _stats(gpt_counts),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Category distribution
# ═══════════════════════════════════════════════════════════════════════════════


def category_distribution(entries: list[dict]) -> dict[str, int]:
    """Count concerns per category across all entries."""
    counts: Counter = Counter()
    for entry in entries:
        for c in entry.get("concerns", []):
            cat = c.get("category", "other") if isinstance(c, dict) else "other"
            counts[cat] += 1
    return dict(counts.most_common())


# ═══════════════════════════════════════════════════════════════════════════════
# Re-evaluate baselines against ensemble GT
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_against_gt(
    gt_entries: list[dict],
    matcher: ConcernMatcher,
) -> dict[str, dict]:
    """Evaluate all available models against a given GT.

    Returns:
        {model_name: {recall, precision, f1, n_articles, n_gt, n_tool, n_matched}}
    """
    gt_by_id = {e["id"]: e for e in gt_entries}
    results: dict[str, dict] = {}

    for model_name, config in MODELS.items():
        tool_path = _REPO_ROOT / config["tool_output"]
        if not tool_path.exists():
            continue

        tool_map = build_tool_map(tool_path)
        article_results: list[EvalResult] = []

        for art_id, gt_entry in gt_by_id.items():
            tool_concerns = tool_map.get(art_id, [])
            gt_concerns = gt_entry.get("concerns", [])
            result = matcher.score_article(tool_concerns, gt_concerns)
            article_results.append(result)

        total_matched = sum(r.n_matched for r in article_results)
        total_gt = sum(r.n_gt_total for r in article_results)
        total_tool = sum(r.n_tool_total for r in article_results)
        recall = total_matched / total_gt if total_gt > 0 else 0.0
        precision = total_matched / total_tool if total_tool > 0 else 0.0
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

        results[model_name] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "n_articles": len(article_results),
            "n_gt": total_gt,
            "n_tool": total_tool,
            "n_matched": total_matched,
        }

    return results


def compute_kendall_tau(rankings_a: list[str], rankings_b: list[str]) -> float:
    """Compute Kendall's tau between two model rankings.

    Both lists contain model names ordered from best to worst.
    """
    n = len(rankings_a)
    if n < 2:
        return 1.0

    rank_a = {name: i for i, name in enumerate(rankings_a)}
    rank_b = {name: i for i, name in enumerate(rankings_b)}
    common = sorted(set(rank_a) & set(rank_b))

    concordant = 0
    discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a_diff = rank_a[common[i]] - rank_a[common[j]]
            b_diff = rank_b[common[i]] - rank_b[common[j]]
            if a_diff * b_diff > 0:
                concordant += 1
            elif a_diff * b_diff < 0:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


@click.command()
@click.option("--haiku-split", required=True, type=click.Path(exists=True),
              help="Path to Haiku-extracted test split JSONL")
@click.option("--ensemble-split", required=True, type=click.Path(exists=True),
              help="Path to ensemble GT JSONL (output of build_ensemble_gt.py)")
@click.option("--gpt-gt", required=True, type=click.Path(exists=True),
              help="Path to GPT-extracted concerns JSONL")
@click.option("--output", "-o", required=True,
              help="Output JSON path for analysis results")
@click.option("--threshold", default=0.65, show_default=True)
def main(
    haiku_split: str,
    ensemble_split: str,
    gpt_gt: str,
    output: str,
    threshold: float,
) -> None:
    """Analyze ensemble GT quality and re-evaluate baselines."""
    haiku_entries = _load_jsonl(Path(haiku_split))
    ensemble_entries = _load_jsonl(Path(ensemble_split))
    gpt_entries = _load_jsonl(Path(gpt_gt))

    click.echo(f"Haiku entries:    {len(haiku_entries)}")
    click.echo(f"Ensemble entries: {len(ensemble_entries)}")
    click.echo(f"GPT entries:      {len(gpt_entries)}")

    # 1. Agreement analysis
    click.echo("\n=== Agreement Analysis ===")
    agreement = analyze_agreement(ensemble_entries)
    for source, stats in agreement.items():
        click.echo(f"  {source}: {stats['total']} concerns, "
                    f"{stats['pct_both']:.0f}% both, "
                    f"borderline: {stats['haiku_only_borderline']}H + {stats['gpt_only_borderline']}G")

    # 2. Volume comparison
    click.echo("\n=== Volume Comparison ===")
    volumes = compare_volumes(haiku_entries, ensemble_entries, gpt_entries)
    for label, stats in volumes.items():
        click.echo(f"  {label}: mean={stats['mean']}, median={stats['median']}, total={stats['total']}")

    # 3. Category distribution
    click.echo("\n=== Category Distribution ===")
    cat_haiku = category_distribution(haiku_entries)
    cat_ensemble = category_distribution(ensemble_entries)
    cat_gpt = category_distribution(gpt_entries)
    click.echo(f"  {'Category':<30} {'Haiku':>8} {'Ensemble':>8} {'GPT':>8}")
    all_cats = sorted(set(cat_haiku) | set(cat_ensemble) | set(cat_gpt))
    for cat in all_cats:
        click.echo(f"  {cat:<30} {cat_haiku.get(cat, 0):>8} {cat_ensemble.get(cat, 0):>8} {cat_gpt.get(cat, 0):>8}")

    # 4. Re-evaluate baselines against three GT variants
    click.echo("\n=== Baseline Evaluation (Haiku GT) ===")
    matcher = ConcernMatcher(threshold=threshold, exclude_figure=True)
    haiku_gt_results = evaluate_against_gt(haiku_entries, matcher)
    for model, m in sorted(haiku_gt_results.items(), key=lambda x: -x[1]["f1"]):
        click.echo(f"  {model:<20} F1={m['f1']:.3f} R={m['recall']:.3f} P={m['precision']:.3f}")

    click.echo("\n=== Baseline Evaluation (Ensemble GT) ===")
    ensemble_gt_results = evaluate_against_gt(ensemble_entries, matcher)
    for model, m in sorted(ensemble_gt_results.items(), key=lambda x: -x[1]["f1"]):
        click.echo(f"  {model:<20} F1={m['f1']:.3f} R={m['recall']:.3f} P={m['precision']:.3f}")

    click.echo("\n=== Baseline Evaluation (GPT GT) ===")
    gpt_gt_results = evaluate_against_gt(gpt_entries, matcher)
    for model, m in sorted(gpt_gt_results.items(), key=lambda x: -x[1]["f1"]):
        click.echo(f"  {model:<20} F1={m['f1']:.3f} R={m['recall']:.3f} P={m['precision']:.3f}")

    # 5. Kendall's tau
    haiku_ranking = [m for m, _ in sorted(haiku_gt_results.items(), key=lambda x: -x[1]["f1"])]
    ensemble_ranking = [m for m, _ in sorted(ensemble_gt_results.items(), key=lambda x: -x[1]["f1"])]
    gpt_ranking = [m for m, _ in sorted(gpt_gt_results.items(), key=lambda x: -x[1]["f1"])]

    tau_haiku_gpt = compute_kendall_tau(haiku_ranking, gpt_ranking)
    tau_haiku_ensemble = compute_kendall_tau(haiku_ranking, ensemble_ranking)
    tau_ensemble_gpt = compute_kendall_tau(ensemble_ranking, gpt_ranking)

    click.echo(f"\n=== Kendall's Tau ===")
    click.echo(f"  Haiku GT vs GPT GT:      tau = {tau_haiku_gpt:.3f}")
    click.echo(f"  Haiku GT vs Ensemble GT:  tau = {tau_haiku_ensemble:.3f}")
    click.echo(f"  Ensemble GT vs GPT GT:    tau = {tau_ensemble_gpt:.3f}")

    # Save results
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    analysis = {
        "agreement": agreement,
        "volumes": volumes,
        "category_distribution": {
            "haiku": cat_haiku,
            "ensemble": cat_ensemble,
            "gpt": cat_gpt,
        },
        "baseline_evaluation": {
            "haiku_gt": haiku_gt_results,
            "ensemble_gt": ensemble_gt_results,
            "gpt_gt": gpt_gt_results,
        },
        "rankings": {
            "haiku_gt": haiku_ranking,
            "ensemble_gt": ensemble_ranking,
            "gpt_gt": gpt_ranking,
        },
        "kendall_tau": {
            "haiku_vs_gpt": tau_haiku_gpt,
            "haiku_vs_ensemble": tau_haiku_ensemble,
            "ensemble_vs_gpt": tau_ensemble_gpt,
        },
    }
    out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    click.echo(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

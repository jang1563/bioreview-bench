"""Temporal (contamination) analysis: compare model performance on pre-2023 vs post-2023 articles.

Motivation: Several evaluated LLMs have training cutoffs in 2023-2024. Articles published
before the cutoff may have appeared in training data, potentially inflating recall scores.
This script splits the test set by publication year and compares per-cohort F1.

Usage:
    uv run python scripts/temporal_analysis.py \
        --split data/splits/v4/test.jsonl \
        --outputs tool_outputs/haiku_test_v4.jsonl tool_outputs/gpt4omini_test_v4.jsonl \
        --labels Haiku-4.5 GPT-4o-mini \
        --cutoff 2023 \
        --output results/v4/temporal_analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioreview_bench.evaluate.metrics import ConcernMatcher


def load_tool_map(path: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            aid = obj.get("article_id", "")
            result[aid] = [c for c in obj.get("concerns", []) if isinstance(c, str)]
    return result


def cohort_label(year: str, cutoff: int) -> str:
    try:
        return f"pre-{cutoff}" if int(year) < cutoff else f"{cutoff}+"
    except (ValueError, TypeError):
        return "unknown"


def run(
    split_path: str,
    output_paths: list[str],
    labels: list[str],
    cutoff: int,
    output: str,
) -> None:
    matcher = ConcernMatcher()

    with open(split_path) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    tool_maps = [load_tool_map(p) for p in output_paths]

    # Per-cohort TP/FP/FN accumulators: {label: {cohort: {metric: int}}}
    stats: dict[str, dict[str, dict[str, float]]] = {
        lbl: defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n_articles": 0})
        for lbl in labels
    }
    # Also per (cohort × source)
    src_stats: dict[str, dict[str, dict[str, float]]] = {
        lbl: defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n_articles": 0})
        for lbl in labels
    }

    for entry in entries:
        aid = entry["id"]
        gt_concerns = entry.get("concerns", [])
        date = entry.get("published_date", "")
        year = date[:4] if date else "unknown"
        cohort = cohort_label(year, cutoff)
        source = entry.get("source", "unknown")
        src_cohort = f"{source}/{cohort}"

        for lbl, tool_map in zip(labels, tool_maps):
            tool_concerns = tool_map.get(aid, [])
            result = matcher.score_article(tool_concerns, gt_concerns)
            tp = result.n_matched
            fp = result.n_tool_total - result.n_matched
            fn = result.n_gt_total - result.n_matched

            for bucket in (stats[lbl][cohort], src_stats[lbl][src_cohort]):
                bucket["tp"] += tp
                bucket["fp"] += fp
                bucket["fn"] += fn
                bucket["n_articles"] += 1

    def micro_f1(bucket: dict) -> dict:
        tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {
            "f1": round(f1, 4),
            "recall": round(r, 4),
            "precision": round(p, 4),
            "n_articles": int(bucket["n_articles"]),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    results = {
        "cutoff_year": cutoff,
        "split": split_path,
        "models": {},
    }

    # Print summary
    print(f"=== Temporal Analysis (cutoff: {cutoff}) ===\n")
    for lbl in labels:
        results["models"][lbl] = {
            "by_cohort": {},
            "by_source_cohort": {},
        }
        print(f"{lbl}:")
        for cohort in sorted(stats[lbl]):
            m = micro_f1(stats[lbl][cohort])
            results["models"][lbl]["by_cohort"][cohort] = m
            print(f"  {cohort:12s}  F1={m['f1']:.3f}  R={m['recall']:.3f}  P={m['precision']:.3f}  n={m['n_articles']}")

        print(f"  {'Per-source':12s}")
        for key in sorted(src_stats[lbl]):
            m = micro_f1(src_stats[lbl][key])
            results["models"][lbl]["by_source_cohort"][key] = m
            print(f"    {key:20s}  F1={m['f1']:.3f}  R={m['recall']:.3f}  n={m['n_articles']}")
        print()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--cutoff", type=int, default=2023)
    parser.add_argument("--output", default="results/v4/temporal_analysis.json")
    args = parser.parse_args()

    if len(args.outputs) != len(args.labels):
        parser.error("--outputs and --labels must have the same number of arguments")

    run(args.split, args.outputs, args.labels, args.cutoff, args.output)


if __name__ == "__main__":
    main()

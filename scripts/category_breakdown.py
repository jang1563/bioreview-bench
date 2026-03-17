"""Generate per-category cross-model comparison table.

Reads all test result JSONs from results/v3/ and produces a markdown + JSON
breakdown of per-category metrics for each model.

Usage:
    python scripts/category_breakdown.py
    python scripts/category_breakdown.py --results-dir results/v3 -o results/v3/category_breakdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from bioreview_bench.models.benchmark import BenchmarkResult

ROOT = Path(__file__).resolve().parents[1]

# Display-friendly category names
CATEGORY_LABELS = {
    "statistical_methodology": "Statistical Methodology",
    "design_flaw": "Design Flaw",
    "missing_experiment": "Missing Experiment",
    "interpretation": "Interpretation",
    "prior_art_novelty": "Prior Art / Novelty",
    "reagent_method_specificity": "Reagent / Method Specificity",
    "writing_clarity": "Writing Clarity",
    "other": "Other",
}


def load_results(results_dir: Path, split: str = "test") -> list[BenchmarkResult]:
    """Load non-dedup BenchmarkResult files, keeping best per (tool_name, tool_version)."""
    raw: list[BenchmarkResult] = []
    for json_file in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        try:
            result = BenchmarkResult.model_validate(data)
        except ValidationError:
            continue
        if result.split != split or result.dedup_gt:
            continue
        raw.append(result)

    # Best per (tool_name, tool_version) by F1
    best: dict[tuple[str, str], BenchmarkResult] = {}
    for r in raw:
        key = (r.tool_name, r.tool_version)
        prev = best.get(key)
        if prev is None or r.f1_micro > prev.f1_micro:
            best[key] = r

    # Sort by F1 descending
    return sorted(best.values(), key=lambda r: r.f1_micro, reverse=True)


def build_table(results: list[BenchmarkResult]) -> tuple[str, dict]:
    """Build markdown table and JSON data for per-category breakdown."""
    # Collect all categories across models
    all_cats = set()
    for r in results:
        all_cats.update(r.per_category.keys())

    # Stable category order
    cat_order = [c for c in CATEGORY_LABELS if c in all_cats]
    cat_order += sorted(all_cats - set(cat_order))

    model_names = [r.tool_name for r in results]

    # Short display names (ensure uniqueness)
    short_names = {
        "Haiku-4.5": "Haiku",
        "GPT-4o-mini": "GPT-4o-mini",
        "Gemini-2.5-Flash": "Gem-Flash",
        "BM25": "BM25",
        "Gemini-2.5-Flash-Lite": "Gem-Lite",
        "Llama-3.3-70B": "Llama-70B",
    }

    # Header
    header = "| Category | n_GT |"
    sep = "|----------|------|"
    for name in model_names:
        short = short_names.get(name, name[:10])
        header += f" {short} R/P/F1 |"
        sep += "----------------|"

    lines = [header, sep]

    json_data: dict = {"models": model_names, "categories": {}}

    for cat in cat_order:
        label = CATEGORY_LABELS.get(cat, cat)

        # Get n_gt from first model that has this category
        n_gt = 0
        for r in results:
            cm = r.per_category.get(cat)
            if cm:
                n_gt = cm.n_human_concerns
                break

        row = f"| {label} | {n_gt} |"
        cat_json: dict = {"n_gt": n_gt, "models": {}}

        # Find best F1 for this category
        best_f1 = 0.0
        for r in results:
            cm = r.per_category.get(cat)
            if cm and cm.f1_micro > best_f1:
                best_f1 = cm.f1_micro

        for r in results:
            cm = r.per_category.get(cat)
            if cm:
                f1_str = f"{cm.f1_micro:.3f}"
                if cm.f1_micro == best_f1 and best_f1 > 0:
                    f1_str = f"**{f1_str}**"
                row += f" {cm.recall:.2f}/{cm.precision:.2f}/{f1_str} |"
                cat_json["models"][r.tool_name] = {
                    "recall": round(cm.recall, 3),
                    "precision": round(cm.precision, 3),
                    "f1": round(cm.f1_micro, 3),
                    "n_matched": cm.n_matched,
                }
            else:
                row += " — |"

        lines.append(row)
        json_data["categories"][cat] = cat_json

    return "\n".join(lines), json_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-category cross-model breakdown table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results" / "v3",
        help="Directory containing BenchmarkResult JSON files.",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=ROOT / "results" / "v3" / "category_breakdown",
        help="Output path prefix (without extension).",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir, args.split)
    if not results:
        print("No results found.")
        return

    print(f"Loaded {len(results)} model results for split={args.split}")
    for r in results:
        print(f"  {r.tool_name} (F1={r.f1_micro:.3f})")

    md_table, json_data = build_table(results)

    md_path = Path(str(args.output) + ".md")
    md_path.write_text(md_table, encoding="utf-8")
    print(f"\nMarkdown table saved to {md_path}")

    json_path = Path(str(args.output) + ".json")
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"JSON data saved to {json_path}")

    print(f"\n{md_table}")


if __name__ == "__main__":
    main()

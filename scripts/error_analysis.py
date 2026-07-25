"""Error analysis: extract and categorize FP and FN examples from benchmark results.

Produces a JSON report with 50 false positives and 50 false negatives from the
best model (Haiku-4.5 by default), along with error type categorization.
The report contains article-level manuscript/review-derived text and is
local-only. Do not commit or upload it; publish aggregate diagnostics such as
``results/v4/measurement_audit.{json,md}`` instead.

Usage:
    arch -arm64 .venv-arm64/bin/python3 scripts/error_analysis.py \
        --tool-output tool_outputs/haiku_test_v4.jsonl \
        --split data/splits/v4/test.jsonl \
        --n 50 \
        --output results/v4/error_analysis.json
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import click

from bioreview_bench.evaluate.metrics import ConcernMatcher

# ── Error taxonomy ─────────────────────────────────────────────────────────

FP_CATEGORIES = {
    "plausible_unstated": "Concern is valid but reviewers didn't raise it",
    "overly_specific": "Tool concern is too narrow/specific vs GT scope",
    "hallucinated": "Concern has no basis in the paper content",
    "duplicate_partial": "Concern partially overlaps with a matched GT concern",
    "figure_related": "Concern about figures (excluded from GT)",
    "stylistic_noise": "Generic writing/style concern not substantive",
    "misinterpretation": "Tool misunderstood the paper content",
}

FN_CATEGORIES = {
    "domain_expert": "Requires deep domain knowledge to detect",
    "subtle_flaw": "Subtle methodological issue hard to spot",
    "cross_reference": "Requires comparing multiple paper sections",
    "writing_quality": "Stylistic concern overlooked by tool",
    "statistical_nuance": "Statistical issue requiring expertise",
    "gt_extraction_artifact": "GT concern may be extraction artifact or too vague",
    "near_miss": "Tool raised a related but insufficiently matching concern",
}


@dataclass
class ErrorExample:
    article_id: str
    source: str
    concern_text: str
    category: str  # GT category or "tool_generated"
    severity: str  # GT severity or "unknown"
    nearest_text: str  # nearest match from the other side
    nearest_score: float  # similarity score of nearest
    error_type: str  # categorized error type
    context: str  # article title for context


def _classify_fp(concern: str, nearest_gt: str, sim: float, gt_categories: list[str]) -> str:
    """Classify false positive type based on content analysis.

    SPECTER2 cosine between concerns from the same paper is inherently high
    (0.83-0.97), so classification is purely content-based.
    """
    concern_lower = concern.lower()

    # Figure-related (excluded from GT, so always FP by design)
    if any(kw in concern_lower for kw in [
        "figure", "panel", "image", "micrograph", "blot", "supplementary fig",
    ]):
        return "figure_related"

    # Stylistic / writing excess
    stylistic_kw = ["writing", "clarity", "grammar", "typo", "spelling",
                     "readability", "formatting", "style", "proofread",
                     "sentence", "paragraph", "reorganize", "verbose",
                     "poorly written", "unclear language"]
    if any(kw in concern_lower for kw in stylistic_kw):
        return "stylistic_excess"

    # Method/reagent over-specificity
    method_kw = ["antibod", "primer", "protocol", "reagent", "catalogue",
                 "dilution", "vendor", "kit", "buffer", "concentration",
                 "catalog number", "lot number", "manufacturer"]
    if any(kw in concern_lower for kw in method_kw):
        return "overcritical_methods"

    # Statistical concern
    stat_kw = ["p-value", "multiple comparison", "bonferroni", "sample size",
               "power analysis", "statistical test", "normality", "correction for",
               "confidence interval", "effect size"]
    if any(kw in concern_lower for kw in stat_kw):
        return "overcritical_statistics"

    # Missing experiment / control
    exp_kw = ["control", "experiment", "validation", "replicate", "verify",
              "additional experiment", "negative control", "positive control"]
    if any(kw in concern_lower for kw in exp_kw):
        return "extra_experiment_requested"

    # Interpretation / overclaiming
    interp_kw = ["overclaim", "overstate", "causal", "correlation",
                 "alternative explanation", "generalizab", "extrapolat"]
    if any(kw in concern_lower for kw in interp_kw):
        return "interpretation_pushback"

    # Prior work / novelty
    prior_kw = ["prior", "previous", "published", "literature", "cite",
                "reference", "novel", "already shown", "known"]
    if any(kw in concern_lower for kw in prior_kw):
        return "prior_work_gap"

    # Default: plausible concern that reviewers simply didn't raise
    return "plausible_unstated"


def _classify_fn(concern: str, nearest_tool: str, sim: float, category: str) -> str:
    """Classify false negative type based on GT category and content analysis."""
    # Very short GT concern (too vague for tool to match)
    if len(concern) < 80:
        return "vague_gt_concern"

    # Classify by GT category
    if category == "writing_clarity":
        return "missed_writing"

    if category == "reagent_method_specificity":
        return "missed_methods_detail"

    if category == "statistical_methodology":
        return "missed_statistics"

    if category == "missing_experiment":
        return "missed_experiment"

    if category == "design_flaw":
        return "missed_design_flaw"

    if category == "interpretation":
        return "missed_interpretation"

    if category == "prior_art_novelty":
        return "missed_prior_work"

    return "other_missed"


@click.command()
@click.option("--tool-output", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--split", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--n", default=50, help="Number of FP and FN examples to sample")
@click.option("--seed", default=42, help="Random seed for sampling")
@click.option("--output", required=True, type=click.Path(path_type=Path))
def main(tool_output: Path, split: Path, n: int, seed: int, output: Path) -> None:
    random.seed(seed)

    # Load data
    gt_by_id: dict[str, dict] = {}
    with open(split) as f:
        for line in f:
            entry = json.loads(line)
            gt_by_id[entry["id"]] = entry

    tool_by_id: dict[str, list[str]] = {}
    with open(tool_output) as f:
        for line in f:
            entry = json.loads(line)
            aid = entry.get("article_id", entry.get("id", ""))
            tool_by_id[aid] = entry.get("concerns", [])

    click.echo(f"GT articles: {len(gt_by_id)}, Tool articles: {len(tool_by_id)}")

    # Initialize matcher
    matcher = ConcernMatcher(
        threshold=0.65,
        exclude_figure=True,
        use_embedding=True,
        algorithm="hungarian",
    )

    all_fps: list[ErrorExample] = []
    all_fns: list[ErrorExample] = []

    common_ids = sorted(set(gt_by_id) & set(tool_by_id))
    click.echo(f"Common articles: {len(common_ids)}")

    for i, aid in enumerate(common_ids):
        if i % 100 == 0:
            click.echo(f"  Processing {i}/{len(common_ids)}...")

        gt_entry = gt_by_id[aid]
        tool_concerns = tool_by_id[aid]
        gt_concerns = gt_entry.get("concerns", [])
        title = gt_entry.get("title", "")
        source = gt_entry.get("source", "unknown")

        # Filter figure concerns
        active_gt = [c for c in gt_concerns if not c.get("requires_figure_reading", False)]
        gt_texts = [c["concern_text"] for c in active_gt]

        if not gt_texts or not tool_concerns:
            continue

        # Compute similarity matrix and matches
        scores = matcher._compute_scores(tool_concerns, gt_texts)
        matches = matcher._match(scores)

        matched_gt_idxs = {m.gt_idx for m in matches}
        matched_tool_idxs = {m.tool_idx for m in matches}

        # Extract FPs: unmatched tool concerns
        for ti in range(len(tool_concerns)):
            if ti in matched_tool_idxs:
                continue
            # Find nearest GT concern
            if scores.matrix:
                row = scores.matrix[ti]
                best_gi = max(range(len(row)), key=lambda j: row[j])
                best_sim = row[best_gi]
                nearest_gt_text = gt_texts[best_gi]
            else:
                best_sim = 0.0
                nearest_gt_text = ""

            gt_cats = [c.get("category", "other") for c in active_gt]
            error_type = _classify_fp(tool_concerns[ti], nearest_gt_text, best_sim, gt_cats)

            all_fps.append(ErrorExample(
                article_id=aid,
                source=source,
                concern_text=tool_concerns[ti],
                category="tool_generated",
                severity="unknown",
                nearest_text=nearest_gt_text,
                nearest_score=round(best_sim, 4),
                error_type=error_type,
                context=title[:120],
            ))

        # Extract FNs: unmatched GT concerns
        for gi in range(len(active_gt)):
            if gi in matched_gt_idxs:
                continue
            gt_c = active_gt[gi]
            # Find nearest tool concern
            if scores.matrix:
                col = [scores.matrix[ti][gi] for ti in range(len(tool_concerns))]
                best_ti = max(range(len(col)), key=lambda j: col[j])
                best_sim = col[best_ti]
                nearest_tool_text = tool_concerns[best_ti]
            else:
                best_sim = 0.0
                nearest_tool_text = ""

            error_type = _classify_fn(
                gt_c["concern_text"], nearest_tool_text, best_sim,
                gt_c.get("category", "other"),
            )

            all_fns.append(ErrorExample(
                article_id=aid,
                source=source,
                concern_text=gt_c["concern_text"],
                category=gt_c.get("category", "other"),
                severity=gt_c.get("severity", "unknown"),
                nearest_text=nearest_tool_text,
                nearest_score=round(best_sim, 4),
                error_type=error_type,
                context=title[:120],
            ))

    click.echo(f"\nTotal FPs: {len(all_fps)}, Total FNs: {len(all_fns)}")

    # Sample n of each, stratified by error type
    sampled_fps = _stratified_sample(all_fps, n, seed)
    sampled_fns = _stratified_sample(all_fns, n, seed)

    # Compute summary statistics
    fp_type_dist = Counter(e.error_type for e in all_fps)
    fn_type_dist = Counter(e.error_type for e in all_fns)
    fn_cat_dist = Counter(e.category for e in all_fns)
    fp_source_dist = Counter(e.source for e in all_fps)
    fn_source_dist = Counter(e.source for e in all_fns)

    # Compute near-miss statistics
    fp_near_miss_pct = sum(1 for e in all_fps if e.nearest_score >= 0.50) / max(len(all_fps), 1)
    fn_near_miss_pct = sum(1 for e in all_fns if e.nearest_score >= 0.50) / max(len(all_fns), 1)

    report = {
        "summary": {
            "total_fps": len(all_fps),
            "total_fns": len(all_fns),
            "sampled_fps": len(sampled_fps),
            "sampled_fns": len(sampled_fns),
            "fp_near_miss_pct": round(fp_near_miss_pct, 3),
            "fn_near_miss_pct": round(fn_near_miss_pct, 3),
        },
        "fp_type_distribution": dict(fp_type_dist.most_common()),
        "fn_type_distribution": dict(fn_type_dist.most_common()),
        "fn_category_distribution": dict(fn_cat_dist.most_common()),
        "fp_source_distribution": dict(fp_source_dist.most_common()),
        "fn_source_distribution": dict(fn_source_dist.most_common()),
        "false_positives": [_example_to_dict(e) for e in sampled_fps],
        "false_negatives": [_example_to_dict(e) for e in sampled_fns],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    click.echo("\nFP type distribution:")
    for t, c in fp_type_dist.most_common():
        click.echo(f"  {t}: {c} ({c/len(all_fps)*100:.1f}%)")

    click.echo("\nFN type distribution:")
    for t, c in fn_type_dist.most_common():
        click.echo(f"  {t}: {c} ({c/len(all_fns)*100:.1f}%)")

    click.echo("\nFN by GT category:")
    for t, c in fn_cat_dist.most_common():
        click.echo(f"  {t}: {c} ({c/len(all_fns)*100:.1f}%)")

    click.echo("\nNear-miss rates (sim >= 0.50):")
    click.echo(f"  FP: {fp_near_miss_pct:.1%}")
    click.echo(f"  FN: {fn_near_miss_pct:.1%}")

    click.echo(f"\nReport saved to: {output}")


def _stratified_sample(
    examples: list[ErrorExample], n: int, seed: int,
) -> list[ErrorExample]:
    """Sample n examples, stratified by error_type."""
    if len(examples) <= n:
        return examples

    rng = random.Random(seed)
    by_type: dict[str, list[ErrorExample]] = defaultdict(list)
    for e in examples:
        by_type[e.error_type].append(e)

    # Proportional allocation
    sampled = []
    for etype, group in by_type.items():
        k = max(1, round(n * len(group) / len(examples)))
        sampled.extend(rng.sample(group, min(k, len(group))))

    # Fill up to n if needed
    if len(sampled) < n:
        remaining = [e for e in examples if e not in set(sampled)]
        sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))

    # Trim if over
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)

    return sampled


def _example_to_dict(e: ErrorExample) -> dict:
    return {
        "article_id": e.article_id,
        "source": e.source,
        "concern_text": e.concern_text[:500],
        "category": e.category,
        "severity": e.severity,
        "nearest_text": e.nearest_text[:500],
        "nearest_score": e.nearest_score,
        "error_type": e.error_type,
        "context": e.context,
    }


if __name__ == "__main__":
    main()

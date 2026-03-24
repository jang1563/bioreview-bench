"""Build ensemble ground truth from Haiku + GPT-4o-mini extractions.

Ensemble rules (intersection with borderline inclusion):
  1. Hungarian match Haiku <-> GPT concerns (threshold=0.65)
  2. Matched pairs (sim >= threshold): INCLUDE, use Haiku text
  3. Unmatched Haiku with max_sim < borderline: EXCLUDE
  4. Unmatched Haiku with borderline <= max_sim < threshold: INCLUDE (borderline)
  5. Unmatched GPT with max_sim < borderline: EXCLUDE
  6. Unmatched GPT with borderline <= max_sim < threshold: INCLUDE (borderline)

Usage:
    uv run python scripts/build_ensemble_gt.py \\
        --haiku-split data/splits/v3/test.jsonl \\
        --gpt-gt results/v3/cross_model_gt.jsonl \\
        --output results/v3/test_ensemble_v3.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioreview_bench.evaluate.metrics import ConcernMatcher


def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _extract_concern_texts(entry: dict) -> list[str]:
    """Extract concern_text strings from an entry's concerns list."""
    concerns = entry.get("concerns", [])
    return [
        c["concern_text"] if isinstance(c, dict) else str(c)
        for c in concerns
    ]


def _max_sim_to_set(
    text: str, other_texts: list[str], matcher: ConcernMatcher
) -> float:
    """Compute max cosine similarity of text against a set of other texts."""
    if not other_texts:
        return 0.0
    scores = matcher._compute_scores([text], other_texts)
    if not scores.matrix or not scores.matrix[0]:
        return 0.0
    return max(scores.matrix[0])


def build_ensemble_concerns(
    haiku_entry: dict,
    gpt_entry: dict,
    matcher: ConcernMatcher,
    threshold: float,
    borderline_threshold: float,
) -> list[dict]:
    """Build ensemble concerns from two extractor outputs for a single article."""
    haiku_concerns = haiku_entry.get("concerns", [])
    gpt_concerns = gpt_entry.get("concerns", [])

    if not haiku_concerns and not gpt_concerns:
        return []

    haiku_texts = [
        c["concern_text"] if isinstance(c, dict) else str(c) for c in haiku_concerns
    ]
    gpt_texts = [
        c["concern_text"] if isinstance(c, dict) else str(c) for c in gpt_concerns
    ]

    # Compute similarity matrix and Hungarian matching
    scores = matcher._compute_scores(haiku_texts, gpt_texts)
    matches = matcher._match(scores)

    matched_haiku = {m.tool_idx for m in matches}
    matched_gpt = {m.gt_idx for m in matches}

    result = []

    # 1. Matched pairs: INCLUDE with Haiku text, resolved category/severity
    for m in matches:
        h_concern = haiku_concerns[m.tool_idx]
        g_concern = gpt_concerns[m.gt_idx] if isinstance(gpt_concerns[m.gt_idx], dict) else {}

        concern = dict(h_concern) if isinstance(h_concern, dict) else {"concern_text": str(h_concern)}

        # Resolve category: agree if same, else use Haiku
        h_cat = h_concern.get("category", "other") if isinstance(h_concern, dict) else "other"
        g_cat = g_concern.get("category", "other") if isinstance(g_concern, dict) else "other"
        concern["category"] = h_cat
        concern["category_agreed"] = h_cat == g_cat

        # Resolve severity: major if either says major (conservative)
        h_sev = h_concern.get("severity", "minor") if isinstance(h_concern, dict) else "minor"
        g_sev = g_concern.get("severity", "minor") if isinstance(g_concern, dict) else "minor"
        if h_sev == "major" or g_sev == "major":
            concern["severity"] = "major"
        else:
            concern["severity"] = h_sev

        concern["ensemble_agreement"] = "both"
        concern["ensemble_match_score"] = round(m.score, 4)
        result.append(concern)

    # 2. Unmatched Haiku concerns: check borderline
    for i, h_concern in enumerate(haiku_concerns):
        if i in matched_haiku:
            continue
        h_text = h_concern["concern_text"] if isinstance(h_concern, dict) else str(h_concern)
        max_sim = _max_sim_to_set(h_text, gpt_texts, matcher)

        if max_sim >= borderline_threshold:
            concern = dict(h_concern) if isinstance(h_concern, dict) else {"concern_text": h_text}
            concern["ensemble_agreement"] = "haiku_only_borderline"
            concern["ensemble_match_score"] = round(max_sim, 4)
            result.append(concern)
        # else: excluded (max_sim < borderline)

    # 3. Unmatched GPT concerns: check borderline
    for j, g_concern in enumerate(gpt_concerns):
        if j in matched_gpt:
            continue
        g_text = g_concern["concern_text"] if isinstance(g_concern, dict) else str(g_concern)
        max_sim = _max_sim_to_set(g_text, haiku_texts, matcher)

        if max_sim >= borderline_threshold:
            concern = dict(g_concern) if isinstance(g_concern, dict) else {"concern_text": g_text}
            concern["ensemble_agreement"] = "gpt_only_borderline"
            concern["ensemble_match_score"] = round(max_sim, 4)
            result.append(concern)
        # else: excluded (max_sim < borderline)

    return result


@click.command()
@click.option("--haiku-split", required=True, type=click.Path(exists=True),
              help="Path to Haiku-extracted test split JSONL")
@click.option("--gpt-gt", required=True, type=click.Path(exists=True),
              help="Path to GPT-extracted concerns JSONL")
@click.option("--output", "-o", required=True,
              help="Output JSONL path for ensemble GT")
@click.option("--threshold", default=0.65, show_default=True,
              help="Similarity threshold for matched pairs")
@click.option("--borderline-threshold", default=0.50, show_default=True,
              help="Similarity threshold for borderline inclusion")
def main(
    haiku_split: str,
    gpt_gt: str,
    output: str,
    threshold: float,
    borderline_threshold: float,
) -> None:
    """Build ensemble GT from two extractor outputs."""
    haiku_entries = _load_jsonl(Path(haiku_split))
    gpt_entries = _load_jsonl(Path(gpt_gt))

    gpt_by_id = {e["id"]: e for e in gpt_entries}

    matcher = ConcernMatcher(threshold=threshold, exclude_figure=False)

    stats = {
        "n_articles": 0,
        "n_haiku_total": 0,
        "n_gpt_total": 0,
        "n_ensemble_total": 0,
        "n_both": 0,
        "n_haiku_borderline": 0,
        "n_gpt_borderline": 0,
        "n_haiku_excluded": 0,
        "n_gpt_excluded": 0,
        "n_no_gpt": 0,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for haiku_entry in haiku_entries:
            article_id = haiku_entry.get("id", "")
            gpt_entry = gpt_by_id.get(article_id)

            n_haiku = len(haiku_entry.get("concerns", []))
            stats["n_haiku_total"] += n_haiku
            stats["n_articles"] += 1

            if gpt_entry is None:
                # No GPT extraction: keep Haiku GT as-is
                stats["n_no_gpt"] += 1
                stats["n_ensemble_total"] += n_haiku
                out_entry = dict(haiku_entry)
                out_entry["ensemble_method"] = "haiku_only_no_gpt"
                f.write(json.dumps(out_entry, ensure_ascii=False, default=str) + "\n")
                continue

            n_gpt = len(gpt_entry.get("concerns", []))
            stats["n_gpt_total"] += n_gpt

            ensemble_concerns = build_ensemble_concerns(
                haiku_entry, gpt_entry, matcher, threshold, borderline_threshold
            )

            # Count by agreement type
            for c in ensemble_concerns:
                ag = c.get("ensemble_agreement", "")
                if ag == "both":
                    stats["n_both"] += 1
                elif ag == "haiku_only_borderline":
                    stats["n_haiku_borderline"] += 1
                elif ag == "gpt_only_borderline":
                    stats["n_gpt_borderline"] += 1

            n_ensemble = len(ensemble_concerns)
            stats["n_ensemble_total"] += n_ensemble
            stats["n_haiku_excluded"] += n_haiku - sum(
                1 for c in ensemble_concerns
                if c.get("ensemble_agreement") in ("both", "haiku_only_borderline")
            )
            stats["n_gpt_excluded"] += n_gpt - sum(
                1 for c in ensemble_concerns
                if c.get("ensemble_agreement") in ("both", "gpt_only_borderline")
            )

            out_entry = dict(haiku_entry)
            out_entry["concerns"] = ensemble_concerns
            out_entry["ensemble_method"] = "haiku_gpt_intersection"
            out_entry["n_haiku_concerns"] = n_haiku
            out_entry["n_gpt_concerns"] = n_gpt

            f.write(json.dumps(out_entry, ensure_ascii=False, default=str) + "\n")

    click.echo(f"\nEnsemble GT Summary:")
    click.echo(f"  Articles: {stats['n_articles']} ({stats['n_no_gpt']} without GPT)")
    click.echo(f"  Haiku concerns: {stats['n_haiku_total']}")
    click.echo(f"  GPT concerns: {stats['n_gpt_total']}")
    click.echo(f"  Ensemble concerns: {stats['n_ensemble_total']}")
    click.echo(f"    Both agreed: {stats['n_both']}")
    click.echo(f"    Haiku-only borderline: {stats['n_haiku_borderline']}")
    click.echo(f"    GPT-only borderline: {stats['n_gpt_borderline']}")
    click.echo(f"    Haiku excluded: {stats['n_haiku_excluded']}")
    click.echo(f"    GPT excluded: {stats['n_gpt_excluded']}")

    reduction = 1.0 - stats["n_ensemble_total"] / max(stats["n_haiku_total"], 1)
    click.echo(f"  Reduction from Haiku-only: {reduction:.1%}")

    # Save stats
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    click.echo(f"\n  Stats saved to {stats_path}")


if __name__ == "__main__":
    main()

"""Cross-model ground truth validation.

Re-extracts concerns from test-split raw review text using a second LLM
(default: GPT-4o-mini) and computes inter-extractor agreement metrics.

Subcommands:
    extract  — Run concern extraction with an alternative model
    analyze  — Compute 5 agreement metrics from existing extraction results

Usage:
    # Step 1: Extract concerns with GPT-4o-mini (~44 min, ~$1.37)
    uv run python scripts/cross_model_validation.py extract \
        --model gpt-4o-mini --provider openai \
        --split-file data/splits/v3/test.jsonl \
        --output results/v3/cross_model_gt.jsonl

    # Step 2: Compute agreement metrics (~30 min for SPECTER2)
    uv run python scripts/cross_model_validation.py analyze \
        --haiku-split data/splits/v3/test.jsonl \
        --alt-gt results/v3/cross_model_gt.jsonl \
        --output results/v3/cross_model_agreement.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioreview_bench.evaluate.metrics import ConcernMatcher  # noqa: E402
from bioreview_bench.parse.concern_extractor import (  # noqa: E402
    ConcernExtractor,
    split_into_reviewer_blocks,
)
from bioreview_bench.parse.jats import ParsedReview  # noqa: E402

console = Console()
log = logging.getLogger(__name__)

# ── Model config (same as source_analysis.py) ────────────────────────────────

MODELS: dict[str, dict] = {
    "Haiku-4.5": {
        "tool_output": "tool_outputs/haiku_test_v3.jsonl",
        "tool_version": "claude-haiku-4-5-20251001",
    },
    "GPT-4o-mini": {
        "tool_output": "tool_outputs/gpt-4o-mini_test.jsonl",
        "tool_version": "gpt-4o-mini",
    },
    "Gemini-2.5-Flash": {
        "tool_output": "tool_outputs/v3/gemini25flash_test_v2.jsonl",
        "tool_version": "gemini-2.5-flash",
    },
    "BM25": {
        "tool_output": "tool_outputs/bm25_test.jsonl",
        "tool_version": "bm25-specter2",
    },
    "Gemini-Flash-Lite": {
        "tool_output": "tool_outputs/gemini-2.5-flash-lite_test_v2.jsonl",
        "tool_version": "gemini-2.5-flash-lite",
    },
    "Llama-3.3-70B": {
        "tool_output": "tool_outputs/v3/llama33_test.jsonl",
        "tool_version": "llama-3.3-70b",
    },
}

# Known Haiku GT F1 values (from results/v3/*_test_v3.json, 4-decimal precision)
HAIKU_GT_F1: dict[str, float] = {
    "Haiku-4.5": 0.6988,
    "GPT-4o-mini": 0.6937,
    "Gemini-2.5-Flash": 0.6865,
    "BM25": 0.6852,
    "Gemini-Flash-Lite": 0.6584,
    "Llama-3.3-70B": 0.6526,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Extract subcommand
# ═══════════════════════════════════════════════════════════════════════════════


def load_existing_ids(output_path: Path) -> set[str]:
    """Load already-processed article IDs from output JSONL for resume support."""
    ids: set[str] = set()
    if not output_path.exists():
        return ids
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    art_id = entry.get("id", "")
                    if art_id:
                        ids.add(art_id)
                except json.JSONDecodeError:
                    pass
    return ids


def extract_article_concerns(
    entry: dict,
    extractor: ConcernExtractor,
) -> list[dict]:
    """Extract concerns from a single article using the given extractor.

    Passes empty author_response_text to skip Stage 2 (resolution classification)
    since we only need concern_text, category, severity for comparison.
    """
    decision_letter_raw = entry.get("decision_letter_raw", "")
    article_doi = entry.get("doi", "")
    article_source = entry.get("source", "elife")

    blocks = split_into_reviewer_blocks(decision_letter_raw)
    if not blocks:
        return []

    all_concerns = []
    for r_idx, block_text in enumerate(blocks, start=1):
        review = ParsedReview(
            reviewer_num=r_idx,
            review_text=block_text,
            author_response_text="",  # Skip Stage 2 deliberately
        )
        concerns = extractor.process_review(
            review,
            article_doi=article_doi,
            article_source=article_source,
        )
        all_concerns.extend(concerns)

    return [c.model_dump() for c in all_concerns]


@click.command("extract")
@click.option("--model", default="gpt-4o-mini", help="LLM model ID")
@click.option("--provider", default="openai", help="API provider")
@click.option(
    "--split-file",
    required=True,
    type=click.Path(exists=True),
    help="Path to test.jsonl",
)
@click.option("--output", "-o", required=True, help="Output JSONL path")
def cmd_extract(model: str, provider: str, split_file: str, output: str) -> None:
    """Extract concerns with an alternative model."""
    split_path = Path(split_file)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Cross-Model GT Extraction[/bold cyan]")
    console.print(f"  model    : {model}")
    console.print(f"  provider : {provider}")
    console.print(f"  split    : {split_path}")
    console.print(f"  output   : {output_path}")
    console.print()

    # Load test split
    entries = []
    with split_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    console.print(f"Loaded {len(entries)} articles from split")

    # Resume support
    existing_ids = load_existing_ids(output_path)
    to_process = [e for e in entries if e["id"] not in existing_ids]
    console.print(f"  {len(existing_ids)} already processed (skip)")
    console.print(f"  {len(to_process)} to process")
    console.print()

    if not to_process:
        console.print("[green]Nothing to do — all articles already processed.")
        return

    extractor = ConcernExtractor(
        model=model,
        manifest_id=f"cross-model-{model}",
        provider=provider,
    )

    stats = {"processed": 0, "failed": 0, "total_concerns": 0}
    start_time = time.time()

    with (
        output_path.open("a", encoding="utf-8") as out_f,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("Extracting...", total=len(to_process))

        for entry in to_process:
            try:
                concerns = extract_article_concerns(entry, extractor)
                out_row = {
                    "id": entry["id"],
                    "source": entry.get("source", ""),
                    "concerns": concerns,
                }
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out_f.flush()
                stats["processed"] += 1
                stats["total_concerns"] += len(concerns)
            except Exception as e:
                stats["failed"] += 1
                console.print(f"[red]Error {entry['id']}: {e}")

            progress.advance(task)

    elapsed = time.time() - start_time
    console.print()
    console.print(f"[green]Done in {elapsed/60:.1f} min[/green]")
    console.print(f"  Processed  : {stats['processed']}")
    console.print(f"  Failed     : {stats['failed']}")
    console.print(f"  Concerns   : {stats['total_concerns']}")
    console.print(f"  Resumed    : {len(existing_ids)}")
    console.print(f"  Output     : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Analyze subcommand
# ═══════════════════════════════════════════════════════════════════════════════


def load_jsonl(path: Path) -> list[dict]:
    """Load all JSON lines from a file."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_concern_overlap(
    haiku_entries: list[dict],
    gpt_entries: list[dict],
    matcher: ConcernMatcher,
) -> dict:
    """M1: Bidirectional concern overlap using SPECTER2 matching."""
    gpt_by_id = {e["id"]: e for e in gpt_entries}

    # Forward: Haiku → GPT (what fraction of Haiku concerns match a GPT concern?)
    fwd_matched = 0
    fwd_total = 0

    # Reverse: GPT → Haiku (what fraction of GPT concerns match a Haiku concern?)
    rev_matched = 0
    rev_total = 0

    for haiku_entry in haiku_entries:
        art_id = haiku_entry["id"]
        gpt_entry = gpt_by_id.get(art_id)
        if gpt_entry is None:
            continue

        haiku_concerns = haiku_entry.get("concerns", [])
        gpt_concerns = gpt_entry.get("concerns", [])

        haiku_texts = [c["concern_text"] for c in haiku_concerns if not c.get("requires_figure_reading")]
        gpt_texts = [c["concern_text"] for c in gpt_concerns if not c.get("requires_figure_reading")]

        if not haiku_texts or not gpt_texts:
            continue

        # Forward: Haiku as "tool", GPT as "GT"
        gpt_as_gt = [{"concern_text": t, "category": "other", "severity": "major", "requires_figure_reading": False} for t in gpt_texts]
        fwd_result = matcher.score_article(haiku_texts, gpt_as_gt)
        fwd_matched += fwd_result.n_matched
        fwd_total += len(haiku_texts)

        # Reverse: GPT as "tool", Haiku as "GT"
        haiku_as_gt = [c for c in haiku_concerns if not c.get("requires_figure_reading")]
        rev_result = matcher.score_article(gpt_texts, haiku_as_gt)
        rev_matched += rev_result.n_matched
        rev_total += len(gpt_texts)

    return {
        "forward": {
            "matched": fwd_matched,
            "total_haiku": fwd_total,
            "frac": fwd_matched / fwd_total if fwd_total else 0.0,
        },
        "reverse": {
            "matched": rev_matched,
            "total_gpt": rev_total,
            "frac": rev_matched / rev_total if rev_total else 0.0,
        },
    }


def compute_category_agreement(
    haiku_entries: list[dict],
    gpt_entries: list[dict],
    matcher: ConcernMatcher,
) -> dict:
    """M2 & M3: Category and severity agreement on matched concern pairs."""
    gpt_by_id = {e["id"]: e for e in gpt_entries}

    haiku_cats: list[str] = []
    gpt_cats: list[str] = []
    haiku_sevs: list[str] = []
    gpt_sevs: list[str] = []

    for haiku_entry in haiku_entries:
        art_id = haiku_entry["id"]
        gpt_entry = gpt_by_id.get(art_id)
        if gpt_entry is None:
            continue

        haiku_concerns = [c for c in haiku_entry.get("concerns", []) if not c.get("requires_figure_reading")]
        gpt_concerns = [c for c in gpt_entry.get("concerns", []) if not c.get("requires_figure_reading")]

        if not haiku_concerns or not gpt_concerns:
            continue

        haiku_texts = [c["concern_text"] for c in haiku_concerns]

        # Extract matched pairs' categories and severities
        scores = matcher._compute_scores(haiku_texts, [c["concern_text"] for c in gpt_concerns])
        matches = matcher._match(scores)

        for m in matches:
            h_cat = haiku_concerns[m.tool_idx].get("category", "other")
            g_cat = gpt_concerns[m.gt_idx].get("category", "other")
            haiku_cats.append(h_cat)
            gpt_cats.append(g_cat)

            h_sev = haiku_concerns[m.tool_idx].get("severity", "minor")
            g_sev = gpt_concerns[m.gt_idx].get("severity", "minor")
            haiku_sevs.append(h_sev)
            gpt_sevs.append(g_sev)

    # Cohen's kappa
    from sklearn.metrics import cohen_kappa_score

    cat_kappa = cohen_kappa_score(haiku_cats, gpt_cats) if haiku_cats else 0.0
    sev_kappa = cohen_kappa_score(haiku_sevs, gpt_sevs) if haiku_sevs else 0.0

    # Confusion matrices
    cat_labels = sorted(set(haiku_cats) | set(gpt_cats))
    cat_confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in cat_labels} for c in cat_labels}
    for h, g in zip(haiku_cats, gpt_cats):
        cat_confusion[h][g] += 1

    sev_labels = sorted(set(haiku_sevs) | set(gpt_sevs))
    sev_confusion: dict[str, dict[str, int]] = {s: {s2: 0 for s2 in sev_labels} for s in sev_labels}
    for h, g in zip(haiku_sevs, gpt_sevs):
        sev_confusion[h][g] += 1

    # Per-category agreement rate
    cat_agreement: dict[str, float] = {}
    for cat in cat_labels:
        pairs_with_cat = sum(1 for h, g in zip(haiku_cats, gpt_cats) if h == cat or g == cat)
        agree = sum(1 for h, g in zip(haiku_cats, gpt_cats) if h == cat and g == cat)
        cat_agreement[cat] = agree / pairs_with_cat if pairs_with_cat else 0.0

    return {
        "category_agreement": {
            "kappa": round(cat_kappa, 4),
            "n_pairs": len(haiku_cats),
            "confusion_matrix": cat_confusion,
            "per_category_agreement": {k: round(v, 4) for k, v in cat_agreement.items()},
        },
        "severity_agreement": {
            "kappa": round(sev_kappa, 4),
            "confusion_matrix": sev_confusion,
        },
    }


def compute_volume_comparison(
    haiku_entries: list[dict],
    gpt_entries: list[dict],
) -> dict:
    """M4: Volume comparison (mean concerns per article, Wilcoxon test)."""
    from scipy.stats import wilcoxon

    gpt_by_id = {e["id"]: e for e in gpt_entries}

    haiku_counts: list[int] = []
    gpt_counts: list[int] = []
    per_source: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"haiku": [], "gpt": []})

    for haiku_entry in haiku_entries:
        art_id = haiku_entry["id"]
        gpt_entry = gpt_by_id.get(art_id)
        if gpt_entry is None:
            continue

        h_n = len([c for c in haiku_entry.get("concerns", []) if not c.get("requires_figure_reading")])
        g_n = len([c for c in gpt_entry.get("concerns", []) if not c.get("requires_figure_reading")])
        haiku_counts.append(h_n)
        gpt_counts.append(g_n)

        source = haiku_entry.get("source", "unknown")
        per_source[source]["haiku"].append(h_n)
        per_source[source]["gpt"].append(g_n)

    # Wilcoxon signed-rank test
    try:
        stat, p_value = wilcoxon(haiku_counts, gpt_counts)
    except ValueError:
        stat, p_value = 0.0, 1.0

    source_summary = {}
    for src, counts in per_source.items():
        source_summary[src] = {
            "haiku_mean": round(sum(counts["haiku"]) / len(counts["haiku"]), 2) if counts["haiku"] else 0,
            "gpt_mean": round(sum(counts["gpt"]) / len(counts["gpt"]), 2) if counts["gpt"] else 0,
            "n_articles": len(counts["haiku"]),
        }

    return {
        "haiku_mean": round(sum(haiku_counts) / len(haiku_counts), 2) if haiku_counts else 0,
        "gpt_mean": round(sum(gpt_counts) / len(gpt_counts), 2) if gpt_counts else 0,
        "n_articles": len(haiku_counts),
        "wilcoxon_statistic": round(float(stat), 2),
        "wilcoxon_p": round(float(p_value), 6),
        "per_source": source_summary,
    }


def compute_ranking_preservation(
    gpt_entries: list[dict],
    matcher: ConcernMatcher,
) -> dict:
    """M5: Re-evaluate all baselines against GPT GT; compute Kendall's tau."""
    from scipy.stats import kendalltau

    gpt_gt_f1: dict[str, float] = {}

    for model_name, model_info in MODELS.items():
        tool_path = _REPO_ROOT / model_info["tool_output"]
        if not tool_path.exists():
            console.print(f"[yellow]Warning: {tool_path} not found, skipping {model_name}")
            continue

        tool_rows = load_jsonl(tool_path)

        # Build tool_results and alt GT in the format expected by score_dataset
        tool_results = []
        for row in tool_rows:
            art_id = row.get("article_id", row.get("id", ""))
            raw_concerns = row.get("concerns", [])
            if raw_concerns and isinstance(raw_concerns[0], str):
                texts = raw_concerns
            else:
                texts = [c.get("text", c.get("concern_text", "")) for c in raw_concerns if isinstance(c, dict)]
            tool_results.append({"article_id": art_id, "concerns": texts})

        result = matcher.score_dataset(tool_results, gpt_entries)
        gpt_gt_f1[model_name] = round(result.f1, 4)
        console.print(f"  {model_name}: F1={result.f1:.4f} (GPT GT)")

    # Compute Kendall's tau
    common_models = sorted(set(HAIKU_GT_F1) & set(gpt_gt_f1))
    if len(common_models) < 2:
        return {
            "haiku_gt_f1": HAIKU_GT_F1,
            "gpt_gt_f1": gpt_gt_f1,
            "kendall_tau": 0.0,
            "kendall_p": 1.0,
            "n_rank_swaps": -1,
            "error": "Too few models for comparison",
        }

    haiku_f1s = [HAIKU_GT_F1[m] for m in common_models]
    gpt_f1s = [gpt_gt_f1[m] for m in common_models]

    # Rank by F1 (1 = best)
    haiku_rank = _rank_list(haiku_f1s)
    gpt_rank = _rank_list(gpt_f1s)

    tau, p_value = kendalltau(haiku_rank, gpt_rank)

    # Count pairwise rank swaps
    n_swaps = 0
    n = len(common_models)
    for i in range(n):
        for j in range(i + 1, n):
            haiku_order = haiku_f1s[i] > haiku_f1s[j]
            gpt_order = gpt_f1s[i] > gpt_f1s[j]
            if haiku_order != gpt_order:
                n_swaps += 1

    return {
        "haiku_gt_f1": {m: HAIKU_GT_F1[m] for m in common_models},
        "gpt_gt_f1": {m: gpt_gt_f1[m] for m in common_models},
        "haiku_gt_rank": {m: r for m, r in zip(common_models, haiku_rank)},
        "gpt_gt_rank": {m: r for m, r in zip(common_models, gpt_rank)},
        "kendall_tau": round(float(tau), 4),
        "kendall_p": round(float(p_value), 6),
        "n_rank_swaps": n_swaps,
        "n_models": len(common_models),
    }


def _rank_list(values: list[float]) -> list[float]:
    """Rank values in descending order (1 = highest), with average rank for ties."""
    indexed = sorted(enumerate(values), key=lambda x: -x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        # Find group of tied values
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        # Assign average rank to all tied items
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


@click.command("analyze")
@click.option(
    "--haiku-split",
    required=True,
    type=click.Path(exists=True),
    help="Path to test.jsonl (Haiku GT)",
)
@click.option(
    "--alt-gt",
    required=True,
    type=click.Path(exists=True),
    help="Path to cross_model_gt.jsonl (alternative GT)",
)
@click.option("--output", "-o", required=True, help="Output JSON path for agreement metrics")
@click.option("--threshold", default=0.65, help="SPECTER2 matching threshold")
def cmd_analyze(haiku_split: str, alt_gt: str, output: str, threshold: float) -> None:
    """Compute inter-extractor agreement metrics."""
    haiku_path = Path(haiku_split)
    alt_path = Path(alt_gt)
    output_path = Path(output)

    console.print("[bold cyan]Cross-Model Agreement Analysis[/bold cyan]")
    console.print(f"  Haiku GT : {haiku_path}")
    console.print(f"  Alt GT   : {alt_path}")
    console.print(f"  Output   : {output_path}")
    console.print(f"  Threshold: {threshold}")
    console.print()

    haiku_entries = load_jsonl(haiku_path)
    gpt_entries = load_jsonl(alt_path)

    gpt_ids = {e["id"] for e in gpt_entries}
    n_common = sum(1 for e in haiku_entries if e["id"] in gpt_ids)

    console.print(f"  Haiku articles: {len(haiku_entries)}")
    console.print(f"  GPT articles  : {len(gpt_entries)}")
    console.print(f"  Common        : {n_common}")
    console.print()

    matcher = ConcernMatcher(threshold=threshold)

    # M1: Bidirectional concern overlap
    console.print("[bold]M1: Computing bidirectional concern overlap...[/bold]")
    overlap = compute_concern_overlap(haiku_entries, gpt_entries, matcher)
    console.print(f"  Forward (Haiku→GPT): {overlap['forward']['frac']:.3f} "
                  f"({overlap['forward']['matched']}/{overlap['forward']['total_haiku']})")
    console.print(f"  Reverse (GPT→Haiku): {overlap['reverse']['frac']:.3f} "
                  f"({overlap['reverse']['matched']}/{overlap['reverse']['total_gpt']})")
    console.print()

    # M2 & M3: Category and severity agreement
    console.print("[bold]M2/M3: Computing category & severity agreement...[/bold]")
    agreement = compute_category_agreement(haiku_entries, gpt_entries, matcher)
    console.print(f"  Category kappa: {agreement['category_agreement']['kappa']:.3f} "
                  f"({agreement['category_agreement']['n_pairs']} pairs)")
    console.print(f"  Severity kappa: {agreement['severity_agreement']['kappa']:.3f}")
    console.print()

    # M4: Volume comparison
    console.print("[bold]M4: Computing volume comparison...[/bold]")
    volume = compute_volume_comparison(haiku_entries, gpt_entries)
    console.print(f"  Haiku mean: {volume['haiku_mean']:.1f} concerns/article")
    console.print(f"  GPT mean  : {volume['gpt_mean']:.1f} concerns/article")
    console.print(f"  Wilcoxon p: {volume['wilcoxon_p']:.6f}")
    console.print()

    # M5: Ranking preservation
    console.print("[bold]M5: Re-evaluating baselines against GPT GT...[/bold]")
    ranking = compute_ranking_preservation(gpt_entries, matcher)
    console.print(f"  Kendall's tau: {ranking['kendall_tau']:.3f} (p={ranking['kendall_p']:.4f})")
    console.print(f"  Rank swaps: {ranking['n_rank_swaps']}/{ranking['n_models'] * (ranking['n_models'] - 1) // 2}")
    console.print()

    # Assemble output
    result = {
        "extractor_models": {
            "primary": "claude-haiku-4-5-20251001",
            "secondary": "gpt-4o-mini",
        },
        "n_articles_haiku": len(haiku_entries),
        "n_articles_gpt": len(gpt_entries),
        "n_articles_common": n_common,
        "threshold": threshold,
        "concern_overlap": overlap,
        "category_agreement": agreement["category_agreement"],
        "severity_agreement": agreement["severity_agreement"],
        "volume": volume,
        "ranking": ranking,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Results saved to {output_path}[/green]")

    # Summary interpretation
    console.print()
    tau = ranking["kendall_tau"]
    cat_k = agreement["category_agreement"]["kappa"]
    fwd = overlap["forward"]["frac"]
    rev = overlap["reverse"]["frac"]

    console.print("[bold]Summary:[/bold]")
    console.print(f"  Concern overlap: {fwd:.1%} (Haiku→GPT), {rev:.1%} (GPT→Haiku)")
    console.print(f"  Category κ = {cat_k:.3f}", end="")
    if cat_k >= 0.8:
        console.print(" [green](almost perfect)[/green]")
    elif cat_k >= 0.6:
        console.print(" [green](substantial)[/green]")
    elif cat_k >= 0.4:
        console.print(" [yellow](moderate)[/yellow]")
    else:
        console.print(" [red](fair or poor)[/red]")

    console.print(f"  Kendall's τ = {tau:.3f}", end="")
    if tau >= 0.87:
        console.print(" [green](≤1 rank swap — robust)[/green]")
    elif tau >= 0.6:
        console.print(" [yellow](2-3 rank swaps — partially robust)[/yellow]")
    else:
        console.print(" [red](≥3 rank swaps — concerning)[/red]")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI group
# ═══════════════════════════════════════════════════════════════════════════════


@click.group()
def cli():
    """Cross-model ground truth validation for bioreview-bench."""
    pass


cli.add_command(cmd_extract)
cli.add_command(cmd_analyze)


if __name__ == "__main__":
    cli()

"""Reproduce a narrow measurement audit for the frozen v4 test set.

The audit is intentionally read-only with respect to benchmark inputs. It:

1. detects F1000Research DOI-stem families crossing train, validation, and test;
2. reconstructs the frozen BM25 baseline for F1000 cross-split-family queries
   with its published defaults;
3. compares frozen current BM25 predictions with query-time candidate-list
   filtering of training versions from the query manuscript family, while
   holding the original BM25 index, IDF, and average document length fixed;
4. evaluates all frozen tool outputs with token Jaccard similarity and
   threshold-aware one-to-one Hungarian assignment across a deterministic
   operating-point sweep; and
5. compares those lexical rankings with the frozen embedding leaderboard.

The public Git repository does not redistribute the benchmark JSONL files or
tool outputs. Point the CLI at an authorized local copy when the defaults are
not present.

Example:
    python scripts/audit_v4_measurement.py \
      --train /path/to/data/splits/v4/train.jsonl \
      --validation /path/to/data/splits/v4/val.jsonl \
      --test /path/to/data/splits/v4/test.jsonl \
      --tool-output-dir /path/to/tool_outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from bioreview_bench.baseline.lexical import (
    BM25ConcernRetriever,
    _article_text,
    _normalize_concern,
    _tokenize,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

BM25_DEFAULTS: dict[str, int | float] = {
    "top_k_docs": 8,
    "max_concerns": 12,
    "max_input_chars": 40_000,
    "k1": 1.5,
    "b": 0.75,
}

CONFIGURED_MATCHER_THRESHOLD = 0.65
JACCARD_THRESHOLD_SCALE = 0.3
DEFAULT_JACCARD_THRESHOLD = CONFIGURED_MATCHER_THRESHOLD * JACCARD_THRESHOLD_SCALE
DEFAULT_JACCARD_SWEEP_THRESHOLDS: tuple[float, ...] = (
    0.05,
    0.075,
    0.1,
    0.125,
    0.15,
    0.175,
    0.195,
    0.2,
    0.225,
    0.25,
    0.275,
    0.3,
)

MODEL_OUTPUT_FILES: dict[str, str] = {
    "Haiku-4.5": "haiku_test_v4.jsonl",
    "Gemini-2.5-Flash": "gemini25flash_test_v4.jsonl",
    "GPT-4o-mini": "gpt4omini_test_v4.jsonl",
    "BM25": "bm25_test_v4.jsonl",
    "Gemini-Flash-Lite": "gemini_flash_lite_test_v4.jsonl",
    "Llama-3.3-70B": "llama33_test_v4.jsonl",
}

_F1000_FAMILY_RE = re.compile(
    r"^(10\.12688/f1000research\.[^.]+)\.(\d+)$",
    re.IGNORECASE,
)
_JACCARD_TOKEN_RE = re.compile(r"\b[a-z0-9]{3,}\b")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file without changing it."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f1000_manuscript_family(entry: dict[str, Any]) -> str | None:
    """Return the version-independent F1000 DOI family for an entry.

    F1000 article versions use DOI suffixes such as ``...12345.1`` and
    ``...12345.2``. Only that source-specific terminal version is removed.
    Other sources return ``None`` because this audit does not claim a general
    cross-publisher family resolver.
    """
    if str(entry.get("source", "")).lower() != "f1000":
        return None
    doi = str(entry.get("doi", "")).strip().lower()
    match = _F1000_FAMILY_RE.fullmatch(doi)
    return match.group(1) if match else None


def _tool_concern_texts(row: dict[str, Any]) -> list[str]:
    raw = row.get("concerns", [])
    if not isinstance(raw, list):
        return []
    texts: list[str] = []
    for concern in raw:
        if isinstance(concern, str):
            text = concern
        elif isinstance(concern, dict):
            text = str(concern.get("text", concern.get("concern_text", "")))
        else:
            continue
        if text.strip():
            texts.append(text.strip())
    return texts


def load_tool_map(path: Path) -> dict[str, list[str]]:
    tool_map: dict[str, list[str]] = {}
    for row in load_jsonl(path):
        article_id = str(row.get("article_id", row.get("id", "")))
        if not article_id:
            continue
        if article_id in tool_map:
            raise ValueError(f"{path}: duplicate tool-output article ID {article_id!r}")
        tool_map[article_id] = _tool_concern_texts(row)
    return tool_map


def validate_tool_output_coverage(
    tool_maps: dict[str, dict[str, list[str]]],
    test_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless every frozen output exactly covers the test ID set."""
    test_ids = [str(entry.get("id", "")) for entry in test_entries]
    if any(not article_id for article_id in test_ids):
        raise ValueError("Test data contains an empty article ID")
    expected_ids = set(test_ids)
    if len(expected_ids) != len(test_ids):
        raise ValueError("Test data contains duplicate article IDs")

    model_summaries: dict[str, dict[str, int | bool]] = {}
    coverage_errors: list[str] = []
    for model, predictions in tool_maps.items():
        actual_ids = set(predictions)
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids
        exact = not missing_ids and not extra_ids
        model_summaries[model] = {
            "n_expected_test_ids": len(expected_ids),
            "n_output_rows": len(predictions),
            "n_missing_test_ids": len(missing_ids),
            "n_extra_article_ids": len(extra_ids),
            "exact_test_id_set": exact,
        }
        if not exact:
            coverage_errors.append(
                f"{model}: missing={len(missing_ids)}, extra={len(extra_ids)}"
            )

    if coverage_errors:
        raise ValueError(
            "Frozen tool-output article IDs must exactly match the test set; "
            + "; ".join(coverage_errors)
        )
    return {
        "n_expected_test_ids": len(expected_ids),
        "n_models": len(tool_maps),
        "all_models_exact_test_id_set": True,
        "models": model_summaries,
    }


def _jaccard_tokens(text: str) -> frozenset[str]:
    return frozenset(_JACCARD_TOKEN_RE.findall(text.lower()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def eligible_hungarian_matches(
    similarities: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Return threshold-aware one-to-one matches.

    Assignments minimize the number of below-threshold edges first because
    those edges receive a large fixed penalty. Similarity breaks ties among
    assignments with the same eligible cardinality.
    """
    if similarities.size == 0:
        return []
    cost = 1.0 - similarities
    cost = cost.copy()
    cost[similarities < threshold] = 1e6
    row_indices, column_indices = linear_sum_assignment(cost)
    return [
        (int(row), int(column), float(similarities[row, column]))
        for row, column in zip(row_indices, column_indices)
        if similarities[row, column] >= threshold
    ]


def evaluate_jaccard_hungarian(
    predictions: dict[str, list[str]],
    entries: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, int | float]:
    """Compute micro metrics with one-to-one Jaccard/Hungarian matching."""
    token_cache: dict[str, frozenset[str]] = {}

    def tokens(text: str) -> frozenset[str]:
        if text not in token_cache:
            token_cache[text] = _jaccard_tokens(text)
        return token_cache[text]

    n_gt = 0
    n_predictions = 0
    n_matched = 0
    for entry in entries:
        article_id = str(entry.get("id", ""))
        predicted = predictions.get(article_id, [])
        gt_texts = [
            str(concern.get("concern_text", "")).strip()
            for concern in entry.get("concerns", [])
            if str(concern.get("concern_text", "")).strip()
            and not concern.get("requires_figure_reading", False)
        ]
        n_gt += len(gt_texts)
        n_predictions += len(predicted)
        if not predicted or not gt_texts:
            continue
        similarities = np.array(
            [
                [_jaccard(tokens(prediction), tokens(gt_text)) for gt_text in gt_texts]
                for prediction in predicted
            ],
            dtype=np.float64,
        )
        n_matched += len(eligible_hungarian_matches(similarities, threshold))

    recall = n_matched / n_gt if n_gt else 0.0
    precision = n_matched / n_predictions if n_predictions else 0.0
    f1 = (
        2 * recall * precision / (recall + precision)
        if recall + precision
        else 0.0
    )
    return {
        "n_articles": len(entries),
        "n_gt": n_gt,
        "n_predictions": n_predictions,
        "n_matched": n_matched,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
    }


def kendall_tau_a(rank_a: list[str], rank_b: list[str]) -> float:
    """Compute Kendall tau-a for two complete rankings without ties."""
    if set(rank_a) != set(rank_b):
        raise ValueError("Kendall rankings must contain the same model names")
    concordant = 0
    discordant = 0
    for index, left in enumerate(rank_a):
        for right in rank_a[index + 1 :]:
            if rank_b.index(left) < rank_b.index(right):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def normalize_jaccard_thresholds(
    thresholds: Iterable[float],
) -> tuple[float, ...]:
    """Validate, deduplicate, and sort Jaccard operating points."""
    normalized: set[float] = set()
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Jaccard sweep thresholds must be finite values in [0, 1]")
        normalized.add(round(threshold, 12))
    if not normalized:
        raise ValueError("At least one Jaccard sweep threshold is required")
    return tuple(sorted(normalized))


def build_jaccard_threshold_sweep(
    *,
    tool_maps: dict[str, dict[str, list[str]]],
    subsets: dict[str, list[dict[str, Any]]],
    embedding_rankings: dict[str, list[dict[str, Any]]],
    thresholds: Iterable[float],
) -> dict[str, Any]:
    """Return deterministic aggregate, text-free matcher-sensitivity results."""
    operating_points = normalize_jaccard_thresholds(thresholds)
    if set(subsets) != set(embedding_rankings):
        raise ValueError("Sweep subsets and embedding rankings must have identical names")
    if not tool_maps:
        raise ValueError("At least one tool output is required for a threshold sweep")

    subset_order = [
        name
        for name in ("full_600", "no_elife_450")
        if name in subsets
    ]
    subset_order.extend(sorted(set(subsets) - set(subset_order)))
    rendered_subsets: dict[str, list[dict[str, Any]]] = {}
    for subset_name in subset_order:
        entries = subsets[subset_name]
        embedding_order = [
            str(row["model"]) for row in embedding_rankings[subset_name]
        ]
        if (
            len(embedding_order) != len(set(embedding_order))
            or set(embedding_order) != set(tool_maps)
        ):
            raise ValueError(
                f"{subset_name}: embedding ranking must contain every tool exactly once"
            )
        points: list[dict[str, Any]] = []
        for threshold in operating_points:
            metrics_by_model = {
                model: evaluate_jaccard_hungarian(
                    predictions,
                    entries,
                    threshold=threshold,
                )
                for model, predictions in sorted(tool_maps.items())
            }
            ranking = _rank_metrics(metrics_by_model)
            jaccard_order = [str(row["model"]) for row in ranking]
            points.append(
                {
                    "threshold": threshold,
                    "aggregate_metrics_by_model": metrics_by_model,
                    "ranking": ranking,
                    "top_system": ranking[0]["model"] if ranking else None,
                    "total_matches_across_models": sum(
                        int(metrics["n_matched"])
                        for metrics in metrics_by_model.values()
                    ),
                    "kendall_tau_a_vs_frozen_embedding": round(
                        kendall_tau_a(embedding_order, jaccard_order),
                        6,
                    ),
                }
            )
        rendered_subsets[subset_name] = points

    return {
        "status": "aggregate_text_free_sensitivity_analysis",
        "operating_point_status": (
            "uncalibrated; no threshold in this sweep is a validated semantic "
            "equivalence cutoff"
        ),
        "ranking_tie_breaker": "descending F1, then ascending model name",
        "thresholds": list(operating_points),
        "subsets": rendered_subsets,
    }


def _f1000_family_rows(
    entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        family = f1000_manuscript_family(entry)
        if family:
            rows[family].append(entry)
    return rows


def summarize_f1000_split_family_matrix(
    train_entries: list[dict[str, Any]],
    validation_entries: list[dict[str, Any]],
    test_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize F1000 DOI-stem overlap across all frozen partitions.

    This is deliberately an aggregate lower bound. It does not emit DOI stems
    or article identifiers and cannot detect cross-source or fuzzy-title
    manuscript relations.
    """
    rows_by_split = {
        "train": _f1000_family_rows(train_entries),
        "validation": _f1000_family_rows(validation_entries),
        "test": _f1000_family_rows(test_entries),
    }
    entries_by_split = {
        "train": train_entries,
        "validation": validation_entries,
        "test": test_entries,
    }

    def pair_summary(left: str, right: str) -> dict[str, int]:
        crossing = set(rows_by_split[left]) & set(rows_by_split[right])
        return {
            "n_crossing_families": len(crossing),
            "n_left_articles_in_crossing_families": sum(
                len(rows_by_split[left][family]) for family in crossing
            ),
            "n_right_articles_in_crossing_families": sum(
                len(rows_by_split[right][family]) for family in crossing
            ),
        }

    family_presence: defaultdict[str, set[str]] = defaultdict(set)
    for split, family_rows in rows_by_split.items():
        for family in family_rows:
            family_presence[family].add(split)
    crossing_any = {
        family for family, splits in family_presence.items() if len(splits) >= 2
    }

    development_families = set(rows_by_split["train"]) | set(
        rows_by_split["validation"]
    )
    development_test_families = development_families & set(rows_by_split["test"])
    development_test_articles = sum(
        len(rows_by_split["test"][family])
        for family in development_test_families
    )
    n_f1000_test = sum(
        1
        for entry in test_entries
        if str(entry.get("source", "")).lower() == "f1000"
    )

    return {
        "scope": "F1000Research terminal numeric DOI-version suffixes only",
        "interpretation": (
            "aggregate text-free lower bound; no fuzzy-title, preprint-to-journal, "
            "or cross-source family resolution"
        ),
        "is_lower_bound": True,
        "n_f1000_articles_by_split": {
            split: sum(
                1
                for entry in entries
                if str(entry.get("source", "")).lower() == "f1000"
            )
            for split, entries in entries_by_split.items()
        },
        "n_articles_with_parseable_doi_family_by_split": {
            split: sum(len(rows) for rows in family_rows.values())
            for split, family_rows in rows_by_split.items()
        },
        "n_f1000_articles_without_parseable_doi_family_by_split": {
            split: sum(
                1
                for entry in entries_by_split[split]
                if str(entry.get("source", "")).lower() == "f1000"
            )
            - sum(len(rows) for rows in family_rows.values())
            for split, family_rows in rows_by_split.items()
        },
        "pairwise": {
            "train_validation": pair_summary("train", "validation"),
            "train_test": pair_summary("train", "test"),
            "validation_test": pair_summary("validation", "test"),
        },
        "development_vs_test": {
            "development_splits": ["train", "validation"],
            "n_crossing_families": len(development_test_families),
            "n_test_articles_in_crossing_families": development_test_articles,
            "n_f1000_test_articles": n_f1000_test,
            "test_article_overlap_fraction": round(
                development_test_articles / n_f1000_test
                if n_f1000_test
                else 0.0,
                6,
            ),
        },
        "n_unique_families_crossing_any_split_boundary": len(crossing_any),
        "within_split_multiversion_families": {
            split: sum(1 for rows in family_rows.values() if len(rows) > 1)
            for split, family_rows in rows_by_split.items()
        },
    }


def detect_f1000_family_overlap(
    train_entries: list[dict[str, Any]],
    test_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    """Summarize version-independent F1000 families crossing the split."""
    train_family_rows = _f1000_family_rows(train_entries)
    test_family_rows = _f1000_family_rows(test_entries)

    overlapping_families = set(train_family_rows) & set(test_family_rows)
    overlapping_test_ids = {
        str(entry.get("id", ""))
        for family in overlapping_families
        for entry in test_family_rows[family]
    }
    n_f1000_test = sum(
        1 for entry in test_entries if str(entry.get("source", "")).lower() == "f1000"
    )
    n_overlap_test = len(overlapping_test_ids)
    summary = {
        "scope": "F1000Research terminal numeric DOI-version suffixes only",
        "is_lower_bound": True,
        "family_definition": (
            "Lower-cased F1000 DOI with the terminal numeric article-version "
            "suffix removed; no fuzzy title matching."
        ),
        "n_f1000_train_articles": sum(len(rows) for rows in train_family_rows.values()),
        "n_f1000_test_articles": n_f1000_test,
        "n_overlapping_families": len(overlapping_families),
        "n_test_articles_in_overlapping_families": n_overlap_test,
        "test_overlap_fraction": round(
            n_overlap_test / n_f1000_test if n_f1000_test else 0.0,
            6,
        ),
        "n_train_articles_in_overlapping_families": sum(
            len(train_family_rows[family]) for family in overlapping_families
        ),
    }
    return summary, overlapping_test_ids


def _concerns_from_ranked_docs(
    retriever: BM25ConcernRetriever,
    ranked_scores: list[tuple[int, float]],
    *,
    article_id: str,
    excluded_family: str | None,
    document_families: list[str | None],
) -> list[str]:
    concern_scores: defaultdict[str, float] = defaultdict(float)
    accepted_rank = 0
    for doc_index, score in ranked_scores:
        if score <= 0:
            continue
        if retriever._doc_ids[doc_index] == article_id:
            continue
        if excluded_family and document_families[doc_index] == excluded_family:
            continue
        accepted_rank += 1
        weight = score / accepted_rank
        for concern in retriever._doc_concerns[doc_index]:
            normalized = _normalize_concern(concern)
            if normalized:
                concern_scores[normalized] += weight
        if accepted_rank >= retriever.top_k_docs:
            break
    ranked_concerns = sorted(
        concern_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [concern for concern, _score in ranked_concerns[: retriever.max_concerns]]


def reconstruct_bm25_and_family_exclusion(
    train_entries: list[dict[str, Any]],
    test_entries: list[dict[str, Any]],
    frozen_predictions: dict[str, list[str]],
    *,
    target_article_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[str]]]:
    """Reconstruct target rows and filter same-family candidates at query time.

    Scoring uses the original index statistics. The intervention removes
    same-family rows only from the ranked candidate list; it does not rebuild
    the corpus, postings, IDF, or average document length.
    """
    retriever = BM25ConcernRetriever(
        train_entries,
        top_k_docs=int(BM25_DEFAULTS["top_k_docs"]),
        max_concerns=int(BM25_DEFAULTS["max_concerns"]),
        max_input_chars=int(BM25_DEFAULTS["max_input_chars"]),
        k1=float(BM25_DEFAULTS["k1"]),
        b=float(BM25_DEFAULTS["b"]),
    )
    train_by_id = {
        str(entry.get("id", entry.get("article_id", ""))): entry
        for entry in train_entries
    }
    document_families = [
        f1000_manuscript_family(train_by_id.get(document_id, {}))
        for document_id in retriever._doc_ids
    ]

    all_test_ids = {str(entry.get("id", "")) for entry in test_entries}
    target_ids = all_test_ids if target_article_ids is None else target_article_ids
    unknown_target_ids = target_ids - all_test_ids
    if unknown_target_ids:
        raise ValueError(
            f"Reconstruction targets include {len(unknown_target_ids)} unknown test IDs"
        )
    target_entries = [
        entry
        for entry in test_entries
        if str(entry.get("id", "")) in target_ids
    ]

    reconstructed: dict[str, list[str]] = {}
    current_predictions = {
        article_id: list(concerns)
        for article_id, concerns in frozen_predictions.items()
    }
    family_excluded = {
        article_id: list(concerns)
        for article_id, concerns in frozen_predictions.items()
    }
    first_family_ranks: list[int] = []
    n_same_family_rank_1 = 0
    n_same_family_top_8 = 0
    n_rows_changed = 0

    for entry in target_entries:
        article_id = str(entry.get("id", ""))
        query_tokens = _tokenize(_article_text(entry, retriever.max_input_chars))
        if not query_tokens:
            reconstructed[article_id] = []
            family_excluded[article_id] = []
            continue
        query_terms = tuple(dict.fromkeys(query_tokens))
        ranked_scores = retriever._score(query_terms)
        family = f1000_manuscript_family(entry)

        current = _concerns_from_ranked_docs(
            retriever,
            ranked_scores[: retriever.top_k_docs],
            article_id=article_id,
            excluded_family=None,
            document_families=document_families,
        )
        reconstructed[article_id] = current

        if family and family in document_families:
            eligible_rank = 0
            first_family_rank: int | None = None
            for doc_index, score in ranked_scores:
                if score <= 0 or retriever._doc_ids[doc_index] == article_id:
                    continue
                eligible_rank += 1
                if document_families[doc_index] == family:
                    first_family_rank = eligible_rank
                    break
            if first_family_rank is not None:
                first_family_ranks.append(first_family_rank)
                if first_family_rank == 1:
                    n_same_family_rank_1 += 1
                if first_family_rank <= retriever.top_k_docs:
                    n_same_family_top_8 += 1

            excluded = _concerns_from_ranked_docs(
                retriever,
                ranked_scores,
                article_id=article_id,
                excluded_family=family,
                document_families=document_families,
            )
        else:
            excluded = current
        family_excluded[article_id] = excluded
        if excluded != current:
            n_rows_changed += 1

    exact_rows = sum(
        reconstructed.get(str(entry.get("id", "")), [])
        == frozen_predictions.get(str(entry.get("id", "")), [])
        for entry in target_entries
    )
    n_targets = len(target_entries)
    n_family_ranked = len(first_family_ranks)
    summary = {
        "defaults": BM25_DEFAULTS,
        "reconstruction_scope": (
            "targeted_f1000_cross_split_family_queries"
            if target_ids != all_test_ids
            else "all_test_queries"
        ),
        "family_intervention": (
            "Query-time candidate-list filtering after BM25 scoring; the "
            "original corpus, postings, IDF, and average document length are fixed."
        ),
        "n_frozen_output_rows": len(frozen_predictions),
        "n_reconstruction_target_rows": n_targets,
        "n_exact_prediction_rows": exact_rows,
        "all_target_prediction_rows_exact": exact_rows == n_targets,
        "n_rows_changed_after_query_time_family_filter": n_rows_changed,
        "same_family_retrieval": {
            "n_queries_with_ranked_same_family_train_article": n_family_ranked,
            "same_family_at_rank_1": n_same_family_rank_1,
            "same_family_at_rank_1_fraction": round(
                n_same_family_rank_1 / n_family_ranked if n_family_ranked else 0.0,
                6,
            ),
            "same_family_within_top_8": n_same_family_top_8,
            "same_family_within_top_8_fraction": round(
                n_same_family_top_8 / n_family_ranked if n_family_ranked else 0.0,
                6,
            ),
            "median_first_same_family_rank": (
                float(median(first_family_ranks)) if first_family_ranks else None
            ),
            "first_same_family_rank_histogram": {
                str(rank): count
                for rank, count in sorted(Counter(first_family_ranks).items())
            },
        },
    }
    return summary, current_predictions, family_excluded


def require_exact_bm25_reconstruction(summary: dict[str, Any]) -> None:
    """Fail closed unless every targeted BM25 row matches the frozen output."""
    n_targets = int(summary["n_reconstruction_target_rows"])
    n_exact = int(summary["n_exact_prediction_rows"])
    if not summary["all_target_prediction_rows_exact"] or n_exact != n_targets:
        raise ValueError(
            "Targeted BM25 reconstruction does not match every frozen row: "
            f"{n_exact}/{n_targets} exact"
        )


def _rank_metrics(metrics_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    names = sorted(
        metrics_by_model,
        key=lambda name: (-float(metrics_by_model[name]["f1"]), name),
    )
    return [
        {
            "rank": rank,
            "model": name,
            "f1": metrics_by_model[name]["f1"],
            "recall": metrics_by_model[name]["recall"],
            "precision": metrics_by_model[name]["precision"],
            "n_matched": metrics_by_model[name]["n_matched"],
            "n_gt": metrics_by_model[name]["n_gt"],
            "n_predictions": metrics_by_model[name]["n_predictions"],
        }
        for rank, name in enumerate(names, start=1)
    ]


def _frozen_embedding_ranking(leaderboard_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{leaderboard_path}: expected a JSON array")
    rows = sorted(raw, key=lambda row: int(row["rank"]))
    return [
        {
            "rank": index,
            "model": str(row["tool_name"]),
            "f1": round(float(row["f1"]), 6),
            "recall": round(float(row["recall"]), 6),
            "precision": round(float(row["precision"]), 6),
        }
        for index, row in enumerate(rows, start=1)
    ]


def _frozen_no_elife_embedding_ranking(
    aggregate_analysis_path: Path,
) -> list[dict[str, Any]]:
    raw = json.loads(aggregate_analysis_path.read_text(encoding="utf-8"))
    try:
        subset = raw["source_robustness"]["subsets"]["no_elife_450"]
        ranking = subset["ranking"]
        metrics = subset["metrics"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{aggregate_analysis_path}: missing "
            "source_robustness.subsets.no_elife_450 ranking/metrics"
        ) from exc
    if not isinstance(ranking, list) or not isinstance(metrics, dict):
        raise ValueError(
            f"{aggregate_analysis_path}: malformed no_elife_450 ranking/metrics"
        )
    return [
        {
            "rank": index,
            "model": str(model),
            "f1": round(float(metrics[model]["f1"]), 6),
            "recall": round(float(metrics[model]["recall"]), 6),
            "precision": round(float(metrics[model]["precision"]), 6),
        }
        for index, model in enumerate(ranking, start=1)
    ]


def build_measurement_audit(
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    tool_output_dir: Path,
    leaderboard_path: Path,
    aggregate_analysis_path: Path,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    jaccard_sweep_thresholds: Iterable[float] = DEFAULT_JACCARD_SWEEP_THRESHOLDS,
) -> dict[str, Any]:
    """Build the complete deterministic audit payload."""
    train_entries = load_jsonl(train_path)
    validation_entries = load_jsonl(validation_path)
    test_entries = load_jsonl(test_path)
    tool_paths = {
        model: tool_output_dir / filename
        for model, filename in MODEL_OUTPUT_FILES.items()
    }
    missing = [str(path) for path in tool_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen tool outputs required for the audit: " + ", ".join(missing)
        )
    tool_maps = {
        model: load_tool_map(path)
        for model, path in tool_paths.items()
    }
    tool_output_coverage = validate_tool_output_coverage(tool_maps, test_entries)

    family_overlap, overlapping_test_ids = detect_f1000_family_overlap(
        train_entries,
        test_entries,
    )
    family_matrix = summarize_f1000_split_family_matrix(
        train_entries,
        validation_entries,
        test_entries,
    )
    bm25_reconstruction, current_bm25, excluded_bm25 = (
        reconstruct_bm25_and_family_exclusion(
            train_entries,
            test_entries,
            tool_maps["BM25"],
            target_article_ids=overlapping_test_ids,
        )
    )
    require_exact_bm25_reconstruction(bm25_reconstruction)

    f1000_entries = [
        entry for entry in test_entries if str(entry.get("source", "")).lower() == "f1000"
    ]
    no_elife_entries = [
        entry for entry in test_entries if str(entry.get("source", "")).lower() != "elife"
    ]
    overlapping_entries = [
        entry for entry in test_entries if str(entry.get("id", "")) in overlapping_test_ids
    ]
    family_exclusion_metrics: dict[str, Any] = {}
    for name, subset in [
        ("full_test", test_entries),
        ("no_elife_450", no_elife_entries),
        ("f1000_test", f1000_entries),
        ("f1000_overlapping_families", overlapping_entries),
    ]:
        current = evaluate_jaccard_hungarian(
            current_bm25,
            subset,
            threshold=jaccard_threshold,
        )
        excluded = evaluate_jaccard_hungarian(
            excluded_bm25,
            subset,
            threshold=jaccard_threshold,
        )
        family_exclusion_metrics[name] = {
            "current": current,
            "family_excluded": excluded,
            "delta_family_excluded_minus_current": {
                "recall": round(
                    float(excluded["recall"]) - float(current["recall"]),
                    6,
                ),
                "precision": round(
                    float(excluded["precision"]) - float(current["precision"]),
                    6,
                ),
                "f1": round(float(excluded["f1"]) - float(current["f1"]), 6),
                "n_matched": int(excluded["n_matched"]) - int(current["n_matched"]),
            },
        }

    subsets = {
        "full_600": test_entries,
        "no_elife_450": no_elife_entries,
    }
    embedding_rankings = {
        "full_600": _frozen_embedding_ranking(leaderboard_path),
        "no_elife_450": _frozen_no_elife_embedding_ranking(
            aggregate_analysis_path
        ),
    }
    sweep_thresholds = normalize_jaccard_thresholds(
        (*jaccard_sweep_thresholds, jaccard_threshold)
    )
    threshold_sweep = build_jaccard_threshold_sweep(
        tool_maps=tool_maps,
        subsets=subsets,
        embedding_rankings=embedding_rankings,
        thresholds=sweep_thresholds,
    )

    ranking_comparison: dict[str, Any] = {}
    for subset_name in ("full_600", "no_elife_450"):
        subset = subsets[subset_name]
        embedding_ranking = embedding_rankings[subset_name]
        jaccard_metrics = {
            model: evaluate_jaccard_hungarian(
                predictions,
                subset,
                threshold=jaccard_threshold,
            )
            for model, predictions in tool_maps.items()
        }
        jaccard_ranking = _rank_metrics(jaccard_metrics)
        embedding_order = [str(row["model"]) for row in embedding_ranking]
        jaccard_order = [str(row["model"]) for row in jaccard_ranking]
        ranking_comparison[subset_name] = {
            "frozen_embedding_ranking": embedding_ranking,
            "one_to_one_jaccard_ranking": jaccard_ranking,
            "kendall_tau_a": round(
                kendall_tau_a(embedding_order, jaccard_order),
                6,
            ),
        }

    return {
        "audit_version": "1.1",
        "scope": (
            "Frozen v4 full-600 primary diagnostic and post-hoc no-eLife-450 "
            "sensitivity subset; aggregate computational audit only."
        ),
        "input_fingerprints": {
            "train_jsonl_sha256": sha256_file(train_path),
            "validation_jsonl_sha256": sha256_file(validation_path),
            "test_jsonl_sha256": sha256_file(test_path),
            "leaderboard_json_sha256": sha256_file(leaderboard_path),
            "aggregate_analysis_json_sha256": sha256_file(aggregate_analysis_path),
            "tool_output_sha256": {
                model: sha256_file(path)
                for model, path in tool_paths.items()
            },
        },
        "protocol": {
            "configured_matcher_threshold": CONFIGURED_MATCHER_THRESHOLD,
            "jaccard_threshold_scale": JACCARD_THRESHOLD_SCALE,
            "jaccard_threshold": jaccard_threshold,
            "jaccard_threshold_derivation": (
                "configured matcher threshold 0.65 multiplied by the frozen "
                "Jaccard scale 0.3"
            ),
            "matcher_operating_point_status": (
                "uncalibrated; neither the historical embedding cutoff nor any "
                "Jaccard cutoff is human-validated"
            ),
            "jaccard_sweep_thresholds": list(sweep_thresholds),
            "tokenization": "lower-case alphanumeric tokens of length >= 3; set Jaccard",
            "assignment": (
                "threshold-aware one-to-one Hungarian; maximize eligible pair "
                "cardinality, then similarity"
            ),
            "figure_policy": "exclude GT concerns with requires_figure_reading=true",
            "averaging": "micro",
            "bm25_family_intervention": (
                "Query-time candidate-list filtering after scoring; original "
                "corpus, postings, IDF, and average document length held fixed "
                "(no corpus reindexing)."
            ),
        },
        "dataset_counts": {
            "train_articles": len(train_entries),
            "validation_articles": len(validation_entries),
            "test_articles": len(test_entries),
            "no_elife_test_articles": len(no_elife_entries),
        },
        "f1000_manuscript_family_overlap": family_overlap,
        "f1000_split_family_matrix": family_matrix,
        "bm25_reconstruction": bm25_reconstruction,
        "bm25_family_exclusion": family_exclusion_metrics,
        "ranking_comparison": ranking_comparison,
        "jaccard_threshold_sweep": threshold_sweep,
        "tool_output_coverage": tool_output_coverage,
        "limitations": [
            (
                "The family matrix is an F1000 DOI-stem lower bound only; it "
                "does not detect fuzzy-title, preprint-to-journal, or cross-source families."
            ),
            (
                "Jaccard thresholds and the historical embedding threshold are "
                "uncalibrated operating points. The sweep is a lexical sensitivity "
                "analysis, not a human-validated semantic-equivalence measure."
            ),
            (
                "The BM25 family intervention is query-time candidate-list "
                "filtering after scoring. The original corpus, postings, IDF, "
                "and average document length are fixed; this is not a rebuilt "
                "leave-family-out index and does not alter any LLM prediction."
            ),
            (
                "The embedding ranking is read from frozen aggregate results and "
                "is not recomputed here; its historical matcher threshold and model "
                "revision remain independently unvalidated."
            ),
            (
                "A same-family high retrieval rank demonstrates a measurement "
                "confound, but does not prove that every copied concern is invalid."
            ),
            (
                "The Git repository does not redistribute benchmark JSONL or tool "
                "outputs; exact reruns require an authorized local copy of those artifacts."
            ),
        ],
    }


def _metric_cell(metrics: dict[str, Any]) -> str:
    return (
        f"R={float(metrics['recall']):.4f}, "
        f"P={float(metrics['precision']):.4f}, "
        f"F1={float(metrics['f1']):.4f}, "
        f"matches={int(metrics['n_matched'])}"
    )


def render_markdown(audit: dict[str, Any]) -> str:
    overlap = audit["f1000_manuscript_family_overlap"]
    reconstruction = audit["bm25_reconstruction"]
    retrieval = reconstruction["same_family_retrieval"]
    coverage = audit["tool_output_coverage"]

    lines = [
        "# BioReview-Bench v4 Measurement Audit",
        "",
        (
            "**Status:** aggregate computational diagnostic. This audit does not "
            "replace the frozen benchmark, semantic-match validation, or human review."
        ),
        "",
        "## Fixed protocol",
        "",
        (
            f"- Scope: {audit['dataset_counts']['train_articles']} train, "
            f"{audit['dataset_counts'].get('validation_articles', 0)} validation, "
            f"and {audit['dataset_counts']['test_articles']} test articles."
        ),
        (
            f"- Selected historical-heuristic Jaccard operating point: "
            f"`{audit['protocol']['jaccard_threshold']}` (`0.65 × 0.3`); "
            "threshold-aware one-to-one Hungarian assignment; micro averaging."
        ),
        (
            "- Matcher operating points are uncalibrated and have not received "
            "independent human-equivalence validation."
        ),
        "- Figure-dependent reference concerns are excluded.",
        (
            f"- Frozen tool-output ID coverage: {coverage['n_models']} / "
            f"{coverage['n_models']} models exactly cover all "
            f"{coverage['n_expected_test_ids']} test article IDs."
        ),
        (
            "- BM25 family intervention: query-time candidate-list filtering "
            "after scoring; original corpus, postings, IDF, and average document "
            "length held fixed (no corpus reindexing)."
        ),
        "",
        "## F1000 DOI-stem family split overlap (lower bound)",
        "",
        (
            f"- {overlap['n_test_articles_in_overlapping_families']} / "
            f"{overlap['n_f1000_test_articles']} F1000 test articles "
            f"({100 * float(overlap['test_overlap_fraction']):.1f}%) share a "
            "version-independent DOI family with training."
        ),
        (
            f"- Overlapping families: {overlap['n_overlapping_families']}; "
            f"training rows in those families: "
            f"{overlap['n_train_articles_in_overlapping_families']}."
        ),
        (
            f"- Same-family training article at BM25 rank 1: "
            f"{retrieval['same_family_at_rank_1']} / "
            f"{retrieval['n_queries_with_ranked_same_family_train_article']} "
            f"({100 * float(retrieval['same_family_at_rank_1_fraction']):.1f}%)."
        ),
        (
            f"- Same-family training article within top 8: "
            f"{retrieval['same_family_within_top_8']} / "
            f"{retrieval['n_queries_with_ranked_same_family_train_article']} "
            f"({100 * float(retrieval['same_family_within_top_8_fraction']):.1f}%)."
        ),
        "",
    ]
    family_matrix = audit.get("f1000_split_family_matrix")
    if family_matrix:
        pairwise = family_matrix["pairwise"]
        development_test = family_matrix["development_vs_test"]
        parseable = family_matrix[
            "n_articles_with_parseable_doi_family_by_split"
        ]
        lines.extend(
            [
                (
                    "- This matrix covers explicit F1000Research DOI-version "
                    "suffixes only; all counts are aggregate and text-free."
                ),
                (
                    "- Parseable DOI-family coverage (train / validation / test): "
                    f"{parseable['train']} / {parseable['validation']} / "
                    f"{parseable['test']} "
                    "articles."
                ),
                "",
                "| Partition relation | Crossing F1000 DOI-stem families |",
                "|---|---:|",
                (
                    f"| train–validation | "
                    f"{pairwise['train_validation']['n_crossing_families']} |"
                ),
                (
                    f"| train–test | "
                    f"{pairwise['train_test']['n_crossing_families']} |"
                ),
                (
                    f"| validation–test | "
                    f"{pairwise['validation_test']['n_crossing_families']} |"
                ),
                (
                    f"| development (train ∪ validation)–test | "
                    f"{development_test['n_crossing_families']} |"
                ),
                "",
                (
                    f"- Unique F1000 DOI-stem families crossing any split boundary: "
                    f"{family_matrix['n_unique_families_crossing_any_split_boundary']}."
                ),
                (
                    f"- Development-overlapping F1000 test articles: "
                    f"{development_test['n_test_articles_in_crossing_families']} / "
                    f"{development_test['n_f1000_test_articles']} "
                    f"({100 * float(development_test['test_article_overlap_fraction']):.1f}%)."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Frozen BM25 reconstruction",
            "",
            (
                "- Reconstruction scope: targeted F1000 test queries whose "
                "version-independent manuscript family crosses the train/test split."
            ),
            (
                f"- Exact targeted prediction rows: "
                f"{reconstruction['n_exact_prediction_rows']} / "
                f"{reconstruction['n_reconstruction_target_rows']}."
            ),
            (
                f"- Rows changed by query-time manuscript-family candidate filtering: "
                f"{reconstruction['n_rows_changed_after_query_time_family_filter']}."
            ),
            (
                "- Defaults: `top_k_docs=8`, `max_concerns=12`, "
                "`max_input_chars=40000`, `k1=1.5`, `b=0.75`."
            ),
            "",
            "## BM25 current vs query-time family candidate filtering",
            "",
            "| Evaluation subset | Frozen current | Query-time family-filtered | ΔF1 | Δmatches |",
            "|---|---|---|---:|---:|",
        ]
    )
    for subset_name in [
        "full_test",
        "no_elife_450",
        "f1000_test",
        "f1000_overlapping_families",
    ]:
        if subset_name not in audit["bm25_family_exclusion"]:
            continue
        values = audit["bm25_family_exclusion"][subset_name]
        delta = values["delta_family_excluded_minus_current"]
        lines.append(
            f"| {subset_name} | {_metric_cell(values['current'])} | "
            f"{_metric_cell(values['family_excluded'])} | "
            f"{float(delta['f1']):+.4f} | {int(delta['n_matched']):+d} |"
        )

    lines.extend(
        [
            "",
        ]
    )
    threshold_sweep = audit.get("jaccard_threshold_sweep")
    if threshold_sweep:
        lines.extend(
            [
                "## Jaccard operating-point sensitivity",
                "",
                (
                    "All rows are aggregate, text-free diagnostics. No threshold "
                    "is a validated semantic-equivalence operating point."
                ),
                "",
            ]
        )
        for subset_name in ["full_600", "no_elife_450"]:
            points = threshold_sweep["subsets"].get(subset_name, [])
            if not points:
                continue
            qualifier = (
                "primary frozen test snapshot"
                if subset_name == "full_600"
                else "post-hoc sensitivity subset"
            )
            lines.extend(
                [
                    f"### {subset_name} ({qualifier})",
                    "",
                    "| Jaccard cutoff | Top system | Kendall tau-a vs frozen "
                    "embedding | Total matches across systems |",
                    "|---:|---|---:|---:|",
                ]
            )
            for point in points:
                lines.append(
                    f"| {float(point['threshold']):.3f} | "
                    f"{point['top_system']} | "
                    f"{float(point['kendall_tau_a_vs_frozen_embedding']):+.4f} | "
                    f"{int(point['total_matches_across_models'])} |"
                )
            lines.append("")

    for subset_name in ["full_600", "no_elife_450"]:
        comparison = audit["ranking_comparison"][subset_name]
        qualifier = (
            "primary frozen test snapshot"
            if subset_name == "full_600"
            else "post-hoc sensitivity subset"
        )
        lines.extend(
            [
                (
                    "## Frozen embedding vs one-to-one Jaccard at the selected "
                    f"0.195 point: {subset_name} ({qualifier})"
                ),
                "",
                "| Rank | Frozen embedding | F1 | One-to-one Jaccard | F1 |",
                "|---:|---|---:|---|---:|",
            ]
        )
        embedding = comparison["frozen_embedding_ranking"]
        jaccard = comparison["one_to_one_jaccard_ranking"]
        for index in range(len(embedding)):
            lines.append(
                f"| {index + 1} | {embedding[index]['model']} | "
                f"{float(embedding[index]['f1']):.4f} | "
                f"{jaccard[index]['model']} | {float(jaccard[index]['f1']):.4f} |"
            )
        lines.extend(
            [
                "",
                f"- Kendall tau-a: **{float(comparison['kendall_tau_a']):.4f}**.",
                "",
            ]
        )
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in audit["limitations"])
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")


def _parse_jaccard_threshold_list(raw: str) -> tuple[float, ...]:
    try:
        values = [float(value.strip()) for value in raw.split(",") if value.strip()]
        return normalize_jaccard_thresholds(values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=_REPO_ROOT / "data/splits/v4/train.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=_REPO_ROOT / "data/splits/v4/val.jsonl",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=_REPO_ROOT / "data/splits/v4/test.jsonl",
    )
    parser.add_argument(
        "--tool-output-dir",
        type=Path,
        default=_REPO_ROOT / "tool_outputs",
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=_REPO_ROOT / "results/v4/leaderboard.json",
    )
    parser.add_argument(
        "--aggregate-analysis",
        type=Path,
        default=_REPO_ROOT / "results/v4/robustness_validity_analysis.json",
        help=(
            "Authorized aggregate analysis containing the frozen no_elife_450 "
            "embedding ranking and metrics."
        ),
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=DEFAULT_JACCARD_THRESHOLD,
        help=(
            "Selected historical-heuristic Jaccard point retained for the "
            "single-point compatibility tables; it is not calibrated."
        ),
    )
    parser.add_argument(
        "--jaccard-sweep-thresholds",
        type=_parse_jaccard_threshold_list,
        default=DEFAULT_JACCARD_SWEEP_THRESHOLDS,
        metavar="T1,T2,...",
        help=(
            "Comma-separated deterministic sensitivity grid. The selected "
            "--jaccard-threshold is always included."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=_REPO_ROOT / "results/v4/measurement_audit.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=_REPO_ROOT / "results/v4/measurement_audit.md",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for input_path in [
        args.train,
        args.validation,
        args.test,
        args.leaderboard,
        args.aggregate_analysis,
    ]:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Required input not found: {input_path}. "
                "Use CLI options to select an authorized local v4 artifact."
            )
    if not 0.0 <= args.jaccard_threshold <= 1.0:
        raise ValueError("--jaccard-threshold must be in [0, 1]")

    audit = build_measurement_audit(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        tool_output_dir=args.tool_output_dir,
        leaderboard_path=args.leaderboard,
        aggregate_analysis_path=args.aggregate_analysis,
        jaccard_threshold=args.jaccard_threshold,
        jaccard_sweep_thresholds=args.jaccard_sweep_thresholds,
    )
    write_outputs(audit, args.output_json, args.output_md)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

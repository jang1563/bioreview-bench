from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.audit_v4_measurement import (
    BM25_DEFAULTS,
    build_jaccard_threshold_sweep,
    detect_f1000_family_overlap,
    eligible_hungarian_matches,
    evaluate_jaccard_hungarian,
    f1000_manuscript_family,
    kendall_tau_a,
    normalize_jaccard_thresholds,
    reconstruct_bm25_and_family_exclusion,
    render_markdown,
    require_exact_bm25_reconstruction,
    summarize_f1000_split_family_matrix,
    validate_tool_output_coverage,
)


def _entry(
    article_id: str,
    *,
    source: str,
    doi: str,
    title: str,
    concern: str,
) -> dict:
    return {
        "id": article_id,
        "source": source,
        "doi": doi,
        "title": title,
        "abstract": title,
        "paper_text_sections": {"methods": title, "results": title},
        "concerns": [
            {
                "concern_text": concern,
                "requires_figure_reading": False,
            }
        ],
    }


def test_f1000_family_removes_only_terminal_version() -> None:
    entry = {
        "source": "f1000",
        "doi": "10.12688/F1000Research.12345.3",
    }
    assert f1000_manuscript_family(entry) == "10.12688/f1000research.12345"
    assert (
        f1000_manuscript_family(
            {"source": "plos", "doi": "10.12688/f1000research.12345.3"}
        )
        is None
    )
    assert (
        f1000_manuscript_family(
            {"source": "f1000", "doi": "10.1371/journal.pone.12345"}
        )
        is None
    )


def test_hungarian_prioritizes_eligible_cardinality() -> None:
    similarities = np.array([[1.0, 0.65], [0.65, 0.64]], dtype=np.float64)
    matches = eligible_hungarian_matches(similarities, threshold=0.65)
    assert sorted(matches) == [(0, 1, 0.65), (1, 0, 0.65)]


def test_family_overlap_and_exclusion_change_bm25_prediction() -> None:
    train = [
        _entry(
            "f1000:family-v1",
            source="f1000",
            doi="10.12688/f1000research.100.1",
            title="single cell immune atlas sepsis cohort",
            concern="Same-family concern copied from version one.",
        ),
        _entry(
            "f1000:other",
            source="f1000",
            doi="10.12688/f1000research.200.1",
            title="single cell immune atlas sepsis validation",
            concern="Independent validation concern from another manuscript.",
        ),
    ]
    test = [
        _entry(
            "f1000:family-v2",
            source="f1000",
            doi="10.12688/f1000research.100.2",
            title="single cell immune atlas sepsis cohort",
            concern="Same-family concern copied from version one.",
        )
    ]
    frozen = {
        "f1000:family-v2": [
            "Same-family concern copied from version one.",
            "Independent validation concern from another manuscript.",
        ]
    }

    overlap, overlapping_ids = detect_f1000_family_overlap(train, test)
    reconstruction, current, excluded = reconstruct_bm25_and_family_exclusion(
        train,
        test,
        frozen,
    )

    assert overlapping_ids == {"f1000:family-v2"}
    assert overlap["n_overlapping_families"] == 1
    assert overlap["n_test_articles_in_overlapping_families"] == 1
    assert reconstruction["defaults"] == BM25_DEFAULTS
    assert reconstruction["all_target_prediction_rows_exact"] is True
    assert reconstruction["same_family_retrieval"]["same_family_at_rank_1"] == 1
    assert current["f1000:family-v2"] == frozen["f1000:family-v2"]
    assert excluded["f1000:family-v2"] == [
        "Independent validation concern from another manuscript."
    ]


def test_f1000_split_family_matrix_is_val_aware_and_aggregate() -> None:
    def family_entry(article_id: str, family: int, version: int) -> dict:
        return _entry(
            article_id,
            source="f1000",
            doi=f"10.12688/f1000research.{family}.{version}",
            title="synthetic article",
            concern="Synthetic concern.",
        )

    train = [
        family_entry("train-a", 100, 1),
        family_entry("train-b", 200, 1),
        family_entry("train-d", 400, 1),
        family_entry("train-only", 900, 1),
        _entry(
            "train-unparseable",
            source="f1000",
            doi="not-a-versioned-f1000-doi",
            title="synthetic article",
            concern="Synthetic concern.",
        ),
    ]
    validation = [
        family_entry("val-a", 100, 2),
        family_entry("val-c", 300, 1),
        family_entry("val-d", 400, 2),
    ]
    test = [
        family_entry("test-b", 200, 2),
        family_entry("test-c", 300, 2),
        family_entry("test-d", 400, 3),
        family_entry("test-e1", 500, 1),
        family_entry("test-e2", 500, 2),
    ]

    matrix = summarize_f1000_split_family_matrix(train, validation, test)

    assert matrix["is_lower_bound"] is True
    assert "F1000Research" in matrix["scope"]
    assert matrix["n_f1000_articles_by_split"]["train"] == 5
    assert matrix["n_articles_with_parseable_doi_family_by_split"]["train"] == 4
    assert (
        matrix["n_f1000_articles_without_parseable_doi_family_by_split"]["train"]
        == 1
    )
    assert matrix["pairwise"]["train_validation"]["n_crossing_families"] == 2
    assert matrix["pairwise"]["train_test"]["n_crossing_families"] == 2
    assert matrix["pairwise"]["validation_test"]["n_crossing_families"] == 2
    assert matrix["development_vs_test"]["n_crossing_families"] == 3
    assert matrix["development_vs_test"]["n_test_articles_in_crossing_families"] == 3
    assert matrix["development_vs_test"]["test_article_overlap_fraction"] == 0.6
    assert matrix["n_unique_families_crossing_any_split_boundary"] == 4
    assert matrix["within_split_multiversion_families"]["test"] == 1
    serialized = json.dumps(matrix, sort_keys=True)
    assert "10.12688" not in serialized
    assert "test-b" not in serialized


def test_jaccard_threshold_sweep_is_deterministic_and_text_free() -> None:
    entries = [
        _entry(
            "article:1",
            source="plos",
            doi="10.1371/journal.pone.1",
            title="paper",
            concern="Missing independent validation cohort.",
        )
    ]
    tool_maps = {
        "Model-A": {
            "article:1": [
                "Missing independent validation cohort.",
                "Unrelated reporting suggestion.",
            ]
        },
        "Model-B": {"article:1": ["Validation cohort absent."]},
    }
    embeddings = {
        "full_600": [
            {"rank": 1, "model": "Model-B", "f1": 0.8},
            {"rank": 2, "model": "Model-A", "f1": 0.7},
        ]
    }

    first = build_jaccard_threshold_sweep(
        tool_maps=tool_maps,
        subsets={"full_600": entries},
        embedding_rankings=embeddings,
        thresholds=[0.5, 0.3, 0.5],
    )
    second = build_jaccard_threshold_sweep(
        tool_maps=tool_maps,
        subsets={"full_600": entries},
        embedding_rankings=embeddings,
        thresholds=[0.3, 0.5],
    )

    assert first == second
    assert first["thresholds"] == [0.3, 0.5]
    low, high = first["subsets"]["full_600"]
    assert low["top_system"] == "Model-B"
    assert low["kendall_tau_a_vs_frozen_embedding"] == 1.0
    assert high["top_system"] == "Model-A"
    assert high["kendall_tau_a_vs_frozen_embedding"] == -1.0
    assert low["total_matches_across_models"] == 2
    assert high["total_matches_across_models"] == 1
    serialized = json.dumps(first, sort_keys=True)
    assert "Missing independent validation cohort" not in serialized
    assert "Validation cohort absent" not in serialized


def test_jaccard_threshold_normalization_fails_closed() -> None:
    assert normalize_jaccard_thresholds([0.2, 0.1, 0.2]) == (0.1, 0.2)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        normalize_jaccard_thresholds([1.1])
    with pytest.raises(ValueError, match="At least one"):
        normalize_jaccard_thresholds([])

    with pytest.raises(ValueError, match="every tool exactly once"):
        build_jaccard_threshold_sweep(
            tool_maps={"model-a": {"article:1": []}},
            subsets={"full_600": []},
            embedding_rankings={
                "full_600": [{"rank": 1, "model": "model-b", "f1": 0.0}]
            },
            thresholds=[0.2],
        )


def test_frozen_output_coverage_and_bm25_reconstruction_fail_closed() -> None:
    entries = [
        _entry(
            "article:1",
            source="plos",
            doi="10.1371/journal.pone.1",
            title="paper one",
            concern="Concern one.",
        ),
        _entry(
            "article:2",
            source="plos",
            doi="10.1371/journal.pone.2",
            title="paper two",
            concern="Concern two.",
        ),
    ]
    exact_maps = {
        "model-a": {"article:1": [], "article:2": []},
        "model-b": {"article:1": [], "article:2": []},
    }
    coverage = validate_tool_output_coverage(exact_maps, entries)
    assert coverage["all_models_exact_test_id_set"] is True
    assert coverage["models"]["model-a"]["exact_test_id_set"] is True

    with pytest.raises(ValueError, match="exactly match the test set"):
        validate_tool_output_coverage(
            {"model-a": {"article:1": [], "article:extra": []}},
            entries,
        )

    with pytest.raises(ValueError, match="1/2 exact"):
        require_exact_bm25_reconstruction(
            {
                "n_reconstruction_target_rows": 2,
                "n_exact_prediction_rows": 1,
                "all_target_prediction_rows_exact": False,
            }
        )


def test_one_to_one_jaccard_and_markdown_are_explicit() -> None:
    entries = [
        _entry(
            "article:1",
            source="plos",
            doi="10.1371/journal.pone.1",
            title="paper",
            concern="Missing independent validation cohort.",
        )
    ]
    entries[0]["concerns"].append(
        {
            "concern_text": "Missing independent validation cohort.",
            "requires_figure_reading": False,
        }
    )
    metrics = evaluate_jaccard_hungarian(
        {"article:1": ["Missing independent validation cohort."]},
        entries,
        threshold=0.15,
    )
    assert metrics["n_gt"] == 2
    assert metrics["n_predictions"] == 1
    assert metrics["n_matched"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1.0

    audit = {
        "dataset_counts": {"train_articles": 2, "test_articles": 1},
        "protocol": {"jaccard_threshold": 0.195},
        "f1000_manuscript_family_overlap": {
            "n_test_articles_in_overlapping_families": 1,
            "n_f1000_test_articles": 1,
            "test_overlap_fraction": 1.0,
            "n_overlapping_families": 1,
            "n_train_articles_in_overlapping_families": 1,
        },
        "bm25_reconstruction": {
            "n_exact_prediction_rows": 1,
            "n_reconstruction_target_rows": 1,
            "n_rows_changed_after_query_time_family_filter": 1,
            "same_family_retrieval": {
                "same_family_at_rank_1": 1,
                "n_queries_with_ranked_same_family_train_article": 1,
                "same_family_at_rank_1_fraction": 1.0,
                "same_family_within_top_8": 1,
                "same_family_within_top_8_fraction": 1.0,
            },
        },
        "tool_output_coverage": {
            "n_expected_test_ids": 1,
            "n_models": 1,
            "all_models_exact_test_id_set": True,
            "models": {
                "A": {
                    "n_expected_test_ids": 1,
                    "n_output_rows": 1,
                    "n_missing_test_ids": 0,
                    "n_extra_article_ids": 0,
                    "exact_test_id_set": True,
                }
            },
        },
        "bm25_family_exclusion": {
            "full_test": {
                "current": metrics,
                "family_excluded": metrics,
                "delta_family_excluded_minus_current": {
                    "f1": 0.0,
                    "n_matched": 0,
                },
            }
        },
        "ranking_comparison": {
            subset: {
                "frozen_embedding_ranking": [
                    {
                        "model": "A",
                        "f1": 0.9,
                    }
                ],
                "one_to_one_jaccard_ranking": [
                    {
                        "model": "A",
                        "f1": 0.1,
                    }
                ],
                "kendall_tau_a": 1.0,
            }
            for subset in ["no_elife_450", "full_600"]
        },
        "limitations": ["Lexical matching is not semantic validation."],
    }
    markdown = render_markdown(audit)
    assert "threshold-aware one-to-one Hungarian" in markdown
    assert "query-time candidate-list filtering" in markdown
    assert "IDF" in markdown
    assert "Kendall tau-a" in markdown
    assert "Lexical matching is not semantic validation." in markdown


def test_kendall_tau_a_detects_reversal() -> None:
    assert kendall_tau_a(["A", "B", "C"], ["A", "B", "C"]) == 1.0
    assert kendall_tau_a(["A", "B", "C"], ["C", "B", "A"]) == -1.0

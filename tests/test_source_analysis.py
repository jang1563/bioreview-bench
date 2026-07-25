from __future__ import annotations

import pytest

from bioreview_bench.evaluate.metrics import EvalResult
from scripts.source_analysis import paired_micro_f1_randomization_test


def _result(n_matched: int, n_reference: int, n_predicted: int) -> EvalResult:
    recall = n_matched / n_reference if n_reference else 0.0
    precision = n_matched / n_predicted if n_predicted else 0.0
    f1 = 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0
    return EvalResult(
        recall=recall,
        precision=precision,
        f1=f1,
        n_gt_total=n_reference,
        n_tool_total=n_predicted,
        n_matched=n_matched,
    )


def test_pairwise_micro_f1_randomization_aligns_by_article_id() -> None:
    results_a = {
        "article-2": _result(1, 1, 1),
        "article-1": _result(1, 1, 1),
    }
    results_b = {
        "article-1": _result(0, 1, 1),
        "article-2": _result(0, 1, 1),
    }

    result = paired_micro_f1_randomization_test(results_a, results_b)

    assert result["method"] == ("article_paired_label_swap_randomization_delta_micro_f1")
    assert result["delta_f1"] == 1.0
    assert result["p_value"] == 0.5
    assert result["n_articles"] == 2
    assert result["n_informative_pairs"] == 2
    assert result["randomization_mode"] == "exact"
    assert result["randomization_draws"] == 4
    assert result["plus_one_correction"] is False
    assert result["seed"] is None
    assert result["multiplicity_adjustment"] == "none"
    assert result["interpretation"] == "exploratory_unadjusted"


def test_pairwise_micro_f1_randomization_rejects_id_mismatch() -> None:
    with pytest.raises(ValueError, match="identical article IDs"):
        paired_micro_f1_randomization_test(
            {"article-1": _result(1, 1, 1)},
            {"article-2": _result(1, 1, 1)},
        )


def test_pairwise_micro_f1_randomization_rejects_reference_mismatch() -> None:
    with pytest.raises(ValueError, match="same reference count"):
        paired_micro_f1_randomization_test(
            {"article-1": _result(1, 1, 1)},
            {"article-1": _result(1, 2, 1)},
        )


def test_pairwise_micro_f1_monte_carlo_cannot_publish_zero_pvalue() -> None:
    results_a = {f"article-{index}": _result(1, 1, 1) for index in range(20)}
    results_b = {f"article-{index}": _result(0, 1, 1) for index in range(20)}

    result = paired_micro_f1_randomization_test(
        results_a,
        results_b,
        n_resamples=2_000,
        seed=7,
        exact_max_pairs=8,
    )

    assert result["randomization_mode"] == "monte_carlo"
    assert result["randomization_draws"] == 2_000
    assert result["plus_one_correction"] is True
    assert result["p_value"] is not None
    assert result["p_value"] >= 1 / 2_001

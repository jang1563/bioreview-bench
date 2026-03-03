"""Tests for dataset-level metric aggregation in evaluate.runner."""

from __future__ import annotations

from bioreview_bench.evaluate.metrics import EvalResult
from bioreview_bench.evaluate.runner import aggregate_results, bootstrap_ci


def _make_eval(
    *,
    recall: float,
    precision: float,
    f1: float,
    n_gt_total: int,
    n_tool_total: int,
    n_matched: int,
) -> EvalResult:
    return EvalResult(
        recall=recall,
        precision=precision,
        f1=f1,
        n_gt_total=n_gt_total,
        n_tool_total=n_tool_total,
        n_matched=n_matched,
    )


def test_aggregate_results_uses_micro_counts_for_overall_metrics():
    """Overall recall/precision/F1 should be count-weighted (micro), not article mean."""
    # Deliberately imbalanced: article means would be 0.5, micro is ~0.0099.
    article_results = [
        _make_eval(recall=1.0, precision=1.0, f1=1.0, n_gt_total=1, n_tool_total=1, n_matched=1),
        _make_eval(recall=0.0, precision=0.0, f1=0.0, n_gt_total=100, n_tool_total=100, n_matched=0),
    ]

    result = aggregate_results(
        article_results=article_results,
        n_bootstrap=0,
        tool_name="tool",
        tool_version="v1",
        git_hash="",
        split="val",
        extraction_manifest_id="em-v1.0",
        n_articles=2,
        n_human_concerns=101,
        n_tool_concerns=101,
        n_figure_excluded=0,
        notes="",
    )

    expected = 1 / 101
    assert result.recall_overall == expected
    assert result.precision_overall == expected
    assert result.f1_micro == expected


def test_bootstrap_ci_uses_micro_counts():
    """Bootstrap CI should run on micro metrics based on concern counts."""
    article_results = [
        _make_eval(recall=1.0, precision=1.0, f1=1.0, n_gt_total=1, n_tool_total=1, n_matched=1),
        _make_eval(recall=0.0, precision=0.0, f1=0.0, n_gt_total=100, n_tool_total=100, n_matched=0),
    ]

    cis = bootstrap_ci(article_results=article_results, n_bootstrap=50, seed=7)
    assert 0.0 <= cis["recall"].lo <= cis["recall"].hi <= 1.0
    assert 0.0 <= cis["precision"].lo <= cis["precision"].hi <= 1.0

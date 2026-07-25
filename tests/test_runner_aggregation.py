"""Tests for dataset-level metric aggregation in evaluate.runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioreview_bench.evaluate.metrics import CategoryMetrics, EvalResult
from bioreview_bench.evaluate.runner import (
    aggregate_results,
    bootstrap_ci,
    build_tool_map,
    load_jsonl,
)


def _make_eval(
    *,
    recall: float,
    precision: float,
    f1: float,
    n_gt_total: int,
    n_tool_total: int,
    n_matched: int,
    n_gt_major: int = 0,
    n_matched_major: int = 0,
    per_category: dict[str, CategoryMetrics] | None = None,
) -> EvalResult:
    return EvalResult(
        recall=recall,
        precision=precision,
        f1=f1,
        n_gt_total=n_gt_total,
        n_tool_total=n_tool_total,
        n_matched=n_matched,
        n_gt_major=n_gt_major,
        n_matched_major=n_matched_major,
        per_category=per_category or {},
    )


def test_aggregate_results_uses_micro_counts_for_overall_metrics():
    """Overall recall/precision/F1 should be count-weighted (micro), not article mean."""
    # Deliberately imbalanced: article means would be 0.5, micro is ~0.0099.
    article_results = [
        _make_eval(recall=1.0, precision=1.0, f1=1.0, n_gt_total=1, n_tool_total=1, n_matched=1),
        _make_eval(
            recall=0.0,
            precision=0.0,
            f1=0.0,
            n_gt_total=100,
            n_tool_total=100,
            n_matched=0,
        ),
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
        _make_eval(
            recall=0.0,
            precision=0.0,
            f1=0.0,
            n_gt_total=100,
            n_tool_total=100,
            n_matched=0,
        ),
    ]

    cis = bootstrap_ci(article_results=article_results, n_bootstrap=50, seed=7)
    assert 0.0 <= cis["recall"].lo <= cis["recall"].hi <= 1.0
    assert 0.0 <= cis["precision"].lo <= cis["precision"].hi <= 1.0


def test_aggregate_results_uses_micro_counts_for_major_recall():
    article_results = [
        _make_eval(
            recall=1.0,
            precision=1.0,
            f1=1.0,
            n_gt_total=1,
            n_tool_total=1,
            n_matched=1,
            n_gt_major=1,
            n_matched_major=1,
        ),
        _make_eval(
            recall=0.0,
            precision=0.0,
            f1=0.0,
            n_gt_total=100,
            n_tool_total=100,
            n_matched=0,
            n_gt_major=100,
            n_matched_major=0,
        ),
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

    assert result.recall_major == pytest.approx(1 / 101)


def test_aggregate_results_populates_unweighted_category_macro_f1() -> None:
    """Macro F1 is the simple mean of dataset-level category F1 values."""
    article_results = [
        _make_eval(
            recall=100 / 101,
            precision=100 / 101,
            f1=100 / 101,
            n_gt_total=101,
            n_tool_total=101,
            n_matched=100,
            per_category={
                "large_category": CategoryMetrics(
                    recall=1.0,
                    precision=1.0,
                    f1=1.0,
                    n_gt=100,
                    n_tool=100,
                    n_matched=100,
                ),
                "small_category": CategoryMetrics(
                    recall=0.0,
                    precision=0.0,
                    f1=0.0,
                    n_gt=1,
                    n_tool=1,
                    n_matched=0,
                ),
            },
        )
    ]

    result = aggregate_results(
        article_results=article_results,
        n_bootstrap=0,
        tool_name="tool",
        tool_version="v1",
        git_hash="",
        split="val",
        extraction_manifest_id="em-v1.0",
        n_articles=1,
        n_human_concerns=101,
        n_tool_concerns=101,
        n_figure_excluded=0,
        notes="",
    )

    assert result.per_category["large_category"].f1_micro == 1.0
    assert result.per_category["small_category"].f1_micro == 0.0
    assert result.per_category["large_category"].f1_macro is None
    assert result.per_category["small_category"].f1_macro is None
    assert result.f1_macro == pytest.approx(0.5)
    assert result.f1_macro != pytest.approx(result.f1_micro)


def test_aggregate_results_uses_zero_macro_f1_without_categories() -> None:
    result = aggregate_results(
        article_results=[
            _make_eval(
                recall=0.0,
                precision=0.0,
                f1=0.0,
                n_gt_total=1,
                n_tool_total=1,
                n_matched=0,
            )
        ],
        n_bootstrap=0,
        tool_name="tool",
        tool_version="v1",
        git_hash="",
        split="val",
        extraction_manifest_id="em-v1.0",
        n_articles=1,
        n_human_concerns=1,
        n_tool_concerns=1,
        n_figure_excluded=0,
        notes="",
    )

    assert result.per_category == {}
    assert result.f1_macro == 0.0


def test_aggregate_results_explicitly_sets_zero_macro_f1_for_empty_input() -> None:
    result = aggregate_results(
        article_results=[],
        n_bootstrap=0,
        tool_name="tool",
        tool_version="v1",
        git_hash="",
        split="val",
        extraction_manifest_id="em-v1.0",
        n_articles=0,
        n_human_concerns=0,
        n_tool_concerns=0,
        n_figure_excluded=0,
        notes="",
    )

    assert result.per_category == {}
    assert result.f1_macro == 0.0


def test_load_jsonl_fails_fast_on_malformed_input(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        load_jsonl(path)


def test_build_tool_map_rejects_duplicate_article_rows(tmp_path: Path) -> None:
    path = tmp_path / "tool.jsonl"
    path.write_text(
        '{"article_id": "a1", "concerns": ["first"]}\n'
        '{"article_id": "a1", "concerns": ["second"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate article_id 'a1'"):
        build_tool_map(path)


def test_build_tool_map_discards_optional_category_and_severity(tmp_path: Path) -> None:
    path = tmp_path / "tool.jsonl"
    path.write_text(
        (
            '{"article_id":"a1","concerns":['
            '{"text":"Concern one","category":"design_flaw","severity":"major"},'
            '{"concern_text":"Concern two","category":"other","severity":"minor"}'
            "]}\n"
        ),
        encoding="utf-8",
    )

    assert build_tool_map(path) == {"a1": ["Concern one", "Concern two"]}

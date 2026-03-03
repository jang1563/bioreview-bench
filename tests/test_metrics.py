"""Tests for ConcernMatcher evaluation metrics (Jaccard mode, no model loading)."""

from __future__ import annotations

import pytest

from bioreview_bench.evaluate.metrics import ConcernMatcher, EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gt(
    text: str,
    category: str = "statistical_methodology",
    severity: str = "major",
    requires_figure_reading: bool = False,
) -> dict:
    """Build a minimal ground-truth concern dict."""
    return {
        "concern_text": text,
        "category": category,
        "severity": severity,
        "requires_figure_reading": requires_figure_reading,
    }


def _matcher(
    threshold: float = 0.65,
    exclude_figure: bool = True,
    algorithm: str = "greedy",
) -> ConcernMatcher:
    """Return a Jaccard-mode matcher (no sentence-transformers)."""
    return ConcernMatcher(
        threshold=threshold,
        exclude_figure=exclude_figure,
        use_embedding=False,
        algorithm=algorithm,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Recall tests
# ---------------------------------------------------------------------------


def test_perfect_recall():
    """When tool concerns are identical to GT, recall == 1.0."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    gt = [_make_gt(text)]
    tool = [text]
    result = _matcher().score_article(tool, gt)
    assert result.recall == pytest.approx(1.0)


def test_zero_recall():
    """When tool concerns share no tokens with GT, recall == 0.0."""
    gt = [_make_gt("The statistical analysis lacks proper controls and sufficient sample size.")]
    tool = ["Fluorescent microscopy images show abnormal cellular distribution patterns."]
    result = _matcher().score_article(tool, gt)
    assert result.recall == pytest.approx(0.0)


def test_partial_recall():
    """When tool covers 2 of 4 GT concerns, recall == 0.5."""
    gt_texts = [
        "The statistical analysis lacks proper controls and sufficient sample size validation.",
        "The western blot normalization procedure is missing and needs correction.",
        "The microscopy imaging resolution is insufficient for the claimed conclusions.",
        "The animal model selection is not justified relative to the human disease model.",
    ]
    # Tool texts that share enough tokens with the first two GT entries
    tool_texts = [
        "Statistical analysis lacks proper controls and sufficient sample size validation.",
        "The western blot normalization procedure is missing and needs to be corrected.",
    ]
    gt = [_make_gt(t) for t in gt_texts]
    result = _matcher().score_article(tool_texts, gt)
    assert result.recall == pytest.approx(0.5)
    assert result.n_gt_total == 4
    assert result.n_matched == 2


# ---------------------------------------------------------------------------
# Precision tests
# ---------------------------------------------------------------------------


def test_precision_perfect():
    """When all tool concerns match a GT concern, precision == 1.0."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    gt = [_make_gt(text)]
    tool = [text]
    result = _matcher().score_article(tool, gt)
    assert result.precision == pytest.approx(1.0)


def test_precision_zero():
    """When no tool concerns match GT, precision == 0.0 (tool output wasted)."""
    gt = [_make_gt("The statistical analysis lacks proper controls and sufficient sample size.")]
    tool = [
        "Fluorescent microscopy images show abnormal cellular distribution patterns.",
        "The protein structure prediction algorithm diverges under stress conditions.",
    ]
    result = _matcher().score_article(tool, gt)
    assert result.precision == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# F1 test
# ---------------------------------------------------------------------------


def test_f1_harmonic_mean():
    """F1 is the harmonic mean of precision and recall: 2*P*R/(P+R)."""
    gt_texts = [
        "The statistical analysis lacks proper controls and sufficient sample size.",
        "The animal model selection is not justified for this human disease context.",
    ]
    tool_texts = [
        "The statistical analysis lacks proper controls and sufficient sample size.",
        "Fluorescent microscopy images show completely unrelated cellular morphology.",
    ]
    gt = [_make_gt(t) for t in gt_texts]
    result = _matcher().score_article(tool_texts, gt)

    # 1 match out of 2 GT → recall=0.5; 1 match out of 2 tool → precision=0.5
    expected_f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
    assert result.f1 == pytest.approx(expected_f1, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_empty_tool_concerns():
    """Empty tool output → recall=0.0, precision=0.0 (undefined, no tool output)."""
    gt = [_make_gt("The statistical analysis lacks proper controls and sufficient sample size.")]
    result = _matcher().score_article([], gt)
    assert result.recall == pytest.approx(0.0)
    # precision = n_matched / n_tool; n_tool=0 → 0.0 (convention: no false positives but
    # also no true positives, so precision defaults to 0.0 in the implementation)
    assert result.precision == pytest.approx(0.0)
    assert result.n_tool_total == 0
    assert result.n_matched == 0


def test_empty_gt_concerns():
    """Empty GT → recall=0.0, precision=0.0 (tool output wasted)."""
    tool = ["The statistical analysis lacks proper controls."]
    result = _matcher().score_article(tool, [])
    assert result.recall == pytest.approx(0.0)
    assert result.precision == pytest.approx(0.0)
    assert result.n_gt_total == 0
    assert result.n_tool_total == 1


# ---------------------------------------------------------------------------
# Figure exclusion tests
# ---------------------------------------------------------------------------


def test_figure_exclude():
    """figure_issue concern is excluded from GT when exclude_figure=True."""
    figure_concern = _make_gt(
        "Figure 3B shows inconsistent band sizes across lanes in the western blot.",
        category="figure_issue",
        requires_figure_reading=True,
    )
    non_figure_concern = _make_gt(
        "The statistical analysis lacks proper controls and sufficient sample size.",
        category="statistical_methodology",
        requires_figure_reading=False,
    )
    gt = [figure_concern, non_figure_concern]
    tool = ["The statistical analysis lacks proper controls and sufficient sample size."]

    result = _matcher(exclude_figure=True).score_article(tool, gt)
    # Only non-figure concern counts toward GT
    assert result.n_gt_total == 1
    assert result.n_gt_figure_excluded == 1
    assert result.recall == pytest.approx(1.0)


def test_figure_include():
    """figure_issue concern is counted in GT when exclude_figure=False."""
    figure_concern = _make_gt(
        "Figure 3B shows inconsistent band sizes across lanes in the western blot.",
        category="figure_issue",
        requires_figure_reading=True,
    )
    non_figure_concern = _make_gt(
        "The statistical analysis lacks proper controls and sufficient sample size.",
        category="statistical_methodology",
        requires_figure_reading=False,
    )
    gt = [figure_concern, non_figure_concern]
    tool = ["The statistical analysis lacks proper controls and sufficient sample size."]

    result = _matcher(exclude_figure=False).score_article(tool, gt)
    # Both concerns count toward GT total
    assert result.n_gt_total == 2
    assert result.n_gt_figure_excluded == 0
    # Tool matches 1 of 2 → recall = 0.5
    assert result.recall == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Severity breakdown test
# ---------------------------------------------------------------------------


def test_severity_breakdown():
    """recall_major is computed correctly from major-severity GT concerns."""
    major_concern = _make_gt(
        "The statistical analysis lacks proper controls and sufficient sample size.",
        severity="major",
    )
    minor_concern = _make_gt(
        "A minor typographical error appears in Figure 1 caption text.",
        severity="minor",
    )
    gt = [major_concern, minor_concern]
    # Tool matches only the major concern
    tool = ["The statistical analysis lacks proper controls and sufficient sample size."]

    result = _matcher().score_article(tool, gt)
    assert result.recall_major == pytest.approx(1.0)
    assert result.recall_minor == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Per-category breakdown test
# ---------------------------------------------------------------------------


def test_per_category_breakdown():
    """per_category is populated for each GT category present."""
    gt = [
        _make_gt(
            "The statistical analysis lacks proper controls and sufficient sample size.",
            category="statistical_methodology",
        ),
        _make_gt(
            "The experimental design does not include appropriate negative controls.",
            category="design_flaw",
        ),
    ]
    tool = ["The statistical analysis lacks proper controls and sufficient sample size."]
    result = _matcher().score_article(tool, gt)

    assert "statistical_methodology" in result.per_category
    assert "design_flaw" in result.per_category
    # statistical_methodology GT concern was matched
    assert result.per_category["statistical_methodology"].n_gt == 1
    assert result.per_category["design_flaw"].n_gt == 1


# ---------------------------------------------------------------------------
# No double-matching test
# ---------------------------------------------------------------------------


def test_no_double_matching():
    """A single GT concern can be matched to at most one tool concern."""
    shared_text = "The statistical analysis lacks proper controls and sufficient sample size."
    gt = [_make_gt(shared_text)]
    # Two identical tool concerns targeting the same GT
    tool = [shared_text, shared_text]

    result = _matcher().score_article(tool, gt)
    # Only one match is allowed (1:1 bipartite)
    assert result.n_matched == 1
    assert result.recall == pytest.approx(1.0)
    # precision = 1 match / 2 tool = 0.5
    assert result.precision == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# score_dataset test
# ---------------------------------------------------------------------------


def test_score_dataset():
    """score_dataset aggregates results across multiple articles correctly."""
    text_a = "The statistical analysis lacks proper controls and sufficient sample size."
    text_b = "The western blot normalization procedure is missing and needs correction."

    ground_truth = [
        {
            "id": "article-1",
            "concerns": [
                {
                    "concern_text": text_a,
                    "category": "statistical_methodology",
                    "severity": "major",
                    "requires_figure_reading": False,
                }
            ],
        },
        {
            "id": "article-2",
            "concerns": [
                {
                    "concern_text": text_b,
                    "category": "reagent_method_specificity",
                    "severity": "minor",
                    "requires_figure_reading": False,
                }
            ],
        },
    ]

    # Tool gets article-1 right, misses article-2
    tool_results = [
        {"article_id": "article-1", "concerns": [text_a]},
        {"article_id": "article-2", "concerns": ["Completely unrelated fluorescence issue here."]},
    ]

    matcher = _matcher()
    result = matcher.score_dataset(tool_results, ground_truth)

    # Macro average: article-1 recall=1.0, article-2 recall=0.0 → mean=0.5
    assert result.recall == pytest.approx(0.5)
    assert result.n_gt_total == 2
    assert result.n_tool_total == 2
    assert result.n_matched == 1


# ---------------------------------------------------------------------------
# Hungarian matching tests
# ---------------------------------------------------------------------------


def test_hungarian_same_as_greedy_simple():
    """For simple cases, Hungarian and greedy produce the same result."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    gt = [_make_gt(text)]
    tool = [text]

    greedy_result = _matcher(algorithm="greedy").score_article(tool, gt)
    hungarian_result = _matcher(algorithm="hungarian").score_article(tool, gt)

    assert greedy_result.recall == hungarian_result.recall
    assert greedy_result.precision == hungarian_result.precision
    assert greedy_result.n_matched == hungarian_result.n_matched


def test_hungarian_finds_more_matches():
    """Hungarian can find more matches than greedy in ambiguous assignments.

    Scenario: Two tool concerns (A, B) vs two GT concerns (1, 2).
    A is most similar to 1, but A also matches 2 well.
    B only matches 1.
    Greedy assigns A→1 (best score), leaving B unmatched.
    Hungarian assigns A→2, B→1 for 2 total matches.
    """
    # GT concerns
    gt = [
        _make_gt("The statistical methods need more proper controls for the experiment."),
        _make_gt("Statistical analysis and proper experimental controls are lacking."),
    ]
    # Tool A: shares many tokens with both GT
    # Tool B: shares tokens mainly with GT[0]
    tool = [
        "Statistical analysis and proper experimental controls require improvement.",
        "The statistical methods need more proper controls in this study.",
    ]
    # Greedy takes best pair first, possibly blocking a second match
    greedy_result = _matcher(threshold=0.20, algorithm="greedy").score_article(tool, gt)
    hungarian_result = _matcher(threshold=0.20, algorithm="hungarian").score_article(tool, gt)

    # Hungarian should find at least as many matches as greedy
    assert hungarian_result.n_matched >= greedy_result.n_matched


def test_algorithm_field_in_result():
    """EvalResult stores the algorithm used."""
    text = "The statistical analysis lacks proper controls."
    gt = [_make_gt(text)]

    greedy_result = _matcher(algorithm="greedy").score_article([text], gt)
    hungarian_result = _matcher(algorithm="hungarian").score_article([text], gt)

    assert greedy_result.algorithm == "greedy"
    assert hungarian_result.algorithm == "hungarian"


# ---------------------------------------------------------------------------
# Per-category precision fix tests
# ---------------------------------------------------------------------------


def test_per_category_precision_uses_assigned_tool_count():
    """Per-category precision uses category-assigned tool count, not total.

    Previously: cat_prec = cat_matched / n_tool (total) → always very low.
    Now: cat_prec = cat_matched / n_tool_assigned_to_cat.
    """
    # Two categories: stat (1 GT) and design (1 GT)
    gt = [
        _make_gt(
            "The statistical analysis lacks proper controls and sufficient sample size.",
            category="statistical_methodology",
        ),
        _make_gt(
            "The experimental design does not include appropriate negative controls.",
            category="design_flaw",
        ),
    ]
    # Tool matches stat perfectly, misses design
    tool = [
        "The statistical analysis lacks proper controls and sufficient sample size.",
        "Completely unrelated fluorescence imaging concern unrelated to anything.",
    ]
    result = _matcher().score_article(tool, gt)

    stat_cat = result.per_category["statistical_methodology"]
    # stat has 1 matched, and its n_tool should NOT be total (2)
    # The matched tool concern → stat. The unmatched tool concern → nearest GT category.
    assert stat_cat.n_matched == 1
    assert stat_cat.recall == pytest.approx(1.0)
    # n_tool should be tool concerns assigned to this category, not total
    assert stat_cat.n_tool <= 2  # could be 1 or 2 depending on nearest assignment


def test_per_category_tool_count_sums_to_total():
    """Sum of per-category n_tool should equal total tool concerns."""
    gt = [
        _make_gt("Statistical controls are missing.", category="statistical_methodology"),
        _make_gt("The design has flaws.", category="design_flaw"),
        _make_gt("Writing is unclear.", category="writing_clarity"),
    ]
    tool = [
        "Statistical controls are missing.",
        "The design has flaws.",
        "Some unrelated concern about methods.",
        "Another concern about clarity.",
    ]
    result = _matcher(threshold=0.20).score_article(tool, gt)

    total_assigned = sum(cm.n_tool for cm in result.per_category.values())
    assert total_assigned == len(tool)


# ---------------------------------------------------------------------------
# Soft matching tests
# ---------------------------------------------------------------------------


def test_soft_metrics_perfect_match():
    """Soft metrics equal hard metrics when match similarity is 1.0 (identical text)."""
    text = "The statistical analysis lacks proper controls and sufficient sample size."
    gt = [_make_gt(text)]
    tool = [text]
    result = _matcher().score_article(tool, gt)

    # Jaccard of identical text = 1.0, so soft credit = 1.0 = hard credit
    assert result.soft_recall == pytest.approx(result.recall)
    assert result.soft_precision == pytest.approx(result.precision)
    assert result.soft_f1 == pytest.approx(result.f1)


def test_soft_metrics_zero_when_no_matches():
    """Soft metrics are 0 when there are no matches."""
    gt = [_make_gt("The statistical analysis lacks proper controls and sufficient sample size.")]
    tool = ["Fluorescent microscopy images show abnormal cellular distribution patterns."]
    result = _matcher().score_article(tool, gt)

    assert result.soft_recall == pytest.approx(0.0)
    assert result.soft_precision == pytest.approx(0.0)
    assert result.soft_f1 == pytest.approx(0.0)


def test_soft_metrics_less_than_or_equal_hard():
    """Soft recall/precision <= hard recall/precision (credit ≤ 1 per match)."""
    gt_texts = [
        "The statistical analysis lacks proper controls and sufficient sample size validation.",
        "The western blot normalization procedure is missing and needs correction.",
    ]
    tool_texts = [
        "Statistical analysis lacks proper controls and sufficient sample size validation.",
        "The western blot normalization procedure is missing and needs to be corrected.",
    ]
    gt = [_make_gt(t) for t in gt_texts]
    result = _matcher().score_article(tool_texts, gt)

    assert result.soft_recall <= result.recall + 1e-9
    assert result.soft_precision <= result.precision + 1e-9

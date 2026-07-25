from __future__ import annotations

from bioreview_bench.validate.agreement import compute_label_agreement
from bioreview_bench.validate.human_subset import build_subset_manifest, sample_human_subset


def test_compute_label_agreement_reports_per_category():
    rows = [
        {
            "concern_id": "c1",
            "concern_text": "Concern one",
            "llm_category": "design_flaw",
            "human_category": "design_flaw",
            "llm_stance": "conceded",
            "human_stance": "conceded",
        },
        {
            "concern_id": "c2",
            "concern_text": "Concern two",
            "llm_category": "design_flaw",
            "human_category": "design_flaw",
            "llm_stance": "partial",
            "human_stance": "rebutted",
        },
        {
            "concern_id": "c3",
            "concern_text": "Concern three",
            "llm_category": "interpretation",
            "human_category": "interpretation",
            "llm_stance": "no_response",
            "human_stance": "no_response",
        },
    ]

    summary = compute_label_agreement(rows)

    assert summary.n_rows == 3
    assert round(summary.category_agreement, 3) == 1.0
    assert round(summary.stance_agreement, 3) == round(2 / 3, 3)
    assert len(summary.per_category) == 2
    assert summary.stance_disagreements[0]["concern_id"] == "c2"


def test_sample_human_subset_preserves_observed_strata():
    entries = []
    for split in ("val", "test"):
        for source in ("elife", "plos"):
            for response in (True, False):
                entries.append(
                    {
                        "id": f"{split}:{source}:{response}",
                        "benchmark_split": split,
                        "source": source,
                        "review_format": "journal",
                        "has_author_response": response,
                        "concerns": [{"concern_text": "x"}],
                    }
                )

    sampled = sample_human_subset(entries, n=8, seed=7)
    manifest = build_subset_manifest(sampled)

    assert len(sampled) == 8
    assert manifest["splits"] == {"test": 4, "val": 4}
    assert manifest["sources"] == {"elife": 4, "plos": 4}
    assert manifest["author_response"] == {"with_response": 4, "without_response": 4}


def test_sample_human_subset_small_n_uses_largest_strata_without_error():
    entries = [
        {
            "id": f"val:elife:{idx}",
            "benchmark_split": "val",
            "source": "elife",
            "review_format": "journal",
            "has_author_response": bool(idx % 2),
            "concerns": [{"concern_text": "x"}],
        }
        for idx in range(10)
    ]
    entries.extend(
        {
            "id": f"test:plos:{idx}",
            "benchmark_split": "test",
            "source": "plos",
            "review_format": "reviewed_preprint",
            "has_author_response": False,
            "concerns": [{"concern_text": "x"}],
        }
        for idx in range(3)
    )

    sampled = sample_human_subset(entries, n=2, seed=7)

    assert len(sampled) == 2

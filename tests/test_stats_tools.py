from __future__ import annotations

import json
from pathlib import Path

from bioreview_bench.stats import check_documentation, summarize_splits


def test_summarize_splits_small_fixture(tmp_path: Path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    train_entry = {
        "id": "a:1",
        "source": "alpha",
        "review_format": "journal",
        "has_author_response": True,
        "concerns": [
            {"category": "design_flaw", "severity": "major", "author_stance": "conceded"},
            {"category": "other", "severity": "minor", "author_stance": "partial"},
        ],
    }
    val_entry = {
        "id": "b:1",
        "source": "beta",
        "review_format": "reviewed_preprint",
        "has_author_response": False,
        "concerns": [
            {"category": "interpretation", "severity": "major", "author_stance": "no_response"},
        ],
    }

    (splits_dir / "train.jsonl").write_text(json.dumps(train_entry) + "\n", encoding="utf-8")
    (splits_dir / "val.jsonl").write_text(json.dumps(val_entry) + "\n", encoding="utf-8")
    (splits_dir / "test.jsonl").write_text("", encoding="utf-8")

    summary = summarize_splits(splits_dir)

    assert summary["total_articles"] == 2
    assert summary["total_concerns"] == 3
    assert summary["splits"]["train"]["articles"] == 1
    assert summary["splits"]["validation"]["concerns"] == 1
    assert summary["severity_distribution"]["major"]["count"] == 2
    assert summary["author_stance_distribution"]["no_response"]["count"] == 1


def test_repo_docs_match_generated_split_stats():
    splits_dir = Path("data/splits/v3")
    summary = summarize_splits(splits_dir)
    errors = check_documentation(summary, [Path("README.md"), Path("DATASHEET.md")])

    assert errors == []

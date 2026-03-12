"""Tests for HuggingFace dataset export pipeline.

Tests cover:
- hf_transforms: default, benchmark, concerns_flat, source_subset transforms
- hf_export: JSONL I/O, stats collection, full config export
- hf_card: DatasetCard YAML/Markdown generation
- hf_push: dry-run upload plan, staging lifecycle
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioreview_bench.collect.hf_transforms import (
    transform_benchmark,
    transform_concerns_flat,
    transform_default,
    transform_source_subset,
)
from bioreview_bench.collect.hf_export import (
    export_all_configs,
    _count_concerns,
    _load_jsonl,
    _save_jsonl,
    _source_distribution,
)
from bioreview_bench.collect.hf_card import generate_dataset_card


# ── Fixtures ──────────────────────────────────────────────────────


def _make_entry(
    eid: str = "elife:100",
    source: str = "elife",
    n_concerns: int = 2,
    has_response: bool = True,
) -> dict:
    """Create a realistic article entry dict for testing."""
    concerns = [
        {
            "concern_id": f"{eid}:R1C{i}",
            "reviewer_num": 1,
            "concern_text": f"Concern number {i} about methodology.",
            "category": "statistical_methodology",
            "severity": "major",
            "author_response_text": f"We addressed concern {i}." if has_response else "",
            "author_stance": "conceded" if has_response else "no_response",
            "evidence_of_change": True if has_response else None,
            "resolution_confidence": 0.85 if has_response else 0.0,
            "was_valid": True,
            "raised_by_multiple": False,
            "requires_figure_reading": False,
            "article_doi": f"10.7554/eLife.{eid.split(':')[1]}",
            "extraction_trace_id": "trace-test",
            "extraction_manifest_id": "em-v1.0",
            "source": source,
            "resolution": "conceded" if has_response else "no_response",
        }
        for i in range(1, n_concerns + 1)
    ]

    return {
        "id": eid,
        "source": source,
        "doi": f"10.7554/eLife.{eid.split(':')[1]}",
        "title": f"Study {eid}",
        "abstract": "This is the abstract.",
        "subjects": ["Neuroscience"],
        "published_date": "2025-06-15",
        "paper_text_sections": {"introduction": "Background...", "methods": "We used..."},
        "structured_references": [{"doi": "10.1234/ref1"}],
        "decision_letter_raw": "eLife Assessment: This is valuable...",
        "author_response_raw": "We thank the reviewers..." if has_response else "",
        "editorial_decision": "unknown",
        "review_format": "reviewed_preprint",
        "has_author_response": has_response,
        "revision_round": 1,
        "schema_version": "1.1",
        "extraction_manifest_id": "em-v1.0",
        "concerns": concerns,
    }


@pytest.fixture
def entries_mixed() -> list[dict]:
    """5 entries: 3 elife, 1 plos, 1 with zero concerns."""
    return [
        _make_entry("elife:100", source="elife", n_concerns=3),
        _make_entry("elife:200", source="elife", n_concerns=2),
        _make_entry("elife:300", source="elife", n_concerns=0),
        _make_entry("plos:10.1371/1", source="plos", n_concerns=4),
        _make_entry("elife:400", source="elife", n_concerns=1, has_response=False),
    ]


@pytest.fixture
def splits_dir(tmp_path: Path, entries_mixed: list[dict]) -> Path:
    """Create a splits directory with train/val/test JSONL files."""
    splits = tmp_path / "splits"
    splits.mkdir()

    # train: first 3 entries
    _write_jsonl(splits / "train.jsonl", entries_mixed[:3])
    # val: 4th entry
    _write_jsonl(splits / "val.jsonl", [entries_mixed[3]])
    # test: 5th entry
    _write_jsonl(splits / "test.jsonl", [entries_mixed[4:]])

    return splits


def _write_jsonl(path: Path, entries: list) -> None:
    """Write entries as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            # Flatten nested lists
            if isinstance(e, list):
                for item in e:
                    f.write(json.dumps(item, default=str) + "\n")
            else:
                f.write(json.dumps(e, default=str) + "\n")


# ── Transform: default ────────────────────────────────────────────


class TestTransformDefault:

    def test_preserves_all_fields(self):
        """Default transform preserves all article fields."""
        entry = _make_entry()
        result = transform_default([entry])
        assert len(result) == 1
        row = result[0]

        # Core fields
        assert row["id"] == "elife:100"
        assert row["source"] == "elife"
        assert row["doi"] == "10.7554/eLife.100"
        assert row["title"] == "Study elife:100"
        assert row["decision_letter_raw"] == "eLife Assessment: This is valuable..."
        assert row["author_response_raw"] == "We thank the reviewers..."
        assert row["paper_text_sections"]["introduction"] == "Background..."

    def test_concerns_as_dicts(self):
        """Concerns are plain dicts, not Pydantic models."""
        entry = _make_entry(n_concerns=2)
        result = transform_default([entry])
        concerns = result[0]["concerns"]
        assert len(concerns) == 2
        assert isinstance(concerns[0], dict)
        assert "concern_text" in concerns[0]

    def test_date_stringified(self):
        """Date values are converted to strings."""
        from datetime import date
        entry = _make_entry()
        entry["published_date"] = date(2025, 6, 15)
        result = transform_default([entry])
        assert result[0]["published_date"] == "2025-06-15"

    def test_none_date(self):
        """None date becomes empty string."""
        entry = _make_entry()
        entry["published_date"] = None
        result = transform_default([entry])
        assert result[0]["published_date"] == ""


# ── Transform: benchmark ──────────────────────────────────────────


class TestTransformBenchmark:

    def test_test_split_no_concerns(self):
        """Test split has concerns=[] to prevent leakage."""
        entry = _make_entry(n_concerns=5)
        result = transform_benchmark([entry], split="test")
        assert result[0]["concerns"] == []

    def test_train_has_simplified_concerns(self):
        """Train split has concerns with only text/category/severity."""
        entry = _make_entry(n_concerns=2)
        result = transform_benchmark([entry], split="train")
        concerns = result[0]["concerns"]
        assert len(concerns) == 2

        for c in concerns:
            assert set(c.keys()) == {"concern_text", "category", "severity"}
            assert c["concern_text"]  # Not empty
            assert c["category"] == "statistical_methodology"
            assert c["severity"] == "major"

    def test_validation_has_simplified_concerns(self):
        """Validation split also has simplified concerns."""
        entry = _make_entry(n_concerns=1)
        result = transform_benchmark([entry], split="validation")
        assert len(result[0]["concerns"]) == 1
        assert set(result[0]["concerns"][0].keys()) == {"concern_text", "category", "severity"}

    def test_no_leakage_fields(self):
        """Benchmark config excludes decision letter, author response, and other leakage fields."""
        entry = _make_entry()
        for split in ("train", "validation", "test"):
            result = transform_benchmark([entry], split=split)
            row = result[0]

            # These fields must NOT be present
            leakage_fields = [
                "decision_letter_raw",
                "author_response_raw",
                "editorial_decision",
                "review_format",
                "has_author_response",
                "revision_round",
                "extraction_manifest_id",
                "schema_version",
            ]
            for field in leakage_fields:
                assert field not in row, f"Leakage field '{field}' found in benchmark {split}"

    def test_benchmark_has_task_fields(self):
        """Benchmark config includes all task-relevant fields."""
        entry = _make_entry()
        result = transform_benchmark([entry], split="train")
        row = result[0]

        expected_fields = [
            "article_id", "doi", "source", "title", "abstract",
            "subjects", "published_date", "paper_text_sections",
            "structured_references", "concerns",
        ]
        for field in expected_fields:
            assert field in row, f"Missing task field: {field}"

    def test_benchmark_article_id_mapping(self):
        """'id' is renamed to 'article_id' in benchmark config."""
        entry = _make_entry("elife:42")
        result = transform_benchmark([entry], split="train")
        assert result[0]["article_id"] == "elife:42"
        assert "id" not in result[0]

    def test_zero_concerns_article(self):
        """Article with zero concerns produces row with empty concerns list."""
        entry = _make_entry(n_concerns=0)
        for split in ("train", "validation", "test"):
            result = transform_benchmark([entry], split=split)
            assert result[0]["concerns"] == []


# ── Transform: concerns_flat ──────────────────────────────────────


class TestTransformConcernsFlat:

    def test_row_count_matches_total_concerns(self):
        """Number of output rows = sum of concerns across all articles."""
        entries = [
            _make_entry("elife:1", n_concerns=3),
            _make_entry("elife:2", n_concerns=2),
            _make_entry("elife:3", n_concerns=0),
        ]
        result = transform_concerns_flat(entries)
        assert len(result) == 5  # 3 + 2 + 0

    def test_zero_concerns_produces_zero_rows(self):
        """Article with no concerns produces no rows in flat config."""
        entries = [_make_entry(n_concerns=0)]
        result = transform_concerns_flat(entries)
        assert len(result) == 0

    def test_article_context_included(self):
        """Each flat row includes article-level context fields."""
        entry = _make_entry("elife:100", n_concerns=1)
        result = transform_concerns_flat([entry])
        row = result[0]

        assert row["article_id"] == "elife:100"
        assert row["doi"] == "10.7554/eLife.100"
        assert row["source"] == "elife"
        assert row["title"] == "Study elife:100"
        assert "paper_text_sections" in row

    def test_concern_fields_included(self):
        """Each flat row includes concern-level fields."""
        entry = _make_entry("elife:100", n_concerns=1)
        result = transform_concerns_flat([entry])
        row = result[0]

        assert row["concern_id"] == "elife:100:R1C1"
        assert row["concern_text"] == "Concern number 1 about methodology."
        assert row["category"] == "statistical_methodology"
        assert row["severity"] == "major"
        assert row["author_stance"] == "conceded"
        assert row["resolution_confidence"] == 0.85

    def test_no_source_duplication(self):
        """source field comes from article, not duplicated from concern."""
        entry = _make_entry("elife:100", source="elife", n_concerns=1)
        result = transform_concerns_flat([entry])
        row = result[0]
        assert row["source"] == "elife"


# ── Transform: source_subset ──────────────────────────────────────


class TestTransformSourceSubset:

    def test_filters_by_source(self, entries_mixed):
        """Only entries matching the source are included."""
        result = transform_source_subset(entries_mixed, "elife")
        assert all(r["source"] == "elife" for r in result)
        assert len(result) == 4  # 3 elife + 1 elife with no response

    def test_plos_subset(self, entries_mixed):
        """PLOS subset contains only PLOS entries."""
        result = transform_source_subset(entries_mixed, "plos")
        assert len(result) == 1
        assert result[0]["source"] == "plos"

    def test_empty_source(self, entries_mixed):
        """Non-existent source returns empty list."""
        result = transform_source_subset(entries_mixed, "f1000")
        assert result == []

    def test_applies_default_transform(self, entries_mixed):
        """Source subset applies default transform (dates stringified, etc.)."""
        result = transform_source_subset(entries_mixed, "elife")
        assert isinstance(result[0]["published_date"], str)
        assert isinstance(result[0]["concerns"], list)


# ── JSONL I/O ─────────────────────────────────────────────────────


class TestJsonlIO:

    def test_save_load_roundtrip(self, tmp_path):
        """Entries survive save → load cycle."""
        entries = [_make_entry(f"elife:{i}") for i in range(3)]
        path = tmp_path / "test.jsonl"
        _save_jsonl(entries, path)
        loaded = _load_jsonl(path)
        assert len(loaded) == 3
        assert loaded[0]["id"] == "elife:0"

    def test_load_missing_file(self, tmp_path):
        """Missing file returns empty list."""
        loaded = _load_jsonl(tmp_path / "nonexistent.jsonl")
        assert loaded == []

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_jsonl creates parent directories."""
        path = tmp_path / "deep" / "nested" / "dir" / "test.jsonl"
        _save_jsonl([{"id": "x"}], path)
        assert path.exists()

    def test_malformed_json_skipped(self, tmp_path):
        """Malformed JSON lines are skipped with logging."""
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"id": "good"}\n'
            'this is not json\n'
            '{"id": "also_good"}\n'
        )
        loaded = _load_jsonl(path)
        assert len(loaded) == 2

    def test_empty_lines_skipped(self, tmp_path):
        """Blank lines are ignored."""
        path = tmp_path / "sparse.jsonl"
        path.write_text('{"id": "a"}\n\n\n{"id": "b"}\n')
        loaded = _load_jsonl(path)
        assert len(loaded) == 2


# ── Stats helpers ─────────────────────────────────────────────────


class TestStatsHelpers:

    def test_count_concerns_nested(self):
        """Count concerns in nested article format."""
        entries = [
            {"concerns": [{"concern_id": "1"}, {"concern_id": "2"}]},
            {"concerns": [{"concern_id": "3"}]},
            {"concerns": []},
        ]
        assert _count_concerns(entries) == 3

    def test_count_concerns_flat(self):
        """Count concerns in flat format (one row per concern)."""
        entries = [
            {"concern_text": "Issue 1", "concern_id": "c1"},
            {"concern_text": "Issue 2", "concern_id": "c2"},
        ]
        assert _count_concerns(entries) == 2

    def test_count_concerns_empty(self):
        """Empty list returns 0."""
        assert _count_concerns([]) == 0

    def test_source_distribution(self):
        """Correct source counts."""
        entries = [
            {"source": "elife"},
            {"source": "elife"},
            {"source": "plos"},
        ]
        dist = _source_distribution(entries)
        assert dist == {"elife": 2, "plos": 1}

    def test_source_distribution_fallback(self):
        """Falls back to article_id prefix when source missing."""
        entries = [{"article_id": "elife:100"}]
        dist = _source_distribution(entries)
        assert dist == {"elife": 1}


# ── Full export ───────────────────────────────────────────────────


class TestExportAllConfigs:

    @pytest.fixture
    def realistic_splits_dir(self, tmp_path: Path) -> Path:
        """Create splits directory with realistic multi-source data."""
        splits = tmp_path / "splits"
        splits.mkdir()

        train = [
            _make_entry("elife:1", source="elife", n_concerns=3),
            _make_entry("elife:2", source="elife", n_concerns=2),
            _make_entry("plos:1", source="plos", n_concerns=4),
        ]
        val = [
            _make_entry("elife:3", source="elife", n_concerns=1),
        ]
        test = [
            _make_entry("elife:4", source="elife", n_concerns=2),
            _make_entry("plos:2", source="plos", n_concerns=3),
        ]

        _write_jsonl(splits / "train.jsonl", train)
        _write_jsonl(splits / "val.jsonl", val)
        _write_jsonl(splits / "test.jsonl", test)

        return splits

    def test_generates_all_configs(self, realistic_splits_dir, tmp_path):
        """Export creates all 6 config directories."""
        output = tmp_path / "output"
        stats = export_all_configs(realistic_splits_dir, output)

        assert "error" not in stats
        expected_configs = {"default", "benchmark", "concerns_flat", "elife", "plos", "f1000", "peerj", "nature"}
        assert set(stats["configs"].keys()) == expected_configs

    def test_generates_all_splits_per_config(self, realistic_splits_dir, tmp_path):
        """Each config has train/validation/test JSONL files."""
        output = tmp_path / "output"
        export_all_configs(realistic_splits_dir, output)

        for config in ["default", "benchmark", "concerns_flat", "elife", "plos", "f1000", "peerj", "nature"]:
            for split in ["train", "validation", "test"]:
                path = output / config / f"{split}.jsonl"
                assert path.exists(), f"Missing: {config}/{split}.jsonl"

    def test_total_articles_correct(self, realistic_splits_dir, tmp_path):
        """Total articles = sum across splits."""
        output = tmp_path / "output"
        stats = export_all_configs(realistic_splits_dir, output)
        assert stats["total_articles"] == 6  # 3 + 1 + 2

    def test_default_config_row_counts(self, realistic_splits_dir, tmp_path):
        """Default config preserves all rows."""
        output = tmp_path / "output"
        stats = export_all_configs(realistic_splits_dir, output)

        default_splits = stats["configs"]["default"]["splits"]
        assert default_splits["train"]["num_rows"] == 3
        assert default_splits["validation"]["num_rows"] == 1
        assert default_splits["test"]["num_rows"] == 2

    def test_benchmark_test_empty_concerns(self, realistic_splits_dir, tmp_path):
        """Benchmark test split has empty concerns for every row."""
        output = tmp_path / "output"
        export_all_configs(realistic_splits_dir, output)

        test_path = output / "benchmark" / "test.jsonl"
        with open(test_path) as f:
            for line in f:
                row = json.loads(line)
                assert row["concerns"] == [], "Test split should have no concerns"

    def test_benchmark_train_has_concerns(self, realistic_splits_dir, tmp_path):
        """Benchmark train split has simplified concerns."""
        output = tmp_path / "output"
        export_all_configs(realistic_splits_dir, output)

        train_path = output / "benchmark" / "train.jsonl"
        with open(train_path) as f:
            for line in f:
                row = json.loads(line)
                if row["concerns"]:
                    c = row["concerns"][0]
                    assert set(c.keys()) == {"concern_text", "category", "severity"}

    def test_concerns_flat_total_rows(self, realistic_splits_dir, tmp_path):
        """concerns_flat total rows = sum of all concerns."""
        output = tmp_path / "output"
        stats = export_all_configs(realistic_splits_dir, output)

        flat_total = stats["configs"]["concerns_flat"]["total_rows"]
        # train: 3+2+4=9, val: 1, test: 2+3=5 → total 15
        assert flat_total == 15

    def test_source_subset_filters(self, realistic_splits_dir, tmp_path):
        """Source subsets only contain entries from that source."""
        output = tmp_path / "output"
        export_all_configs(realistic_splits_dir, output)

        # elife config should not have plos entries
        for split in ["train", "validation", "test"]:
            path = output / "elife" / f"{split}.jsonl"
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    assert row["source"] == "elife"

    def test_f1000_subset_empty(self, realistic_splits_dir, tmp_path):
        """f1000 config is empty when no f1000 entries exist."""
        output = tmp_path / "output"
        stats = export_all_configs(realistic_splits_dir, output)

        assert stats["configs"]["f1000"]["total_rows"] == 0

    def test_no_data_returns_error(self, tmp_path):
        """Empty splits directory returns error dict."""
        empty_splits = tmp_path / "empty_splits"
        empty_splits.mkdir()
        output = tmp_path / "output"

        stats = export_all_configs(empty_splits, output)
        assert "error" in stats

    def test_no_decision_letter_in_benchmark(self, realistic_splits_dir, tmp_path):
        """Benchmark config rows never contain decision_letter_raw."""
        output = tmp_path / "output"
        export_all_configs(realistic_splits_dir, output)

        for split in ["train", "validation", "test"]:
            path = output / "benchmark" / f"{split}.jsonl"
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    assert "decision_letter_raw" not in row
                    assert "author_response_raw" not in row

    def test_stats_concern_counts(self, realistic_splits_dir, tmp_path):
        """Stats correctly count concerns per config."""
        output = tmp_path / "output"
        stats = export_all_configs(realistic_splits_dir, output)

        default_concerns = stats["configs"]["default"]["total_concerns"]
        assert default_concerns == 15  # 3+2+4+1+2+3

        flat_concerns = stats["configs"]["concerns_flat"]["total_concerns"]
        assert flat_concerns == 15  # Same (each row is one concern)

    def test_source_distribution_in_stats(self, realistic_splits_dir, tmp_path):
        """Stats include per-split source distributions."""
        output = tmp_path / "output"
        stats = export_all_configs(realistic_splits_dir, output)

        train_dist = stats["configs"]["default"]["splits"]["train"]["source_distribution"]
        assert train_dist["elife"] == 2
        assert train_dist["plos"] == 1


# ── DatasetCard ───────────────────────────────────────────────────


class TestDatasetCard:

    @pytest.fixture
    def sample_stats(self) -> dict:
        """Minimal stats dict for card generation."""
        return {
            "total_articles": 100,
            "configs": {
                "default": {
                    "total_rows": 100,
                    "total_concerns": 500,
                    "splits": {
                        "train": {
                            "num_rows": 70,
                            "num_concerns": 350,
                            "source_distribution": {"elife": 50, "plos": 20},
                        },
                        "validation": {
                            "num_rows": 15,
                            "num_concerns": 75,
                            "source_distribution": {"elife": 10, "plos": 5},
                        },
                        "test": {
                            "num_rows": 15,
                            "num_concerns": 75,
                            "source_distribution": {"elife": 10, "plos": 5},
                        },
                    },
                },
                "benchmark": {"total_rows": 100, "total_concerns": 350, "splits": {
                    "train": {"num_rows": 70, "num_concerns": 350},
                    "validation": {"num_rows": 15, "num_concerns": 75},
                    "test": {"num_rows": 15, "num_concerns": 0},
                }},
                "concerns_flat": {"total_rows": 500, "total_concerns": 500, "splits": {
                    "train": {"num_rows": 350, "num_concerns": 350},
                    "validation": {"num_rows": 75, "num_concerns": 75},
                    "test": {"num_rows": 75, "num_concerns": 75},
                }},
                "elife": {"total_rows": 70, "total_concerns": 350, "splits": {
                    "train": {"num_rows": 50, "num_concerns": 250},
                    "validation": {"num_rows": 10, "num_concerns": 50},
                    "test": {"num_rows": 10, "num_concerns": 50},
                }},
                "plos": {"total_rows": 30, "total_concerns": 150, "splits": {
                    "train": {"num_rows": 20, "num_concerns": 100},
                    "validation": {"num_rows": 5, "num_concerns": 25},
                    "test": {"num_rows": 5, "num_concerns": 25},
                }},
                "f1000": {"total_rows": 0, "total_concerns": 0, "splits": {
                    "train": {"num_rows": 0, "num_concerns": 0},
                    "validation": {"num_rows": 0, "num_concerns": 0},
                    "test": {"num_rows": 0, "num_concerns": 0},
                }},
            },
        }

    def test_has_yaml_front_matter(self, sample_stats):
        """Card starts with YAML front matter fences."""
        card = generate_dataset_card(sample_stats)
        assert card.startswith("---\n")
        # Find closing fence
        second_fence = card.index("---", 4)
        assert second_fence > 4

    def test_yaml_contains_configs(self, sample_stats):
        """YAML block declares all 6 configs."""
        card = generate_dataset_card(sample_stats)
        for cfg in ["default", "benchmark", "concerns_flat", "elife", "plos", "f1000"]:
            assert f"config_name: {cfg}" in card

    def test_yaml_default_flag(self, sample_stats):
        """default config is marked as default: true."""
        card = generate_dataset_card(sample_stats)
        assert "default: true" in card

    def test_yaml_data_files_paths(self, sample_stats):
        """Data files paths follow expected pattern."""
        card = generate_dataset_card(sample_stats)
        assert 'path: "data/default/train.jsonl"' in card
        assert 'path: "data/benchmark/test.jsonl"' in card

    def test_body_contains_sections(self, sample_stats):
        """Card body contains all expected sections."""
        card = generate_dataset_card(sample_stats)
        assert "# BioReview-Bench" in card
        assert "## Configs" in card
        assert "## Quick start" in card
        assert "## Schema" in card
        assert "## License" in card
        assert "## Citation" in card

    def test_dynamic_stats_in_body(self, sample_stats):
        """Card body includes dynamic stats from the data."""
        card = generate_dataset_card(sample_stats)
        assert "100 articles" in card
        assert "500 reviewer concerns" in card

    def test_source_distribution_in_overview(self, sample_stats):
        """Overview section lists source distribution."""
        card = generate_dataset_card(sample_stats)
        assert "elife" in card
        assert "plos" in card

    def test_license_is_source_specific(self, sample_stats):
        """Card reflects source-specific licensing instead of blanket CC-BY."""
        card = generate_dataset_card(sample_stats)
        assert "license: other" in card
        assert "source content follows per-source terms" in card
        assert "LICENSE_MATRIX.md" in card

    def test_size_category_small(self, sample_stats):
        """Size category is n<1K for 100 articles."""
        card = generate_dataset_card(sample_stats)
        assert "n<1K" in card


# ── Push (dry run) ────────────────────────────────────────────────


class TestPushDryRun:

    @pytest.fixture
    def data_dir_with_splits(self, tmp_path: Path) -> Path:
        """Create complete data directory structure for push testing."""
        data = tmp_path / "data"

        # v2 splits
        splits_v2 = data / "splits" / "v2"
        splits_v2.mkdir(parents=True)

        entries = [
            _make_entry("elife:1", source="elife", n_concerns=2),
            _make_entry("elife:2", source="elife", n_concerns=1),
        ]
        _write_jsonl(splits_v2 / "train.jsonl", entries)
        _write_jsonl(splits_v2 / "val.jsonl", [entries[0]])
        _write_jsonl(splits_v2 / "test.jsonl", [entries[1]])

        # Manifests
        manifests = data / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "em-v1.0.json").write_text('{"version": "1.0"}')

        # Frozen test IDs
        splits_root = data / "splits"
        (splits_root / "test_ids_frozen_v2.json").write_text('{"ids": ["elife:2"]}')

        # Split metadata
        (splits_v2 / "split_meta_v2.json").write_text('{"seed": 42}')

        return data

    def test_dry_run_returns_upload_plan(self, data_dir_with_splits):
        """Dry run generates upload plan without uploading."""
        from bioreview_bench.collect.hf_push import push_to_hub

        result = push_to_hub(data_dir=data_dir_with_splits, dry_run=True)

        assert result["dry_run"] is True
        assert len(result["uploaded"]) > 0
        assert "stats" in result

    def test_dry_run_includes_readme(self, data_dir_with_splits):
        """Upload plan includes README.md."""
        from bioreview_bench.collect.hf_push import push_to_hub

        result = push_to_hub(data_dir=data_dir_with_splits, dry_run=True)
        assert "README.md" in result["uploaded"]

    def test_dry_run_includes_all_configs(self, data_dir_with_splits):
        """Upload plan includes JSONL files for all 6 configs."""
        from bioreview_bench.collect.hf_push import push_to_hub

        result = push_to_hub(data_dir=data_dir_with_splits, dry_run=True)
        uploaded = result["uploaded"]

        for config in ["default", "benchmark", "concerns_flat", "elife"]:
            config_files = [f for f in uploaded if f"/{config}/" in f]
            assert len(config_files) == 3, f"Config {config} should have 3 splits"

    def test_dry_run_includes_auxiliary_files(self, data_dir_with_splits):
        """Upload plan includes manifests and metadata."""
        from bioreview_bench.collect.hf_push import push_to_hub

        result = push_to_hub(data_dir=data_dir_with_splits, dry_run=True)
        uploaded_str = " ".join(result["uploaded"])

        assert "manifests/" in uploaded_str
        assert "metadata/test_ids_frozen_v2.json" in uploaded_str
        assert "metadata/split_meta_v2.json" in uploaded_str

    def test_dry_run_preserves_staging_dir(self, data_dir_with_splits):
        """Dry run preserves staging directory for inspection."""
        from bioreview_bench.collect.hf_push import push_to_hub

        result = push_to_hub(data_dir=data_dir_with_splits, dry_run=True)
        staging = Path(result["staging_dir"])
        assert staging.exists()
        assert (staging / "README.md").exists()

    def test_empty_splits_returns_error(self, tmp_path):
        """Push with no data returns error without crashing."""
        from bioreview_bench.collect.hf_push import push_to_hub

        data = tmp_path / "empty_data"
        (data / "splits" / "v2").mkdir(parents=True)

        result = push_to_hub(data_dir=data, dry_run=True)
        assert result["uploaded"] == []
        assert "error" in result["stats"]

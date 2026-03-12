"""Tests for the baseline reviewer module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bioreview_bench.baseline.reviewer import BaselineReviewer, _try_parse_string_array
from bioreview_bench.baseline.runner import (
    _get_id,
    _get_pricing,
    estimate_cost,
    load_existing_ids,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_entry() -> dict:
    """Minimal paper entry for testing."""
    return {
        "id": "elife:12345",
        "title": "A Novel Method for Gene Expression Analysis",
        "abstract": "We present a new method for analyzing gene expression data "
                    "using machine learning approaches. Our method achieves state-of-the-art "
                    "performance on benchmark datasets.",
        "paper_text_sections": {
            "introduction": "Gene expression analysis is fundamental to biology. "
                           "Previous methods have limitations.",
            "methods": "We used a random forest classifier with 500 trees. "
                      "Data was collected from GEO database.",
            "results": "Our method achieved 95% accuracy on the test set. "
                      "Figure 1 shows the comparison.",
            "discussion": "Our results demonstrate the superiority of ML approaches. "
                         "Future work will explore deep learning.",
        },
        "concerns": [{"concern_text": "dummy concern"}],
    }


@pytest.fixture
def large_entry() -> dict:
    """Entry with very long sections for truncation testing."""
    return {
        "id": "elife:99999",
        "title": "Large Paper",
        "abstract": "Abstract " * 100,
        "paper_text_sections": {
            "introduction": "Intro text. " * 5000,
            "methods": "Method text. " * 5000,
            "results": "Result text. " * 5000,
        },
    }


@pytest.fixture
def reviewer() -> BaselineReviewer:
    """BaselineReviewer instance (no API calls — mock required)."""
    return BaselineReviewer(
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        max_input_chars=80_000,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_concerns / _try_parse_string_array
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseConcerns:
    """Test JSON parsing from LLM output."""

    def test_clean_array(self):
        text = '["concern one", "concern two", "concern three"]'
        assert BaselineReviewer._parse_concerns(text) == [
            "concern one", "concern two", "concern three"
        ]

    def test_fenced_json(self):
        text = 'Here are the concerns:\n```json\n["c1", "c2"]\n```\nDone.'
        assert BaselineReviewer._parse_concerns(text) == ["c1", "c2"]

    def test_fenced_no_language(self):
        text = '```\n["c1", "c2"]\n```'
        assert BaselineReviewer._parse_concerns(text) == ["c1", "c2"]

    def test_trailing_text(self):
        text = '["concern A", "concern B"]\n\nThese are the main issues I identified.'
        assert BaselineReviewer._parse_concerns(text) == ["concern A", "concern B"]

    def test_embedded_in_text(self):
        text = 'After reviewing the paper, I found:\n["issue 1", "issue 2"]\nPlease address.'
        result = BaselineReviewer._parse_concerns(text)
        assert result == ["issue 1", "issue 2"]

    def test_empty_array(self):
        assert BaselineReviewer._parse_concerns("[]") == []

    def test_no_json(self):
        assert BaselineReviewer._parse_concerns("No concerns found.") == []

    def test_empty_strings_filtered(self):
        text = '["concern", "", "  ", "another"]'
        assert BaselineReviewer._parse_concerns(text) == ["concern", "another"]

    def test_dict_format_accepted(self):
        text = '[{"text": "concern 1"}, {"text": "concern 2"}]'
        assert BaselineReviewer._parse_concerns(text) == ["concern 1", "concern 2"]

    def test_dict_concern_text_key(self):
        text = '[{"concern_text": "issue A"}]'
        assert BaselineReviewer._parse_concerns(text) == ["issue A"]

    def test_first_bracket_to_last_bracket(self):
        """Greedy fallback: first [ to last ] when bracket match is incomplete."""
        # Non-greedy [.*?] would match ["a"] but the full array spans to the last ]
        text = 'Here: ["a", "b", "c"]  \n\nI hope this helps.'
        result = BaselineReviewer._parse_concerns(text)
        assert result == ["a", "b", "c"]

    def test_multiline_json_array(self):
        """JSON array split across lines (common LLM output)."""
        text = '[\n  "concern one",\n  "concern two"\n]'
        result = BaselineReviewer._parse_concerns(text)
        assert result == ["concern one", "concern two"]


class TestTryParseStringArray:
    """Test the helper directly."""

    def test_valid_array(self):
        assert _try_parse_string_array('["a", "b"]') == ["a", "b"]

    def test_not_a_list(self):
        assert _try_parse_string_array('{"key": "value"}') is None

    def test_invalid_json(self):
        assert _try_parse_string_array("not json") is None

    def test_mixed_types(self):
        text = '["string", {"text": "dict concern"}, 42]'
        result = _try_parse_string_array(text)
        assert result == ["string", "dict concern"]


# ═══════════════════════════════════════════════════════════════════════════════
# _format_paper_input
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatPaperInput:
    """Test paper text formatting and truncation."""

    def test_basic_formatting(self, reviewer, sample_entry):
        result = reviewer._format_paper_input(sample_entry)
        assert "# A Novel Method" in result
        assert "## Abstract" in result
        assert "## Methods" in result or "## methods" in result.lower()
        assert "## Results" in result or "## results" in result.lower()

    def test_truncation_respects_limit(self, reviewer, large_entry):
        result = reviewer._format_paper_input(large_entry)
        assert len(result) <= reviewer.max_input_chars

    def test_small_limit(self, large_entry):
        reviewer = BaselineReviewer(max_input_chars=500)
        result = reviewer._format_paper_input(large_entry)
        assert len(result) <= 500

    def test_empty_entry(self, reviewer):
        result = reviewer._format_paper_input({})
        assert result == ""

    def test_title_only(self, reviewer):
        result = reviewer._format_paper_input({"title": "Test Title"})
        assert "# Test Title" in result

    def test_sections_key_fallback(self, reviewer):
        """Accept 'sections' key (from to_task_input) as well as 'paper_text_sections'."""
        entry = {
            "title": "Test",
            "abstract": "Abstract.",
            "sections": {"methods": "Method text."},
        }
        result = reviewer._format_paper_input(entry)
        assert "Method text" in result

    def test_methods_before_discussion(self, reviewer, sample_entry):
        """Methods should appear before discussion in the output."""
        result = reviewer._format_paper_input(sample_entry)
        methods_pos = result.lower().find("method")
        discussion_pos = result.lower().find("discussion")
        if methods_pos != -1 and discussion_pos != -1:
            assert methods_pos < discussion_pos


class TestPrioritizeSections:
    """Test section ordering logic."""

    def test_methods_first(self):
        sections = {
            "discussion": "disc",
            "methods": "meth",
            "introduction": "intro",
        }
        ordered = BaselineReviewer._prioritize_sections(sections)
        names = [n for n, _ in ordered]
        assert names.index("methods") < names.index("discussion")
        assert names.index("methods") < names.index("introduction")

    def test_all_sections_included(self):
        sections = {"a": "1", "b": "2", "c": "3"}
        ordered = BaselineReviewer._prioritize_sections(sections)
        assert len(ordered) == 3

    def test_case_insensitive_match(self):
        sections = {
            "Materials and Methods": "meth",
            "Results": "res",
        }
        ordered = BaselineReviewer._prioritize_sections(sections)
        names = [n for n, _ in ordered]
        assert names[0] == "Materials and Methods"


# ═══════════════════════════════════════════════════════════════════════════════
# review_article (mocked API)
# ═══════════════════════════════════════════════════════════════════════════════

class TestReviewArticle:
    """Test review_article with mocked LLM calls."""

    def test_anthropic_provider(self, sample_entry):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='["concern A", "concern B"]')]

        reviewer = BaselineReviewer(model="claude-haiku-4-5-20251001", provider="anthropic")
        with patch.object(reviewer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_response
            concerns = reviewer.review_article(sample_entry)

        assert concerns == ["concern A", "concern B"]

    def test_openai_provider(self, sample_entry):
        mock_message = MagicMock()
        mock_message.content = '["concern X", "concern Y"]'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        reviewer = BaselineReviewer(model="gpt-4o-mini", provider="openai")
        with patch.object(reviewer, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_response
            concerns = reviewer.review_article(sample_entry)

        assert concerns == ["concern X", "concern Y"]

    def test_empty_input_returns_empty(self):
        reviewer = BaselineReviewer()
        concerns = reviewer.review_article({})
        assert concerns == []

    def test_output_format_matches_benchmark(self, sample_entry):
        """Output should be plain string list matching run_benchmark.py expectations."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='["The study lacks controls."]')]

        reviewer = BaselineReviewer(provider="anthropic")
        with patch.object(reviewer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_response
            concerns = reviewer.review_article(sample_entry)

        # Must be list[str] (not list[dict])
        assert all(isinstance(c, str) for c in concerns)

    def test_unsupported_provider(self):
        reviewer = BaselineReviewer(provider="unsupported")
        with pytest.raises(ValueError, match="Unsupported provider"):
            reviewer._get_client()


# ═══════════════════════════════════════════════════════════════════════════════
# Runner: load_existing_ids
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadExistingIds:
    """Test resume logic."""

    def test_nonexistent_file(self, tmp_path):
        assert load_existing_ids(tmp_path / "nope.jsonl") == set()

    def test_loads_ids(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text(
            '{"article_id": "elife:1", "concerns": ["c1"]}\n'
            '{"article_id": "elife:2", "concerns": ["c2"]}\n'
        )
        assert load_existing_ids(p) == {"elife:1", "elife:2"}

    def test_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text(
            '{"article_id": "elife:1", "concerns": []}\n'
            'not json\n'
            '{"article_id": "elife:3", "concerns": []}\n'
        )
        assert load_existing_ids(p) == {"elife:1", "elife:3"}

    def test_skips_empty_ids(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text('{"article_id": "", "concerns": []}\n')
        assert load_existing_ids(p) == set()


# ═══════════════════════════════════════════════════════════════════════════════
# Runner: estimate_cost
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimateCost:
    """Test cost estimation."""

    def test_basic_estimate(self, sample_entry):
        cost = estimate_cost([sample_entry], "claude-haiku-4-5-20251001")
        assert cost["n_articles"] == 1
        assert cost["est_input_tokens"] > 0
        assert cost["est_output_tokens"] > 0
        assert cost["est_cost_usd"] >= 0

    def test_empty_articles(self):
        cost = estimate_cost([], "claude-haiku-4-5-20251001")
        assert cost["n_articles"] == 0
        assert cost["est_cost_usd"] == 0

    def test_openai_cheaper(self, sample_entry):
        cost_anthropic = estimate_cost([sample_entry], "claude-haiku-4-5-20251001", "anthropic")
        cost_openai = estimate_cost([sample_entry], "gpt-4o-mini", "openai")
        assert cost_openai["est_cost_usd"] <= cost_anthropic["est_cost_usd"]

    def test_caps_at_max_input_chars(self, large_entry):
        cost = estimate_cost([large_entry], "claude-haiku-4-5-20251001", max_input_chars=1000)
        # With 1000 char cap, input tokens should be ~250
        assert cost["est_input_tokens"] <= 300


class TestGetPricing:
    """Test pricing lookup."""

    def test_known_model(self):
        p = _get_pricing("claude-haiku-4-5-20251001", "anthropic")
        assert p["input"] == 0.80
        assert p["output"] == 4.00

    def test_openai_model(self):
        p = _get_pricing("gpt-4o-mini", "openai")
        assert p["input"] == 0.15

    def test_unknown_model_fallback(self):
        p = _get_pricing("unknown-model", "anthropic")
        assert "input" in p
        assert "output" in p

    def test_unknown_openai_fallback(self):
        p = _get_pricing("unknown-model", "openai")
        assert p["input"] == 0.50


class TestGetId:
    """Test article ID extraction."""

    def test_id_key(self):
        assert _get_id({"id": "elife:123"}) == "elife:123"

    def test_article_id_key(self):
        assert _get_id({"article_id": "elife:456"}) == "elife:456"

    def test_id_preferred_over_article_id(self):
        assert _get_id({"id": "a", "article_id": "b"}) == "a"

    def test_empty_dict(self):
        assert _get_id({}) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Runner: run_baseline (integration-style with mocked reviewer)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunBaseline:
    """Test the concurrent runner with mocked reviewer."""

    def test_basic_run(self, tmp_path, sample_entry):
        from bioreview_bench.baseline.runner import run_baseline

        mock_reviewer = MagicMock()
        mock_reviewer.review_article.return_value = ["concern 1", "concern 2"]

        output_path = tmp_path / "output.jsonl"
        stats = run_baseline(
            reviewer=mock_reviewer,
            articles=[sample_entry],
            output_path=output_path,
            concurrency=1,
        )

        assert stats["processed"] == 1
        assert stats["failed"] == 0
        assert stats["total_concerns"] == 2

        # Verify JSONL output
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["article_id"] == "elife:12345"
        assert row["concerns"] == ["concern 1", "concern 2"]

    def test_resume_skips_existing(self, tmp_path, sample_entry):
        from bioreview_bench.baseline.runner import run_baseline

        mock_reviewer = MagicMock()
        mock_reviewer.review_article.return_value = ["c"]

        output_path = tmp_path / "output.jsonl"
        stats = run_baseline(
            reviewer=mock_reviewer,
            articles=[sample_entry],
            output_path=output_path,
            concurrency=1,
            resume_ids={"elife:12345"},
        )

        assert stats["skipped"] == 1
        assert stats["processed"] == 0
        mock_reviewer.review_article.assert_not_called()

    def test_failed_article_counted(self, tmp_path, sample_entry):
        from bioreview_bench.baseline.runner import run_baseline

        mock_reviewer = MagicMock()
        mock_reviewer.review_article.side_effect = RuntimeError("API error")

        output_path = tmp_path / "output.jsonl"
        stats = run_baseline(
            reviewer=mock_reviewer,
            articles=[sample_entry],
            output_path=output_path,
            concurrency=1,
        )

        assert stats["failed"] == 1
        assert stats["processed"] == 0

    def test_multiple_articles(self, tmp_path):
        from bioreview_bench.baseline.runner import run_baseline

        articles = [
            {"id": f"elife:{i}", "title": f"Paper {i}", "abstract": "abs", "concerns": [{"text": "c"}]}
            for i in range(5)
        ]
        mock_reviewer = MagicMock()
        mock_reviewer.review_article.return_value = ["c1"]

        output_path = tmp_path / "output.jsonl"
        stats = run_baseline(
            reviewer=mock_reviewer,
            articles=articles,
            output_path=output_path,
            concurrency=3,
        )

        assert stats["processed"] == 5
        assert stats["total_concerns"] == 5

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLI:
    """Test the Click CLI commands."""

    def test_cli_help(self):
        from click.testing import CliRunner
        from bioreview_bench.scripts.run_baseline import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--split" in result.output
        assert "--model" in result.output
        assert "--dry-run" in result.output

    def test_cli_missing_split_file(self, tmp_path):
        from click.testing import CliRunner
        from bioreview_bench.scripts.run_baseline import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "--splits-dir", str(tmp_path),
            "--split", "val",
        ])
        assert result.exit_code != 0

    def test_cli_dry_run(self, tmp_path):
        """Dry run should print cost estimate and exit without API calls."""
        from click.testing import CliRunner
        from bioreview_bench.scripts.run_baseline import main

        # Create a minimal split file
        split_file = tmp_path / "val.jsonl"
        entry = {
            "id": "elife:1",
            "title": "Test Paper",
            "abstract": "Abstract.",
            "concerns": [{"concern_text": "test"}],
        }
        split_file.write_text(json.dumps(entry) + "\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "--splits-dir", str(tmp_path),
            "--split", "val",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "Cost estimate" in result.output

    def test_cli_default_splits_dir_tracks_v3(self):
        from bioreview_bench.scripts.run_baseline import _DEFAULT_SPLITS_DIR

        assert _DEFAULT_SPLITS_DIR == Path("data/splits/v3")

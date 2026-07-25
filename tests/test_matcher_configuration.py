"""Tests for reproducible matcher configuration and explicit fallback policy."""

from __future__ import annotations

import numpy as np
import pytest

from bioreview_bench.evaluate import metrics
from bioreview_bench.evaluate.metrics import (
    DEFAULT_EMBEDDING_MODEL,
    ConcernMatcher,
    EmbeddingModelError,
)
from bioreview_bench.evaluate.runner import aggregate_results, evaluate_articles


def _gt(text: str) -> dict:
    return {
        "concern_text": text,
        "category": "statistical_methodology",
        "severity": "major",
        "requires_figure_reading": False,
    }


class _ConstantEmbeddingModel:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        del normalize_embeddings
        return np.array(
            [[self.value, 0.0] for _ in texts],
            dtype=np.float32,
        )


class _BrokenEmbeddingModel:
    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        del texts, normalize_embeddings
        raise OSError("synthetic encoding failure")


class _FailsAfterFirstPairModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        del normalize_embeddings
        self.calls += 1
        if self.calls > 2:
            raise OSError("synthetic late encoding failure")
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_default_embedding_model_id_is_preserved() -> None:
    matcher = ConcernMatcher(use_embedding=False)
    assert matcher.embedding_model == "allenai/specter2_base"
    assert DEFAULT_EMBEDDING_MODEL == "allenai/specter2_base"


def test_configured_embedding_model_is_loaded_and_recorded(monkeypatch) -> None:
    requested: list[str] = []

    def fake_loader(model_id: str, revision: str | None = None) -> object:
        requested.append(f"{model_id}@{revision or 'default'}")
        return _ConstantEmbeddingModel()

    monkeypatch.setattr(metrics, "_get_embed_model", fake_loader)

    with pytest.warns(UserWarning, match="model-specific"):
        matcher = ConcernMatcher(embedding_model="org/custom-review-encoder")
    result = matcher.score_article(
        ["The analysis lacks an appropriate negative control."],
        [_gt("The experiment lacks an appropriate negative control.")],
    )

    assert requested == ["org/custom-review-encoder@default"]
    assert result.matching_method == "embedding"
    assert result.embedding_model == "org/custom-review-encoder"
    assert result.threshold == pytest.approx(0.65)
    assert result.effective_threshold == pytest.approx(0.65)


def test_embedding_load_failure_is_fail_closed_by_default(monkeypatch) -> None:
    def unavailable(model_id: str, revision: str | None = None) -> object:
        del revision
        raise EmbeddingModelError(f"{model_id} unavailable")

    monkeypatch.setattr(metrics, "_get_embed_model", unavailable)
    matcher = ConcernMatcher()

    with pytest.raises(EmbeddingModelError, match="Silent substitution"):
        matcher.score_article(
            ["The analysis lacks an appropriate negative control."],
            [_gt("The experiment lacks an appropriate negative control.")],
        )

    assert matcher.matching_method == "embedding"


def test_embedding_load_failure_falls_back_only_when_explicit(monkeypatch) -> None:
    def unavailable(model_id: str, revision: str | None = None) -> object:
        del revision
        raise EmbeddingModelError(f"{model_id} unavailable")

    monkeypatch.setattr(metrics, "_get_embed_model", unavailable)
    matcher = ConcernMatcher(allow_fallback=True)
    text = "The analysis lacks an appropriate negative control."

    with pytest.warns(RuntimeWarning, match="separate calibration"):
        result = matcher.score_article([text], [_gt(text)])

    assert result.matching_method == "jaccard"
    assert result.embedding_model is None
    assert result.embedding_revision is None
    assert result.threshold == pytest.approx(0.65)
    assert result.effective_threshold == pytest.approx(0.65 * 0.3)


def test_explicit_jaccard_mode_never_loads_embedding_model(monkeypatch) -> None:
    def unexpected_loader(model_id: str, revision: str | None = None) -> object:
        del revision
        raise AssertionError(f"unexpected embedding load: {model_id}")

    monkeypatch.setattr(metrics, "_get_embed_model", unexpected_loader)
    matcher = ConcernMatcher(use_embedding=False)
    text = "The analysis lacks an appropriate negative control."
    result = matcher.score_article([text], [_gt(text)])

    assert result.recall == pytest.approx(1.0)
    assert result.matching_method == "jaccard"
    assert result.embedding_model is None
    assert result.threshold == pytest.approx(0.65)
    assert result.effective_threshold == pytest.approx(0.65 * 0.3)


def test_encoding_failure_is_fail_closed_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        metrics,
        "_get_embed_model",
        lambda model_id, revision=None: _BrokenEmbeddingModel(),
    )
    matcher = ConcernMatcher()

    with pytest.raises(EmbeddingModelError, match="text encoding"):
        matcher.score_article(
            ["The analysis lacks an appropriate negative control."],
            [_gt("The experiment lacks an appropriate negative control.")],
        )


def test_encoding_failure_can_use_explicit_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        metrics,
        "_get_embed_model",
        lambda model_id, revision=None: _BrokenEmbeddingModel(),
    )
    matcher = ConcernMatcher(allow_fallback=True)
    text = "The analysis lacks an appropriate negative control."

    with pytest.warns(RuntimeWarning, match="using Jaccard"):
        result = matcher.score_article([text], [_gt(text)])

    assert result.matching_method == "jaccard"
    assert result.recall == pytest.approx(1.0)


def test_late_failure_never_mixes_embedding_and_jaccard(monkeypatch) -> None:
    model = _FailsAfterFirstPairModel()
    monkeypatch.setattr(
        metrics,
        "_get_embed_model",
        lambda model_id, revision=None: model,
    )
    matcher = ConcernMatcher(allow_fallback=True)

    matcher.score_article(
        ["First unique prediction about statistical controls."],
        [_gt("First unique reference about statistical controls.")],
    )

    with pytest.raises(EmbeddingModelError, match="Refusing to mix"):
        matcher.score_article(
            ["Second unique prediction about sample size."],
            [_gt("Second unique reference about sample size.")],
        )


def test_embedding_cache_is_scoped_by_model_id() -> None:
    metrics._EMBED_CACHE.clear()
    text = "Identical text must not reuse another model's vector."

    first = metrics._encode_with_cache(
        _ConstantEmbeddingModel(1.0),
        [text],
        "model-a",
    )
    second = metrics._encode_with_cache(
        _ConstantEmbeddingModel(2.0),
        [text],
        "model-b",
    )

    assert first[0, 0] == pytest.approx(1.0)
    assert second[0, 0] == pytest.approx(2.0)


def test_coverage_and_benchmark_notes_record_actual_matcher(monkeypatch) -> None:
    monkeypatch.setattr(
        metrics,
        "_get_embed_model",
        lambda model_id, revision=None: _ConstantEmbeddingModel(),
    )
    model_id = "org/provenance-test-model"
    with pytest.warns(UserWarning, match="requires|Calibrate"):
        matcher = ConcernMatcher(embedding_model=model_id)
    text = "The analysis lacks an appropriate negative control."
    article_results, coverage = evaluate_articles(
        {"article-1": [text]},
        [{"id": "article-1", "concerns": [_gt(text)]}],
        matcher,
    )

    assert coverage[0]["matching_method"] == "embedding"
    assert coverage[0]["embedding_model"] == model_id
    assert coverage[0]["effective_threshold"] == pytest.approx(0.65)

    benchmark = aggregate_results(
        article_results=article_results,
        n_bootstrap=0,
        tool_name="test-tool",
        tool_version="test-version",
        git_hash="",
        split="val",
        extraction_manifest_id="test-manifest",
        n_articles=1,
        n_human_concerns=1,
        n_tool_concerns=1,
        n_figure_excluded=0,
        notes="",
    )
    assert "actual_method=embedding" in benchmark.notes
    assert f"embedding_model={model_id}" in benchmark.notes
    assert "require calibration" in benchmark.notes
    assert benchmark.matching_stats is not None
    assert benchmark.matching_stats.threshold == pytest.approx(0.65)
    assert benchmark.matching_stats.configured_threshold == pytest.approx(0.65)
    assert benchmark.matching_stats.method == "embedding"
    assert benchmark.matching_stats.embedding_model == model_id
    assert benchmark.matching_stats.embedding_revision is None


def test_jaccard_result_serializes_effective_threshold() -> None:
    text = "The analysis lacks an appropriate negative control."
    matcher = ConcernMatcher(use_embedding=False, threshold=0.65)
    article_results, _ = evaluate_articles(
        {"article-1": [text]},
        [{"id": "article-1", "concerns": [_gt(text)]}],
        matcher,
    )

    benchmark = aggregate_results(
        article_results=article_results,
        n_bootstrap=0,
        tool_name="test-tool",
        tool_version="test-version",
        git_hash="",
        split="val",
        extraction_manifest_id="test-manifest",
        n_articles=1,
        n_human_concerns=1,
        n_tool_concerns=1,
        n_figure_excluded=0,
        notes="",
    )

    assert benchmark.matching_stats is not None
    assert benchmark.matching_stats.threshold == pytest.approx(0.65 * 0.3)
    assert benchmark.matching_stats.configured_threshold == pytest.approx(0.65)
    assert benchmark.matching_stats.method == "jaccard"
    assert benchmark.matching_stats.embedding_model is None


def test_embedding_revision_is_loaded_cached_and_recorded(monkeypatch) -> None:
    requested: list[tuple[str, str | None]] = []

    def fake_loader(model_id: str, revision: str | None = None) -> object:
        requested.append((model_id, revision))
        return _ConstantEmbeddingModel()

    monkeypatch.setattr(metrics, "_get_embed_model", fake_loader)
    matcher = ConcernMatcher(embedding_revision="abc123")
    text = "The analysis lacks an appropriate negative control."
    result = matcher.score_article([text], [_gt(text)])

    assert requested == [(DEFAULT_EMBEDDING_MODEL, "abc123")]
    assert result.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert result.embedding_revision == "abc123"


def test_jaccard_mode_rejects_embedding_only_deduplication() -> None:
    with pytest.raises(ValueError, match="dedup_gt requires embedding"):
        ConcernMatcher(use_embedding=False, dedup_gt=True)


def test_fallback_refuses_to_skip_requested_deduplication(monkeypatch) -> None:
    def unavailable(model_id: str, revision: str | None = None) -> object:
        del revision
        raise EmbeddingModelError(f"{model_id} unavailable")

    monkeypatch.setattr(metrics, "_get_embed_model", unavailable)
    matcher = ConcernMatcher(allow_fallback=True, dedup_gt=True)

    with pytest.raises(EmbeddingModelError, match="dedup_gt=True"):
        matcher.score_article(
            ["The analysis lacks an appropriate negative control."],
            [_gt("The experiment lacks an appropriate negative control.")],
        )

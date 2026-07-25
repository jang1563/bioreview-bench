"""Benchmark evaluation metrics.

Compares AI tool-generated concerns against ground-truth concerns to compute
recall, precision, F1, and per-category breakdowns.

Matching strategies:
- Primary: sentence-transformers cosine similarity (SPECTER2 or a configured model)
- Explicit alternative: Jaccard token overlap
- Optional fallback: Jaccard, only when ``allow_fallback=True``

Note on thresholds:
  - Embedding mode: cosine similarity; 0.65 is the historical SPECTER2 threshold
  - Jaccard mode: token overlap, auto-scaled to ~0.20 (30% of embedding threshold)
    because Jaccard scores are much lower for paraphrased text.
  - Thresholds are model- and method-specific. Any model change requires
    calibration before results are compared or reported.

Usage:
    from bioreview_bench.evaluate.metrics import ConcernMatcher, quick_eval

    matcher = ConcernMatcher()
    result = matcher.score_article(tool_concern_texts, gt_concerns_dicts)
    print(result.recall, result.precision, result.f1)
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy

# --- Embedding model (optional dependency) -----------------------------------

DEFAULT_EMBEDDING_MODEL = "allenai/specter2_base"

_EMBED_MODELS: dict[tuple[str, str | None], object] = {}
_EMBED_LOAD_ERRORS: dict[tuple[str, str | None], str] = {}
_EMBED_CACHE: dict[tuple[str, str], list[float]] = {}


class EmbeddingModelError(RuntimeError):
    """Raised when embedding matching cannot run as configured."""


def _get_embed_model(
    model_id: str = DEFAULT_EMBEDDING_MODEL,
    revision: str | None = None,
) -> object:
    """Load and cache a sentence-transformers model by ID and optional revision.

    Loading failures are raised rather than converted to a different metric.
    ``ConcernMatcher`` decides whether an explicitly enabled fallback is allowed.
    """
    model_key = (model_id, revision)
    if model_key in _EMBED_MODELS:
        return _EMBED_MODELS[model_key]
    if model_key in _EMBED_LOAD_ERRORS:
        raise EmbeddingModelError(_EMBED_LOAD_ERRORS[model_key])

    try:
        from sentence_transformers import SentenceTransformer

        if revision is None:
            model = SentenceTransformer(model_id)
        else:
            model = SentenceTransformer(model_id, revision=revision)
    except Exception as exc:
        revision_label = revision or "default"
        message = (
            f"Embedding model '{model_id}' at revision '{revision_label}' "
            "could not be loaded "
            f"({type(exc).__name__}: {exc})."
        )
        _EMBED_LOAD_ERRORS[model_key] = message
        raise EmbeddingModelError(message) from exc

    _EMBED_MODELS[model_key] = model
    return model


def _encode_with_cache(
    model: object,
    texts: list[str],
    model_id: str = DEFAULT_EMBEDDING_MODEL,
) -> numpy.ndarray:
    """Encode texts with a model-scoped cache. Uncached texts are batch-encoded."""
    import numpy as np

    uncached_indices: list[int] = []
    for i, t in enumerate(texts):
        if (model_id, t) not in _EMBED_CACHE:
            uncached_indices.append(i)

    # Batch-encode uncached texts
    if uncached_indices:
        uncached_texts = [texts[i] for i in uncached_indices]
        new_embs = model.encode(uncached_texts, normalize_embeddings=True)  # type: ignore
        for j, idx in enumerate(uncached_indices):
            _EMBED_CACHE[(model_id, texts[idx])] = new_embs[j].tolist()

    # Assemble result in original order
    result = np.array(
        [_EMBED_CACHE[(model_id, text)] for text in texts],
        dtype=np.float32,
    )
    return result


# --- Data classes ------------------------------------------------------------

@dataclass
class MatchResult:
    """Result of a single concern pair match."""
    tool_idx: int
    gt_idx: int
    score: float
    method: Literal["embedding", "jaccard"]


@dataclass
class PairwiseScores:
    """Similarity matrix between tool concerns and ground-truth concerns."""
    matrix: list[list[float]]      # [tool_i][gt_j] = similarity score
    method: Literal["embedding", "jaccard"]
    threshold: float               # effective threshold for this method


@dataclass
class CategoryMetrics:
    """Per-category evaluation metrics."""
    recall: float
    precision: float
    f1: float
    n_gt: int           # number of ground truth concerns in this category
    n_tool: int         # number of tool concerns assigned to this category
    n_matched: int      # number of matched concerns


@dataclass
class EvalResult:
    """Evaluation result for a single article or the full dataset."""
    # Overall metrics
    recall: float           # TP / (TP + FN)
    precision: float        # TP / (TP + FP)
    f1: float               # 2 * P * R / (P + R)

    # Counts
    n_gt_total: int         # total ground truth concerns
    n_tool_total: int       # total tool concerns
    n_matched: int          # matched pairs (TP)

    # Severity breakdown
    recall_major: float = 0.0
    recall_minor: float = 0.0
    n_gt_major: int = 0
    n_matched_major: int = 0
    n_gt_minor: int = 0
    n_matched_minor: int = 0

    # Category breakdown
    per_category: dict[str, CategoryMetrics] = field(default_factory=dict)

    # Figure concern handling
    n_gt_figure_excluded: int = 0   # figure concerns excluded from GT

    # Soft matching metrics (similarity-weighted credit for matched pairs)
    soft_recall: float = 0.0
    soft_precision: float = 0.0
    soft_f1: float = 0.0

    # Metadata
    matching_method: Literal["embedding", "jaccard"] = "jaccard"
    threshold: float = 0.65  # configured threshold (backward-compatible field)
    effective_threshold: float = 0.65
    embedding_model: str | None = None
    embedding_revision: str | None = None
    algorithm: Literal["hungarian", "greedy"] = "hungarian"
    figure_policy: Literal["exclude", "include"] = "exclude"


# --- Core matcher ------------------------------------------------------------

class ConcernMatcher:
    """Matcher for AI tool concerns vs. ground truth concerns.

    Args:
        threshold: Similarity threshold for a match to be accepted.
            - Embedding mode: historical SPECTER2 default 0.65
            - Jaccard mode/fallback: scaled to threshold * 0.3 (~0.20)
            Both values are matcher-specific and require calibration.
        exclude_figure: If True, figure_issue concerns are removed from GT
            before scoring (they require viewing actual figures to assess).
        use_embedding: If True, attempt to use SPECTER2 embeddings first.
            Embedding failures raise by default rather than changing the metric.
        embedding_model: Exact sentence-transformers model ID. Defaults to the
            historical ``allenai/specter2_base`` matcher.
        embedding_revision: Optional Hugging Face revision or commit SHA. New
            reproducible runs should pin this value.
        allow_fallback: If True, an embedding load/encode failure may switch to
            Jaccard with a prominent warning. False by default (fail closed).
        algorithm: Matching algorithm. "hungarian" (optimal, default) or
            "greedy" (legacy, faster but suboptimal).
    """

    # Historical heuristic: Jaccard token-overlap scores are typically lower
    # than embedding similarity. This is not a model-independent calibration.
    _JACCARD_THRESHOLD_SCALE: float = 0.3

    def __init__(
        self,
        threshold: float = 0.65,
        exclude_figure: bool = True,
        use_embedding: bool = True,
        algorithm: Literal["hungarian", "greedy"] = "hungarian",
        dedup_gt: bool = False,
        dedup_threshold: float = 0.95,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_revision: str | None = None,
        allow_fallback: bool = False,
    ) -> None:
        if not embedding_model.strip():
            raise ValueError("embedding_model must be a non-empty model ID")
        if embedding_revision is not None and not embedding_revision.strip():
            raise ValueError("embedding_revision must be non-empty when provided")
        if dedup_gt and not use_embedding:
            raise ValueError(
                "dedup_gt requires embedding matching; it cannot be combined "
                "with explicit Jaccard mode."
            )

        self.threshold = threshold
        self.exclude_figure = exclude_figure
        self.use_embedding = use_embedding
        self._algorithm = algorithm
        self.dedup_gt = dedup_gt
        self.dedup_threshold = dedup_threshold
        self.embedding_model = embedding_model
        self.embedding_revision = embedding_revision
        self.allow_fallback = allow_fallback
        self._resolved_model: object | None = None
        self._model_resolution_attempted = False
        self._effective_method: Literal["embedding", "jaccard"] = (
            "embedding" if use_embedding else "jaccard"
        )
        self._embedding_results_emitted = 0

        if use_embedding and embedding_model != DEFAULT_EMBEDDING_MODEL:
            warnings.warn(
                f"Embedding threshold {threshold:.4g} is model-specific and has "
                f"not been calibrated automatically for '{embedding_model}'. "
                "Calibrate the threshold before comparing or reporting results.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def matching_method(self) -> Literal["embedding", "jaccard"]:
        """Actual similarity method selected for this matcher."""
        return self._effective_method

    @property
    def effective_threshold(self) -> float:
        """Threshold actually applied by the selected similarity method."""
        if self.matching_method == "jaccard":
            return self.threshold * self._JACCARD_THRESHOLD_SCALE
        return self.threshold

    def _fallback_or_raise(
        self,
        error: Exception,
        *,
        operation: str,
    ) -> None:
        model_label = self.embedding_model
        if self.embedding_revision:
            model_label = f"{model_label}@{self.embedding_revision}"
        message = (
            f"Embedding matching with model '{model_label}' failed "
            f"during {operation}: {error}"
        )
        if not self.allow_fallback:
            raise EmbeddingModelError(
                f"{message}. Silent substitution with Jaccard is disabled. "
                "Fix the embedding configuration, select explicit Jaccard mode "
                "(use_embedding=False), or explicitly set allow_fallback=True."
            ) from error
        if self._embedding_results_emitted:
            raise EmbeddingModelError(
                f"{message}. Refusing to mix embedding and Jaccard scores in one "
                "evaluation after embedding results were already produced."
            ) from error
        if self.dedup_gt:
            raise EmbeddingModelError(
                f"{message}. Refusing Jaccard fallback because dedup_gt=True "
                "requires the configured embedding model."
            ) from error

        self._effective_method = "jaccard"
        self._resolved_model = None
        self._model_resolution_attempted = True
        warnings.warn(
            f"{message}. Explicit fallback is enabled; using Jaccard similarity. "
            "Jaccard and embedding thresholds are not interchangeable, and the "
            "effective Jaccard threshold requires separate calibration.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _resolve_embedding_model(self) -> object | None:
        """Resolve the configured model once, applying explicit fallback policy."""
        if not self.use_embedding or self._effective_method == "jaccard":
            return None
        if self._model_resolution_attempted:
            return self._resolved_model

        self._model_resolution_attempted = True
        try:
            self._resolved_model = _get_embed_model(
                self.embedding_model,
                self.embedding_revision,
            )
        except Exception as exc:
            # A fail-closed error must remain fail-closed if the caller retries.
            self._model_resolution_attempted = False
            self._fallback_or_raise(exc, operation="model loading")
        return self._resolved_model

    def validate_configuration(self) -> Literal["embedding", "jaccard"]:
        """Resolve the requested similarity backend before an evaluation run."""
        self._resolve_embedding_model()
        return self.matching_method

    # -- Text preprocessing --------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple tokenization: lowercase alphanumeric tokens of length >= 3."""
        return set(re.findall(r"\b[a-z0-9]{3,}\b", text.lower()))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # -- Similarity matrix ---------------------------------------------------

    def _compute_scores(
        self,
        tool_texts: list[str],
        gt_texts: list[str],
    ) -> PairwiseScores:
        """Compute (n_tool x n_gt) similarity matrix."""
        if not tool_texts or not gt_texts:
            return PairwiseScores(
                [],
                self.matching_method,
                self.effective_threshold,
            )

        model = self._resolve_embedding_model()

        if model is not None:
            try:
                cache_identity = self.embedding_model
                if self.embedding_revision:
                    cache_identity = (
                        f"{self.embedding_model}@{self.embedding_revision}"
                    )
                tool_emb = _encode_with_cache(
                    model,
                    tool_texts,
                    cache_identity,
                )
                gt_emb = _encode_with_cache(
                    model,
                    gt_texts,
                    cache_identity,
                )
                sim_matrix = (tool_emb @ gt_emb.T).tolist()
                self._effective_method = "embedding"
                self._embedding_results_emitted += 1
                return PairwiseScores(sim_matrix, "embedding", self.threshold)
            except Exception as exc:
                self._fallback_or_raise(exc, operation="text encoding")

        # Explicit Jaccard mode or explicitly enabled fallback.
        jaccard_threshold = self.effective_threshold
        tool_tokens = [self._tokenize(t) for t in tool_texts]
        gt_tokens = [self._tokenize(t) for t in gt_texts]
        matrix = [
            [self._jaccard(tt, gt) for gt in gt_tokens]
            for tt in tool_tokens
        ]
        return PairwiseScores(matrix, "jaccard", jaccard_threshold)

    # -- Greedy bipartite matching -------------------------------------------

    @staticmethod
    def _greedy_match(scores: PairwiseScores) -> list[MatchResult]:
        """Greedy 1:1 bipartite matching (highest-score pairs first)."""
        matrix = scores.matrix
        threshold = scores.threshold
        if not matrix or not matrix[0]:
            return []

        candidates = []
        for i, row in enumerate(matrix):
            for j, score in enumerate(row):
                if score >= threshold:
                    candidates.append((score, i, j))
        candidates.sort(reverse=True)

        matched_tool: set[int] = set()
        matched_gt: set[int] = set()
        results = []

        for score, i, j in candidates:
            if i in matched_tool or j in matched_gt:
                continue
            matched_tool.add(i)
            matched_gt.add(j)
            results.append(MatchResult(i, j, score, scores.method))

        return results

    # -- Hungarian bipartite matching ----------------------------------------

    @staticmethod
    def _hungarian_match(scores: PairwiseScores) -> list[MatchResult]:
        """Threshold-aware 1:1 matching via the Hungarian algorithm.

        The assignment first maximizes the number of above-threshold pairs and,
        among those assignments, maximizes total similarity. This is the frozen
        leaderboard behavior; it differs from unconstrained maximum-weight
        assignment followed by threshold filtering.
        """
        matrix = scores.matrix
        threshold = scores.threshold
        if not matrix or not matrix[0]:
            return []

        import numpy as np
        from scipy.optimize import linear_sum_assignment

        sim = np.array(matrix, dtype=np.float64)
        cost = 1.0 - sim
        cost[sim < threshold] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)

        results = []
        for i, j in zip(row_ind, col_ind):
            if matrix[i][j] >= threshold:
                results.append(
                    MatchResult(int(i), int(j), float(matrix[i][j]), scores.method)
                )
        return results

    # -- Matching dispatch ---------------------------------------------------

    def _match(self, scores: PairwiseScores) -> list[MatchResult]:
        """Dispatch to the configured matching algorithm."""
        if self._algorithm == "hungarian":
            try:
                return self._hungarian_match(scores)
            except ImportError:
                pass  # fall back to greedy if scipy missing
        return self._greedy_match(scores)

    # -- GT dedup ------------------------------------------------------------

    def _dedup_concerns(
        self, concerns: list[dict], threshold: float
    ) -> list[dict]:
        """Remove near-duplicate GT concerns (greedy, intra-article).

        Keeps the first occurrence; removes later concerns whose cosine
        similarity to any earlier kept concern is >= *threshold*.
        """
        texts = [c["concern_text"] for c in concerns]
        model = self._resolve_embedding_model()
        if len(texts) < 2:
            return concerns
        if model is None:
            raise EmbeddingModelError(
                "Ground-truth deduplication requires an embedding model; "
                "deduplication was not applied."
            )

        try:
            cache_identity = self.embedding_model
            if self.embedding_revision:
                cache_identity = f"{self.embedding_model}@{self.embedding_revision}"
            embs = _encode_with_cache(model, texts, cache_identity)
        except Exception as exc:
            self._fallback_or_raise(exc, operation="ground-truth deduplication")
            return concerns
        sim = embs @ embs.T  # (n, n) cosine similarity

        keep = [True] * len(concerns)
        for i in range(len(concerns)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(concerns)):
                if not keep[j]:
                    continue
                if sim[i, j] >= threshold:
                    keep[j] = False
        return [c for c, k in zip(concerns, keep) if k]

    # -- Public API ----------------------------------------------------------

    def score_article(
        self,
        tool_concerns: list[str],
        gt_concerns: list[dict],
    ) -> EvalResult:
        """Evaluate tool concerns against ground truth for a single article.

        Args:
            tool_concerns: Concern texts generated by the AI tool.
            gt_concerns: Ground truth concerns as dicts (ReviewerConcern.model_dump()).
                Each dict must have keys: concern_text, category, severity,
                requires_figure_reading.

        Returns:
            EvalResult with recall, precision, F1, and breakdowns.
        """
        if self.exclude_figure:
            active_gt = [c for c in gt_concerns if not c.get("requires_figure_reading", False)]
            n_excluded = len(gt_concerns) - len(active_gt)
        else:
            active_gt = gt_concerns
            n_excluded = 0

        if self.dedup_gt and len(active_gt) > 1:
            active_gt = self._dedup_concerns(active_gt, self.dedup_threshold)

        gt_texts = [c["concern_text"] for c in active_gt]

        if not gt_texts:
            return EvalResult(
                recall=0.0,
                precision=0.0,
                f1=0.0,
                n_gt_total=0,
                n_tool_total=len(tool_concerns),
                n_matched=0,
                n_gt_figure_excluded=n_excluded,
                matching_method=self.matching_method,
                threshold=self.threshold,
                effective_threshold=self.effective_threshold,
                embedding_model=(
                    self.embedding_model
                    if self.matching_method == "embedding"
                    else None
                ),
                embedding_revision=(
                    self.embedding_revision
                    if self.matching_method == "embedding"
                    else None
                ),
                algorithm=self._algorithm,
                figure_policy="exclude" if self.exclude_figure else "include",
            )

        scores = self._compute_scores(tool_concerns, gt_texts)
        matches = self._match(scores)
        matched_gt_idxs = {m.gt_idx for m in matches}
        matched_tool_idxs = {m.tool_idx for m in matches}

        n_gt = len(gt_texts)
        n_tool = len(tool_concerns)
        n_matched = len(matches)

        recall = n_matched / n_gt if n_gt > 0 else 0.0
        precision = n_matched / n_tool if n_tool > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Soft matching: use similarity scores as fractional credit
        soft_credit = sum(m.score for m in matches)
        soft_recall = soft_credit / n_gt if n_gt > 0 else 0.0
        soft_precision = soft_credit / n_tool if n_tool > 0 else 0.0
        soft_f1 = (
            (2 * soft_precision * soft_recall / (soft_precision + soft_recall))
            if (soft_precision + soft_recall) > 0
            else 0.0
        )

        # Recall by severity
        major_gt = [c for c in active_gt if c.get("severity") == "major"]
        major_matched = sum(
            1 for i, c in enumerate(active_gt)
            if i in matched_gt_idxs and c.get("severity") == "major"
        )
        minor_gt = [c for c in active_gt if c.get("severity") == "minor"]
        minor_matched = sum(
            1 for i, c in enumerate(active_gt)
            if i in matched_gt_idxs and c.get("severity") == "minor"
        )

        recall_major = major_matched / len(major_gt) if major_gt else 0.0
        recall_minor = minor_matched / len(minor_gt) if minor_gt else 0.0

        # Per-category metrics
        # Assign each tool concern to a category:
        #   - Matched tool concerns inherit the category of their GT match
        #   - Unmatched tool concerns are assigned to the category of their
        #     most similar GT concern (nearest neighbour by similarity)
        per_category: dict[str, CategoryMetrics] = {}
        cat_gt: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(active_gt):
            cat_gt[c.get("category", "other")].append(i)

        tool_cat_count: dict[str, int] = defaultdict(int)
        for m in matches:
            cat = active_gt[m.gt_idx].get("category", "other")
            tool_cat_count[cat] += 1

        if scores.matrix:
            for i in range(n_tool):
                if i not in matched_tool_idxs:
                    row = scores.matrix[i]
                    best_j = max(range(len(row)), key=lambda j: row[j])
                    cat = active_gt[best_j].get("category", "other")
                    tool_cat_count[cat] += 1

        for cat, gt_idxs in cat_gt.items():
            cat_matched = sum(1 for idx in gt_idxs if idx in matched_gt_idxs)
            cat_gt_n = len(gt_idxs)
            cat_recall = cat_matched / cat_gt_n if cat_gt_n > 0 else 0.0
            cat_tool_n = tool_cat_count.get(cat, 0)
            cat_prec = cat_matched / max(cat_tool_n, 1)
            cat_f1 = (
                2 * cat_prec * cat_recall / (cat_prec + cat_recall)
                if (cat_prec + cat_recall) > 0
                else 0.0
            )
            per_category[cat] = CategoryMetrics(
                recall=cat_recall,
                precision=cat_prec,
                f1=cat_f1,
                n_gt=cat_gt_n,
                n_tool=cat_tool_n,
                n_matched=cat_matched,
            )

        return EvalResult(
            recall=recall,
            precision=precision,
            f1=f1,
            n_gt_total=n_gt,
            n_tool_total=n_tool,
            n_matched=n_matched,
            recall_major=recall_major,
            recall_minor=recall_minor,
            n_gt_major=len(major_gt),
            n_matched_major=major_matched,
            n_gt_minor=len(minor_gt),
            n_matched_minor=minor_matched,
            per_category=per_category,
            n_gt_figure_excluded=n_excluded,
            soft_recall=soft_recall,
            soft_precision=soft_precision,
            soft_f1=soft_f1,
            matching_method=scores.method,
            threshold=self.threshold,
            effective_threshold=scores.threshold,
            embedding_model=(
                self.embedding_model if scores.method == "embedding" else None
            ),
            embedding_revision=(
                self.embedding_revision if scores.method == "embedding" else None
            ),
            algorithm=self._algorithm,
            figure_policy="exclude" if self.exclude_figure else "include",
        )

    def score_dataset(
        self,
        tool_results: list[dict],
        ground_truth: list[dict],
    ) -> EvalResult:
        """Evaluate tool outputs across a full dataset.

        Args:
            tool_results: List of dicts with keys 'article_id' (or 'id')
                and 'concerns' (list of concern text strings or dicts).
            ground_truth: List of OpenPeerReviewEntry dicts (JSONL rows).

        Returns:
            Micro-averaged EvalResult across all ground-truth articles.
        """
        gt_by_id: dict[str, list[dict]] = {}
        for entry in ground_truth:
            art_id = entry.get("id", "")
            gt_by_id[art_id] = entry.get("concerns", [])

        article_results = []
        tool_by_id: dict[str, list[str]] = {}
        for tool_row in tool_results:
            art_id = tool_row.get("article_id", tool_row.get("id", ""))
            if not art_id:
                continue
            tool_texts = tool_row.get("concerns", [])
            if tool_texts and isinstance(tool_texts[0], dict):
                tool_texts = [c.get("text", c.get("concern_text", "")) for c in tool_texts]
            tool_by_id[art_id] = tool_texts

        for art_id, gt in gt_by_id.items():
            result = self.score_article(tool_by_id.get(art_id, []), gt)
            article_results.append(result)

        if not article_results:
            return EvalResult(0.0, 0.0, 0.0, 0, 0, 0)

        total_matched = sum(r.n_matched for r in article_results)
        total_gt = sum(r.n_gt_total for r in article_results)
        total_tool = sum(r.n_tool_total for r in article_results)
        total_gt_major = sum(r.n_gt_major for r in article_results)
        total_matched_major = sum(r.n_matched_major for r in article_results)
        total_gt_minor = sum(r.n_gt_minor for r in article_results)
        total_matched_minor = sum(r.n_matched_minor for r in article_results)
        n = len(article_results)
        recall = total_matched / total_gt if total_gt > 0 else 0.0
        precision = total_matched / total_tool if total_tool > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        recall_major = total_matched_major / total_gt_major if total_gt_major > 0 else 0.0
        recall_minor = total_matched_minor / total_gt_minor if total_gt_minor > 0 else 0.0
        soft_recall = sum(r.soft_recall for r in article_results) / n
        soft_precision = sum(r.soft_precision for r in article_results) / n
        soft_f1 = (
            (2 * soft_precision * soft_recall / (soft_precision + soft_recall))
            if (soft_precision + soft_recall) > 0
            else 0.0
        )

        agg_cat: dict[str, list[CategoryMetrics]] = defaultdict(list)
        for r in article_results:
            for cat, cm in r.per_category.items():
                agg_cat[cat].append(cm)

        per_category = {}
        for cat, cms in agg_cat.items():
            cat_n_gt = sum(m.n_gt for m in cms)
            cat_n_tool = sum(m.n_tool for m in cms)
            cat_n_matched = sum(m.n_matched for m in cms)
            cat_recall = cat_n_matched / cat_n_gt if cat_n_gt > 0 else 0.0
            cat_prec = cat_n_matched / cat_n_tool if cat_n_tool > 0 else 0.0
            cat_f1 = (
                (2 * cat_prec * cat_recall / (cat_prec + cat_recall))
                if (cat_prec + cat_recall) > 0
                else 0.0
            )
            per_category[cat] = CategoryMetrics(
                recall=cat_recall,
                precision=cat_prec,
                f1=cat_f1,
                n_gt=cat_n_gt,
                n_tool=cat_n_tool,
                n_matched=cat_n_matched,
            )

        return EvalResult(
            recall=recall,
            precision=precision,
            f1=f1,
            n_gt_total=sum(r.n_gt_total for r in article_results),
            n_tool_total=sum(r.n_tool_total for r in article_results),
            n_matched=sum(r.n_matched for r in article_results),
            recall_major=recall_major,
            recall_minor=recall_minor,
            n_gt_major=total_gt_major,
            n_matched_major=total_matched_major,
            n_gt_minor=total_gt_minor,
            n_matched_minor=total_matched_minor,
            per_category=per_category,
            n_gt_figure_excluded=sum(r.n_gt_figure_excluded for r in article_results),
            soft_recall=soft_recall,
            soft_precision=soft_precision,
            soft_f1=soft_f1,
            matching_method=self.matching_method,
            threshold=self.threshold,
            effective_threshold=self.effective_threshold,
            embedding_model=(
                self.embedding_model
                if self.matching_method == "embedding"
                else None
            ),
            embedding_revision=(
                self.embedding_revision
                if self.matching_method == "embedding"
                else None
            ),
            algorithm=article_results[0].algorithm,
            figure_policy="exclude" if self.exclude_figure else "include",
        )


# --- Convenience function ----------------------------------------------------

def quick_eval(
    tool_concerns: list[str],
    gt_entry: dict,
    threshold: float = 0.65,
) -> EvalResult:
    """Quick evaluation for a single article.

    Args:
        tool_concerns: Concern texts produced by the AI tool.
        gt_entry: OpenPeerReviewEntry dict (a JSONL row).
        threshold: Cosine similarity threshold (embedding mode).

    Example::

        import json
        entry = json.loads(open("data/splits/val.jsonl").readline())
        result = quick_eval(
            ["The statistical analysis is insufficient"],
            entry,
        )
        print(f"Recall: {result.recall:.2f}, Precision: {result.precision:.2f}")
    """
    matcher = ConcernMatcher(threshold=threshold)
    return matcher.score_article(tool_concerns, gt_entry.get("concerns", []))

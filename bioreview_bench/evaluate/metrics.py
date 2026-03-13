"""Benchmark evaluation metrics.

Compares AI tool-generated concerns against ground-truth concerns to compute
recall, precision, F1, and per-category breakdowns.

Matching strategies:
- Primary: sentence-transformers cosine similarity (SPECTER2 or compatible model)
- Fallback: Jaccard token overlap (used when sentence-transformers is not installed)

Note on thresholds:
  - Embedding mode: cosine similarity, recommended threshold = 0.65
  - Jaccard fallback: token overlap, auto-scaled to ~0.20 (30% of embedding threshold)
    because Jaccard scores are much lower for paraphrased text.

Usage:
    from bioreview_bench.evaluate.metrics import ConcernMatcher, quick_eval

    matcher = ConcernMatcher()
    result = matcher.score_article(tool_concern_texts, gt_concerns_dicts)
    print(result.recall, result.precision, result.f1)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

# --- Embedding model (optional dependency) -----------------------------------

_EMBED_MODEL: object | None = None
_EMBED_AVAILABLE: bool | None = None
_EMBED_CACHE: dict[str, list[float]] = {}  # text → embedding vector cache


def _get_embed_model() -> object | None:
    """Load sentence-transformers model. Returns None on failure (Jaccard fallback)."""
    global _EMBED_MODEL, _EMBED_AVAILABLE
    if _EMBED_AVAILABLE is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _EMBED_MODEL = SentenceTransformer("allenai/specter2_base")
        _EMBED_AVAILABLE = True
    except Exception:
        _EMBED_MODEL = None
        _EMBED_AVAILABLE = False
    return _EMBED_MODEL


def _encode_with_cache(model: object, texts: list[str]) -> "numpy.ndarray":
    """Encode texts with module-level cache. Uncached texts are batch-encoded."""
    import numpy as np

    cached_indices: list[int] = []
    uncached_indices: list[int] = []
    for i, t in enumerate(texts):
        if t in _EMBED_CACHE:
            cached_indices.append(i)
        else:
            uncached_indices.append(i)

    # Batch-encode uncached texts
    if uncached_indices:
        uncached_texts = [texts[i] for i in uncached_indices]
        new_embs = model.encode(uncached_texts, normalize_embeddings=True)  # type: ignore
        for j, idx in enumerate(uncached_indices):
            _EMBED_CACHE[texts[idx]] = new_embs[j].tolist()

    # Assemble result in original order
    result = np.array([_EMBED_CACHE[t] for t in texts], dtype=np.float32)
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
    threshold: float = 0.65
    algorithm: Literal["hungarian", "greedy"] = "hungarian"


# --- Core matcher ------------------------------------------------------------

class ConcernMatcher:
    """Matcher for AI tool concerns vs. ground truth concerns.

    Args:
        threshold: Similarity threshold for a match to be accepted.
            - Embedding mode (cosine similarity): recommended 0.65
            - Jaccard fallback: auto-scaled to threshold * 0.3 (~0.20)
        exclude_figure: If True, figure_issue concerns are removed from GT
            before scoring (they require viewing actual figures to assess).
        use_embedding: If True, attempt to use SPECTER2 embeddings first.
            Falls back to Jaccard if sentence-transformers is not installed.
        algorithm: Matching algorithm. "hungarian" (optimal, default) or
            "greedy" (legacy, faster but suboptimal).
    """

    # Jaccard threshold is scaled down from embedding threshold
    # because token overlap scores are much lower for paraphrased text.
    _JACCARD_THRESHOLD_SCALE: float = 0.3

    def __init__(
        self,
        threshold: float = 0.65,
        exclude_figure: bool = True,
        use_embedding: bool = True,
        algorithm: Literal["hungarian", "greedy"] = "hungarian",
        dedup_gt: bool = False,
        dedup_threshold: float = 0.90,
    ) -> None:
        self.threshold = threshold
        self.exclude_figure = exclude_figure
        self.use_embedding = use_embedding
        self._algorithm = algorithm
        self.dedup_gt = dedup_gt
        self.dedup_threshold = dedup_threshold

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
            method: Literal["embedding", "jaccard"] = "jaccard"
            return PairwiseScores([], method, self.threshold)

        model = _get_embed_model() if self.use_embedding else None

        if model is not None:
            try:
                tool_emb = _encode_with_cache(model, tool_texts)
                gt_emb = _encode_with_cache(model, gt_texts)
                sim_matrix = (tool_emb @ gt_emb.T).tolist()
                return PairwiseScores(sim_matrix, "embedding", self.threshold)
            except Exception:
                pass  # fall through to Jaccard

        # Jaccard fallback: scale threshold down since Jaccard scores are lower
        jaccard_threshold = self.threshold * self._JACCARD_THRESHOLD_SCALE
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
        """Optimal 1:1 bipartite matching via the Hungarian algorithm.

        Uses ``scipy.optimize.linear_sum_assignment`` to find the assignment
        that maximises total similarity while respecting the threshold.
        """
        matrix = scores.matrix
        threshold = scores.threshold
        if not matrix or not matrix[0]:
            return []

        import numpy as np
        from scipy.optimize import linear_sum_assignment

        sim = np.array(matrix, dtype=np.float64)
        cost = 1.0 - sim
        # Penalise below-threshold pairs so they are only used as last resort
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
        model = _get_embed_model() if self.use_embedding else None
        if model is None or len(texts) < 2:
            return concerns

        embs = _encode_with_cache(model, texts)
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
            per_category=per_category,
            n_gt_figure_excluded=n_excluded,
            soft_recall=soft_recall,
            soft_precision=soft_precision,
            soft_f1=soft_f1,
            matching_method=scores.method,
            threshold=self.threshold,
            algorithm=self._algorithm,
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

        n = len(article_results)
        total_matched = sum(r.n_matched for r in article_results)
        total_gt = sum(r.n_gt_total for r in article_results)
        total_tool = sum(r.n_tool_total for r in article_results)
        recall = total_matched / total_gt if total_gt > 0 else 0.0
        precision = total_matched / total_tool if total_tool > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        recall_major = sum(r.recall_major for r in article_results) / n
        recall_minor = sum(r.recall_minor for r in article_results) / n
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

        per_category = {
            cat: CategoryMetrics(
                recall=sum(m.recall for m in cms) / len(cms),
                precision=sum(m.precision for m in cms) / len(cms),
                f1=sum(m.f1 for m in cms) / len(cms),
                n_gt=sum(m.n_gt for m in cms),
                n_tool=sum(m.n_tool for m in cms),
                n_matched=sum(m.n_matched for m in cms),
            )
            for cat, cms in agg_cat.items()
        }

        return EvalResult(
            recall=recall,
            precision=precision,
            f1=f1,
            n_gt_total=sum(r.n_gt_total for r in article_results),
            n_tool_total=sum(r.n_tool_total for r in article_results),
            n_matched=sum(r.n_matched for r in article_results),
            recall_major=recall_major,
            recall_minor=recall_minor,
            per_category=per_category,
            n_gt_figure_excluded=sum(r.n_gt_figure_excluded for r in article_results),
            soft_recall=soft_recall,
            soft_precision=soft_precision,
            soft_f1=soft_f1,
            matching_method=article_results[0].matching_method,
            threshold=article_results[0].threshold,
            algorithm=article_results[0].algorithm,
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

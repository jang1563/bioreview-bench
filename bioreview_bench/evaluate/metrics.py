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
    n_tool: int         # number of tool concerns (total, not category-specific)
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

    # Metadata
    matching_method: Literal["embedding", "jaccard"] = "jaccard"
    threshold: float = 0.65


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
    """

    # Jaccard threshold is scaled down from embedding threshold
    # because token overlap scores are much lower for paraphrased text.
    _JACCARD_THRESHOLD_SCALE: float = 0.3

    def __init__(
        self,
        threshold: float = 0.65,
        exclude_figure: bool = True,
        use_embedding: bool = True,
    ) -> None:
        self.threshold = threshold
        self.exclude_figure = exclude_figure
        self.use_embedding = use_embedding

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
                all_texts = tool_texts + gt_texts
                embeddings = model.encode(all_texts, normalize_embeddings=True)  # type: ignore
                tool_emb = embeddings[:len(tool_texts)]
                gt_emb = embeddings[len(tool_texts):]
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

        gt_texts = [c["concern_text"] for c in active_gt]

        if not gt_texts:
            return EvalResult(
                recall=0.0,
                precision=1.0 if not tool_concerns else 0.0,
                f1=0.0,
                n_gt_total=0,
                n_tool_total=len(tool_concerns),
                n_matched=0,
                n_gt_figure_excluded=n_excluded,
            )

        scores = self._compute_scores(tool_concerns, gt_texts)
        matches = self._greedy_match(scores)
        matched_gt_idxs = {m.gt_idx for m in matches}

        n_gt = len(gt_texts)
        n_tool = len(tool_concerns)
        n_matched = len(matches)

        recall = n_matched / n_gt if n_gt > 0 else 0.0
        precision = n_matched / n_tool if n_tool > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

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
        per_category: dict[str, CategoryMetrics] = {}
        cat_gt: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(active_gt):
            cat_gt[c.get("category", "other")].append(i)

        for cat, gt_idxs in cat_gt.items():
            cat_matched = sum(1 for idx in gt_idxs if idx in matched_gt_idxs)
            cat_gt_n = len(gt_idxs)
            cat_recall = cat_matched / cat_gt_n if cat_gt_n > 0 else 0.0
            # Category-level precision is approximated since tool outputs lack category labels
            cat_prec = cat_matched / max(n_tool, 1)
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
                n_tool=n_tool,
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
            matching_method=scores.method,
            threshold=self.threshold,
        )

    def score_dataset(
        self,
        tool_results: list[dict],
        ground_truth: list[dict],
    ) -> EvalResult:
        """Evaluate tool outputs across a full dataset (macro-average).

        Args:
            tool_results: List of dicts with keys 'article_id' (or 'id')
                and 'concerns' (list of concern text strings or dicts).
            ground_truth: List of OpenPeerReviewEntry dicts (JSONL rows).

        Returns:
            Macro-averaged EvalResult across all articles.
        """
        gt_by_id: dict[str, list[dict]] = {}
        for entry in ground_truth:
            art_id = entry.get("id", "")
            gt_by_id[art_id] = entry.get("concerns", [])

        article_results = []
        for tool_row in tool_results:
            art_id = tool_row.get("article_id", tool_row.get("id", ""))
            tool_texts = tool_row.get("concerns", [])
            # Accept both string lists and dict lists
            if tool_texts and isinstance(tool_texts[0], dict):
                tool_texts = [c.get("text", c.get("concern_text", "")) for c in tool_texts]

            gt = gt_by_id.get(art_id, [])
            result = self.score_article(tool_texts, gt)
            article_results.append(result)

        if not article_results:
            return EvalResult(0.0, 0.0, 0.0, 0, 0, 0)

        n = len(article_results)
        recall = sum(r.recall for r in article_results) / n
        precision = sum(r.precision for r in article_results) / n
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        recall_major = sum(r.recall_major for r in article_results) / n
        recall_minor = sum(r.recall_minor for r in article_results) / n

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
            matching_method=article_results[0].matching_method,
            threshold=article_results[0].threshold,
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

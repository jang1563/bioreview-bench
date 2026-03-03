"""BenchmarkResult — evaluation result schema (includes bootstrap CI + bipartite stats)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class MatchingStats(BaseModel):
    """Bipartite matching result details (CRITICAL_REVIEW B3)."""
    n_tool_concerns: int
    n_human_concerns: int
    n_matched_pairs: int
    threshold: float
    algorithm: str = "hungarian"

    @property
    def recall(self) -> float:
        return self.n_matched_pairs / self.n_human_concerns if self.n_human_concerns > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.n_matched_pairs / self.n_tool_concerns if self.n_tool_concerns > 0 else 0.0


class ConfidenceInterval(BaseModel):
    lo: float = Field(description="95% CI lower bound")
    hi: float = Field(description="95% CI upper bound")
    n_bootstrap: int = 1000


class CategoryMetrics(BaseModel):
    recall: float
    precision: float
    f1_micro: float
    f1_macro: float = 0.0
    aucpr: float = 0.0
    brier_score: float | None = None
    n_human_concerns: int
    n_matched: int
    ci_recall: ConfidenceInterval | None = None
    ci_precision: ConfidenceInterval | None = None


class BenchmarkResult(BaseModel):
    """Complete benchmark evaluation result for a single tool."""

    # ── Tool info ─────────────────────────────────────────
    tool_name: str
    tool_version: str
    git_hash: str = ""

    # ── Benchmark info ────────────────────────────────────
    benchmark_version: str = "1.0"
    extraction_manifest_id: str = Field(
        description="ExtractionManifest ID used to extract human concerns"
    )
    split: Literal["train", "val", "test"] = "val"
    run_date: datetime = Field(default_factory=datetime.utcnow)

    # ── Overall metrics ───────────────────────────────────
    recall_overall: float
    precision_overall: float
    f1_micro: float
    recall_major: float = 0.0
    f1_macro: float = 0.0

    # ── Soft matching (similarity-weighted) ────────────────
    soft_recall_overall: float = 0.0
    soft_precision_overall: float = 0.0
    soft_f1: float = 0.0

    # ── Bootstrap CI (CRITICAL_REVIEW B4) ────────────────
    ci_recall: ConfidenceInterval | None = None
    ci_precision: ConfidenceInterval | None = None
    bootstrap_n: int = 1000

    # ── Per-category metrics ──────────────────────────────
    per_category: dict[str, CategoryMetrics] = Field(default_factory=dict)

    # ── Bipartite matching stats (CRITICAL_REVIEW B3) ─────
    matching_stats: MatchingStats | None = None

    # ── Counts ────────────────────────────────────────────
    n_articles: int
    n_human_concerns: int
    n_tool_concerns: int
    excluded_figure_concerns: int = 0

    notes: str = ""

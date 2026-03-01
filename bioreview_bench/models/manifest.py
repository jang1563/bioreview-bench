"""ExtractionManifest — extraction reproducibility guarantee (CRITICAL_REVIEW B2)."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class ExtractionManifest(BaseModel):
    """Metadata to guarantee reproducibility of LLM-based concern extraction.

    Frozen per release. Issue a new manifest_id when prompts or parsing logic changes.
    """

    manifest_id: str = Field(description="e.g. 'em-v1.0'")
    model_id: str = Field(description="e.g. 'claude-haiku-4-5-20251001'")
    model_date: str = Field(description="Model reference date (YYYY-MM-DD)")
    prompt_hash: str = Field(description="SHA-256 of system+user prompt templates")
    parsing_rule_hash: str = Field(description="SHA-256 of JSON parsing logic")
    temperature: float = 0.0
    seed: int = 42
    retry_policy: str = "max_3_attempts_exponential_backoff"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    cost_per_article_usd: float = Field(description="Estimated cost per article (USD)")
    n_articles_processed: int = 0
    intra_run_agreement: float | None = Field(
        default=None,
        description="Cohen's kappa from 3 repeated extractions of same input. None=not measured"
    )

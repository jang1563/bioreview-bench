"""Concurrent baseline runner — processes multiple articles with ThreadPoolExecutor.

Supports resume (skipping already-processed articles) and cost estimation.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .reviewer import BaselineReviewer

log = logging.getLogger(__name__)


def load_existing_ids(output_path: Path) -> set[str]:
    """Load article IDs from an existing output JSONL file for resume."""
    if not output_path.exists():
        return set()

    ids: set[str] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                art_id = row.get("article_id", "")
                if art_id:
                    ids.add(art_id)
            except json.JSONDecodeError:
                continue
    return ids


def estimate_cost(
    articles: list[dict],
    model: str,
    provider: str = "anthropic",
    max_input_chars: int = 80_000,
    input_price_per_mtok: float | None = None,
    output_price_per_mtok: float | None = None,
) -> dict:
    """Estimate API cost for a batch of articles.

    Returns:
        Dict with: n_articles, est_input_tokens, est_output_tokens, est_cost_usd.
    """
    total_chars = 0
    for entry in articles:
        article_chars = len(entry.get("title", ""))
        article_chars += len(entry.get("abstract", ""))
        sections = entry.get("paper_text_sections") or entry.get("sections") or {}
        for text in sections.values():
            article_chars += len(text)
        # Cap per article
        total_chars += min(article_chars, max_input_chars)

    # Rough token estimate: 1 token ~ 4 chars
    est_input_tokens = int(total_chars / 4)
    est_output_tokens = len(articles) * 1500  # ~1500 output tokens per article

    pricing = (
        {"input": input_price_per_mtok, "output": output_price_per_mtok}
        if input_price_per_mtok is not None and output_price_per_mtok is not None
        else _get_pricing(model, provider)
    )
    est_cost = (
        est_input_tokens * pricing["input"] / 1_000_000
        + est_output_tokens * pricing["output"] / 1_000_000
    )

    return {
        "n_articles": len(articles),
        "est_input_tokens": est_input_tokens,
        "est_output_tokens": est_output_tokens,
        "est_cost_usd": round(est_cost, 2),
        "model": model,
        "provider": provider,
        "pricing_input_per_mtok": pricing["input"],
        "pricing_output_per_mtok": pricing["output"],
    }


def _get_pricing(model: str, provider: str) -> dict[str, float]:
    """Get pricing per 1M tokens for known models.

    These values are lightweight defaults for local dry runs. Exact prices can
    change; callers can override them via ``estimate_cost(..., *_price_per_mtok)``.
    """
    pricing_map: dict[str, dict[str, float]] = {
        # Anthropic (as of 2026-03)
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-sonnet-4-6-20250620": {"input": 3.00, "output": 15.00},
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-opus-4-6-20250619": {"input": 15.00, "output": 75.00},
        # OpenAI
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        # Google Gemini
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        # Groq
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    }

    if model in pricing_map:
        return pricing_map[model]

    # Prefix match (e.g. "claude-haiku-4-5" matches "claude-haiku-4-5-20251001")
    for key, prices in pricing_map.items():
        if model.startswith(key.rsplit("-", 1)[0]):
            return prices

    # Fallback
    if provider == "openai":
        return {"input": 0.50, "output": 2.00}
    if provider == "google":
        return {"input": 0.20, "output": 1.00}
    if provider == "groq":
        return {"input": 0.20, "output": 0.50}
    return {"input": 1.00, "output": 5.00}


def run_baseline(
    reviewer: BaselineReviewer,
    articles: list[dict],
    output_path: Path,
    concurrency: int = 5,
    resume_ids: set[str] | None = None,
) -> dict:
    """Process articles concurrently and write results to JSONL.

    Args:
        reviewer: BaselineReviewer instance.
        articles: List of paper entry dicts.
        output_path: Path to output JSONL file (appended if resuming).
        concurrency: Max concurrent API calls.
        resume_ids: Set of article IDs to skip (already processed).

    Returns:
        Dict with: processed, skipped, failed, total_concerns.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
    )

    resume_ids = resume_ids or set()
    to_process = [a for a in articles if _get_id(a) not in resume_ids]
    skipped = len(articles) - len(to_process)

    if skipped:
        log.info("Skipping %d already-processed articles", skipped)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume_ids else "w"
    stats = {"processed": 0, "skipped": skipped, "failed": 0, "total_concerns": 0}

    with (
        open(output_path, mode, encoding="utf-8") as fh,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        ) as progress,
    ):
        task = progress.add_task("Reviewing articles", total=len(to_process))

        def _process_one(entry: dict) -> tuple[str, list[str] | None]:
            art_id = _get_id(entry)
            try:
                concerns = reviewer.review_article(entry)
                return art_id, concerns
            except Exception as exc:
                log.error("Failed to process %s: %s", art_id, exc)
                return art_id, None

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_process_one, entry): entry
                for entry in to_process
            }

            for future in as_completed(futures):
                art_id, concerns = future.result()
                progress.advance(task)

                if concerns is None:
                    stats["failed"] += 1
                    continue

                row = {"article_id": art_id, "concerns": concerns}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()

                stats["processed"] += 1
                stats["total_concerns"] += len(concerns)

    return stats


def _get_id(entry: dict) -> str:
    """Extract article ID from entry dict."""
    return entry.get("id", entry.get("article_id", ""))

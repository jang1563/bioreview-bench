"""Export JSONL files for all HuggingFace dataset configs.

Reads split JSONL files, applies per-config transforms, and writes the
results to a staging directory in the layout expected by HF Hub::

    {output_dir}/
      default/          train.jsonl  validation.jsonl  test.jsonl
      benchmark/        train.jsonl  validation.jsonl  test.jsonl
      concerns_flat/    train.jsonl  validation.jsonl  test.jsonl
      elife/            train.jsonl  validation.jsonl  test.jsonl
      plos/             train.jsonl  validation.jsonl  test.jsonl
      f1000/            train.jsonl  validation.jsonl  test.jsonl

No external dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .hf_transforms import (
    transform_benchmark,
    transform_concerns_flat,
    transform_default,
    transform_source_subset,
)

log = logging.getLogger(__name__)

_SPLIT_NAMES = ("train", "validation", "test")

# Source-specific configs to generate
_SOURCE_CONFIGS = ("elife", "plos", "f1000")


# ── I/O helpers ────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file; return empty list if file missing."""
    if not path.exists():
        log.warning("JSONL file not found: %s", path)
        return []
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed JSON at %s:%d: %s", path, lineno, exc)
                    continue
    return entries


def _save_jsonl(entries: list[dict], path: Path) -> int:
    """Write entries as JSONL.  Returns number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    return len(entries)


# ── Stats collection ───────────────────────────────────────────────

def _count_concerns(entries: list[dict]) -> int:
    """Count total concerns across entries.

    Handles both nested format (article with concerns list) and flat
    format (one row per concern, no ``concerns`` key).
    """
    if not entries:
        return 0
    if "concern_text" in entries[0] and "concerns" not in entries[0]:
        return len(entries)  # flat: each row is one concern
    return sum(len(e.get("concerns") or []) for e in entries)


def _source_distribution(entries: list[dict]) -> dict[str, int]:
    """Return {source: count} mapping."""
    dist: dict[str, int] = {}
    for e in entries:
        src = e.get("source", e.get("article_id", "unknown").split(":")[0])
        dist[src] = dist.get(src, 0) + 1
    return dist


# ── Main export function ───────────────────────────────────────────

def export_all_configs(
    splits_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate JSONL files for all 6 HF configs.

    Args:
        splits_dir: Directory containing train.jsonl, val.jsonl, test.jsonl.
        output_dir: Staging directory for HF upload.

    Returns:
        Statistics dict with per-config split sizes and concern counts.
    """
    # Map split file names: val.jsonl → "validation" config name
    split_file_map = {
        "train": "train.jsonl",
        "validation": "val.jsonl",
        "test": "test.jsonl",
    }

    # Load all splits
    raw_splits: dict[str, list[dict]] = {}
    for split_name, filename in split_file_map.items():
        raw_splits[split_name] = _load_jsonl(splits_dir / filename)
        log.info(
            "Loaded %s: %d entries",
            split_name,
            len(raw_splits[split_name]),
        )

    total_articles = sum(len(v) for v in raw_splits.values())
    if total_articles == 0:
        log.error("No entries loaded from %s", splits_dir)
        return {"error": "no data", "configs": {}}

    stats: dict[str, Any] = {
        "total_articles": total_articles,
        "configs": {},
    }

    # ── Config: default ────────────────────────────────────────────
    _export_config(
        "default", raw_splits, output_dir, stats,
        transform_fn=lambda entries, split: transform_default(entries),
    )

    # ── Config: benchmark ──────────────────────────────────────────
    _export_config(
        "benchmark", raw_splits, output_dir, stats,
        transform_fn=transform_benchmark,
    )

    # ── Config: concerns_flat ──────────────────────────────────────
    _export_config(
        "concerns_flat", raw_splits, output_dir, stats,
        transform_fn=lambda entries, split: transform_concerns_flat(entries),
    )

    # ── Config: source subsets ─────────────────────────────────────
    for source in _SOURCE_CONFIGS:
        _export_config(
            source, raw_splits, output_dir, stats,
            transform_fn=lambda entries, split, s=source: transform_source_subset(entries, s),
        )

    log.info(
        "Export complete: %d configs, %d total files",
        len(stats["configs"]),
        sum(
            len(cfg_stats.get("splits", {}))
            for cfg_stats in stats["configs"].values()
        ),
    )
    return stats


def _export_config(
    config_name: str,
    raw_splits: dict[str, list[dict]],
    output_dir: Path,
    stats: dict[str, Any],
    transform_fn: Any,
) -> None:
    """Export a single config's splits and collect stats."""
    cfg_stats: dict[str, Any] = {"splits": {}}

    for split_name in _SPLIT_NAMES:
        entries = raw_splits.get(split_name, [])
        transformed = transform_fn(entries, split_name)

        out_path = output_dir / config_name / f"{split_name}.jsonl"
        n_rows = _save_jsonl(transformed, out_path)

        split_stat = {
            "num_rows": n_rows,
            "num_concerns": _count_concerns(transformed),
            "source_distribution": _source_distribution(transformed),
        }
        cfg_stats["splits"][split_name] = split_stat

        log.info(
            "  %s/%s: %d rows",
            config_name,
            split_name,
            n_rows,
        )

    cfg_stats["total_rows"] = sum(
        s["num_rows"] for s in cfg_stats["splits"].values()
    )
    cfg_stats["total_concerns"] = sum(
        s["num_concerns"] for s in cfg_stats["splits"].values()
    )
    stats["configs"][config_name] = cfg_stats

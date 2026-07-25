"""Helpers for building a stratified human-reference subset."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

_SPLIT_FILE_MAP = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def load_entries_for_subset(
    splits_dir: Path,
    splits: Sequence[str],
) -> list[dict[str, Any]]:
    """Load entries from one or more benchmark splits."""
    rows: list[dict[str, Any]] = []
    for split in splits:
        path = splits_dir / _SPLIT_FILE_MAP[split]
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["benchmark_split"] = split
                rows.append(row)
    return rows


def sample_human_subset(
    entries: Sequence[dict[str, Any]],
    n: int,
    seed: int = 42,
    stratify_fields: Sequence[str] = (
        "benchmark_split",
        "source",
        "review_format",
        "has_author_response",
    ),
) -> list[dict[str, Any]]:
    """Sample a stratified subset with at least one example per observed stratum."""
    if n <= 0:
        return []
    if n >= len(entries):
        return [dict(entry) for entry in entries]

    rng = random.Random(seed)
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        strata[_stratum_key(entry, stratify_fields)].append(dict(entry))

    allocations = {key: 0 for key in strata}

    if n < len(strata):
        for key, _rows in sorted(
            strata.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:n]:
            allocations[key] = 1
        remaining = 0
    else:
        for key in strata:
            allocations[key] = 1
        remaining = n - len(strata)

    total_pool = sum(len(rows) - 1 for rows in strata.values())

    if remaining > 0 and total_pool > 0:
        fractional: list[tuple[float, tuple[str, ...]]] = []
        for key, rows in strata.items():
            capacity = max(len(rows) - 1, 0)
            share = remaining * capacity / total_pool if total_pool else 0.0
            extra = min(int(share), capacity)
            allocations[key] += extra
            fractional.append((share - int(share), key))

        used = sum(allocations.values())
        for _fraction, key in sorted(fractional, reverse=True):
            if used >= n:
                break
            if allocations[key] < len(strata[key]):
                allocations[key] += 1
                used += 1

    sampled: list[dict[str, Any]] = []
    for key, rows in strata.items():
        picks = min(allocations[key], len(rows))
        sampled.extend(rng.sample(rows, picks))

    rng.shuffle(sampled)
    return sampled[:n]


def build_subset_manifest(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a sampled subset for tracking and review assignment."""
    rows = list(entries)
    split_counts = Counter(str(row.get("benchmark_split", "unknown")) for row in rows)
    source_counts = Counter(str(row.get("source", "unknown")) for row in rows)
    format_counts = Counter(str(row.get("review_format", "unknown")) for row in rows)
    response_counts = Counter(
        "with_response" if row.get("has_author_response") else "without_response"
        for row in rows
    )
    return {
        "n_articles": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "review_formats": dict(sorted(format_counts.items())),
        "author_response": dict(sorted(response_counts.items())),
    }


def _stratum_key(entry: dict[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        value = entry.get(field, "unknown")
        if field == "has_author_response":
            value = "with_response" if value else "without_response"
        values.append(str(value))
    return tuple(values)

"""Dataset statistics helpers for split summaries and documentation checks."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_SPLIT_FILE_MAP: tuple[tuple[str, str], ...] = (
    ("train", "train.jsonl"),
    ("validation", "val.jsonl"),
    ("test", "test.jsonl"),
)

_README_SOURCE_LABELS = {
    "f1000": "F1000Research",
    "elife": "eLife",
    "plos": "PLOS",
    "peerj": "PeerJ",
    "nature": "Nature",
}


def summarize_splits(splits_dir: Path) -> dict[str, Any]:
    """Return aggregate dataset statistics for the canonical split layout."""
    split_stats: dict[str, dict[str, Any]] = {}
    totals = {
        "articles": 0,
        "concerns": 0,
    }
    source_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    stance_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    review_format_counts: Counter[str] = Counter()
    response_counts: Counter[str] = Counter()

    for split_name, filename in _SPLIT_FILE_MAP:
        rows = _load_jsonl(splits_dir / filename)
        split_concerns = 0
        split_sources: Counter[str] = Counter()

        for row in rows:
            source = str(row.get("source", "unknown"))
            split_sources[source] += 1
            source_counts[source] += 1

            review_format = str(row.get("review_format", "unknown"))
            review_format_counts[review_format] += 1

            response_key = "with_response" if row.get("has_author_response") else "without_response"
            response_counts[response_key] += 1

            for concern in row.get("concerns", []):
                split_concerns += 1
                severity_counts[str(concern.get("severity", "unknown"))] += 1
                stance_counts[str(concern.get("author_stance", "unknown"))] += 1
                category_counts[str(concern.get("category", "unknown"))] += 1

        split_stats[split_name] = {
            "articles": len(rows),
            "concerns": split_concerns,
            "avg_concerns_per_article": round(split_concerns / len(rows), 1) if rows else 0.0,
            "source_distribution": dict(sorted(split_sources.items())),
        }
        totals["articles"] += len(rows)
        totals["concerns"] += split_concerns

    return {
        "splits": split_stats,
        "total_articles": totals["articles"],
        "total_concerns": totals["concerns"],
        "avg_concerns_per_article": (
            round(totals["concerns"] / totals["articles"], 1)
            if totals["articles"]
            else 0.0
        ),
        "source_distribution": dict(sorted(source_counts.items())),
        "severity_distribution": _to_distribution(severity_counts, totals["concerns"]),
        "author_stance_distribution": _to_distribution(stance_counts, totals["concerns"]),
        "category_distribution": _to_distribution(category_counts, totals["concerns"]),
        "review_format_distribution": dict(sorted(review_format_counts.items())),
        "author_response_distribution": dict(sorted(response_counts.items())),
    }


def render_markdown_summary(summary: dict[str, Any]) -> str:
    """Render a compact markdown summary for generated stats artifacts."""
    lines = [
        "# Split Summary",
        "",
        f"- Articles: {summary['total_articles']:,}",
        f"- Concerns: {summary['total_concerns']:,}",
        "",
        "## Splits",
        "",
        "| Split | Articles | Concerns | Avg concerns/article |",
        "|-------|----------|----------|----------------------|",
    ]
    for split_name in ("train", "validation", "test"):
        split_stats = summary["splits"][split_name]
        lines.append(
            "| "
            f"{split_name} | {split_stats['articles']:,} | {split_stats['concerns']:,} | "
            f"{split_stats['avg_concerns_per_article']:.1f} |"
        )

    lines.extend([
        "",
        "## Sources",
        "",
        "| Source | Articles |",
        "|--------|----------|",
    ])
    for source, count in summary["source_distribution"].items():
        lines.append(f"| {source} | {count:,} |")

    return "\n".join(lines) + "\n"


def check_documentation(summary: dict[str, Any], doc_paths: list[Path]) -> list[str]:
    """Return validation errors for docs that should reflect split statistics."""
    expected = _expected_doc_patterns(summary)
    errors: list[str] = []

    for path in doc_paths:
        patterns = expected.get(path.name)
        if not patterns:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if re.search(pattern, text, flags=re.MULTILINE) is None:
                errors.append(f"{path}: missing pattern {pattern}")

    return errors


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_distribution(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "count": count,
            "percent": round(count / total * 100, 1) if total else 0.0,
        }
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    }


def _row_pattern(*cells: str) -> str:
    escaped = [re.escape(cell) for cell in cells]
    return r"\|\s*" + r"\s*\|\s*".join(escaped) + r"\s*\|"


def _expected_doc_patterns(summary: dict[str, Any]) -> dict[str, list[str]]:
    splits = summary["splits"]
    readme_patterns = [
        re.escape(f"**{summary['total_articles']:,} articles**"),
        re.escape(f"**{summary['total_concerns']:,} reviewer concerns**"),
        re.escape(
            f'train = dataset["train"]       # {splits["train"]["articles"]:,} articles, '
            f'{splits["train"]["concerns"]:,} concerns'
        ),
        re.escape(
            f'val   = dataset["validation"]  # {splits["validation"]["articles"]:,} articles, '
            f'{splits["validation"]["concerns"]:,} concerns'
        ),
        re.escape(
            f'test  = dataset["test"]        # {splits["test"]["articles"]:,} articles, '
            f'{splits["test"]["concerns"]:,} concerns'
        ),
        _row_pattern(
            "train",
            f"{splits['train']['articles']:,}",
            f"{splits['train']['concerns']:,}",
            f"{splits['train']['avg_concerns_per_article']:.1f}",
        ),
        _row_pattern(
            "validation",
            f"{splits['validation']['articles']:,}",
            f"{splits['validation']['concerns']:,}",
            f"{splits['validation']['avg_concerns_per_article']:.1f}",
        ),
        _row_pattern(
            "test",
            f"{splits['test']['articles']:,}",
            f"{splits['test']['concerns']:,}",
            f"{splits['test']['avg_concerns_per_article']:.1f}",
        ),
    ]

    for source, count in summary["source_distribution"].items():
        label = _README_SOURCE_LABELS.get(source, source)
        readme_patterns.append(_row_pattern(label, f"{count:,}"))
    for severity, values in summary["severity_distribution"].items():
        readme_patterns.append(_row_pattern(severity, f"{values['count']:,}", f"{values['percent']:.1f}%"))
    for stance, values in summary["author_stance_distribution"].items():
        readme_patterns.append(_row_pattern(stance, f"{values['count']:,}", f"{values['percent']:.1f}%"))
    for category, values in summary["category_distribution"].items():
        readme_patterns.append(_row_pattern(category, f"{values['count']:,}", f"{values['percent']:.1f}%"))

    datasheet_patterns = [
        re.escape(
            f"The current repository snapshot contains {summary['total_articles']:,} articles "
            f"(instances) and {summary['total_concerns']:,}"
        ),
        _row_pattern("train", f"{splits['train']['articles']:,}", f"{splits['train']['concerns']:,}"),
        _row_pattern("validation", f"{splits['validation']['articles']:,}", f"{splits['validation']['concerns']:,}"),
        _row_pattern("test", f"{splits['test']['articles']:,}", f"{splits['test']['concerns']:,}"),
        _row_pattern("**Total**", f"**{summary['total_articles']:,}**", f"**{summary['total_concerns']:,}**"),
    ]

    return {
        "README.md": readme_patterns,
        "DATASHEET.md": datasheet_patterns,
    }

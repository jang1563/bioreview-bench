"""Dataset statistics helpers for split summaries and documentation checks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_jsonl

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


def paired_sign_flip_pvalue(
    values_a: Iterable[float],
    values_b: Iterable[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    exact_max_pairs: int = 16,
) -> float | None:
    """Return a two-sided paired sign-flip/randomization p-value.

    Under the sharp null of exchangeable tool labels within each pair, changing
    the sign of every paired difference is equally likely. Small samples are
    enumerated exactly. Larger samples use a deterministic Monte Carlo estimate
    with the standard plus-one correction.

    ``None`` is returned when there are no pairs. Pairs with exactly zero
    difference do not affect the randomization distribution; an all-tie sample
    returns ``1.0``.
    """
    left = np.asarray(tuple(values_a), dtype=np.float64)
    right = np.asarray(tuple(values_b), dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("Paired values must be one-dimensional")
    if left.size != right.size:
        raise ValueError("Paired values must have the same length")
    if left.size == 0:
        return None
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Paired values must be finite")
    if exact_max_pairs < 0:
        raise ValueError("exact_max_pairs must be non-negative")

    differences = left - right
    differences = differences[differences != 0.0]
    if differences.size == 0:
        return 1.0

    observed = abs(float(differences.sum()))
    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * differences.size

    if differences.size <= exact_max_pairs:
        extreme = 0
        total = 0
        for sign_pattern in product(
            (-1.0, 1.0),
            repeat=int(differences.size),
        ):
            statistic = abs(float(np.dot(differences, sign_pattern)))
            extreme += statistic + tolerance >= observed
            total += 1
        return extreme / total

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive for Monte Carlo testing")
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = n_resamples
    batch_size = 4096
    while remaining:
        batch = min(batch_size, remaining)
        random_bits = rng.integers(
            0,
            2,
            size=(batch, differences.size),
            dtype=np.int8,
        )
        signed = random_bits.astype(np.float64) * 2.0 - 1.0
        statistics = np.abs(signed @ differences)
        extreme += int(np.count_nonzero(statistics + tolerance >= observed))
        remaining -= batch
    return (extreme + 1) / (n_resamples + 1)


def paired_micro_f1_randomization_pvalue(
    counts_a: Iterable[tuple[int, int, int]],
    counts_b: Iterable[tuple[int, int, int]],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    exact_max_pairs: int = 16,
) -> float | None:
    """Test a paired difference in micro-F1 by swapping tool labels per article.

    Each input row is ``(n_matched, n_reference, n_predicted)`` for one
    article. The two sequences must already be aligned by article and must use
    the same reference count within every pair. Under the sharp null of
    exchangeable tool labels, a randomization swaps the complete sufficient
    statistics for a pair and then recomputes both dataset-level micro-F1
    values.

    Small numbers of informative pairs are enumerated exactly. Larger samples
    use a deterministic Monte Carlo estimate with a plus-one correction.
    ``None`` is returned when there are no article pairs.
    """

    def _count_array(
        rows: Iterable[tuple[int, int, int]],
        label: str,
    ) -> np.ndarray:
        materialized = tuple(tuple(row) for row in rows)
        if not materialized:
            return np.empty((0, 3), dtype=np.int64)
        values = np.asarray(materialized, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{label} rows must be (n_matched, n_reference, n_predicted)")
        if not np.isfinite(values).all():
            raise ValueError(f"{label} counts must be finite")
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{label} counts must be non-negative integers")
        integer_values = values.astype(np.int64)
        if (integer_values[:, 0] > integer_values[:, 1]).any() or (
            integer_values[:, 0] > integer_values[:, 2]
        ).any():
            raise ValueError(f"{label} n_matched cannot exceed reference or predicted counts")
        return integer_values

    left = _count_array(counts_a, "counts_a")
    right = _count_array(counts_b, "counts_b")
    if left.shape[0] != right.shape[0]:
        raise ValueError("Paired count rows must have the same length")
    if left.shape[0] == 0:
        return None
    if not np.array_equal(left[:, 1], right[:, 1]):
        raise ValueError("Paired tools must use the same reference count per article")
    if exact_max_pairs < 0:
        raise ValueError("exact_max_pairs must be non-negative")

    def _micro_f1(totals: np.ndarray) -> float:
        denominator = int(totals[1] + totals[2])
        return 2.0 * float(totals[0]) / denominator if denominator else 0.0

    left_totals = left.sum(axis=0)
    right_totals = right.sum(axis=0)
    observed = abs(_micro_f1(left_totals) - _micro_f1(right_totals))

    deltas = right - left
    deltas = deltas[np.any(deltas != 0, axis=1)]
    if deltas.shape[0] == 0:
        return 1.0

    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * max(1, int(deltas.shape[0]))

    def _is_extreme(swap_totals: np.ndarray) -> bool:
        permuted_left = left_totals + swap_totals
        permuted_right = right_totals - swap_totals
        statistic = abs(_micro_f1(permuted_left) - _micro_f1(permuted_right))
        return bool(statistic + tolerance >= observed)

    if deltas.shape[0] <= exact_max_pairs:
        extreme = 0
        total = 0
        for swap_pattern in product(
            (0, 1),
            repeat=int(deltas.shape[0]),
        ):
            swap_totals = np.asarray(swap_pattern, dtype=np.int64) @ deltas
            extreme += _is_extreme(swap_totals)
            total += 1
        return extreme / total

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive for Monte Carlo testing")
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = n_resamples
    batch_size = 4096
    while remaining:
        batch = min(batch_size, remaining)
        swap_patterns = rng.integers(
            0,
            2,
            size=(batch, deltas.shape[0]),
            dtype=np.int8,
        )
        swap_totals = swap_patterns.astype(np.int64) @ deltas
        for row in swap_totals:
            extreme += _is_extreme(row)
        remaining -= batch
    return (extreme + 1) / (n_resamples + 1)


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
        rows = load_jsonl(splits_dir / filename)
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
            round(totals["concerns"] / totals["articles"], 1) if totals["articles"] else 0.0
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

    lines.extend(
        [
            "",
            "## Sources",
            "",
            "| Source | Articles |",
            "|--------|----------|",
        ]
    )
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
        _row_pattern(
            "`index`",
            f"{summary['total_articles']:,}",
            "train / validation / test",
            "Text-free article ID/source/DOI/date/schema rows",
        ),
        _row_pattern(
            "`annotations`",
            f"{summary['total_concerns'] - splits['test']['concerns']:,}",
            "train / validation",
            "Text-free category, severity, and stance rows",
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
        readme_patterns.append(
            _row_pattern(
                severity,
                f"{values['count']:,}",
                f"{values['percent']:.1f}%",
            )
        )
    for stance, values in summary["author_stance_distribution"].items():
        readme_patterns.append(
            _row_pattern(
                stance,
                f"{values['count']:,}",
                f"{values['percent']:.1f}%",
            )
        )
    for category, values in summary["category_distribution"].items():
        readme_patterns.append(
            _row_pattern(
                category,
                f"{values['count']:,}",
                f"{values['percent']:.1f}%",
            )
        )

    datasheet_patterns = [
        re.escape(
            f"The current repository snapshot contains {summary['total_articles']:,} articles "
            f"(instances) and {summary['total_concerns']:,}"
        ),
        _row_pattern(
            "train",
            f"{splits['train']['articles']:,}",
            f"{splits['train']['concerns']:,}",
        ),
        _row_pattern(
            "validation",
            f"{splits['validation']['articles']:,}",
            f"{splits['validation']['concerns']:,}",
        ),
        _row_pattern("test", f"{splits['test']['articles']:,}", f"{splits['test']['concerns']:,}"),
        _row_pattern(
            "**Total**",
            f"**{summary['total_articles']:,}**",
            f"**{summary['total_concerns']:,}**",
        ),
    ]

    return {
        "README.md": readme_patterns,
        "DATASHEET.md": datasheet_patterns,
    }

"""Leaderboard generation from BenchmarkResult files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from pydantic import ValidationError

from bioreview_bench.evaluate.metrics import DEFAULT_EMBEDDING_MODEL
from bioreview_bench.models.benchmark import BenchmarkResult
from bioreview_bench.project_defaults import DEFAULT_SOFTWARE_RELEASE_VERSION

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardEntry:
    """A single ranked row in the leaderboard table."""

    rank: int
    tool_name: str
    tool_version: str
    split: str
    recall: float
    precision: float
    f1: float
    recall_major: float
    n_articles: int
    run_date: str        # ISO-formatted date string
    notes: str
    result_file: str     # path relative to the source results directory
    ci_recall_lo: float | None = None
    ci_recall_hi: float | None = None
    ci_precision_lo: float | None = None
    ci_precision_hi: float | None = None


@dataclass(frozen=True)
class MatcherSignature:
    """Fields that must agree before benchmark scores can be ranked together."""

    method: str
    embedding_model: str | None
    embedding_revision: str | None
    effective_threshold: float
    algorithm: str
    figure_policy: str


# The historical v4 scores used this model ID, but did not record the resolved
# Hub commit. The missing revision is retained explicitly rather than invented.
FROZEN_MATCHER_SIGNATURE = MatcherSignature(
    method="embedding",
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    embedding_revision=None,
    effective_threshold=0.65,
    algorithm="hungarian",
    figure_policy="exclude",
)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

class Leaderboard:
    """Load, rank, and render benchmark results from a results directory."""

    def __init__(self, results_dir: Path, split: str = "val") -> None:
        """Load all ``*.json`` files from *results_dir*, filter to *split*,
        and sort by F1 (descending) then recall (descending).

        Args:
            results_dir: Directory containing BenchmarkResult JSON files.
            split: Dataset split to filter for (``"train"``, ``"val"``,
                   or ``"test"``).
        """
        self._split = split
        self._results_dir = Path(results_dir)
        self._entries: list[LeaderboardEntry] = []
        self._excluded_incompatible: list[str] = []
        self._load(self._results_dir)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[LeaderboardEntry]:
        """Ranked list of LeaderboardEntry objects."""
        return self._entries

    @property
    def excluded_incompatible(self) -> list[str]:
        """Result filenames excluded for a non-frozen matcher signature."""
        return list(self._excluded_incompatible)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the leaderboard as a GitHub-flavored markdown table.

        Returns:
            Multi-line string with a header, table, and footer note.
        """
        today = date.today().isoformat()
        has_ci = any(e.ci_recall_lo is not None for e in self._entries)

        lines: list[str] = [
            f"# bioreview-bench Frozen Score Snapshot ({self._split} split)",
            "",
            f"*Last updated: {today}. Ranked by F1.*",
            "",
        ]

        if has_ci:
            lines.append(
                "| Rank | Tool | Version | Recall | 95% CI | Precision | 95% CI "
                "| F1 | Major Recall | Articles | Date |"
            )
            lines.append(
                "|------|------|---------|--------|--------|-----------|--------|----|--------------|----------|------|"
            )
        else:
            lines.append(
                "| Rank | Tool | Version | Recall | Precision | F1 "
                "| Major Recall | Articles | Date |"
            )
            lines.append(
                "|------|------|---------|--------|-----------|-----|--------------|----------|------|"
            )

        for e in self._entries:
            if has_ci:
                ci_r = (
                    f"[{e.ci_recall_lo:.3f}, {e.ci_recall_hi:.3f}]"
                    if e.ci_recall_lo is not None else "—"
                )
                ci_p = (
                    f"[{e.ci_precision_lo:.3f}, {e.ci_precision_hi:.3f}]"
                    if e.ci_precision_lo is not None else "—"
                )
                lines.append(
                    f"| {e.rank} "
                    f"| {e.tool_name} "
                    f"| {e.tool_version} "
                    f"| {e.recall:.3f} "
                    f"| {ci_r} "
                    f"| {e.precision:.3f} "
                    f"| {ci_p} "
                    f"| {e.f1:.3f} "
                    f"| {e.recall_major:.3f} "
                    f"| {e.n_articles} "
                    f"| {e.run_date} |"
                )
            else:
                lines.append(
                    f"| {e.rank} "
                    f"| {e.tool_name} "
                    f"| {e.tool_version} "
                    f"| {e.recall:.3f} "
                    f"| {e.precision:.3f} "
                    f"| {e.f1:.3f} "
                    f"| {e.recall_major:.3f} "
                    f"| {e.n_articles} "
                    f"| {e.run_date} |"
                )

        lines.extend(self._footer_lines())
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        """Serialize the leaderboard as a JSON array of entry dicts.

        Returns:
            Pretty-printed JSON string.
        """
        return json.dumps([asdict(e) for e in self._entries], indent=2)

    def save(self, output_dir: Path) -> None:
        """Write ``leaderboard.md`` and ``leaderboard.json`` to *output_dir*.

        The directory is created if it does not already exist.

        Args:
            output_dir: Destination directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / "leaderboard.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")

        json_path = output_dir / "leaderboard.json"
        json_path.write_text(self.to_json(), encoding="utf-8")

    def _footer_lines(self) -> list[str]:
        signature = FROZEN_MATCHER_SIGNATURE
        revision = signature.embedding_revision or "unrecorded in historical runs"
        lines = [
            "",
            (
                f"> Frozen historical matcher wrapper: unadapted "
                f"`{signature.embedding_model}` with automatic mean pooling "
                f"(checkpoint revision: {revision}), "
                f"threshold={signature.effective_threshold:.2f}; "
                "threshold-aware cardinality-first Hungarian matching."
            ),
            "> Figure-issue concerns excluded from ground truth "
            "(require visual inspection).",
            "> Scores are matcher-dependent and provisional pending independent "
            "human validation of the matching threshold.",
            "> The six frozen raw result files store `f1_macro=0.0` as an "
            "unpopulated legacy sentinel; historical category-macro F1 is "
            "invalid and unreported.",
            "> Historical snapshot only; this is not an open public leaderboard. "
            "See [KNOWN_ISSUES.md]"
            "(https://github.com/jang1563/bioreview-bench/blob/v4.1.3/KNOWN_ISSUES.md).",
            f"> [bioreview-bench {DEFAULT_SOFTWARE_RELEASE_VERSION}]"
            "(https://github.com/jang1563/bioreview-bench)",
        ]
        if self._excluded_incompatible:
            lines.append(
                f"> Excluded {len(self._excluded_incompatible)} result file(s) with "
                "a non-frozen or incomplete matcher signature."
            )
        return lines

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, results_dir: Path) -> None:
        """Read all JSON files, parse BenchmarkResult objects, and rank."""
        raw: list[tuple[BenchmarkResult, str]] = []

        for json_file in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # Skip files that cannot be read or parsed
                continue

            try:
                result = BenchmarkResult.model_validate(data)
            except ValidationError:
                # Skip files that do not match the BenchmarkResult schema
                continue

            if result.split != self._split:
                continue
            if result.dedup_gt:
                continue

            signature = _matcher_signature(result)
            if signature != FROZEN_MATCHER_SIGNATURE:
                self._excluded_incompatible.append(
                    json_file.relative_to(results_dir).as_posix()
                )
                continue

            raw.append((result, json_file.relative_to(results_dir).as_posix()))

        # Keep the strongest non-dedup result for each tool/version pair.
        best_by_tool: dict[tuple[str, str], tuple[BenchmarkResult, str]] = {}
        for result, file_path in raw:
            key = (result.tool_name, result.tool_version)
            existing = best_by_tool.get(key)
            if existing is None:
                best_by_tool[key] = (result, file_path)
                continue

            prev, _ = existing
            prev_key = (prev.f1_micro, prev.recall_overall, prev.run_date)
            curr_key = (result.f1_micro, result.recall_overall, result.run_date)
            if curr_key > prev_key:
                best_by_tool[key] = (result, file_path)

        raw = list(best_by_tool.values())

        # Sort: primary = f1_micro descending, secondary = recall_overall descending
        raw.sort(key=lambda pair: (pair[0].f1_micro, pair[0].recall_overall), reverse=True)

        self._entries = []
        for rank, (result, file_path) in enumerate(raw, start=1):
            run_date_str = _format_date(result.run_date)
            ci_r_lo = result.ci_recall.lo if result.ci_recall else None
            ci_r_hi = result.ci_recall.hi if result.ci_recall else None
            ci_p_lo = result.ci_precision.lo if result.ci_precision else None
            ci_p_hi = result.ci_precision.hi if result.ci_precision else None
            self._entries.append(
                LeaderboardEntry(
                    rank=rank,
                    tool_name=result.tool_name,
                    tool_version=result.tool_version,
                    split=result.split,
                    recall=result.recall_overall,
                    precision=result.precision_overall,
                    f1=result.f1_micro,
                    recall_major=result.recall_major,
                    n_articles=result.n_articles,
                    run_date=run_date_str,
                    notes=result.notes,
                    result_file=file_path,
                    ci_recall_lo=ci_r_lo,
                    ci_recall_hi=ci_r_hi,
                    ci_precision_lo=ci_p_lo,
                    ci_precision_hi=ci_p_hi,
                )
            )


def _matcher_signature(result: BenchmarkResult) -> MatcherSignature | None:
    """Return the fully specified matcher signature recorded by a result."""
    matching = result.matching_stats
    if matching is None or matching.method is None:
        return None
    if matching.method == "embedding" and matching.embedding_model is None:
        return None
    return MatcherSignature(
        method=matching.method,
        embedding_model=matching.embedding_model,
        embedding_revision=matching.embedding_revision,
        effective_threshold=matching.threshold,
        algorithm=matching.algorithm,
        figure_policy=matching.figure_policy,
    )


def _format_date(dt: datetime) -> str:
    """Return an ISO-8601 date string (YYYY-MM-DD) from a datetime."""
    try:
        return dt.date().isoformat()
    except AttributeError:
        return str(dt)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def update_leaderboard(
    results_dir: Path,
    split: str = "val",
    output_dir: Path | None = None,
) -> Leaderboard:
    """Load results, build a leaderboard, and save it to *output_dir*.

    Args:
        results_dir: Directory containing ``*.json`` BenchmarkResult files.
        split: Dataset split to include (``"train"``, ``"val"``, ``"test"``).
        output_dir: Where to write ``leaderboard.md`` and
                    ``leaderboard.json``.  Defaults to *results_dir*.

    Returns:
        The constructed :class:`Leaderboard` instance.
    """
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir
    lb = Leaderboard(results_dir=results_dir, split=split)
    lb.save(output_dir)
    return lb


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Usage: python -m bioreview_bench.evaluate.leaderboard \
    #            --results-dir results/ --split val [--output-dir .]
    parser = argparse.ArgumentParser(
        description="Build and save the bioreview-bench leaderboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing BenchmarkResult *.json files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to include in the leaderboard.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where leaderboard.md and leaderboard.json are written. "
            "Defaults to --results-dir."
        ),
    )
    args = parser.parse_args()

    lb = update_leaderboard(
        results_dir=args.results_dir,
        split=args.split,
        output_dir=args.output_dir,
    )

    out = args.output_dir if args.output_dir is not None else args.results_dir
    print(f"Leaderboard saved to {out.resolve()}")
    print(f"  {len(lb.entries)} tool(s) ranked for split='{args.split}'")
    if lb.entries:
        top = lb.entries[0]
        print(f"  Top entry: {top.tool_name} v{top.tool_version}  F1={top.f1:.3f}")

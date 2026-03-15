"""Leaderboard generation from BenchmarkResult files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from bioreview_bench.models.benchmark import BenchmarkResult


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
    result_file: str     # absolute path of the source JSON file
    ci_recall_lo: float | None = None
    ci_recall_hi: float | None = None
    ci_precision_lo: float | None = None
    ci_precision_hi: float | None = None


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
        self._entries: list[LeaderboardEntry] = []
        self._matching_thresholds: set[float] = set()
        self._matching_algorithms: set[str] = set()
        self._load(Path(results_dir))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[LeaderboardEntry]:
        """Ranked list of LeaderboardEntry objects."""
        return self._entries

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
            f"# bioreview-bench Leaderboard ({self._split} split)",
            "",
            f"*Last updated: {today}. Ranked by F1.*",
            "",
        ]

        if has_ci:
            lines.append(
                "| Rank | Tool | Version | Recall | 95% CI | Precision | 95% CI | F1 | Major Recall | Articles | Date |"
            )
            lines.append(
                "|------|------|---------|--------|--------|-----------|--------|----|--------------|----------|------|"
            )
        else:
            lines.append(
                "| Rank | Tool | Version | Recall | Precision | F1 | Major Recall | Articles | Date |"
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
        return "\n".join(lines)

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
        if len(self._matching_thresholds) == 1 and len(self._matching_algorithms) == 1:
            threshold = next(iter(self._matching_thresholds))
            algorithm = next(iter(self._matching_algorithms))
            matching_line = (
                f"> Matching: SPECTER2 cosine similarity, threshold={threshold:.2f}, "
                f"{algorithm} bipartite matching."
            )
        else:
            matching_line = (
                "> Matching settings vary across result files. "
                "See each result JSON for exact threshold and algorithm."
            )

        return [
            "",
            matching_line,
            "> Figure-issue concerns excluded from ground truth "
            "(require visual inspection).",
            "> [bioreview-bench v1.0](https://github.com/jang1563/bioreview-bench)",
        ]

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

            if result.matching_stats is not None:
                self._matching_thresholds.add(result.matching_stats.threshold)
                self._matching_algorithms.add(result.matching_stats.algorithm)

            raw.append((result, str(json_file.resolve())))

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

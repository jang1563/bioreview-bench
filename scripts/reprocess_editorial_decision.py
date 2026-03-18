"""Re-infer editorial_decision from stored decision_letter_raw in JSONL files.

No re-downloading needed — just re-runs the updated _infer_decision() logic
on existing data.

Usage:
    python scripts/reprocess_editorial_decision.py
    python scripts/reprocess_editorial_decision.py --dry-run
    python scripts/reprocess_editorial_decision.py --sources elife
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import click

from bioreview_bench.parse.jats import JATSParser

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

_SOURCES = {
    "elife": ["elife_v1.1.jsonl", "elife_legacy_v1.jsonl"],
    "plos": ["plos_v1.jsonl"],
    "f1000": ["f1000_v1.jsonl"],
    "peerj": ["peerj_v1.jsonl"],
    "nature": ["nature_v1.jsonl"],
}


def reprocess_file(path: Path, dry_run: bool) -> dict[str, int]:
    """Re-infer editorial_decision for all entries in a JSONL file."""
    parser = JATSParser()
    stats: dict[str, int] = {"total": 0, "changed": 0, "had_dl": 0}
    transitions: Counter[str] = Counter()

    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    for entry in entries:
        stats["total"] += 1
        dl = entry.get("decision_letter_raw", "")
        if not dl:
            continue
        stats["had_dl"] += 1

        old = entry.get("editorial_decision", "unknown")
        # Only re-infer for "unknown" entries — preserve existing decisions
        # to avoid regressions (decision_letter_raw is full concatenated text,
        # while original _infer_decision ran on sub-article text)
        if old != "unknown":
            continue

        new = parser._infer_decision(dl)

        if new != "unknown":
            stats["changed"] += 1
            transitions[f"unknown -> {new}"] += 1
            entry["editorial_decision"] = new

    if not dry_run and stats["changed"] > 0:
        backup = path.with_suffix(".jsonl.bak")
        shutil.copy2(path, backup)
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats["transitions"] = dict(transitions)
    return stats


@click.command()
@click.option("--sources", "-s", multiple=True, default=list(_SOURCES.keys()),
              help="Sources to reprocess")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show changes without writing")
def main(sources: tuple[str, ...], dry_run: bool) -> None:
    """Re-infer editorial_decision from decision_letter_raw."""
    if dry_run:
        click.echo("[DRY RUN] No files will be modified.\n")

    for source in sources:
        filenames = _SOURCES.get(source, [])
        for fname in filenames:
            path = PROCESSED_DIR / fname
            if not path.exists():
                continue

            stats = reprocess_file(path, dry_run)
            click.echo(f"{fname}:")
            click.echo(f"  total={stats['total']}, had_dl={stats['had_dl']}, changed={stats['changed']}")
            if stats.get("transitions"):
                for trans, count in sorted(stats["transitions"].items()):
                    click.echo(f"    {trans}: {count}")
            click.echo()


if __name__ == "__main__":
    main()

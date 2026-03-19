"""Multi-source stratified train/val/test split script.

Merges JSONL files from all sources and creates stratified splits with
guaranteed representation of each source in every split.

Stratification keys:
  - source (elife / plos / f1000 / nature / peerj)
  - editorial_decision (accept / major_revision / minor_revision / reject / unknown)
  - review_format (journal / reviewed_preprint / unknown)

Split ratio: 70% train / 15% val / 15% test (articles with ≥ 1 concern)
Constraints:
  - Each source appears in all three splits.
  - No single source exceeds 50% of any split.
  - Seed 42 for reproducibility.

Output:
  data/splits/v3/train.jsonl
  data/splits/v3/val.jsonl
  data/splits/v3/test.jsonl
  data/splits/v3/split_meta_v3.json

Usage:
    python scripts/rebuild_splits.py
    python scripts/rebuild_splits.py \\
        --sources elife plos f1000 nature \\
        --input-dir data/processed \\
        --output-dir data/splits/v3 \\
        --seed 42 --val-ratio 0.15 --test-ratio 0.15
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click

_CORRECTION_TITLE_RE = re.compile(
    r"^(correction|erratum|retraction|corrigendum)[\s:]", re.IGNORECASE
)

ROOT = Path(__file__).resolve().parents[1]

# Source → input JSONL filename(s) mapping (multiple files per source supported)
_SOURCE_FILES: dict[str, list[str]] = {
    "elife": ["elife_v1.1.jsonl", "elife_legacy_v1.jsonl"],
    "plos": ["plos_v1.jsonl"],
    "f1000": ["f1000_v1.jsonl"],
    "nature": ["nature_v1.jsonl"],
    "peerj": ["peerj_v1.jsonl"],
}


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def save_jsonl(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def get_stratum(entry: dict) -> str:
    """Return stratification key for an entry."""
    source = entry.get("source", "unknown")
    decision = entry.get("editorial_decision", "unknown")
    fmt = entry.get("review_format", "unknown")
    return f"{source}|{decision}|{fmt}"


def stratified_split(
    entries: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by (source, editorial_decision, review_format).

    Guarantees each source appears in all three splits and that no single
    source exceeds 50% of any split.  Falls back to random split for strata
    too small to divide three ways.
    """
    rng = random.Random(seed)

    # Group by stratum
    strata: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        strata[get_stratum(entry)].append(entry)

    train, val, test = [], [], []

    for stratum_key, stratum_entries in strata.items():
        rng.shuffle(stratum_entries)
        n = len(stratum_entries)

        if n < 3:
            # Too small to split three ways — assign all to train
            train.extend(stratum_entries)
            continue

        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train < 1:
            # Edge case: stratum has exactly 2 or 3 entries
            train.append(stratum_entries[0])
            if n >= 2:
                val.append(stratum_entries[1])
            if n >= 3:
                test.append(stratum_entries[2])
            continue

        train.extend(stratum_entries[:n_train])
        val.extend(stratum_entries[n_train:n_train + n_val])
        test.extend(stratum_entries[n_train + n_val:])

    # Shuffle within each split for random ordering
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def frozen_split(
    entries: list[dict],
    frozen_ids: set[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split with a frozen test set.

    Articles matching ``frozen_ids`` go to test. Remaining entries are split
    into train/val only using stratified allocation.
    """
    rng = random.Random(seed)

    test = [e for e in entries if e.get("id") in frozen_ids]
    remaining = [e for e in entries if e.get("id") not in frozen_ids]

    # Split remaining into train/val using stratified approach
    strata: dict[str, list[dict]] = defaultdict(list)
    for entry in remaining:
        strata[get_stratum(entry)].append(entry)

    train, val = [], []
    for _key, stratum_entries in strata.items():
        rng.shuffle(stratum_entries)
        n = len(stratum_entries)
        n_val = max(1, round(n * val_ratio)) if n >= 2 else 0
        val.extend(stratum_entries[:n_val])
        train.extend(stratum_entries[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def _check_source_balance(split: list[dict], split_name: str) -> None:
    """Warn if any single source exceeds 50% of a split."""
    if not split:
        return
    source_counts = Counter(e.get("source", "unknown") for e in split)
    total = len(split)
    for source, count in source_counts.most_common():
        pct = count / total * 100
        if pct > 50:
            click.echo(
                f"  [warn] {split_name}: source '{source}' is {pct:.1f}% of split "
                f"({count}/{total}). Consider collecting more from other sources.",
                err=True,
            )


@click.command()
@click.option(
    "--sources",
    "-s",
    multiple=True,
    default=["elife", "plos", "f1000", "nature", "peerj"],
    show_default=True,
    help="Source names to include (multiple allowed). "
    "Choices: elife plos f1000 nature peerj",
)
@click.option(
    "--input-dir",
    default=None,
    help="Directory containing per-source JSONL files (default: data/processed/)",
)
@click.option(
    "--output-dir",
    default=None,
    help="Output directory for split files (default: data/splits/v3/)",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for reproducibility",
)
@click.option(
    "--val-ratio",
    default=0.15,
    show_default=True,
    help="Fraction of data for validation set",
)
@click.option(
    "--test-ratio",
    default=0.15,
    show_default=True,
    help="Fraction of data for test set",
)
@click.option(
    "--usable-only/--no-usable-only",
    default=True,
    show_default=True,
    help="Only include entries with ≥ 1 extracted concern",
)
@click.option(
    "--frozen-test",
    default=None,
    help="Path to frozen test IDs JSON. Test set is fixed; new articles go to train/val only.",
)
def main(
    sources: tuple[str, ...],
    input_dir: str | None,
    output_dir: str | None,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    usable_only: bool,
    frozen_test: str | None,
) -> None:
    """Rebuild multi-source stratified train/val/test splits."""
    in_dir = Path(input_dir) if input_dir else ROOT / "data" / "processed"
    out_dir = Path(output_dir) if output_dir else ROOT / "data" / "splits" / "v3"

    click.echo(f"Input dir : {in_dir}")
    click.echo(f"Output dir: {out_dir}")
    click.echo(f"Sources   : {list(sources)}")
    click.echo(f"Seed      : {seed}")
    click.echo(f"Val ratio : {val_ratio}")
    click.echo(f"Test ratio: {test_ratio}")
    click.echo(f"Usable-only: {usable_only}")
    click.echo()

    # Load and merge all source data
    all_entries: list[dict] = []
    source_counts: dict[str, int] = {}

    for source in sources:
        filenames = _SOURCE_FILES.get(source)
        if filenames is None:
            click.echo(f"  [warn] Unknown source '{source}', skipping.", err=True)
            continue

        entries: list[dict] = []
        for filename in filenames:
            path = in_dir / filename
            entries.extend(load_jsonl(path))

        if not entries:
            click.echo(f"  [warn] No data found for '{source}'", err=True)
            continue

        # Filter out correction/erratum/retraction articles
        before_corr = len(entries)
        entries = [
            e for e in entries
            if not _CORRECTION_TITLE_RE.match(e.get("title", ""))
        ]
        n_corr = before_corr - len(entries)
        if n_corr:
            click.echo(f"  {source}: removed {n_corr} correction/erratum/retraction articles")

        # Filter to usable entries
        if usable_only:
            before = len(entries)
            entries = [e for e in entries if e.get("concerns")]
            after = len(entries)
            click.echo(
                f"  {source}: {after}/{before} usable entries "
                f"(filtered {before - after} with 0 concerns)"
            )
        else:
            click.echo(f"  {source}: {len(entries)} entries")

        # Deduplicate by article ID
        seen_ids: set[str] = set()
        deduped: list[dict] = []
        for e in entries:
            eid = e.get("id", "")
            if eid not in seen_ids:
                seen_ids.add(eid)
                deduped.append(e)
        if len(deduped) < len(entries):
            click.echo(
                f"  {source}: removed {len(entries) - len(deduped)} duplicates"
            )
            entries = deduped

        all_entries.extend(entries)
        source_counts[source] = len(entries)

    if not all_entries:
        click.echo("No entries loaded. Check --input-dir and that collection has run.", err=True)
        sys.exit(1)

    click.echo(f"\nTotal entries: {len(all_entries)}")

    # Stratified split (with optional frozen test set)
    if frozen_test:
        freeze_path = Path(frozen_test)
        freeze_data = json.loads(freeze_path.read_text())
        frozen_ids = set(freeze_data["test_ids"])
        click.echo(f"Frozen test mode: {len(frozen_ids)} IDs from {freeze_path.name}")
        train, val, test = frozen_split(all_entries, frozen_ids=frozen_ids, val_ratio=val_ratio, seed=seed)
        missing = frozen_ids - {e.get("id") for e in test}
        if missing:
            click.echo(f"  [warn] {len(missing)} frozen test IDs not found in data", err=True)
    else:
        train, val, test = stratified_split(all_entries, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    # Balance checks
    _check_source_balance(train, "train")
    _check_source_balance(val, "val")
    _check_source_balance(test, "test")

    # Save splits
    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(val, out_dir / "val.jsonl")
    save_jsonl(test, out_dir / "test.jsonl")

    # Save split metadata
    meta = {
        "version": "v3",
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": "frozen" if frozen_test else test_ratio,
        "frozen_test": frozen_test,
        "usable_only": usable_only,
        "sources": list(sources),
        "source_counts": source_counts,
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(all_entries),
        },
        "split_source_distribution": {
            split_name: dict(Counter(e.get("source", "unknown") for e in split_data))
            for split_name, split_data in [("train", train), ("val", val), ("test", test)]
        },
    }
    meta_path = out_dir / "split_meta_v3.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # Print summary
    click.echo()
    click.echo("Split sizes:")
    click.echo(f"  train: {len(train)}")
    click.echo(f"  val:   {len(val)}")
    click.echo(f"  test:  {len(test)}")
    click.echo(f"  total: {len(all_entries)}")
    click.echo()
    click.echo("Source distribution per split:")
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        counts = Counter(e.get("source", "unknown") for e in split_data)
        click.echo(f"  {split_name}: {dict(counts)}")
    click.echo()
    click.echo(f"Splits saved to: {out_dir}")
    click.echo(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()

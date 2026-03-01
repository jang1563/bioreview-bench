"""Train / val / test split script.

Merges JSONL files and creates stratified splits for benchmark use.

Stratification criteria:
- review_format (journal / reviewed_preprint)
- Stance distribution (no_response-heavy vs. conceded/rebutted-heavy)
- Subject diversity

Split ratio: 70% train / 15% val / 15% test (articles with >= 1 concern)
Test set is saved separately to prevent label leakage.

Frozen test mode (--frozen-test):
  When a frozen test ID file exists, the test set is fixed. New articles
  are distributed only to train/val, preserving benchmark comparability.

Usage:
    # Fresh split (no frozen test)
    python scripts/create_splits.py

    # Incremental split with frozen test
    python scripts/create_splits.py --frozen-test data/splits/test_ids_frozen.json

    # Multi-source with explicit input files
    python scripts/create_splits.py --input-files data/processed/elife_v1.1.jsonl \
        data/processed/elife_legacy_v1.jsonl data/processed/plos_v1.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_jsonl(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def get_stratum(entry: dict) -> str:
    """Generate stratification key for an entry."""
    fmt = entry.get("review_format", "unknown")

    concerns = entry.get("concerns", [])
    if not concerns:
        return f"{fmt}:no_concerns"

    stances = [c.get("author_stance", "unclear") for c in concerns]
    conceded_pct = stances.count("conceded") / len(stances)
    no_resp_pct = stances.count("no_response") / len(stances)

    if no_resp_pct >= 0.8:
        stance_bucket = "high_noresp"
    elif conceded_pct >= 0.3:
        stance_bucket = "high_concede"
    else:
        stance_bucket = "mixed"

    # Primary subject bucket
    subj = entry.get("subjects", [])
    if subj:
        primary = subj[0]
        if "neuro" in primary.lower():
            subj_bucket = "neuro"
        elif "cell" in primary.lower():
            subj_bucket = "cell"
        else:
            subj_bucket = "other"
    else:
        subj_bucket = "unknown"

    return f"{fmt}:{stance_bucket}:{subj_bucket}"


def stratified_split(
    entries: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split into (train, val, test).

    Only articles with >= 1 concern are split; zero-concern articles
    go entirely into train (useful as negative examples).
    """
    rng = random.Random(seed)

    usable = [e for e in entries if len(e.get("concerns", [])) >= 1]
    unusable = [e for e in entries if len(e.get("concerns", [])) == 0]

    print(f"Usable entries: {len(usable)} / {len(entries)}")
    print(f"Zero-concern entries: {len(unusable)} -> added to train split")

    strata: dict[str, list[dict]] = defaultdict(list)
    for e in usable:
        strata[get_stratum(e)].append(e)

    train, val, test = [], [], []

    for stratum_key, stratum_entries in sorted(strata.items()):
        rng.shuffle(stratum_entries)
        n = len(stratum_entries)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train < 0:
            # Stratum too small: all goes to train
            train.extend(stratum_entries)
            continue

        test.extend(stratum_entries[:n_test])
        val.extend(stratum_entries[n_test:n_test + n_val])
        train.extend(stratum_entries[n_test + n_val:])

    # Zero-concern entries go to train
    train.extend(unusable)

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

    Articles matching frozen_ids go to test unchanged. Remaining articles
    are split into train/val only (no new test entries). Zero-concern
    articles go to train.
    """
    rng = random.Random(seed)

    test = [e for e in entries if e.get("id") in frozen_ids]
    remaining = [e for e in entries if e.get("id") not in frozen_ids]

    # Check for missing frozen IDs (articles removed from data)
    found_ids = {e["id"] for e in test}
    missing = frozen_ids - found_ids
    if missing:
        print(f"[warn] {len(missing)} frozen test IDs not found in data")

    usable = [e for e in remaining if len(e.get("concerns", [])) >= 1]
    unusable = [e for e in remaining if len(e.get("concerns", [])) == 0]

    print(f"Frozen test: {len(test)} articles (fixed)")
    print(f"Remaining usable: {len(usable)} / {len(remaining)}")
    print(f"Zero-concern: {len(unusable)} -> added to train")

    # Check stratum coverage
    test_strata = {get_stratum(e) for e in test if e.get("concerns")}
    remaining_strata = {get_stratum(e) for e in usable}
    uncovered = remaining_strata - test_strata
    if uncovered:
        print(f"[warn] {len(uncovered)} new strata not in frozen test: {sorted(uncovered)}")

    # Split remaining into train/val only
    strata: dict[str, list[dict]] = defaultdict(list)
    for e in usable:
        strata[get_stratum(e)].append(e)

    train, val = [], []
    for stratum_key, stratum_entries in sorted(strata.items()):
        rng.shuffle(stratum_entries)
        n = len(stratum_entries)
        n_val = max(1, round(n * val_ratio))

        if n - n_val < 1:
            # Stratum too small: all goes to train
            train.extend(stratum_entries)
            continue

        val.extend(stratum_entries[:n_val])
        train.extend(stratum_entries[n_val:])

    train.extend(unusable)
    rng.shuffle(train)
    rng.shuffle(val)

    return train, val, test


def print_split_stats(name: str, entries: list[dict]) -> None:
    concerns = [c for e in entries for c in e.get("concerns", [])]
    fmts = Counter(e.get("review_format", "unknown") for e in entries)
    stances = Counter(c.get("author_stance", "?") for c in concerns)

    print(f"\n  [{name}] {len(entries)} articles, {len(concerns)} concerns")
    print(f"    review_format: {dict(fmts)}")
    if concerns:
        print(f"    stances: {dict(stances)}")


@click.command()
@click.option("--new-format", default=None, help="New-format JSONL (elife_v1.1.jsonl)")
@click.option("--legacy", default=None, help="Legacy-format JSONL (elife_legacy_v1.jsonl)")
@click.option(
    "--input-files", "-i",
    multiple=True,
    help="Input JSONL files (replaces --new-format/--legacy when specified)",
)
@click.option(
    "--frozen-test",
    default=None,
    help="Path to frozen test IDs JSON (test set preserved, new articles → train/val only)",
)
@click.option("--output-dir", default=None, help="Output directory (default: data/splits/)")
@click.option("--seed", default=42, show_default=True, help="Random seed")
@click.option("--val-ratio", default=0.15, show_default=True)
@click.option("--test-ratio", default=0.15, show_default=True)
def main(
    new_format: str | None,
    legacy: str | None,
    input_files: tuple[str, ...],
    frozen_test: str | None,
    output_dir: str | None,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Create stratified train/val/test splits from collected JSONL files."""
    out_dir = Path(output_dir) if output_dir else ROOT / "data" / "splits"

    # Determine input files
    if input_files:
        paths = [Path(p) for p in input_files]
    else:
        new_path = Path(new_format) if new_format else ROOT / "data" / "processed" / "elife_v1.1.jsonl"
        legacy_path = Path(legacy) if legacy else ROOT / "data" / "processed" / "elife_legacy_v1.jsonl"
        paths = [new_path, legacy_path]

    all_entries: list[dict] = []
    source_paths: list[str] = []
    for path in paths:
        if path.exists():
            entries = load_jsonl(path)
            print(f"Loaded {len(entries)} from {path.name}")
            all_entries.extend(entries)
            source_paths.append(str(path))
        else:
            print(f"[warn] {path} not found — skipping")

    if not all_entries:
        print("ERROR: No entries loaded.")
        sys.exit(1)

    # Deduplicate by ID (in case of overlapping files)
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for e in all_entries:
        eid = e.get("id", "")
        if eid not in seen_ids:
            seen_ids.add(eid)
            deduped.append(e)
    if len(deduped) < len(all_entries):
        print(f"Deduplication: {len(all_entries)} → {len(deduped)} entries")
    all_entries = deduped

    print(f"\nTotal: {len(all_entries)} entries")

    # Split: frozen test mode vs. fresh split
    if frozen_test:
        freeze_path = Path(frozen_test)
        freeze_data = json.loads(freeze_path.read_text())
        frozen_ids = set(freeze_data["test_ids"])
        print(f"\nFrozen test mode: {len(frozen_ids)} IDs from {freeze_path.name}")

        train, val, test = frozen_split(all_entries, frozen_ids, val_ratio, seed)
    else:
        train, val, test = stratified_split(all_entries, val_ratio, test_ratio, seed)

    print("\n=== Split summary ===")
    print_split_stats("train", train)
    print_split_stats("val", val)
    print_split_stats("test", test)

    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(val, out_dir / "val.jsonl")
    save_jsonl(test, out_dir / "test.jsonl")

    print(f"\nSaved splits to {out_dir}/")
    print(f"  train.jsonl : {len(train)} entries")
    print(f"  val.jsonl   : {len(val)} entries")
    print(f"  test.jsonl  : {len(test)} entries")

    meta = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "frozen_test": str(frozen_test) if frozen_test else None,
        "total": len(all_entries),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "sources": source_paths,
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

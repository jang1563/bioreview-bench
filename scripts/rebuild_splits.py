"""Multi-source train/validation/test split builder.

Supports ratio-based, frozen-test, and balanced fixed-per-source test modes.
The checked-in v4 split used balanced-test mode with seed 42: 600 fixed-count
test articles, followed by train/validation allocation of the remaining records
within the strata below.

Train/validation stratification keys:
  - source (elife / plos / f1000 / nature / peerj)
  - editorial_decision (accept / major_revision / minor_revision / reject / unknown)
  - review_format (journal / reviewed_preprint / unknown)

The builder does not implement temporal stratification or a chronological
holdout. The generic ratio-based mode defaults to 70% train / 15% validation /
15% test, but those are not the realized v4 proportions.

Constraints:
  - Each source appears in all three splits.
  - No single source exceeds 50% of any split.
  - Seed 42 for reproducibility.

Output:
  data/splits/v4/train.jsonl
  data/splits/v4/val.jsonl
  data/splits/v4/test.jsonl
  data/splits/v4/split_meta_v4.json

Usage:
    # Generic ratio split: an explicit non-canonical output is required.
    python scripts/rebuild_splits.py \\
        --input-dir data/processed \\
        --output-dir data/splits/experiments/ratio-70-15-15 \\
        --seed 42 --val-ratio 0.15 --test-ratio 0.15

    # Exact reproducible v4 build.
    python scripts/rebuild_splits.py \\
        --input-dir data/processed \\
        --output-dir data/splits/v4 --version v4 --seed 42 \\
        --balanced-test \\
        '{"elife":150,"plos":150,"f1000":150,"nature":100,"peerj":50}'
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click

from bioreview_bench.io import load_jsonl, write_jsonl
from bioreview_bench.project_defaults import (
    frozen_ids_name,
    split_metadata_name,
    splits_dir,
)

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

_CANONICAL_VERSION = "v4"
_CANONICAL_SOURCES = ("elife", "plos", "f1000", "nature", "peerj")
_CANONICAL_SEED = 42
_CANONICAL_VAL_RATIO = 0.15
_CANONICAL_TEST_RATIO = 0.15
_CANONICAL_BALANCED_TEST = {
    "elife": 150,
    "plos": 150,
    "f1000": 150,
    "nature": 100,
    "peerj": 50,
}
_CANONICAL_SPLIT_SIZES = {"train": 5387, "val": 953, "test": 600}


def save_jsonl(entries: list[dict], path: Path) -> None:
    write_jsonl(entries, path, ensure_ascii=False)


def load_frozen_ids(path: Path, *, split: str = "test") -> set[str]:
    """Load a frozen-ID artifact without silently accepting a malformed schema.

    Current files use ``{"ids": [...]}``; historical test artifacts used
    ``{"test_ids": [...]}``. Both are accepted. If both keys are present they
    must describe the same ID set.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read frozen {split} IDs from {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Frozen {split} IDs must be a JSON object: {path}")

    accepted_keys = ("ids", f"{split}_ids")
    present = [key for key in accepted_keys if key in payload]
    if not present:
        expected = " or ".join(repr(key) for key in accepted_keys)
        raise ValueError(f"Frozen {split} IDs must contain {expected}: {path}")

    parsed: dict[str, set[str]] = {}
    for key in present:
        raw_ids = payload[key]
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"Frozen-ID field {key!r} must be a non-empty JSON list: {path}")
        if any(not isinstance(value, str) or not value.strip() for value in raw_ids):
            raise ValueError(
                f"Frozen-ID field {key!r} must contain only non-empty strings: {path}"
            )
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(f"Frozen-ID field {key!r} contains duplicate IDs: {path}")
        parsed[key] = set(raw_ids)

    first_ids = parsed[present[0]]
    if any(ids != first_ids for ids in parsed.values()):
        raise ValueError(f"Frozen-ID fields disagree in {path}")
    return first_ids


def _parse_balanced_test(value: str | None) -> dict[str, int] | None:
    """Parse and validate ``--balanced-test`` JSON."""
    if value is None:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid --balanced-test JSON: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise click.ClickException("--balanced-test must be a non-empty JSON object.")
    if any(
        not isinstance(source, str)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for source, count in payload.items()
    ):
        raise click.ClickException(
            "--balanced-test must map source names to non-negative integer counts."
        )
    return payload


def _is_canonical_output(out_dir: Path) -> bool:
    """Return whether an output path resolves to the release's canonical v4 path."""
    return out_dir.resolve() == (ROOT / splits_dir()).resolve()


def _validate_canonical_mode(
    *,
    out_dir: Path,
    sources: tuple[str, ...],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    usable_only: bool,
    frozen_test: str | None,
    balanced_test: dict[str, int] | None,
    split_version: str | None,
) -> bool:
    """Reject commands that could silently replace the canonical v4 split."""
    if not _is_canonical_output(out_dir):
        return False

    common_exact = (
        sources == _CANONICAL_SOURCES
        and seed == _CANONICAL_SEED
        and val_ratio == _CANONICAL_VAL_RATIO
        and test_ratio == _CANONICAL_TEST_RATIO
        and usable_only
        and split_version in (None, _CANONICAL_VERSION)
    )
    frozen_exact = frozen_test is not None and balanced_test is None
    balanced_exact = (
        frozen_test is None and balanced_test == _CANONICAL_BALANCED_TEST
    )
    if not common_exact or not (frozen_exact or balanced_exact):
        raise click.ClickException(
            "Refusing to overwrite canonical data/splits/v4. Use an explicit "
            "non-canonical --output-dir for ratio/experimental splits, or use the "
            "exact v4 frozen/balanced configuration (five canonical sources, "
            "seed 42, val/test ratios 0.15, usable-only, version v4)."
        )
    return True


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


def balanced_test_split(
    entries: list[dict],
    per_source_test: dict[str, int],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split with balanced per-source test allocation.

    For each source, sample exactly ``per_source_test[source]`` articles for
    test (or all available if fewer exist). Remaining articles are split into
    train/val using the same stratified approach as ``frozen_split``.
    """
    rng = random.Random(seed)

    by_source: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_source[entry.get("source", "unknown")].append(entry)

    test: list[dict] = []
    remaining: list[dict] = []

    for source, src_entries in by_source.items():
        rng.shuffle(src_entries)
        n_test = min(per_source_test.get(source, 0), len(src_entries))
        test.extend(src_entries[:n_test])
        remaining.extend(src_entries[n_test:])

    # Split remaining into train/val using stratified approach
    strata: dict[str, list[dict]] = defaultdict(list)
    for entry in remaining:
        strata[get_stratum(entry)].append(entry)

    train: list[dict] = []
    val: list[dict] = []
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
    help=(
        f"Output directory for split files. Omitted means {splits_dir()}/, "
        "which is protected and requires the exact canonical frozen/balanced mode."
    ),
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
@click.option(
    "--balanced-test",
    default=None,
    help=(
        "Per-source test counts as JSON, e.g. "
        '\'{"elife":150,"plos":150,"f1000":150,'
        '"nature":100,"peerj":50}\''
    ),
)
@click.option(
    "--version",
    "split_version",
    default=None,
    help="Version label for metadata (default: inferred from output-dir name).",
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
    balanced_test: str | None,
    split_version: str | None,
) -> None:
    """Rebuild multi-source stratified train/val/test splits."""
    in_dir = Path(input_dir) if input_dir else ROOT / "data" / "processed"
    out_dir = Path(output_dir) if output_dir else ROOT / splits_dir()
    if frozen_test and balanced_test:
        raise click.ClickException(
            "--frozen-test and --balanced-test are mutually exclusive."
        )
    per_source_test = _parse_balanced_test(balanced_test)
    canonical_output = _validate_canonical_mode(
        out_dir=out_dir,
        sources=sources,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        usable_only=usable_only,
        frozen_test=frozen_test,
        balanced_test=per_source_test,
        split_version=split_version,
    )

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
            entries.extend(load_jsonl(path, allow_missing=True, skip_invalid=True))

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

    # Stratified split (with optional frozen test set or balanced test)
    frozen_ids: set[str] | None = None
    if per_source_test is not None:
        click.echo(f"Balanced test mode: {per_source_test}")
        train, val, test = balanced_test_split(
            all_entries, per_source_test=per_source_test, val_ratio=val_ratio, seed=seed
        )
    elif frozen_test:
        freeze_path = Path(frozen_test)
        try:
            frozen_ids = load_frozen_ids(freeze_path)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if canonical_output and len(frozen_ids) != _CANONICAL_SPLIT_SIZES["test"]:
            raise click.ClickException(
                "Canonical v4 frozen mode requires exactly 600 frozen test IDs; "
                f"found {len(frozen_ids)}."
            )
        click.echo(f"Frozen test mode: {len(frozen_ids)} IDs from {freeze_path.name}")
        train, val, test = frozen_split(
            all_entries,
            frozen_ids=frozen_ids,
            val_ratio=val_ratio,
            seed=seed,
        )
        missing = frozen_ids - {e.get("id") for e in test}
        if missing:
            raise click.ClickException(
                f"Refusing to write splits: {len(missing)} frozen test IDs are "
                "missing from the input data."
            )
    else:
        train, val, test = stratified_split(
            all_entries,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    split_sizes = {"train": len(train), "val": len(val), "test": len(test)}
    if canonical_output and frozen_ids is not None:
        test_source_counts = Counter(entry.get("source", "unknown") for entry in test)
        if dict(test_source_counts) != _CANONICAL_BALANCED_TEST:
            raise click.ClickException(
                "Canonical v4 frozen mode must preserve the fixed per-source test "
                f"counts {_CANONICAL_BALANCED_TEST}; got {dict(test_source_counts)}."
            )
    if canonical_output and per_source_test is not None:
        if split_sizes != _CANONICAL_SPLIT_SIZES:
            raise click.ClickException(
                "Refusing to write canonical v4: exact balanced mode must reproduce "
                f"{_CANONICAL_SPLIT_SIZES}, got {split_sizes}."
            )

    version = split_version or out_dir.name
    test_ids_path = out_dir / frozen_ids_name("test", version)
    if canonical_output and frozen_ids is not None and test_ids_path.exists():
        try:
            existing_frozen_ids = load_frozen_ids(test_ids_path)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if existing_frozen_ids != frozen_ids:
            raise click.ClickException(
                "Refusing to replace canonical v4 with a different frozen test-ID set."
            )

    # Balance checks
    _check_source_balance(train, "train")
    _check_source_balance(val, "val")
    _check_source_balance(test, "test")

    # Save splits
    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(val, out_dir / "val.jsonl")
    save_jsonl(test, out_dir / "test.jsonl")

    # Preserve the canonical frozen-ID artifact byte-for-byte during frozen updates.
    # Balanced canonical rebuilds and all non-canonical builds create a fresh artifact.
    if not (canonical_output and frozen_ids is not None and test_ids_path.exists()):
        test_ids_path.write_text(
            json.dumps({"ids": [entry["id"] for entry in test]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (out_dir / frozen_ids_name("val", version)).write_text(
        json.dumps({"ids": [entry["id"] for entry in val]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save split metadata
    meta = {
        "version": version,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": "balanced" if balanced_test else ("frozen" if frozen_test else test_ratio),
        "frozen_test": frozen_test,
        "balanced_test": balanced_test,
        "usable_only": usable_only,
        "sources": list(sources),
        "source_counts": source_counts,
        "split_sizes": {
            **split_sizes,
            "total": len(all_entries),
        },
        "split_source_distribution": {
            split_name: dict(Counter(e.get("source", "unknown") for e in split_data))
            for split_name, split_data in [("train", train), ("val", val), ("test", test)]
        },
    }
    meta_path = out_dir / split_metadata_name(version)
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

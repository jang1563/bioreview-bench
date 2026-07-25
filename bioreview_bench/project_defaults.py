"""Shared version and path defaults for supported bioreview-bench workflows."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
SPLITS_ROOT = DATA_DIR / "splits"
RESULTS_ROOT = Path("results")
STATS_ROOT = DATA_DIR / "stats"
VALIDATION_ROOT = DATA_DIR / "validation"

# Package-supported CLIs and release tooling target the current public benchmark.
DEFAULT_BENCHMARK_SPLIT_VERSION = "v4"
DEFAULT_PUBLIC_RELEASE_VERSION = "v4"
DEFAULT_SOFTWARE_RELEASE_VERSION = "v4.1.3"
LEGACY_SPLIT_VERSION = "v3"


def splits_dir(version: str = DEFAULT_BENCHMARK_SPLIT_VERSION) -> Path:
    """Return the canonical split directory for a benchmark version."""
    return SPLITS_ROOT / version


def results_dir(version: str = DEFAULT_PUBLIC_RELEASE_VERSION) -> Path:
    """Return the versioned results directory for a public release."""
    return RESULTS_ROOT / version


def split_metadata_name(version: str = DEFAULT_BENCHMARK_SPLIT_VERSION) -> str:
    """Return the metadata filename for a split version."""
    return f"split_meta_{version}.json"


def frozen_ids_name(split: str, version: str = DEFAULT_BENCHMARK_SPLIT_VERSION) -> str:
    """Return the frozen split ID filename for a split/version pair."""
    return f"{split}_ids_frozen_{version}.json"


def resolve_frozen_ids_path(
    splits_root: Path,
    *,
    split: str,
    version: str = DEFAULT_BENCHMARK_SPLIT_VERSION,
) -> Path | None:
    """Resolve the best available frozen split-id file.

    Prefer the current version's file, but fall back to historical files while the
    repository transitions older metadata forward.
    """
    preferred_name = frozen_ids_name(split, version)
    candidates = [
        splits_root / version / preferred_name,
        splits_root / preferred_name,
    ]

    legacy_name = frozen_ids_name(split, LEGACY_SPLIT_VERSION)
    if legacy_name != preferred_name:
        candidates.extend([
            splits_root / version / legacy_name,
            splits_root / legacy_name,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


DEFAULT_BENCHMARK_SPLITS_DIR = splits_dir()
DEFAULT_PUBLIC_RESULTS_DIR = results_dir()
DEFAULT_STATS_JSON = STATS_ROOT / f"{DEFAULT_BENCHMARK_SPLIT_VERSION}_summary.json"
DEFAULT_STATS_MD = STATS_ROOT / f"{DEFAULT_BENCHMARK_SPLIT_VERSION}_summary.md"
DEFAULT_HUMAN_SUBSET_SPLITS_DIR = DEFAULT_BENCHMARK_SPLITS_DIR
DEFAULT_HUMAN_SUBSET_OUTPUT = VALIDATION_ROOT / "human_subset_v1.jsonl"
DEFAULT_HUMAN_SUBSET_MANIFEST = VALIDATION_ROOT / "human_subset_v1_manifest.json"

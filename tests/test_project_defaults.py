from __future__ import annotations

from pathlib import Path

from bioreview_bench.project_defaults import (
    DEFAULT_BENCHMARK_SPLIT_VERSION,
    DEFAULT_BENCHMARK_SPLITS_DIR,
    DEFAULT_PUBLIC_RELEASE_VERSION,
    DEFAULT_PUBLIC_RESULTS_DIR,
    DEFAULT_SOFTWARE_RELEASE_VERSION,
    resolve_frozen_ids_path,
)


def test_public_defaults_track_v4() -> None:
    assert DEFAULT_BENCHMARK_SPLIT_VERSION == "v4"
    assert DEFAULT_PUBLIC_RELEASE_VERSION == "v4"
    assert DEFAULT_SOFTWARE_RELEASE_VERSION == "v4.1.3"
    assert DEFAULT_BENCHMARK_SPLITS_DIR == Path("data/splits/v4")
    assert DEFAULT_PUBLIC_RESULTS_DIR == Path("results/v4")


def test_resolve_frozen_ids_prefers_current_version(tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"
    current = splits_root / "v4" / "test_ids_frozen_v4.json"
    current.parent.mkdir(parents=True)
    current.write_text('{"ids": ["a"]}', encoding="utf-8")
    legacy = splits_root / "test_ids_frozen_v3.json"
    legacy.write_text('{"ids": ["legacy"]}', encoding="utf-8")

    assert resolve_frozen_ids_path(splits_root, split="test") == current


def test_resolve_frozen_ids_falls_back_to_legacy(tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"
    legacy = splits_root / "test_ids_frozen_v3.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"ids": ["legacy"]}', encoding="utf-8")

    assert resolve_frozen_ids_path(splits_root, split="test") == legacy

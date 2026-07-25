from __future__ import annotations

from pathlib import Path


def test_scripts_readme_exists() -> None:
    assert Path("scripts/README.md").exists()


def test_public_docs_and_root_scripts_have_no_stale_v3_defaults() -> None:
    stale_patterns = (
        "data/splits/v3",
        "results/v3",
        "test_ids_frozen_v3",
        "split_meta_v3",
        "haiku_test_v3",
        "_test_v3",
    )
    targets = [
        Path("README.md"),
        Path("DATASHEET.md"),
        *sorted(Path("scripts").glob("*.py")),
        *sorted(Path(".github/workflows").glob("*.yml")),
    ]

    offenders: list[str] = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for pattern in stale_patterns:
            if pattern in text:
                offenders.append(f"{path}: {pattern}")

    assert offenders == []

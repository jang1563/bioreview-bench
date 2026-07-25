from __future__ import annotations

from pathlib import Path

import pytest

from bioreview_bench.collect.postprocess import load_known_ids


def test_load_known_ids_reads_existing_ids(tmp_path: Path) -> None:
    path = tmp_path / "existing.jsonl"
    path.write_text('{"id": "a1"}\n{"id": "a2"}\n', encoding="utf-8")

    assert load_known_ids(path) == {"a1", "a2"}


def test_load_known_ids_fails_on_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a1"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        load_known_ids(path)

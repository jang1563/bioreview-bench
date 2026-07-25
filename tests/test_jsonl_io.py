from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioreview_bench.io import load_jsonl, write_jsonl


def test_load_jsonl_allow_missing(tmp_path: Path) -> None:
    assert load_jsonl(tmp_path / "missing.jsonl", allow_missing=True) == []


def test_load_jsonl_strict_malformed_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        load_jsonl(path)


def test_load_jsonl_skip_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text('{"id": "a"}\nnot-json\n{"id": "b"}\n', encoding="utf-8")

    rows = load_jsonl(path, skip_invalid=True)

    assert [row["id"] for row in rows] == ["a", "b"]


def test_write_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]

    count = write_jsonl(rows, path)

    assert count == 2
    assert load_jsonl(path) == rows

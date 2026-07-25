"""Shared JSONL readers/writers with explicit strictness controls."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable


def _format_json_error(path: Path, lineno: int, exc: json.JSONDecodeError) -> str:
    return f"Malformed JSON in {path} at line {lineno}: {exc.msg}"


def load_jsonl(
    path: Path,
    *,
    allow_missing: bool = False,
    skip_invalid: bool = False,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Load JSONL rows from *path*.

    Args:
        path: JSONL file path.
        allow_missing: Return ``[]`` instead of raising when the file is absent.
        skip_invalid: Skip malformed JSON rows instead of raising ``ValueError``.
        logger: Logger used when ``allow_missing`` or ``skip_invalid`` is enabled.
    """
    if not path.exists():
        if allow_missing:
            if logger is not None:
                logger.warning("JSONL file not found: %s", path)
            return []
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                message = _format_json_error(path, lineno, exc)
                if skip_invalid:
                    if logger is not None:
                        logger.warning(message)
                    continue
                raise ValueError(message) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path} at line {lineno}")
            rows.append(payload)
    return rows


def write_jsonl(
    entries: Iterable[Any],
    path: Path,
    *,
    ensure_ascii: bool = False,
    default: Any = str,
) -> int:
    """Write *entries* as JSONL and return the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=ensure_ascii, default=default) + "\n")
            count += 1
    return count

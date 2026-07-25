"""Build the rights-minimized public Hugging Face release.

The private research corpus contains publisher article text, peer-review prose,
and author responses whose redistribution terms vary by article. This module
creates a clean public index using strict, text-free whitelists:

- bibliographic/benchmark identifiers for all splits; and
- project-created categorical label rows for train and validation.

It does not export article/abstract text, normalized concern text, raw review or
response text, reviewer/data-row names, emails, or explicit identity fields,
test targets, or internal traces. Stable concern IDs may retain source-local
review ordinals; those ordinals do not identify a person. This data-row policy
does not exclude project-maintainer contact metadata in ``CITATION.cff``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from bioreview_bench.project_defaults import DEFAULT_SOFTWARE_RELEASE_VERSION

PUBLIC_INDEX_FIELDS: tuple[str, ...] = (
    "id",
    "source",
    "doi",
    "published_date",
    "schema_version",
)

PUBLIC_ANNOTATION_FIELDS: tuple[str, ...] = (
    "concern_id",
    "category",
    "severity",
    "author_stance",
)

PUBLIC_ANNOTATION_OUTPUT_FIELDS: tuple[str, ...] = (
    "article_id",
    "source",
    *PUBLIC_ANNOTATION_FIELDS,
)

_CANONICAL_SOURCES: frozenset[str] = frozenset(
    {"elife", "f1000", "nature", "peerj", "plos"}
)
_CONCERN_CATEGORIES: frozenset[str] = frozenset(
    {
        "design_flaw",
        "statistical_methodology",
        "missing_experiment",
        "figure_issue",
        "prior_art_novelty",
        "writing_clarity",
        "reagent_method_specificity",
        "interpretation",
        "other",
    }
)
_CONCERN_SEVERITIES: frozenset[str] = frozenset(
    {"major", "minor", "optional"}
)
_AUTHOR_STANCES: frozenset[str] = frozenset(
    {"conceded", "rebutted", "partial", "unclear", "no_response"}
)
_ARTICLE_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "elife": re.compile(r"elife:[1-9][0-9]*"),
    "f1000": re.compile(
        r"f1000:10\.12688_f1000research\.[1-9][0-9]*\.[1-9][0-9]*"
    ),
    "nature": re.compile(
        r"nature:s[0-9]{5}-[0-9]{3}-[0-9]{5}-[a-z0-9]"
    ),
    "peerj": re.compile(r"peerj:[1-9][0-9]*"),
    "plos": re.compile(
        r"plos:10\.1371/journal\.(?:pbio|pcbi|pgen|pmed)\.[0-9]+"
    ),
}
_DOI_PATTERN = re.compile(
    r"10\.[0-9]{4,9}/[A-Za-z0-9][A-Za-z0-9._;()/:\-]*"
)
_CONCERN_ORDINAL_PATTERN = re.compile(r"R[1-9][0-9]*C[1-9][0-9]*")
_FROZEN_SPLIT_FILES: dict[str, str] = {
    "validation": "data/splits/v4/val_ids_frozen_v4.json",
    "test": "data/splits/v4/test_ids_frozen_v4.json",
}

BLOCKED_PUBLIC_FIELDS: frozenset[str] = frozenset(
    {
        "title",
        "abstract",
        "subjects",
        "paper_text_sections",
        "paper_text_v1_sections",
        "structured_references",
        "decision_letter_raw",
        "author_response_raw",
        "concerns",
        "concern_text",
        "author_response_text",
        "evidence_of_change",
        "resolution",
        "resolution_confidence",
        "was_valid",
        "raised_by_multiple",
        "requires_figure_reading",
        "reviewer_num",
        "reviewer_name",
        "reviewer_email",
        "review_format",
        "revision_round",
        "has_author_response",
        "editorial_decision",
        "extraction_manifest_id",
        "extraction_trace_id",
        "authors",
    }
)

_SPLIT_FILES: tuple[tuple[str, str], ...] = (
    ("train", "train.jsonl"),
    ("validation", "val.jsonl"),
    ("test", "test.jsonl"),
)

_PUBLIC_MANIFEST_PATH = "PUBLIC_RELEASE_MANIFEST.json"
_PUBLIC_MANIFEST_SCHEMA_VERSION = 1
_PUBLIC_MANIFEST_SELF_HASH_POLICY = (
    "PUBLIC_RELEASE_MANIFEST.json is exact-allowlisted but intentionally "
    "omitted from artifacts because a manifest cannot contain its own final "
    "hash and size"
)

_CANONICAL_HF_GITATTRIBUTES_PATH = Path(__file__).with_name(
    "hf_public.gitattributes"
)
_CANONICAL_HF_GITATTRIBUTES_BYTES = (
    _CANONICAL_HF_GITATTRIBUTES_PATH.read_bytes()
)
_CANONICAL_HF_GITATTRIBUTES_SHA256 = (
    "4aa1947088aa553e4f5553ae7587b440a0e240df1574fc653199289077297a59"
)
if (
    len(_CANONICAL_HF_GITATTRIBUTES_BYTES) != 2_569
    or hashlib.sha256(_CANONICAL_HF_GITATTRIBUTES_BYTES).hexdigest()
    != _CANONICAL_HF_GITATTRIBUTES_SHA256
):
    raise RuntimeError(
        "Packaged canonical Hugging Face .gitattributes asset has drifted"
    )

_AUXILIARY_FILES: tuple[tuple[str, str], ...] = (
    ("data/splits/v4/split_meta_v4.json", "metadata/split_meta_v4.json"),
    ("data/splits/v4/val_ids_frozen_v4.json", "metadata/val_ids_frozen_v4.json"),
    ("data/splits/v4/test_ids_frozen_v4.json", "metadata/test_ids_frozen_v4.json"),
    ("data/stats/v4_summary.json", "metadata/v4_summary.json"),
    ("data/stats/v4_summary.md", "metadata/v4_summary.md"),
    ("results/v4/leaderboard.json", "results/v4/leaderboard.json"),
    ("results/v4/leaderboard.md", "results/v4/leaderboard.md"),
    ("results/v4/measurement_audit.json", "results/v4/measurement_audit.json"),
    ("results/v4/measurement_audit.md", "results/v4/measurement_audit.md"),
    ("results/v4/release_manifest.json", "results/v4/release_manifest.json"),
    ("CITATION.cff", "CITATION.cff"),
    ("DATASHEET.md", "docs/DATASHEET.md"),
    ("EVALUATION_PROTOCOL.md", "docs/EVALUATION_PROTOCOL.md"),
    ("LICENSE_MATRIX.md", "docs/LICENSE_MATRIX.md"),
    ("LIMITATIONS_AND_ETHICS.md", "docs/LIMITATIONS_AND_ETHICS.md"),
    ("KNOWN_ISSUES.md", "docs/KNOWN_ISSUES.md"),
    ("NEURIPS_2026_REVIEW_RESPONSE.md", "docs/NEURIPS_2026_REVIEW_RESPONSE.md"),
    ("RELEASE_NOTES_v4.1.md", "docs/RELEASE_NOTES_v4.1.md"),
    ("RELEASE_NOTES_v4.1.2.md", "docs/RELEASE_NOTES_v4.1.2.md"),
    ("RELEASE_NOTES_v4.1.3.md", "docs/RELEASE_NOTES_v4.1.3.md"),
)

_PUBLIC_PACKAGE_FILES: frozenset[str] = frozenset(
    {
        ".gitattributes",
        "README.md",
        _PUBLIC_MANIFEST_PATH,
        *(destination for _, destination in _AUXILIARY_FILES),
        *(f"data/index/{split}.jsonl" for split, _ in _SPLIT_FILES),
        "data/annotations/train.jsonl",
        "data/annotations/validation.jsonl",
    }
)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _concern_dicts(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    concerns = entry.get("concerns") or []
    return [
        concern.model_dump() if hasattr(concern, "model_dump") else concern for concern in concerns
    ]


def sanitize_index_entry(
    entry: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Return one text-free article index row."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    _require_fields(entry, PUBLIC_INDEX_FIELDS, context="article index")
    row = {field: _json_safe(entry.get(field)) for field in PUBLIC_INDEX_FIELDS}
    _raise_if_blocked(row)
    _validate_public_index_row(row)
    return row


def sanitize_annotation_rows(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return text-free project-created label rows for one article."""
    article_id = _json_safe(entry.get("id"))
    source = _json_safe(entry.get("source"))
    rows: list[dict[str, Any]] = []

    for concern in _concern_dicts(entry):
        _require_fields(
            concern,
            PUBLIC_ANNOTATION_FIELDS,
            context=f"concern annotation for {article_id}",
        )
        row = {
            "article_id": article_id,
            "source": source,
            **{field: _json_safe(concern.get(field)) for field in PUBLIC_ANNOTATION_FIELDS},
        }
        _raise_if_blocked(row)
        _validate_public_annotation_row(row)
        rows.append(row)
    return rows


def _require_safe_string(
    value: Any,
    *,
    field: str,
    max_length: int,
    forbid_whitespace: bool = False,
) -> str:
    """Require a bounded, printable, non-email string."""
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected a string")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{field}: control or non-printing characters are not allowed")
    if "@" in value:
        raise ValueError(f"{field}: email-like data is not allowed")
    if forbid_whitespace and any(character.isspace() for character in value):
        raise ValueError(f"{field}: whitespace is not allowed")
    if not value or len(value) > max_length:
        raise ValueError(
            f"{field}: length must be between 1 and {max_length} characters"
        )
    return value


def _validate_article_identity(article_id: Any, source: Any) -> tuple[str, str]:
    source_value = _require_safe_string(
        source,
        field="source",
        max_length=16,
        forbid_whitespace=True,
    )
    if source_value not in _CANONICAL_SOURCES:
        raise ValueError(f"source: unsupported canonical source: {source_value!r}")

    article_id_value = _require_safe_string(
        article_id,
        field="id",
        max_length=128,
        forbid_whitespace=True,
    )
    if _ARTICLE_ID_PATTERNS[source_value].fullmatch(article_id_value) is None:
        raise ValueError(
            f"id/source mismatch or non-canonical article ID: "
            f"{article_id_value!r} / {source_value!r}"
        )
    return article_id_value, source_value


def _concern_owner_stem(article_id: str, source: str) -> str:
    """Return the canonical concern-ID stem used by each source collector."""
    if source == "f1000":
        return "f1000:" + article_id.removeprefix("f1000:10.12688_")
    if source == "plos":
        return "plos:" + article_id.removeprefix("plos:10.1371/")
    if source == "peerj":
        return "peerj:peerj." + article_id.removeprefix("peerj:")
    return article_id


def _validate_public_index_row(row: Mapping[str, Any]) -> None:
    article_id, _ = _validate_article_identity(row.get("id"), row.get("source"))

    doi = _require_safe_string(
        row.get("doi"),
        field=f"doi for {article_id}",
        max_length=255,
        forbid_whitespace=True,
    )
    if _DOI_PATTERN.fullmatch(doi) is None:
        raise ValueError(f"doi for {article_id}: invalid DOI shape")

    published_date = _require_safe_string(
        row.get("published_date"),
        field=f"published_date for {article_id}",
        max_length=10,
        forbid_whitespace=True,
    )
    try:
        parsed_date = date.fromisoformat(published_date)
    except ValueError as exc:
        raise ValueError(
            f"published_date for {article_id}: expected an ISO calendar date"
        ) from exc
    if parsed_date.isoformat() != published_date:
        raise ValueError(
            f"published_date for {article_id}: expected YYYY-MM-DD"
        )

    schema_version = _require_safe_string(
        row.get("schema_version"),
        field=f"schema_version for {article_id}",
        max_length=16,
        forbid_whitespace=True,
    )
    if schema_version != "1.1":
        raise ValueError(
            f"schema_version for {article_id}: expected canonical version '1.1'"
        )


def _validate_enum_value(
    value: Any,
    *,
    field: str,
    allowed: frozenset[str],
    max_length: int,
) -> str:
    enum_value = _require_safe_string(
        value,
        field=field,
        max_length=max_length,
        forbid_whitespace=True,
    )
    if enum_value not in allowed:
        raise ValueError(f"{field}: unsupported value: {enum_value!r}")
    return enum_value


def _validate_public_annotation_row(row: Mapping[str, Any]) -> None:
    article_id, source = _validate_article_identity(
        row.get("article_id"),
        row.get("source"),
    )
    concern_id = _require_safe_string(
        row.get("concern_id"),
        field=f"concern_id for {article_id}",
        max_length=128,
        forbid_whitespace=True,
    )
    owner_stem = _concern_owner_stem(article_id, source)
    owner_prefix = f"{owner_stem}:"
    if not concern_id.startswith(owner_prefix):
        raise ValueError(
            f"concern_id for {article_id}: concern does not belong to article/source"
        )
    ordinal = concern_id.removeprefix(owner_prefix)
    if _CONCERN_ORDINAL_PATTERN.fullmatch(ordinal) is None:
        raise ValueError(
            f"concern_id for {article_id}: expected canonical RnCm ordinal"
        )

    _validate_enum_value(
        row.get("category"),
        field=f"category for {concern_id}",
        allowed=_CONCERN_CATEGORIES,
        max_length=64,
    )
    _validate_enum_value(
        row.get("severity"),
        field=f"severity for {concern_id}",
        allowed=_CONCERN_SEVERITIES,
        max_length=16,
    )
    _validate_enum_value(
        row.get("author_stance"),
        field=f"author_stance for {concern_id}",
        allowed=_AUTHOR_STANCES,
        max_length=32,
    )


def find_blocked_field_paths(value: Any, path: str = "$") -> list[str]:
    """Return nested paths whose key is forbidden in a public release."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            is_safe_aggregate_count = (
                str(key) == "concerns"
                and isinstance(child, (int, float))
                and not isinstance(child, bool)
            )
            if str(key) in BLOCKED_PUBLIC_FIELDS and not is_safe_aggregate_count:
                found.append(child_path)
            found.extend(find_blocked_field_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_blocked_field_paths(child, f"{path}[{index}]"))
    return found


def _raise_if_blocked(value: Any) -> None:
    blocked = find_blocked_field_paths(value)
    if blocked:
        raise ValueError(f"Blocked fields survived sanitization: {blocked}")


def _require_fields(
    value: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    context: str,
) -> None:
    missing = [field for field in fields if field not in value or value.get(field) in (None, "")]
    if missing:
        raise ValueError(f"{context}: required public fields missing: {missing}")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _preflight_split_rows(splits_dir: Path) -> dict[str, set[str]]:
    """Validate split identity and all values that can enter public rows."""
    split_ids: dict[str, set[str]] = {}
    article_split_by_id: dict[str, str] = {}
    seen_concern_ids: set[str] = set()

    for split, filename in _SPLIT_FILES:
        source_path = splits_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing canonical split: {source_path}")

        article_ids: set[str] = set()
        for entry in _iter_jsonl(source_path):
            index_row = sanitize_index_entry(entry, split=split)
            article_id = index_row["id"]
            if article_id in article_ids:
                raise ValueError(
                    f"{split} split contains duplicate article ID: {article_id}"
                )
            previous_split = article_split_by_id.get(article_id)
            if previous_split is not None:
                raise ValueError(
                    f"Duplicate article ID across {previous_split}/{split} "
                    f"splits: {article_id}"
                )
            article_ids.add(article_id)
            article_split_by_id[article_id] = split

            if split == "test":
                continue
            for annotation_row in sanitize_annotation_rows(entry):
                concern_id = annotation_row["concern_id"]
                if concern_id in seen_concern_ids:
                    raise ValueError(f"Duplicate concern ID: {concern_id}")
                seen_concern_ids.add(concern_id)

        if not article_ids:
            raise ValueError(f"{split} split must contain at least one article ID")
        split_ids[split] = article_ids

    split_names = tuple(split_ids)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap = split_ids[left_name] & split_ids[right_name]
            if overlap:
                raise ValueError(
                    f"Duplicate article ID across {left_name}/{right_name} "
                    f"splits ({len(overlap)} overlap)"
                )
    return split_ids


def _load_frozen_id_set(path: Path, *, split: str) -> set[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing frozen {split} ID file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc

    if not isinstance(value, dict) or set(value) != {"ids"}:
        raise ValueError(f"{path}: expected exactly one 'ids' field")
    raw_ids = value["ids"]
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"{path}: ids must be a nonempty list")

    frozen_ids: set[str] = set()
    for index, article_id in enumerate(raw_ids):
        if not isinstance(article_id, str) or ":" not in article_id:
            raise ValueError(f"{path}: ids[{index}] is not a canonical article ID")
        source = article_id.split(":", 1)[0]
        validated_id, _ = _validate_article_identity(article_id, source)
        if validated_id in frozen_ids:
            raise ValueError(f"{path}: duplicate frozen article ID: {validated_id}")
        frozen_ids.add(validated_id)
    return frozen_ids


def _validate_frozen_split_ids(
    split_ids: Mapping[str, set[str]],
    frozen_split_paths: Mapping[str, Path],
) -> None:
    if set(frozen_split_paths) != set(_FROZEN_SPLIT_FILES):
        raise ValueError(
            "Frozen split paths must contain exactly validation and test"
        )
    for split in ("validation", "test"):
        frozen_ids = _load_frozen_id_set(frozen_split_paths[split], split=split)
        actual_ids = split_ids[split]
        if actual_ids != frozen_ids:
            raise ValueError(
                f"{split} split IDs do not match frozen v4 IDs "
                f"(missing={len(frozen_ids - actual_ids)}, "
                f"unexpected={len(actual_ids - frozen_ids)})"
            )


def _write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def export_public_data(
    *,
    splits_dir: Path,
    output_dir: Path,
    frozen_split_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Stream public index and annotation JSONL files."""
    split_ids = _preflight_split_rows(splits_dir)
    if frozen_split_paths is not None:
        _validate_frozen_split_ids(split_ids, frozen_split_paths)

    index_dir = output_dir / "index"
    annotations_dir = output_dir / "annotations"
    index_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, Any] = {}
    total_articles = 0
    published_annotations = 0
    withheld_test_targets = 0
    seen_article_ids: set[str] = set()
    seen_concern_ids: set[str] = set()

    for split, filename in _SPLIT_FILES:
        source_path = splits_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing canonical split: {source_path}")

        index_path = index_dir / f"{split}.jsonl"
        annotations_path = annotations_dir / f"{split}.jsonl"
        source_counts: Counter[str] = Counter()
        article_count = 0
        source_concern_count = 0
        annotation_count = 0

        with index_path.open("w", encoding="utf-8") as index_output:
            annotations_output = (
                annotations_path.open("w", encoding="utf-8") if split != "test" else None
            )
            try:
                for entry in _iter_jsonl(source_path):
                    article_id = str(entry.get("id") or "")
                    if article_id in seen_article_ids:
                        raise ValueError(f"Duplicate article ID: {article_id}")
                    seen_article_ids.add(article_id)
                    article_count += 1
                    source_counts[str(entry.get("source") or "unknown")] += 1
                    concerns = _concern_dicts(entry)
                    source_concern_count += len(concerns)
                    _write_jsonl_row(
                        index_output,
                        sanitize_index_entry(entry, split=split),
                    )
                    if annotations_output is not None:
                        annotation_rows = sanitize_annotation_rows(entry)
                        annotation_count += len(annotation_rows)
                        for row in annotation_rows:
                            concern_id = str(row["concern_id"])
                            if concern_id in seen_concern_ids:
                                raise ValueError(f"Duplicate concern ID: {concern_id}")
                            seen_concern_ids.add(concern_id)
                            _write_jsonl_row(annotations_output, row)
            finally:
                if annotations_output is not None:
                    annotations_output.close()

        total_articles += article_count
        published_annotations += annotation_count
        if split == "test":
            withheld_test_targets = source_concern_count
        split_stats[split] = {
            "articles": article_count,
            "published_annotations": annotation_count,
            "withheld_test_targets": (source_concern_count if split == "test" else 0),
            "source_distribution": dict(sorted(source_counts.items())),
            "index": _file_metadata(index_path, output_dir.parent),
            "annotations": (
                _file_metadata(annotations_path, output_dir.parent)
                if annotations_path.exists()
                else None
            ),
        }

    return {
        "manifest_schema_version": _PUBLIC_MANIFEST_SCHEMA_VERSION,
        "manifest_self_hash_policy": _PUBLIC_MANIFEST_SELF_HASH_POLICY,
        "release": DEFAULT_SOFTWARE_RELEASE_VERSION,
        "scope": "rights-minimized-public-index",
        "total_articles": total_articles,
        "published_annotations": published_annotations,
        "withheld_test_targets": withheld_test_targets,
        "splits": split_stats,
        "index_fields": list(PUBLIC_INDEX_FIELDS),
        "annotation_fields": [
            *PUBLIC_ANNOTATION_OUTPUT_FIELDS,
        ],
        "blocked_fields": sorted(BLOCKED_PUBLIC_FIELDS),
        "reviewer_identity_policy": (
            "reviewer/data-row names, emails, and explicit identity fields "
            "excluded; project-maintainer contact metadata in CITATION.cff is "
            "not a reviewer identity field; "
            "stable concern IDs may retain source-local review ordinals that "
            "do not identify a person"
        ),
    }


def build_public_package(
    *,
    repo_root: Path,
    splits_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a complete local folder ready for one clean HF upload."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = export_public_data(
        splits_dir=splits_dir,
        output_dir=output_dir / "data",
        frozen_split_paths={
            split: repo_root / relative_path
            for split, relative_path in _FROZEN_SPLIT_FILES.items()
        },
    )

    for source_name, destination_name in _AUXILIARY_FILES:
        source = repo_root / source_name
        if not source.exists():
            raise FileNotFoundError(f"Missing public release artifact: {source}")
        destination = output_dir / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    readme = generate_public_dataset_card(stats)
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / ".gitattributes").write_bytes(
        _CANONICAL_HF_GITATTRIBUTES_BYTES
    )
    stats["exact_package_allowlist"] = sorted(_PUBLIC_PACKAGE_FILES)
    stats["artifacts"] = _package_artifact_metadata(output_dir)
    (output_dir / _PUBLIC_MANIFEST_PATH).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_public_package(output_dir)
    return stats


def validate_public_package(package_dir: Path) -> dict[str, int]:
    """Fail closed on paths, schemas, blocked fields, or artifact drift."""
    totals = {
        "files": 0,
        "index_rows": 0,
        "annotation_rows": 0,
        "test_annotation_rows": 0,
    }
    seen_paths: set[str] = set()
    index_sources: dict[str, dict[str, str]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    annotation_owners: list[tuple[str, str, str]] = []
    seen_annotation_ids: set[str] = set()
    for path in sorted(package_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in a public package: {path}")
        if not path.is_file():
            continue
        relative_path = path.relative_to(package_dir).as_posix()
        if relative_path not in _PUBLIC_PACKAGE_FILES:
            raise ValueError(f"Unexpected public package path: {relative_path}")
        if relative_path == ".gitattributes":
            content = path.read_bytes()
            if content != _CANONICAL_HF_GITATTRIBUTES_BYTES:
                raise ValueError(
                    "Public package .gitattributes does not match the canonical "
                    "Hugging Face LFS policy"
                )
        seen_paths.add(relative_path)
        totals["files"] += 1
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON") from exc
            _raise_if_blocked(value)
        if path.suffix != ".jsonl":
            continue
        is_index = path.parent.name == "index"
        is_annotations = path.parent.name == "annotations"
        for row in _iter_jsonl(path):
            _raise_if_blocked(row)
            if is_index:
                if set(row) != set(PUBLIC_INDEX_FIELDS):
                    raise ValueError(f"{path}: public index schema mismatch: {sorted(row)}")
                _validate_public_index_row(row)
                split = path.stem
                article_id = row["id"]
                if article_id in index_sources[split]:
                    raise ValueError(
                        f"{path}: duplicate public index article ID: {article_id}"
                    )
                index_sources[split][article_id] = row["source"]
                totals["index_rows"] += 1
            elif is_annotations:
                if set(row) != set(PUBLIC_ANNOTATION_OUTPUT_FIELDS):
                    raise ValueError(f"{path}: public annotation schema mismatch: {sorted(row)}")
                _validate_public_annotation_row(row)
                concern_id = row["concern_id"]
                if concern_id in seen_annotation_ids:
                    raise ValueError(
                        f"{path}: duplicate public annotation concern ID: "
                        f"{concern_id}"
                    )
                seen_annotation_ids.add(concern_id)
                annotation_owners.append(
                    (path.stem, row["article_id"], row["source"])
                )
                totals["annotation_rows"] += 1
                if path.name == "test.jsonl":
                    totals["test_annotation_rows"] += 1

    missing_paths = _PUBLIC_PACKAGE_FILES - seen_paths
    if missing_paths:
        raise ValueError(f"Public package files missing: {sorted(missing_paths)}")
    if totals["index_rows"] == 0:
        raise ValueError(f"No public index rows found in {package_dir}")
    if totals["annotation_rows"] == 0:
        raise ValueError(f"No public annotation rows found in {package_dir}")
    if totals["test_annotation_rows"]:
        raise ValueError("Public test targets were not withheld")
    all_index_ids: set[str] = set()
    for split, sources in index_sources.items():
        overlap = all_index_ids & sources.keys()
        if overlap:
            raise ValueError(
                f"Public index split IDs are not disjoint at {split} "
                f"({len(overlap)} overlap)"
            )
        all_index_ids.update(sources)
    for split, article_id, source in annotation_owners:
        indexed_source = index_sources[split].get(article_id)
        if indexed_source is None:
            raise ValueError(
                f"Public annotation article is absent from {split} index: "
                f"{article_id}"
            )
        if source != indexed_source:
            raise ValueError(
                f"Public annotation/index source mismatch for {article_id}"
            )
    _validate_manifest_artifacts(package_dir, seen_paths)
    return totals


def generate_public_dataset_card(stats: Mapping[str, Any]) -> str:
    """Return the metadata-first dataset card."""
    splits = stats["splits"]
    yaml_lines = [
        "---",
        "language:",
        "  - en",
        "license: other",
        "tags:",
        "  - peer-review",
        "  - biomedical",
        "  - benchmark",
        "  - metadata",
        "  - silver-standard",
        "pretty_name: BioReview-Bench Public Index",
        "configs:",
        "  - config_name: index",
        "    default: true",
        "    data_files:",
    ]
    for split in ("train", "validation", "test"):
        yaml_lines.extend(
            [
                f"      - split: {split}",
                f"        path: data/index/{split}.jsonl",
            ]
        )
    yaml_lines.extend(
        [
            "  - config_name: annotations",
            "    data_files:",
            "      - split: train",
            "        path: data/annotations/train.jsonl",
            "      - split: validation",
            "        path: data/annotations/validation.jsonl",
            "dataset_info:",
            "  - config_name: index",
            "    splits:",
        ]
    )
    for split in ("train", "validation", "test"):
        yaml_lines.extend(
            [
                f"      - name: {split}",
                f"        num_examples: {splits[split]['articles']}",
            ]
        )
    yaml_lines.extend(
        [
            "  - config_name: annotations",
            "    splits:",
            "      - name: train",
            f"        num_examples: {splits['train']['published_annotations']}",
            "      - name: validation",
            f"        num_examples: {splits['validation']['published_annotations']}",
            "---",
            "",
        ]
    )

    body = f"""# BioReview-Bench {DEFAULT_SOFTWARE_RELEASE_VERSION} — public index

BioReview-Bench is a silver-standard resource for studying concern patterns
recorded in published biomedical peer review. This rights-minimized snapshot is
an index and label-distribution release. It is not a self-contained text
benchmark, a public test set, or an open leaderboard.

- **{stats["total_articles"]:,} text-free article index rows**
- **{stats["published_annotations"]:,} text-free train/validation label rows**
- **{stats["withheld_test_targets"]:,} test targets withheld**
- Audited Hugging Face history with no article/abstract text, normalized
  concern text, raw review/decision text, raw author response, reviewer/data-row
  names/emails or explicit identity fields, internal traces, or manuscript
  files. Project-maintainer contact metadata in `CITATION.cff` is not a
  reviewer/data-row identity field.

## Important claim boundary

Concern segmentation, category, severity, and author-stance fields are
LLM-derived silver annotations, not expert-adjudicated scientific truth. The
frozen scores used an unadapted `allenai/specter2_base` checkpoint with
automatic mean pooling, no recorded task adapter, and no recorded checkpoint
revision. That historical wrapper and its 0.65 threshold have not completed
independent human validation. The published scores are provisional and
matcher-dependent. The F1000 DOI-stem audit is a source-specific lower bound:
it finds 115 train–validation, 41 train–test, and 9 validation–test crossing
families. Development data overlap 49/150 F1000 test articles across 48
families. A same-family training record was BM25 rank 1 for all 42 affected
train–test queries.

At the uncalibrated Jaccard cutoff 0.195, query-time family filtering reduces
full-600 BM25 matched-record F1 from 0.0577 to 0.0148. The corresponding
450-article non-eLife result, 0.0690 to 0.0171, is a post-hoc sensitivity
analysis. These are diagnostics under one operating point, not corrected
historical embedding scores or a replacement split.

The six-system score table is a frozen historical 600-article snapshot. The
450-article non-eLife analysis is a post-hoc source-subset diagnostic, not a
replacement test set or a basis for reranking that snapshot. See the public
[measurement audit](results/v4/measurement_audit.md) and
[known issues](docs/KNOWN_ISSUES.md).

All six frozen raw result JSONs store `f1_macro=0.0` as an invalid,
unpopulated legacy sentinel; historical category-macro F1 is unreported.
Nested per-category `aucpr=0.0` fields are also uncomputed sentinels.

The index publishes test membership together with stable article IDs and DOIs.
Because source peer reviews are publicly retrievable, withholding target rows
from this package does not make the test set secret or blind. Do not use this
snapshot for new blind evaluation or claim hidden-test generalization.

`author_stance` describes an LLM-derived response-alignment label; 91.1% of the
full corpus is `no_response`. It must not be interpreted as article quality,
reviewer validity, or whether a scientific concern is objectively correct.

The public [NeurIPS 2026 review record and revision
response](https://github.com/jang1563/bioreview-bench/blob/main/NEURIPS_2026_REVIEW_RESPONSE.md)
documents the external review and v4.1 corrections; v4.1.2 added the public
artifact boundary and aggregate measurement audit. v4.1.3 adds deterministic
operating-point and split-family audit tooling plus exact package-tree
validation. The initial reviews were three scores of 2/6; this is a
review-stage result, not the final decision.

## Configs

- `index` (default): stable IDs, DOI/source metadata, publication date, and
  schema version for all 6,940 articles. The Hugging Face split name records
  train/validation/test membership.
- `annotations`: text-free project-created categorical label rows for train
  and validation only. Stable concern IDs may retain source-local review
  ordinals such as `R1`; those ordinals do not identify a person.

```python
from datasets import load_dataset

index = load_dataset("jang1563/bioreview-bench", "index", revision="v4.1.3")
labels = load_dataset("jang1563/bioreview-bench", "annotations", revision="v4.1.3")
```

## Public release boundary

This snapshot intentionally excludes titles, abstracts, article body text,
`paper_text_sections`, `paper_text_v1_sections`, normalized `concern_text`,
`decision_letter_raw`, `author_response_raw`, concern-level response/evidence
text, structured references, reviewer/data-row names, emails, and explicit
identity fields, internal extraction traces, and all test target labels.
Project-maintainer citation/contact metadata is outside that data-row identity
exclusion.

Publisher materials must be obtained from the DOI/source record under the
original publisher terms. Source licenses are not replaced by the benchmark
annotation license.

There is no public test-scoring service or open result-submission channel. The
evaluation CLI requires an authorized local full-schema copy.

## Licensing and documentation

Project-created annotations are released under CC BY-NC 4.0. Code is
Apache-2.0. Bibliographic metadata and publisher materials remain subject to
source-specific terms.

- [License matrix](docs/LICENSE_MATRIX.md)
- [Datasheet](docs/DATASHEET.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Limitations and ethics](docs/LIMITATIONS_AND_ETHICS.md)
- [Known issues](docs/KNOWN_ISSUES.md)
- [Measurement audit](results/v4/measurement_audit.md)
- [GitHub](https://github.com/jang1563/bioreview-bench)
- [v4.1.3 release](https://github.com/jang1563/bioreview-bench/releases/tag/v4.1.3)

## Citation

See `CITATION.cff` and the citation section in the
[GitHub README](https://github.com/jang1563/bioreview-bench#citation).
"""
    return "\n".join(yaml_lines) + body


def _file_metadata(path: Path, package_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(package_dir)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _package_artifact_metadata(package_dir: Path) -> list[dict[str, Any]]:
    """Hash every package file except the self-referential manifest."""
    artifacts: list[dict[str, Any]] = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(package_dir).as_posix()
        if relative_path == _PUBLIC_MANIFEST_PATH:
            continue
        artifacts.append(_file_metadata(path, package_dir))
    return artifacts


def _validate_manifest_artifacts(
    package_dir: Path,
    seen_paths: set[str],
) -> None:
    """Verify the manifest's exact non-self file set, sizes, and hashes."""
    manifest_path = package_dir / _PUBLIC_MANIFEST_PATH
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Public package manifest missing: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: manifest must be a JSON object")
    if manifest.get("manifest_schema_version") != _PUBLIC_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path}: unsupported manifest_schema_version"
        )
    if manifest.get("manifest_self_hash_policy") != _PUBLIC_MANIFEST_SELF_HASH_POLICY:
        raise ValueError(
            f"{manifest_path}: invalid manifest_self_hash_policy"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"{manifest_path}: artifacts must be a list")
    exact_allowlist = manifest.get("exact_package_allowlist")
    expected_allowlist = sorted(_PUBLIC_PACKAGE_FILES)
    if exact_allowlist != expected_allowlist:
        raise ValueError(
            f"{manifest_path}: exact_package_allowlist must equal "
            "the complete sorted public package allowlist"
        )

    recorded: dict[str, dict[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or set(artifact) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise ValueError(
                f"{manifest_path}: artifact {index} must contain exactly "
                "path, sha256, and bytes"
            )
        relative_path = artifact["path"]
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"{manifest_path}: artifact {index} has invalid path")
        if relative_path == _PUBLIC_MANIFEST_PATH:
            raise ValueError(
                f"{manifest_path}: self-referential manifest hash is not allowed"
            )
        if relative_path in recorded:
            raise ValueError(
                f"{manifest_path}: duplicate artifact path: {relative_path}"
            )
        recorded[relative_path] = artifact

    actual_paths = seen_paths - {_PUBLIC_MANIFEST_PATH}
    recorded_paths = set(recorded)
    if recorded_paths != actual_paths:
        missing = sorted(actual_paths - recorded_paths)
        unexpected = sorted(recorded_paths - actual_paths)
        raise ValueError(
            f"{manifest_path}: artifact paths mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for relative_path in sorted(actual_paths):
        artifact = recorded[relative_path]
        path = package_dir / relative_path
        recorded_bytes = artifact["bytes"]
        if (
            not isinstance(recorded_bytes, int)
            or isinstance(recorded_bytes, bool)
            or recorded_bytes < 0
        ):
            raise ValueError(
                f"{manifest_path}: invalid byte count for {relative_path}"
            )
        actual_bytes = path.stat().st_size
        if recorded_bytes != actual_bytes:
            raise ValueError(
                f"{manifest_path}: byte count mismatch for {relative_path}: "
                f"recorded={recorded_bytes}, actual={actual_bytes}"
            )

        recorded_sha256 = artifact["sha256"]
        actual_sha256 = _sha256(path)
        if recorded_sha256 != actual_sha256:
            raise ValueError(
                f"{manifest_path}: SHA-256 mismatch for {relative_path}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bioreview_bench.collect import hf_public
from bioreview_bench.collect.hf_public import (
    BLOCKED_PUBLIC_FIELDS,
    PUBLIC_ANNOTATION_OUTPUT_FIELDS,
    PUBLIC_INDEX_FIELDS,
    build_public_package,
    export_public_data,
    find_blocked_field_paths,
    generate_public_dataset_card,
    sanitize_annotation_rows,
    sanitize_index_entry,
    validate_public_package,
)

_TRAIN_ID = "elife:1"
_VALIDATION_ID = "plos:10.1371/journal.pcbi.1010001"
_TEST_ID = "nature:s41467-024-12345-6"


def _concern_id(article_id: str) -> str:
    source = article_id.split(":", 1)[0]
    if source == "f1000":
        stem = "f1000:" + article_id.removeprefix("f1000:10.12688_")
    elif source == "plos":
        stem = "plos:" + article_id.removeprefix("plos:10.1371/")
    elif source == "peerj":
        stem = "peerj:peerj." + article_id.removeprefix("peerj:")
    else:
        stem = article_id
    return f"{stem}:R1C1"


def _entry(article_id: str = _TRAIN_ID) -> dict:
    return {
        "id": article_id,
        "source": article_id.split(":", 1)[0],
        "doi": "10.0000/example",
        "title": "Private-to-release article title",
        "abstract": "Publisher abstract text",
        "subjects": ["biology"],
        "published_date": "2025-01-01",
        "review_format": "journal",
        "revision_round": 1,
        "has_author_response": True,
        "editorial_decision": "major_revision",
        "schema_version": "1.1",
        "extraction_manifest_id": "em-v1.0",
        "paper_text_sections": {"methods": "Publisher article text"},
        "paper_text_v1_sections": {"methods": "Earlier article text"},
        "structured_references": [{"authors": ["Example Author"]}],
        "decision_letter_raw": "Raw review text",
        "author_response_raw": "Raw response text",
        "concerns": [
            {
                "concern_id": _concern_id(article_id),
                "reviewer_num": 1,
                "concern_text": "Normalized concern text",
                "category": "design_flaw",
                "severity": "major",
                "author_response_text": "Response excerpt",
                "author_stance": "partial",
                "evidence_of_change": "Change excerpt",
                "resolution": "partial",
                "resolution_confidence": 0.8,
                "was_valid": True,
                "raised_by_multiple": False,
                "requires_figure_reading": False,
                "extraction_trace_id": "trace-private",
            }
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepare_fixture_inputs(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    for source_name, _destination_name in hf_public._AUXILIARY_FILES:
        path = repo_root / source_name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "{}" if path.suffix == ".json" else "public fixture\n"
        path.write_text(content, encoding="utf-8")

    splits = tmp_path / "splits"
    _write_jsonl(splits / "train.jsonl", [_entry(_TRAIN_ID)])
    _write_jsonl(splits / "val.jsonl", [_entry(_VALIDATION_ID)])
    _write_jsonl(splits / "test.jsonl", [_entry(_TEST_ID)])
    (repo_root / "data/splits/v4/val_ids_frozen_v4.json").write_text(
        json.dumps({"ids": [_VALIDATION_ID]}),
        encoding="utf-8",
    )
    (repo_root / "data/splits/v4/test_ids_frozen_v4.json").write_text(
        json.dumps({"ids": [_TEST_ID]}),
        encoding="utf-8",
    )
    return repo_root, splits


def _build_fixture_package(tmp_path: Path) -> Path:
    repo_root, splits = _prepare_fixture_inputs(tmp_path)
    package = tmp_path / "package"
    build_public_package(
        repo_root=repo_root,
        splits_dir=splits,
        output_dir=package,
    )
    return package


def test_index_sanitizer_is_text_free() -> None:
    row = sanitize_index_entry(_entry(), split="train")

    assert row["id"] == "elife:1"
    assert set(row) == {"id", "source", "doi", "published_date", "schema_version"}
    assert find_blocked_field_paths(row) == []
    assert BLOCKED_PUBLIC_FIELDS.isdisjoint(row)


def test_annotation_sanitizer_removes_text_and_reviewer_identifier() -> None:
    rows = sanitize_annotation_rows(_entry())

    assert rows == [
        {
            "article_id": "elife:1",
            "source": "elife",
            "concern_id": "elife:1:R1C1",
            "category": "design_flaw",
            "severity": "major",
            "author_stance": "partial",
        }
    ]
    assert find_blocked_field_paths(rows) == []


def test_sanitizer_rejects_missing_required_label() -> None:
    entry = _entry()
    del entry["concerns"][0]["author_stance"]

    with pytest.raises(ValueError, match="author_stance"):
        sanitize_annotation_rows(entry)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "plos:10.1371/journal.pcbi.1010001", "id/source mismatch"),
        ("source", "reviewer@example.org", "email-like"),
        ("doi", "10.0000/reviewer@example.org", "email-like"),
        ("doi", "10.0000/private note", "whitespace"),
        ("doi", 100001, "expected a string"),
        ("published_date", "2025-02-30", "ISO calendar date"),
        ("published_date", "2025-01-01\nprivate", "control"),
        ("schema_version", "1.0", "canonical version"),
        ("schema_version", "1.1\tprivate", "control"),
    ],
)
def test_index_sanitizer_rejects_unsafe_whitelisted_values(
    field: str,
    value: object,
    message: str,
) -> None:
    entry = _entry()
    entry[field] = value

    with pytest.raises(ValueError, match=message):
        sanitize_index_entry(entry, split="train")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("concern_id", "elife:2:R1C1", "does not belong"),
        ("concern_id", "reviewer@example.org", "email-like"),
        ("concern_id", "elife:1:R1C1\nprivate", "control"),
        ("concern_id", "elife:1:R" + ("1" * 128) + "C1", "length"),
        ("concern_id", "elife:1:R0C1", "canonical RnCm"),
        ("category", "reviewer@example.org", "email-like"),
        ("category", "x" * 65, "length"),
        ("category", "unsupported", "unsupported value"),
        ("severity", "\nmajor", "control"),
        ("severity", "critical", "unsupported value"),
        ("author_stance", ["partial"], "expected a string"),
        ("author_stance", "accepted", "unsupported value"),
    ],
)
def test_annotation_sanitizer_rejects_unsafe_whitelisted_values(
    field: str,
    value: object,
    message: str,
) -> None:
    entry = _entry()
    entry["concerns"][0][field] = value

    with pytest.raises(ValueError, match=message):
        sanitize_annotation_rows(entry)


@pytest.mark.parametrize(
    "article_id",
    [
        "elife:84798",
        "f1000:10.12688_f1000research.171192.2",
        "nature:s41467-024-12345-6",
        "peerj:10547",
        "plos:10.1371/journal.pgen.1010233",
    ],
)
def test_annotation_ownership_accepts_each_canonical_source(
    article_id: str,
) -> None:
    rows = sanitize_annotation_rows(_entry(article_id))

    assert rows[0]["concern_id"] == _concern_id(article_id)


def test_export_withholds_test_targets(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    _write_jsonl(splits / "train.jsonl", [_entry(_TRAIN_ID)])
    _write_jsonl(splits / "val.jsonl", [_entry(_VALIDATION_ID)])
    _write_jsonl(splits / "test.jsonl", [_entry(_TEST_ID)])

    package = tmp_path / "package"
    stats = export_public_data(
        splits_dir=splits,
        output_dir=package / "data",
    )

    assert stats["total_articles"] == 3
    assert stats["published_annotations"] == 2
    assert stats["withheld_test_targets"] == 1
    assert stats["release"] == "v4.1.3"
    assert not (package / "data" / "annotations" / "test.jsonl").exists()

    test_index = json.loads((package / "data" / "index" / "test.jsonl").read_text(encoding="utf-8"))
    assert test_index["id"] == _TEST_ID
    assert set(test_index) == {"id", "source", "doi", "published_date", "schema_version"}
    assert not (package / "data" / "annotations" / "test.jsonl").exists()


def test_export_rejects_duplicate_article_ids(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    duplicate_id = "elife:2"
    _write_jsonl(splits / "train.jsonl", [_entry(duplicate_id)])
    _write_jsonl(splits / "val.jsonl", [_entry(duplicate_id)])
    _write_jsonl(splits / "test.jsonl", [_entry(_TEST_ID)])

    with pytest.raises(ValueError, match="Duplicate article ID"):
        export_public_data(
            splits_dir=splits,
            output_dir=tmp_path / "package" / "data",
        )


def test_build_rejects_swapped_validation_and_test_files_before_writing(
    tmp_path: Path,
) -> None:
    repo_root, splits = _prepare_fixture_inputs(tmp_path)
    _write_jsonl(splits / "val.jsonl", [_entry(_TEST_ID)])
    _write_jsonl(splits / "test.jsonl", [_entry(_VALIDATION_ID)])
    package = tmp_path / "package"

    with pytest.raises(ValueError, match="validation split IDs do not match frozen"):
        build_public_package(
            repo_root=repo_root,
            splits_dir=splits,
            output_dir=package,
        )

    assert not (package / "data").exists()


def test_build_rejects_stale_frozen_id_file_before_writing(
    tmp_path: Path,
) -> None:
    repo_root, splits = _prepare_fixture_inputs(tmp_path)
    (repo_root / "data/splits/v4/val_ids_frozen_v4.json").write_text(
        json.dumps({"ids": ["elife:99"]}),
        encoding="utf-8",
    )
    package = tmp_path / "package"

    with pytest.raises(ValueError, match="validation split IDs do not match frozen"):
        build_public_package(
            repo_root=repo_root,
            splits_dir=splits,
            output_dir=package,
        )

    assert not (package / "data").exists()


def test_export_rejects_empty_split_before_writing(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    _write_jsonl(splits / "train.jsonl", [])
    _write_jsonl(splits / "val.jsonl", [_entry(_VALIDATION_ID)])
    _write_jsonl(splits / "test.jsonl", [_entry(_TEST_ID)])
    output = tmp_path / "package" / "data"

    with pytest.raises(ValueError, match="train split must contain"):
        export_public_data(splits_dir=splits, output_dir=output)

    assert not output.exists()


def test_validator_rejects_blocked_jsonl_key(tmp_path: Path) -> None:
    path = tmp_path / "data" / "index" / "train.jsonl"
    _write_jsonl(path, [{"id": "x", "decision_letter_raw": "not public"}])

    with pytest.raises(ValueError, match="Blocked fields"):
        validate_public_package(tmp_path)


def test_validator_rejects_extra_whitelisted_row_key(tmp_path: Path) -> None:
    path = tmp_path / "data" / "index" / "train.jsonl"
    row = {field: "value" for field in PUBLIC_INDEX_FIELDS}
    row["unexpected"] = "not allowed"
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="index schema mismatch"):
        validate_public_package(tmp_path)


def test_validator_rejects_unsafe_whitelisted_row_value(tmp_path: Path) -> None:
    path = tmp_path / "data" / "index" / "train.jsonl"
    row = sanitize_index_entry(_entry(), split="train")
    row["doi"] = "10.0000/reviewer@example.org"
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="email-like"):
        validate_public_package(tmp_path)


def test_validator_rejects_blocked_auxiliary_json_key(tmp_path: Path) -> None:
    path = tmp_path / "metadata" / "v4_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"reviewer_email": "private@example.org"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Blocked fields"):
        validate_public_package(tmp_path)


def test_numeric_concern_count_is_safe_but_concern_rows_are_blocked() -> None:
    assert find_blocked_field_paths({"splits": {"train": {"concerns": 10}}}) == []
    assert find_blocked_field_paths({"concerns": [{"concern_text": "private"}]}) == [
        "$.concerns",
        "$.concerns[0].concern_text",
    ]


def test_annotation_output_schema_is_exact() -> None:
    assert PUBLIC_ANNOTATION_OUTPUT_FIELDS == (
        "article_id",
        "source",
        "concern_id",
        "category",
        "severity",
        "author_stance",
    )


def test_card_has_only_index_and_text_free_annotations() -> None:
    stats = {
        "total_articles": 3,
        "published_annotations": 2,
        "withheld_test_targets": 1,
        "splits": {
            "train": {"articles": 1, "published_annotations": 1},
            "validation": {"articles": 1, "published_annotations": 1},
            "test": {"articles": 1, "published_annotations": 0},
        },
    }

    card = generate_public_dataset_card(stats)

    assert "config_name: index" in card
    assert "config_name: annotations" in card
    assert "data/annotations/test.jsonl" not in card
    assert "normalized `concern_text`" in card
    assert "text-free train/validation label rows" in card
    assert "ordinals do not identify a person" in card
    assert "# BioReview-Bench v4.1.3 — public index" in card
    assert "not a self-contained text" in card
    assert "benchmark, a public test set, or an open leaderboard" in card
    assert "49/150 F1000 test articles across 48" in card
    assert "115 train–validation, 41 train–test, and 9 validation–test" in card
    assert "source-specific lower bound" in card
    assert "measurement audit" in card
    assert "automatic mean pooling" in card
    assert "600-article snapshot" in card
    assert "450-article non-eLife analysis is a post-hoc" in card
    assert "historical category-macro F1 is unreported" in card
    assert "per-category `aucpr=0.0` fields are also uncomputed" in card
    assert "test membership together with stable article IDs and DOIs" in card
    assert "does not make the test set secret or blind" in card
    assert "Project-maintainer contact metadata" in card
    assert "releases/tag/v4.1.3" in card


def test_package_manifest_hashes_every_non_self_file(tmp_path: Path) -> None:
    package = _build_fixture_package(tmp_path)
    manifest = json.loads(
        (package / "PUBLIC_RELEASE_MANIFEST.json").read_text(encoding="utf-8")
    )

    artifacts = manifest["artifacts"]
    assert manifest["manifest_schema_version"] == 1
    assert (
        manifest["manifest_self_hash_policy"]
        == hf_public._PUBLIC_MANIFEST_SELF_HASH_POLICY
    )
    assert manifest["exact_package_allowlist"] == sorted(
        hf_public._PUBLIC_PACKAGE_FILES
    )
    recorded_paths = {artifact["path"] for artifact in artifacts}
    actual_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "PUBLIC_RELEASE_MANIFEST.json"
    }
    assert actual_paths | {
        "PUBLIC_RELEASE_MANIFEST.json"
    } == hf_public._PUBLIC_PACKAGE_FILES
    assert recorded_paths == actual_paths
    assert "PUBLIC_RELEASE_MANIFEST.json" not in recorded_paths
    assert ".gitattributes" in recorded_paths
    assert "results/v4/measurement_audit.json" in recorded_paths
    assert "results/v4/measurement_audit.md" in recorded_paths

    for artifact in artifacts:
        path = package / artifact["path"]
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    attributes = package / ".gitattributes"
    assert attributes.read_bytes() == hf_public._CANONICAL_HF_GITATTRIBUTES_BYTES
    assert attributes.stat().st_size == 2_569
    assert (
        hashlib.sha256(attributes.read_bytes()).hexdigest()
        == "4aa1947088aa553e4f5553ae7587b440a0e240df1574fc653199289077297a59"
    )


def test_package_validator_requires_canonical_gitattributes(
    tmp_path: Path,
) -> None:
    package = _build_fixture_package(tmp_path)
    attributes = package / ".gitattributes"
    attributes.write_text("*.jsonl filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical Hugging Face LFS policy"):
        validate_public_package(package)


def test_package_validator_requires_exact_manifest_allowlist(
    tmp_path: Path,
) -> None:
    package = _build_fixture_package(tmp_path)
    manifest_path = package / "PUBLIC_RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exact_package_allowlist"].remove(".gitattributes")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact_package_allowlist"):
        validate_public_package(package)


def test_package_validator_requires_manifest_self_hash_policy(
    tmp_path: Path,
) -> None:
    package = _build_fixture_package(tmp_path)
    manifest_path = package / "PUBLIC_RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_self_hash_policy"] = "self-hash omitted"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest_self_hash_policy"):
        validate_public_package(package)


def test_package_validator_rejects_same_size_artifact_tampering(
    tmp_path: Path,
) -> None:
    package = _build_fixture_package(tmp_path)
    readme = package / "README.md"
    original = readme.read_text(encoding="utf-8")
    readme.write_text(original.replace("BioReview", "XioReview", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch for README.md"):
        validate_public_package(package)


def test_package_validator_rejects_manifest_path_omission(tmp_path: Path) -> None:
    package = _build_fixture_package(tmp_path)
    manifest_path = package / "PUBLIC_RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        artifact for artifact in manifest["artifacts"]
        if artifact["path"] != "README.md"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact paths mismatch"):
        validate_public_package(package)

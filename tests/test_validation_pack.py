from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bioreview_bench.scripts.create_validation_pack import main
from bioreview_bench.validate.validation_pack import (
    CONCERN_FIDELITY,
    OMISSION_AUDIT,
    VALID_CATEGORIES,
    ValidationPackError,
    build_validation_pack,
    cohen_kappa,
    detect_suspicious_exact_identity,
    load_jsonl_entries,
    summarize_validation_pack,
    validate_annotation_rows,
    write_validation_pack,
)


def _entries() -> list[dict]:
    categories = (
        "design_flaw",
        "statistical_methodology",
        "missing_experiment",
        "interpretation",
    )
    severities = ("major", "minor", "optional")
    entries: list[dict] = []
    index = 0
    for source in ("elife", "plos"):
        for category in categories:
            for severity in severities:
                index += 1
                article_id = f"{source}:{index}"
                entries.append(
                    {
                        "id": article_id,
                        "source": source,
                        "title": f"Article {index}",
                        "benchmark_split": "val" if index % 2 else "test",
                        "decision_letter_raw": (
                            f"Reviewer source text for article {index}. "
                            "The methods and interpretation require careful review."
                        ),
                        "concerns": [
                            {
                                "concern_id": f"{article_id}:R1C1",
                                "concern_text": (
                                    f"Concern {index} identifies a substantive issue "
                                    "that should be evaluated independently."
                                ),
                                "category": category,
                                "severity": severity,
                                "author_stance": "partial",
                            }
                        ],
                    }
                )
    return entries


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _answer_key(pack_dir: Path) -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in (pack_dir / "coordinator" / "answer_key.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return {row["item_id"]: row for row in rows}


def _complete_annotations(pack_dir: Path) -> dict[str, dict]:
    key = _answer_key(pack_dir)
    for slot in ("rater_1", "rater_2"):
        concern_path = (
            pack_dir / "annotations" / f"{slot}_{CONCERN_FIDELITY}.csv"
        )
        concerns = _read_csv(concern_path)
        for row in concerns:
            row["human_fidelity"] = "supported"
            row["human_category"] = key[row["item_id"]]["extracted_category"]
            row["human_severity"] = key[row["item_id"]]["extracted_severity"]
            row["human_notes"] = "Independently checked against the review."
        _write_csv(concern_path, concerns)

        omission_path = (
            pack_dir / "annotations" / f"{slot}_{OMISSION_AUDIT}.csv"
        )
        omissions = _read_csv(omission_path)
        for row in omissions:
            row["human_omission_status"] = "none"
            row["human_omitted_concerns_json"] = "[]"
            row["human_notes"] = "Full review checked."
        _write_csv(omission_path, omissions)
    return key


def test_pack_is_deterministic_stratified_and_label_blinded() -> None:
    pack_a = build_validation_pack(
        _entries(),
        concern_sample_size=24,
        omission_sample_size=12,
        annotators=("expert_a", "expert_b"),
        seed=91,
    )
    pack_b = build_validation_pack(
        list(reversed(_entries())),
        concern_sample_size=24,
        omission_sample_size=12,
        annotators=("expert_a", "expert_b"),
        seed=91,
    )

    assert pack_a == pack_b
    concern_sampling = pack_a["manifest"]["sampling"][CONCERN_FIDELITY]
    assert concern_sampling["selected"] == 24
    assert len(concern_sampling["realized_strata"]) == 24
    assert len(concern_sampling["eligible_strata"]) == 24
    assert {
        row["source"] for row in concern_sampling["realized_strata"]
    } == {"elife", "plos"}
    label_blinding = pack_a["manifest"]["label_blinding"]
    assert label_blinding["scope"] == "system_labels_only"
    assert label_blinding["source_identity_visible"] is True
    assert label_blinding["cryptographic_anonymity_claimed"] is False
    assert pack_a["manifest"]["estimation"]["population_inference_supported"] is False

    for slot in ("rater_1", "rater_2"):
        concern_rows = pack_a["annotations"][slot][CONCERN_FIDELITY]
        omission_rows = pack_a["annotations"][slot][OMISSION_AUDIT]
        assert all(row["human_fidelity"] == "" for row in concern_rows)
        assert all(row["human_category"] == "" for row in concern_rows)
        assert all(row["human_severity"] == "" for row in concern_rows)
        assert all(row["human_omission_status"] == "" for row in omission_rows)
        assert all("extracted_category" not in row for row in concern_rows)
        assert all("extracted_severity" not in row for row in concern_rows)

    assert {
        row["item_id"]
        for row in pack_a["annotations"]["rater_1"][CONCERN_FIDELITY]
    } == {
        row["item_id"]
        for row in pack_a["annotations"]["rater_2"][CONCERN_FIDELITY]
    }
    assert all(
        row["stimulus_sha256"].startswith("sha256:")
        for row in pack_a["answer_key"]
    )


def test_input_fingerprint_covers_review_and_concern_text() -> None:
    original = build_validation_pack(
        _entries(),
        concern_sample_size=8,
        omission_sample_size=4,
        seed=19,
    )
    changed_review_entries = _entries()
    changed_review_entries[0]["decision_letter_raw"] += " Added source text."
    changed_review = build_validation_pack(
        changed_review_entries,
        concern_sample_size=8,
        omission_sample_size=4,
        seed=19,
    )
    changed_concern_entries = _entries()
    changed_concern_entries[0]["concerns"][0]["concern_text"] += " Added concern text."
    changed_concern = build_validation_pack(
        changed_concern_entries,
        concern_sample_size=8,
        omission_sample_size=4,
        seed=19,
    )

    fingerprint = original["manifest"]["input_fingerprint"]
    assert changed_review["manifest"]["input_fingerprint"] != fingerprint
    assert changed_concern["manifest"]["input_fingerprint"] != fingerprint


def test_pack_rejects_invalid_design_inputs() -> None:
    with pytest.raises(ValidationPackError, match="distinct"):
        build_validation_pack(
            _entries(),
            concern_sample_size=5,
            omission_sample_size=5,
            annotators=("same", "same"),
        )

    with pytest.raises(ValidationPackError, match="exceeds eligible concerns"):
        build_validation_pack(
            _entries(),
            concern_sample_size=1_000,
            omission_sample_size=5,
        )

    entries = _entries()
    entries[0]["concerns"][0]["category"] = "invented_category"
    with pytest.raises(ValidationPackError, match="invalid category"):
        build_validation_pack(
            entries,
            concern_sample_size=5,
            omission_sample_size=5,
        )


def test_write_pack_keeps_answer_key_out_of_annotation_files(tmp_path: Path) -> None:
    pack = build_validation_pack(
        _entries(),
        concern_sample_size=8,
        omission_sample_size=4,
        annotators=("expert_a", "expert_b"),
        seed=3,
    )
    output_dir = tmp_path / "pack"
    write_validation_pack(pack, output_dir)

    header = (
        output_dir / "annotations" / "rater_1_concern_fidelity.csv"
    ).read_text(encoding="utf-8").splitlines()[0]
    assert "human_category" in header
    assert "extracted_category" not in header
    assert (output_dir / "coordinator" / "answer_key.jsonl").is_file()

    with pytest.raises(ValidationPackError, match="refusing to overwrite"):
        write_validation_pack(pack, output_dir)


def test_annotation_validation_requires_explicit_unleaked_labels() -> None:
    pack = build_validation_pack(
        _entries(),
        concern_sample_size=2,
        omission_sample_size=2,
    )
    rows = pack["annotations"]["rater_1"][CONCERN_FIDELITY]
    with pytest.raises(ValidationPackError, match="human_fidelity"):
        validate_annotation_rows(
            rows,
            audit_type=CONCERN_FIDELITY,
            expected_annotator="annotator_1",
        )

    leaked = [dict(row) for row in rows]
    leaked[0]["extracted_category"] = "design_flaw"
    with pytest.raises(ValidationPackError, match="hidden system-label"):
        validate_annotation_rows(
            leaked,
            audit_type=CONCERN_FIDELITY,
            expected_annotator="annotator_1",
            require_complete=False,
        )


def test_omission_validation_requires_consistent_explicit_json() -> None:
    pack = build_validation_pack(
        _entries(),
        concern_sample_size=2,
        omission_sample_size=2,
    )
    row = dict(pack["annotations"]["rater_1"][OMISSION_AUDIT][0])
    row["human_omission_status"] = "none"
    row["human_omitted_concerns_json"] = '["A missing concern"]'

    with pytest.raises(ValidationPackError, match="status is 'none'"):
        validate_annotation_rows(
            [row],
            audit_type=OMISSION_AUDIT,
            expected_annotator="annotator_1",
        )


def test_completed_pack_summary_reports_descriptive_clustered_agreement(
    tmp_path: Path,
) -> None:
    pack = build_validation_pack(
        _entries(),
        concern_sample_size=12,
        omission_sample_size=6,
        annotators=("expert_a", "expert_b"),
        seed=7,
    )
    pack_dir = tmp_path / "pack"
    write_validation_pack(pack, pack_dir)
    key = _answer_key(pack_dir)
    omission_target = min(
        item_id
        for item_id, row in key.items()
        if row["audit_type"] == OMISSION_AUDIT
    )

    for slot_index, slot in enumerate(("rater_1", "rater_2")):
        concern_path = (
            pack_dir / "annotations" / f"{slot}_{CONCERN_FIDELITY}.csv"
        )
        concerns = _read_csv(concern_path)
        for row_index, row in enumerate(concerns):
            row["human_fidelity"] = "supported"
            row["human_category"] = key[row["item_id"]]["extracted_category"]
            row["human_severity"] = key[row["item_id"]]["extracted_severity"]
            row["human_notes"] = "Independently checked against the review."
            if slot_index == 1 and row_index == 0:
                row["human_fidelity"] = "unsupported"
                row["human_category"] = next(
                    category
                    for category in VALID_CATEGORIES
                    if category != row["human_category"]
                )
        _write_csv(concern_path, concerns)

        omission_path = (
            pack_dir / "annotations" / f"{slot}_{OMISSION_AUDIT}.csv"
        )
        omissions = _read_csv(omission_path)
        for row in omissions:
            if row["item_id"] == omission_target:
                row["human_omission_status"] = "one_or_more"
                row["human_omitted_concerns_json"] = json.dumps(
                    ["A missing control-related concern."]
                )
            else:
                row["human_omission_status"] = "none"
                row["human_omitted_concerns_json"] = "[]"
            row["human_notes"] = "Full review checked."
        _write_csv(omission_path, omissions)

    summary = summarize_validation_pack(
        pack_dir,
        bootstrap_samples=100,
        seed=11,
    )

    assert summary["n_concern_items"] == 12
    assert summary["n_omission_items"] == 6
    category = summary["inter_annotator"]["category"]
    assert category["observed_agreement"] == pytest.approx(11 / 12)
    assert category["observed_agreement_cluster_bootstrap_interval"][0] < 11 / 12
    assert category["n_articles"] == 12
    assert category["cohen_kappa"] is not None
    assert summary["inter_annotator"]["omission_status"]["observed_agreement"] == 1
    assert summary["audit_rates"]["rater_1"]["supported_concern_rate"][
        "proportion"
    ] == 1
    assert summary["audit_rates"]["rater_1"]["supported_concern_rate"][
        "population_confidence_interval"
    ] is None
    assert "not population confidence intervals" in summary["estimand_notes"][
        "agreement"
    ]


def test_summary_rejects_tampered_rater_stimulus(tmp_path: Path) -> None:
    pack = build_validation_pack(
        _entries(),
        concern_sample_size=8,
        omission_sample_size=4,
        annotators=("expert_a", "expert_b"),
        seed=13,
    )
    pack_dir = tmp_path / "pack"
    write_validation_pack(pack, pack_dir)
    _complete_annotations(pack_dir)

    concern_path = (
        pack_dir / "annotations" / f"rater_2_{CONCERN_FIDELITY}.csv"
    )
    rows = _read_csv(concern_path)
    rows[0]["source_review_text"] += " Tampered after assignment."
    _write_csv(concern_path, rows)

    with pytest.raises(ValidationPackError, match="failed integrity check"):
        summarize_validation_pack(pack_dir, bootstrap_samples=10)


def test_concern_agreement_bootstrap_clusters_by_article(tmp_path: Path) -> None:
    entries = _entries()[:2]
    for entry in entries:
        original = entry["concerns"][0]
        entry["concerns"].append(
            {
                **original,
                "concern_id": f"{entry['id']}:R1C2",
                "concern_text": f"Second concern for {entry['id']}.",
            }
        )
    pack = build_validation_pack(
        entries,
        concern_sample_size=4,
        omission_sample_size=2,
        seed=23,
    )
    pack_dir = tmp_path / "clustered_pack"
    write_validation_pack(pack, pack_dir)
    _complete_annotations(pack_dir)

    summary = summarize_validation_pack(
        pack_dir,
        bootstrap_samples=25,
        seed=29,
    )

    agreement = summary["inter_annotator"]["category"]
    assert agreement["n"] == 4
    assert agreement["n_articles"] == 2
    assert "article-cluster bootstrap" in agreement["interval_note"]


def test_degenerate_kappa_is_reported_as_not_identified() -> None:
    kappa, note = cohen_kappa(["major", "major"], ["major", "major"])
    assert kappa is None
    assert note is not None
    assert "no marginal variation" in note


def test_suspicious_identity_detector_flags_legacy_prefill_pattern() -> None:
    rows = [
        {
            "llm_category": "design_flaw" if index % 2 else "interpretation",
            "human_category": (
                "design_flaw" if index % 2 else "interpretation"
            ),
            "llm_stance": "conceded" if index % 3 else "partial",
            "human_stance": "conceded" if index % 3 else "partial",
            "notes": "",
        }
        for index in range(25)
    ]

    check = detect_suspicious_exact_identity(rows)

    assert check.suspicious is True
    assert check.notes_blank_fraction == 1
    assert sum(pair.suspicious for pair in check.pair_checks) == 2
    assert "should not be cited as independent validation" in check.message


def test_load_jsonl_and_cli_create_pack(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    entries = _entries()
    for split in ("val", "test"):
        selected = [entry for entry in entries if entry["benchmark_split"] == split]
        (splits_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(entry) + "\n" for entry in selected),
            encoding="utf-8",
        )

    loaded = load_jsonl_entries(
        [splits_dir / "val.jsonl", splits_dir / "test.jsonl"]
    )
    assert len(loaded) == len(entries)

    output_dir = tmp_path / "cli_pack"
    result = CliRunner().invoke(
        main,
        [
            "create",
            "--splits-dir",
            str(splits_dir),
            "--concern-sample-size",
            "8",
            "--omission-sample-size",
            "4",
            "--annotator",
            "expert_a",
            "--annotator",
            "expert_b",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "manifest.json").is_file()

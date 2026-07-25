"""System-label-blinded, two-annotator validation packs for label audits.

The validation design separates two questions that should not be conflated:

* ``concern_fidelity``: Is an extracted concern supported by the source review,
  and are its category and severity labels reproducible?
* ``omission_audit``: Did the extraction omit one or more substantive concerns
  from an article's source review?

System category and severity labels are written only to a coordinator-held
answer key. Annotator files contain empty human-label fields, but retain source
and article context; this is label blinding, not source anonymization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bioreview_bench.models.concern import ConcernCategory

CONCERN_FIDELITY = "concern_fidelity"
OMISSION_AUDIT = "omission_audit"

VALID_FIDELITY_LABELS = ("supported", "unsupported", "unclear")
VALID_OMISSION_LABELS = ("none", "one_or_more", "unclear")
VALID_SEVERITIES = ("major", "minor", "optional")
VALID_CATEGORIES = tuple(category.value for category in ConcernCategory)

CONCERN_ANNOTATION_COLUMNS = (
    "item_id",
    "audit_type",
    "annotator_id",
    "source",
    "benchmark_split",
    "article_title",
    "source_review_text",
    "extracted_concern_text",
    "human_fidelity",
    "human_category",
    "human_severity",
    "human_notes",
)

OMISSION_ANNOTATION_COLUMNS = (
    "item_id",
    "audit_type",
    "annotator_id",
    "source",
    "benchmark_split",
    "article_title",
    "source_review_text",
    "extracted_concerns_json",
    "human_omission_status",
    "human_omitted_concerns_json",
    "human_notes",
)

_STIMULUS_FIELDS = {
    CONCERN_FIDELITY: (
        "item_id",
        "audit_type",
        "source",
        "benchmark_split",
        "article_title",
        "source_review_text",
        "extracted_concern_text",
    ),
    OMISSION_AUDIT: (
        "item_id",
        "audit_type",
        "source",
        "benchmark_split",
        "article_title",
        "source_review_text",
        "extracted_concerns_json",
    ),
}

_FORBIDDEN_BLINDED_FIELDS = frozenset(
    {
        "extracted_category",
        "extracted_severity",
        "extracted_author_stance",
        "llm_category",
        "llm_severity",
        "llm_stance",
        "system_category",
        "system_severity",
        "system_stance",
    }
)

_DEFAULT_IDENTITY_FIELD_PAIRS = (
    ("extracted_category", "human_category"),
    ("llm_category", "human_category"),
    ("system_category", "human_category"),
    ("extracted_severity", "human_severity"),
    ("llm_severity", "human_severity"),
    ("system_severity", "human_severity"),
    ("extracted_author_stance", "human_stance"),
    ("llm_stance", "human_stance"),
    ("system_stance", "human_stance"),
)


class ValidationPackError(ValueError):
    """Raised when pack inputs or completed annotations are invalid."""


@dataclass(frozen=True)
class AgreementEstimate:
    """Descriptive agreement with article-cluster stability intervals."""

    n: int
    n_articles: int
    observed_agreement: float
    observed_agreement_cluster_bootstrap_interval: tuple[float, float] | None
    cohen_kappa: float | None
    cohen_kappa_cluster_bootstrap_interval: tuple[float, float] | None
    kappa_note: str | None
    interval_note: str


@dataclass(frozen=True)
class IdentityPairCheck:
    """Exact-identity diagnostic for one system/human field pair."""

    system_field: str
    human_field: str
    n_complete: int
    n_exact: int
    exact_fraction: float
    suspicious: bool


@dataclass(frozen=True)
class IdentityCheck:
    """Diagnostic that flags perfect copied-looking human/system labels."""

    suspicious: bool
    n_rows: int
    notes_blank_fraction: float | None
    pair_checks: tuple[IdentityPairCheck, ...]
    message: str


def load_jsonl_entries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load benchmark entries, adding a split name inferred from each filename."""
    if not paths:
        raise ValidationPackError("At least one JSONL input path is required.")

    entries: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ValidationPackError(f"Input JSONL does not exist: {path}")
        split_name = path.stem
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationPackError(
                        f"{path}:{line_number}: invalid JSON ({exc.msg})."
                    ) from exc
                if not isinstance(row, dict):
                    raise ValidationPackError(
                        f"{path}:{line_number}: expected a JSON object."
                    )
                row = dict(row)
                row.setdefault("benchmark_split", split_name)
                entries.append(row)
    return entries


def build_validation_pack(
    entries: Sequence[Mapping[str, Any]],
    *,
    concern_sample_size: int,
    omission_sample_size: int,
    annotators: Sequence[str] = ("annotator_1", "annotator_2"),
    seed: int = 42,
) -> dict[str, Any]:
    """Build a deterministic, system-label-blinded validation pack in memory.

    Every selected item is assigned to both annotators so that inter-annotator
    agreement can be estimated. Concern items are stratified on
    ``(source, category, severity)``. Omission items are stratified on
    ``(source, dominant category, dominant severity)`` at article level.
    Source, title, and review text remain visible to annotators.
    """
    annotator_ids = _validate_annotators(annotators)
    if concern_sample_size <= 0 or omission_sample_size <= 0:
        raise ValidationPackError(
            "Concern-fidelity and omission-audit sample sizes must both be positive."
        )

    normalized_entries = _validate_and_normalize_entries(entries)
    concern_pool = _flatten_concerns(normalized_entries)
    omission_pool = [_omission_candidate(entry) for entry in normalized_entries]

    if concern_sample_size > len(concern_pool):
        raise ValidationPackError(
            "Concern sample size exceeds eligible concerns "
            f"({concern_sample_size} requested; {len(concern_pool)} available)."
        )
    if omission_sample_size > len(omission_pool):
        raise ValidationPackError(
            "Omission sample size exceeds eligible articles "
            f"({omission_sample_size} requested; {len(omission_pool)} available)."
        )

    sampled_concerns = _stratified_sample(
        concern_pool,
        n=concern_sample_size,
        stratum=lambda row: (
            str(row["source"]),
            str(row["extracted_category"]),
            str(row["extracted_severity"]),
        ),
        stable_id=lambda row: str(row["concern_id"]),
        seed=seed,
    )
    sampled_omissions = _stratified_sample(
        omission_pool,
        n=omission_sample_size,
        stratum=lambda row: (
            str(row["source"]),
            str(row["dominant_category"]),
            str(row["dominant_severity"]),
        ),
        stable_id=lambda row: str(row["article_id"]),
        seed=seed + 1,
    )

    concern_rows, concern_keys = _make_concern_rows(sampled_concerns, seed=seed)
    omission_rows, omission_keys = _make_omission_rows(sampled_omissions, seed=seed)
    answer_key = sorted(concern_keys + omission_keys, key=lambda row: row["item_id"])

    annotations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    annotation_files: list[dict[str, str]] = []
    for index, annotator_id in enumerate(annotator_ids, start=1):
        slot = f"rater_{index}"
        rater_rng = random.Random(seed + index * 10_007)
        rater_concerns = [
            {**row, "annotator_id": annotator_id} for row in concern_rows
        ]
        rater_omissions = [
            {**row, "annotator_id": annotator_id} for row in omission_rows
        ]
        rater_rng.shuffle(rater_concerns)
        rater_rng.shuffle(rater_omissions)
        annotations[slot] = {
            CONCERN_FIDELITY: rater_concerns,
            OMISSION_AUDIT: rater_omissions,
        }
        annotation_files.extend(
            [
                {
                    "slot": slot,
                    "annotator_id": annotator_id,
                    "audit_type": CONCERN_FIDELITY,
                    "path": f"annotations/{slot}_{CONCERN_FIDELITY}.csv",
                },
                {
                    "slot": slot,
                    "annotator_id": annotator_id,
                    "audit_type": OMISSION_AUDIT,
                    "path": f"annotations/{slot}_{OMISSION_AUDIT}.csv",
                },
            ]
        )

    manifest = {
        "schema_version": "1.1",
        "seed": seed,
        "annotators": [
            {"slot": f"rater_{index}", "annotator_id": annotator_id}
            for index, annotator_id in enumerate(annotator_ids, start=1)
        ],
        "sampling": {
            CONCERN_FIDELITY: {
                "unit": "extracted_concern",
                "requested": concern_sample_size,
                "selected": len(sampled_concerns),
                "eligible": len(concern_pool),
                "stratification": ["source", "category", "severity"],
                "eligible_strata": _count_strata(
                    concern_pool,
                    fields=("source", "extracted_category", "extracted_severity"),
                ),
                "realized_strata": _count_strata(
                    sampled_concerns,
                    fields=("source", "extracted_category", "extracted_severity"),
                ),
            },
            OMISSION_AUDIT: {
                "unit": "article",
                "requested": omission_sample_size,
                "selected": len(sampled_omissions),
                "eligible": len(omission_pool),
                "stratification": [
                    "source",
                    "dominant_concern_category",
                    "dominant_concern_severity",
                ],
                "eligible_strata": _count_strata(
                    omission_pool,
                    fields=("source", "dominant_category", "dominant_severity"),
                ),
                "realized_strata": _count_strata(
                    sampled_omissions,
                    fields=("source", "dominant_category", "dominant_severity"),
                ),
            },
        },
        "estimation": {
            "audit_rate_estimand": (
                "unweighted_descriptive_rate_in_realized_stratified_sample"
            ),
            "population_inference_supported": False,
            "reason": (
                "Unequal stratum inclusion probabilities are not accompanied "
                "by survey weights."
            ),
            "agreement_interval_method": (
                "article_cluster_bootstrap_within_realized_sample"
            ),
        },
        "label_blinding": {
            "scope": "system_labels_only",
            "annotation_files_exclude_system_category_and_severity": True,
            "answer_key": "coordinator/answer_key.jsonl",
            "answer_key_access": "coordinator_only_until_annotations_are_locked",
            "human_labels_prefilled": False,
            "source_identity_visible": True,
            "cryptographic_anonymity_claimed": False,
            "item_ids": "deterministic_join_keys_not_anonymization",
        },
        "annotation_files": annotation_files,
        "input_fingerprint": _input_fingerprint(normalized_entries),
    }
    return {
        "manifest": manifest,
        "answer_key": answer_key,
        "annotations": annotations,
    }


def write_validation_pack(pack: Mapping[str, Any], output_dir: Path) -> None:
    """Write a pack to an empty directory without overwriting prior work."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValidationPackError(
            f"Output directory is not empty; refusing to overwrite annotations: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = output_dir / "annotations"
    coordinator_dir = output_dir / "coordinator"
    annotations_dir.mkdir(exist_ok=True)
    coordinator_dir.mkdir(exist_ok=True)

    manifest = pack["manifest"]
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(coordinator_dir / "answer_key.jsonl", pack["answer_key"])

    for file_record in manifest["annotation_files"]:
        slot = file_record["slot"]
        audit_type = file_record["audit_type"]
        path = output_dir / file_record["path"]
        rows = pack["annotations"][slot][audit_type]
        columns = (
            CONCERN_ANNOTATION_COLUMNS
            if audit_type == CONCERN_FIDELITY
            else OMISSION_ANNOTATION_COLUMNS
        )
        _write_csv(path, rows, columns)


def validate_annotation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    audit_type: str,
    expected_annotator: str | None = None,
    require_complete: bool = True,
) -> None:
    """Validate one rater's system-label-blinded annotation rows."""
    if audit_type not in (CONCERN_FIDELITY, OMISSION_AUDIT):
        raise ValidationPackError(f"Unknown audit type: {audit_type}")
    if not rows:
        raise ValidationPackError(f"No rows supplied for {audit_type}.")

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        leaked = _FORBIDDEN_BLINDED_FIELDS.intersection(row)
        if leaked:
            raise ValidationPackError(
                f"Row {row_number} contains hidden system-label field(s): "
                + ", ".join(sorted(leaked))
            )

        item_id = _clean(row.get("item_id"))
        if not item_id:
            raise ValidationPackError(f"Row {row_number}: item_id is required.")
        if item_id in seen:
            raise ValidationPackError(f"Duplicate item_id in annotation file: {item_id}")
        seen.add(item_id)

        if _clean(row.get("audit_type")) != audit_type:
            raise ValidationPackError(
                f"Row {row_number}: audit_type must be {audit_type!r}."
            )
        annotator_id = _clean(row.get("annotator_id"))
        if not annotator_id:
            raise ValidationPackError(f"Row {row_number}: annotator_id is required.")
        if expected_annotator is not None and annotator_id != expected_annotator:
            raise ValidationPackError(
                f"Row {row_number}: expected annotator {expected_annotator!r}, "
                f"found {annotator_id!r}."
            )

        if not require_complete:
            continue
        if audit_type == CONCERN_FIDELITY:
            _validate_choice(
                row,
                row_number,
                "human_fidelity",
                VALID_FIDELITY_LABELS,
            )
            _validate_choice(
                row,
                row_number,
                "human_category",
                VALID_CATEGORIES,
            )
            _validate_choice(
                row,
                row_number,
                "human_severity",
                VALID_SEVERITIES,
            )
        else:
            omission_status = _validate_choice(
                row,
                row_number,
                "human_omission_status",
                VALID_OMISSION_LABELS,
            )
            raw_omissions = _clean(row.get("human_omitted_concerns_json"))
            if not raw_omissions:
                raise ValidationPackError(
                    f"Row {row_number}: human_omitted_concerns_json must be explicit "
                    "([] when no omission was found)."
                )
            try:
                omissions = json.loads(raw_omissions)
            except json.JSONDecodeError as exc:
                raise ValidationPackError(
                    f"Row {row_number}: human_omitted_concerns_json is invalid JSON."
                ) from exc
            if not isinstance(omissions, list) or any(
                not isinstance(item, str) or not item.strip() for item in omissions
            ):
                raise ValidationPackError(
                    f"Row {row_number}: human_omitted_concerns_json must be a JSON "
                    "array of non-empty strings."
                )
            if omission_status == "none" and omissions:
                raise ValidationPackError(
                    f"Row {row_number}: omission status is 'none' but omitted "
                    "concerns are listed."
                )
            if omission_status == "one_or_more" and not omissions:
                raise ValidationPackError(
                    f"Row {row_number}: status 'one_or_more' requires at least one "
                    "omitted concern."
                )


def summarize_validation_pack(
    pack_dir: Path,
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Validate completed files and summarize descriptive agreement.

    Audit rates describe the realized stratified sample without population
    confidence intervals. Agreement stability intervals use an article-cluster
    bootstrap so that multiple concerns from one article are not treated as
    independent draws.
    """
    if bootstrap_samples < 0:
        raise ValidationPackError("bootstrap_samples must be non-negative.")
    _validate_confidence(confidence)

    manifest_path = pack_dir / "manifest.json"
    answer_key_path = pack_dir / "coordinator" / "answer_key.jsonl"
    if not manifest_path.is_file() or not answer_key_path.is_file():
        raise ValidationPackError(
            "Pack directory must contain manifest.json and coordinator/answer_key.jsonl."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rater_records = manifest.get("annotators", [])
    if len(rater_records) != 2:
        raise ValidationPackError("Exactly two annotators are required for agreement.")

    by_slot: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    for file_record in manifest.get("annotation_files", []):
        slot = str(file_record["slot"])
        audit_type = str(file_record["audit_type"])
        rows = _read_csv(pack_dir / str(file_record["path"]))
        expected = next(
            (
                str(record["annotator_id"])
                for record in rater_records
                if record["slot"] == slot
            ),
            None,
        )
        validate_annotation_rows(
            rows,
            audit_type=audit_type,
            expected_annotator=expected,
            require_complete=True,
        )
        by_slot[slot][audit_type] = rows

    slots = [str(record["slot"]) for record in rater_records]
    for slot in slots:
        for audit_type in (CONCERN_FIDELITY, OMISSION_AUDIT):
            if audit_type not in by_slot[slot]:
                raise ValidationPackError(
                    f"Missing {audit_type} annotation file for {slot}."
                )

    answer_key_rows = _read_jsonl(answer_key_path)
    answer_key = {str(row["item_id"]): row for row in answer_key_rows}
    if len(answer_key) != len(answer_key_rows):
        raise ValidationPackError("Coordinator answer key contains duplicate item_id.")
    concern_a, concern_b = _align_rows(
        by_slot[slots[0]][CONCERN_FIDELITY],
        by_slot[slots[1]][CONCERN_FIDELITY],
        audit_type=CONCERN_FIDELITY,
    )
    omission_a, omission_b = _align_rows(
        by_slot[slots[0]][OMISSION_AUDIT],
        by_slot[slots[1]][OMISSION_AUDIT],
        audit_type=OMISSION_AUDIT,
    )
    _validate_key_coverage(answer_key, concern_a + omission_a)
    _validate_stimulus_integrity(
        answer_key,
        concern_a + concern_b + omission_a + omission_b,
    )

    concern_article_ids = [
        _clean(answer_key[row["item_id"]]["article_id"]) for row in concern_a
    ]
    omission_article_ids = [
        _clean(answer_key[row["item_id"]]["article_id"]) for row in omission_a
    ]

    inter_annotator = {
        "fidelity": _agreement_as_dict(
            _agreement_estimate(
                [_clean(row["human_fidelity"]) for row in concern_a],
                [_clean(row["human_fidelity"]) for row in concern_b],
                cluster_ids=concern_article_ids,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        ),
        "category": _agreement_as_dict(
            _agreement_estimate(
                [_clean(row["human_category"]) for row in concern_a],
                [_clean(row["human_category"]) for row in concern_b],
                cluster_ids=concern_article_ids,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1,
            )
        ),
        "severity": _agreement_as_dict(
            _agreement_estimate(
                [_clean(row["human_severity"]) for row in concern_a],
                [_clean(row["human_severity"]) for row in concern_b],
                cluster_ids=concern_article_ids,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 2,
            )
        ),
        "omission_status": _agreement_as_dict(
            _agreement_estimate(
                [_clean(row["human_omission_status"]) for row in omission_a],
                [_clean(row["human_omission_status"]) for row in omission_b],
                cluster_ids=omission_article_ids,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 3,
            )
        ),
    }

    extracted_agreement: dict[str, Any] = {}
    rates: dict[str, Any] = {}
    identity_checks: dict[str, Any] = {}
    for slot_index, (slot, concern_rows, omission_rows) in enumerate(
        (
            (slots[0], concern_a, omission_a),
            (slots[1], concern_b, omission_b),
        )
    ):
        system_categories = [
            _clean(answer_key[row["item_id"]]["extracted_category"])
            for row in concern_rows
        ]
        system_severities = [
            _clean(answer_key[row["item_id"]]["extracted_severity"])
            for row in concern_rows
        ]
        extracted_agreement[slot] = {
            "category": _agreement_as_dict(
                _agreement_estimate(
                    [_clean(row["human_category"]) for row in concern_rows],
                    system_categories,
                    cluster_ids=concern_article_ids,
                    confidence=confidence,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 10 + slot_index,
                )
            ),
            "severity": _agreement_as_dict(
                _agreement_estimate(
                    [_clean(row["human_severity"]) for row in concern_rows],
                    system_severities,
                    cluster_ids=concern_article_ids,
                    confidence=confidence,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 20 + slot_index,
                )
            ),
        }
        supported = sum(
            _clean(row["human_fidelity"]) == "supported" for row in concern_rows
        )
        omissions_found = sum(
            _clean(row["human_omission_status"]) == "one_or_more"
            for row in omission_rows
        )
        rates[slot] = {
            "supported_concern_rate": _descriptive_rate(
                supported,
                len(concern_rows),
            ),
            "articles_with_detected_omissions": _descriptive_rate(
                omissions_found,
                len(omission_rows),
            ),
        }
        joined_for_identity = [
            {
                **row,
                "extracted_category": answer_key[row["item_id"]][
                    "extracted_category"
                ],
                "extracted_severity": answer_key[row["item_id"]][
                    "extracted_severity"
                ],
            }
            for row in concern_rows
        ]
        identity_checks[slot] = asdict(
            detect_suspicious_exact_identity(joined_for_identity)
        )

    return {
        "schema_version": "1.1",
        "cluster_bootstrap_interval_level": confidence,
        "bootstrap_samples": bootstrap_samples,
        "n_concern_items": len(concern_a),
        "n_concern_articles": len(set(concern_article_ids)),
        "n_omission_items": len(omission_a),
        "estimand_notes": {
            "audit_rates": (
                "Unweighted descriptive rates in the realized stratified sample. "
                "No population confidence interval is reported because inclusion "
                "weights were not computed."
            ),
            "agreement": (
                "Agreement values are descriptive. Intervals resample whole "
                "articles and measure stability within the realized sample; they "
                "are not population confidence intervals."
            ),
        },
        "inter_annotator": inter_annotator,
        "agreement_with_extracted_labels": extracted_agreement,
        "audit_rates": rates,
        "identity_checks": identity_checks,
    }


def cohen_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> tuple[float | None, str | None]:
    """Compute Cohen's kappa, returning ``None`` when it is not identified."""
    if len(labels_a) != len(labels_b):
        raise ValidationPackError("Paired label vectors must have equal length.")
    n = len(labels_a)
    if n < 2:
        return None, "Kappa requires at least two paired observations."

    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    labels = set(counts_a) | set(counts_b)
    expected = sum(
        (counts_a[label] / n) * (counts_b[label] / n) for label in labels
    )
    if math.isclose(expected, 1.0):
        return (
            None,
            "Kappa is undefined because both label vectors have no marginal variation.",
        )
    return (observed - expected) / (1.0 - expected), None


def _descriptive_rate(successes: int, n: int) -> dict[str, Any]:
    if n <= 0:
        raise ValidationPackError("A descriptive rate requires n > 0.")
    if successes < 0 or successes > n:
        raise ValidationPackError("successes must be between 0 and n.")
    return {
        "successes": successes,
        "n": n,
        "proportion": successes / n,
        "estimand": "unweighted_rate_in_realized_stratified_sample",
        "population_confidence_interval": None,
        "inference_note": (
            "Descriptive only; unequal stratum inclusion probabilities were not "
            "converted to survey weights."
        ),
    }


def detect_suspicious_exact_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    field_pairs: Sequence[tuple[str, str]] = _DEFAULT_IDENTITY_FIELD_PAIRS,
    min_rows: int = 20,
) -> IdentityCheck:
    """Flag perfect human/system identity as a protocol-audit warning.

    Perfect identity does not prove labels were copied. It does mean that
    independence cannot be inferred from the file itself and that annotation
    provenance should be checked, especially when notes are also absent.
    """
    if min_rows <= 0:
        raise ValidationPackError("min_rows must be positive.")
    pair_checks: list[IdentityPairCheck] = []
    for system_field, human_field in field_pairs:
        comparable = [
            (_clean(row.get(system_field)), _clean(row.get(human_field)))
            for row in rows
            if _clean(row.get(system_field)) and _clean(row.get(human_field))
        ]
        if not comparable:
            continue
        n_exact = sum(system == human for system, human in comparable)
        n_complete = len(comparable)
        pair_checks.append(
            IdentityPairCheck(
                system_field=system_field,
                human_field=human_field,
                n_complete=n_complete,
                n_exact=n_exact,
                exact_fraction=n_exact / n_complete,
                suspicious=n_complete >= min_rows and n_exact == n_complete,
            )
        )

    note_rows = [row for row in rows if "human_notes" in row or "notes" in row]
    notes_blank_fraction = None
    if note_rows:
        notes_blank_fraction = sum(
            not _clean(row.get("human_notes", row.get("notes"))) for row in note_rows
        ) / len(note_rows)

    suspicious_pairs = [check for check in pair_checks if check.suspicious]
    suspicious = bool(suspicious_pairs)
    if suspicious:
        fields = ", ".join(
            f"{check.system_field}→{check.human_field}" for check in suspicious_pairs
        )
        message = (
            "Perfect system/human identity was found for "
            f"{fields} across at least {min_rows} rows. This does not prove copying, "
            "but the labels should not be cited as independent validation until "
            "system-label blinding and annotation provenance are verified."
        )
    else:
        message = "No perfect system/human identity pattern met the warning threshold."
    return IdentityCheck(
        suspicious=suspicious,
        n_rows=len(rows),
        notes_blank_fraction=notes_blank_fraction,
        pair_checks=tuple(pair_checks),
        message=message,
    )


def _validate_annotators(annotators: Sequence[str]) -> tuple[str, str]:
    cleaned = tuple(_clean(annotator) for annotator in annotators)
    if len(cleaned) != 2:
        raise ValidationPackError("Exactly two annotators are required.")
    if not all(cleaned):
        raise ValidationPackError("Annotator identifiers must be non-empty.")
    if cleaned[0] == cleaned[1]:
        raise ValidationPackError("Annotator identifiers must be distinct.")
    return cleaned


def _validate_and_normalize_entries(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not entries:
        raise ValidationPackError("No benchmark entries were supplied.")
    normalized: list[dict[str, Any]] = []
    seen_articles: set[str] = set()
    seen_concerns: set[str] = set()
    for index, raw_entry in enumerate(entries, start=1):
        entry = dict(raw_entry)
        article_id = _clean(entry.get("id"))
        source = _clean(entry.get("source"))
        review_text = _clean(entry.get("decision_letter_raw"))
        title = _clean(entry.get("title"))
        split = _clean(entry.get("benchmark_split")) or "unknown"
        if not article_id or not source or not title or not review_text:
            raise ValidationPackError(
                f"Entry {index} requires non-empty id, source, title, and "
                "decision_letter_raw fields."
            )
        if article_id in seen_articles:
            raise ValidationPackError(f"Duplicate article id: {article_id}")
        seen_articles.add(article_id)

        concerns = entry.get("concerns")
        if not isinstance(concerns, list) or not concerns:
            raise ValidationPackError(
                f"Entry {article_id} must contain at least one concern."
            )
        normalized_concerns: list[dict[str, Any]] = []
        for concern_index, raw_concern in enumerate(concerns, start=1):
            if not isinstance(raw_concern, Mapping):
                raise ValidationPackError(
                    f"Entry {article_id} concern {concern_index} must be an object."
                )
            concern = dict(raw_concern)
            concern_id = _clean(concern.get("concern_id"))
            concern_text = _clean(concern.get("concern_text"))
            category = _clean(concern.get("category"))
            severity = _clean(concern.get("severity"))
            if not concern_id or not concern_text:
                raise ValidationPackError(
                    f"Entry {article_id} concern {concern_index} requires "
                    "concern_id and concern_text."
                )
            if concern_id in seen_concerns:
                raise ValidationPackError(f"Duplicate concern id: {concern_id}")
            seen_concerns.add(concern_id)
            if category not in VALID_CATEGORIES:
                raise ValidationPackError(
                    f"Concern {concern_id} has invalid category {category!r}."
                )
            if severity not in VALID_SEVERITIES:
                raise ValidationPackError(
                    f"Concern {concern_id} has invalid severity {severity!r}."
                )
            concern.update(
                {
                    "concern_id": concern_id,
                    "concern_text": concern_text,
                    "category": category,
                    "severity": severity,
                }
            )
            normalized_concerns.append(concern)
        entry.update(
            {
                "id": article_id,
                "source": source,
                "title": title,
                "decision_letter_raw": review_text,
                "benchmark_split": split,
                "concerns": normalized_concerns,
            }
        )
        normalized.append(entry)
    return sorted(normalized, key=lambda row: str(row["id"]))


def _flatten_concerns(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        for concern in entry["concerns"]:
            rows.append(
                {
                    "article_id": entry["id"],
                    "source": entry["source"],
                    "benchmark_split": entry["benchmark_split"],
                    "article_title": entry["title"],
                    "source_review_text": entry["decision_letter_raw"],
                    "concern_id": concern["concern_id"],
                    "extracted_concern_text": concern["concern_text"],
                    "extracted_category": concern["category"],
                    "extracted_severity": concern["severity"],
                    "extracted_author_stance": concern.get("author_stance", ""),
                }
            )
    return rows


def _omission_candidate(entry: Mapping[str, Any]) -> dict[str, Any]:
    concerns = entry["concerns"]
    categories = [str(concern["category"]) for concern in concerns]
    severities = [str(concern["severity"]) for concern in concerns]
    return {
        "article_id": entry["id"],
        "source": entry["source"],
        "benchmark_split": entry["benchmark_split"],
        "article_title": entry["title"],
        "source_review_text": entry["decision_letter_raw"],
        "extracted_concerns": [str(concern["concern_text"]) for concern in concerns],
        "dominant_category": _dominant_label(categories),
        "dominant_severity": _dominant_label(severities),
    }


def _dominant_label(labels: Sequence[str]) -> str:
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], label))


def _stratified_sample(
    rows: Sequence[dict[str, Any]],
    *,
    n: int,
    stratum: Callable[[dict[str, Any]], tuple[str, ...]],
    stable_id: Callable[[dict[str, Any]], str],
    seed: int,
) -> list[dict[str, Any]]:
    if n == 0:
        return []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=stable_id)

    keys = sorted(groups)
    allocations = {key: 0 for key in keys}
    if n < len(keys):
        marginal_counts: list[Counter[str]] = [
            Counter() for _ in range(len(keys[0]))
        ]
        remaining_keys = set(keys)
        for _ in range(n):
            chosen = max(
                remaining_keys,
                key=lambda key: (
                    sum(
                        1.0 / (1 + marginal_counts[index][value])
                        for index, value in enumerate(key)
                    ),
                    len(groups[key]),
                    tuple(_reverse_lexical(value) for value in key),
                ),
            )
            allocations[chosen] = 1
            remaining_keys.remove(chosen)
            for index, value in enumerate(chosen):
                marginal_counts[index][value] += 1
    else:
        for key in keys:
            allocations[key] = 1
        remaining = n - len(keys)
        target = {key: n * len(groups[key]) / len(rows) for key in keys}
        while remaining:
            eligible = [
                key for key in keys if allocations[key] < len(groups[key])
            ]
            chosen = max(
                eligible,
                key=lambda key: (
                    target[key] - allocations[key],
                    len(groups[key]) - allocations[key],
                    tuple(_reverse_lexical(value) for value in key),
                ),
            )
            allocations[chosen] += 1
            remaining -= 1

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for key in keys:
        allocation = allocations[key]
        if allocation:
            sampled.extend(rng.sample(groups[key], allocation))
    rng.shuffle(sampled)
    return sampled


def _reverse_lexical(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)


def _make_concern_rows(
    sampled: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotation_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for row in sampled:
        item_id = _deterministic_item_id(
            CONCERN_FIDELITY,
            str(row["concern_id"]),
            seed,
        )
        annotation_row = {
            "item_id": item_id,
            "audit_type": CONCERN_FIDELITY,
            "annotator_id": "",
            "source": row["source"],
            "benchmark_split": row["benchmark_split"],
            "article_title": row["article_title"],
            "source_review_text": row["source_review_text"],
            "extracted_concern_text": row["extracted_concern_text"],
            "human_fidelity": "",
            "human_category": "",
            "human_severity": "",
            "human_notes": "",
        }
        annotation_rows.append(annotation_row)
        key_rows.append(
            {
                "item_id": item_id,
                "audit_type": CONCERN_FIDELITY,
                "article_id": row["article_id"],
                "concern_id": row["concern_id"],
                "source": row["source"],
                "benchmark_split": row["benchmark_split"],
                "extracted_category": row["extracted_category"],
                "extracted_severity": row["extracted_severity"],
                "extracted_author_stance": row["extracted_author_stance"],
                "stimulus_sha256": _stimulus_sha256(
                    annotation_row,
                    audit_type=CONCERN_FIDELITY,
                ),
            }
        )
    return annotation_rows, key_rows


def _make_omission_rows(
    sampled: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotation_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for row in sampled:
        item_id = _deterministic_item_id(
            OMISSION_AUDIT,
            str(row["article_id"]),
            seed,
        )
        annotation_row = {
            "item_id": item_id,
            "audit_type": OMISSION_AUDIT,
            "annotator_id": "",
            "source": row["source"],
            "benchmark_split": row["benchmark_split"],
            "article_title": row["article_title"],
            "source_review_text": row["source_review_text"],
            "extracted_concerns_json": json.dumps(
                row["extracted_concerns"], ensure_ascii=False
            ),
            "human_omission_status": "",
            "human_omitted_concerns_json": "",
            "human_notes": "",
        }
        annotation_rows.append(annotation_row)
        key_rows.append(
            {
                "item_id": item_id,
                "audit_type": OMISSION_AUDIT,
                "article_id": row["article_id"],
                "source": row["source"],
                "benchmark_split": row["benchmark_split"],
                "dominant_category": row["dominant_category"],
                "dominant_severity": row["dominant_severity"],
                "n_extracted_concerns": len(row["extracted_concerns"]),
                "stimulus_sha256": _stimulus_sha256(
                    annotation_row,
                    audit_type=OMISSION_AUDIT,
                ),
            }
        )
    return annotation_rows, key_rows


def _agreement_estimate(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    *,
    cluster_ids: Sequence[str],
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> AgreementEstimate:
    if (
        len(labels_a) != len(labels_b)
        or len(labels_a) != len(cluster_ids)
        or not labels_a
    ):
        raise ValidationPackError(
            "Agreement requires non-empty, equally sized paired labels and "
            "article-cluster identifiers."
        )
    n = len(labels_a)
    n_agree = sum(a == b for a, b in zip(labels_a, labels_b, strict=True))
    kappa, note = cohen_kappa(labels_a, labels_b)
    observed_interval, kappa_interval = _cluster_bootstrap_agreement_intervals(
        labels_a,
        labels_b,
        cluster_ids=cluster_ids,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    if bootstrap_samples == 0:
        interval_note = "Article-cluster bootstrap intervals were not requested."
    elif len(set(cluster_ids)) < 2:
        interval_note = (
            "Article-cluster bootstrap intervals require at least two articles."
        )
    else:
        interval_note = (
            "Percentile interval from an article-cluster bootstrap of the "
            "realized sample; descriptive stability interval only, not a "
            "population confidence interval."
        )
    return AgreementEstimate(
        n=n,
        n_articles=len(set(cluster_ids)),
        observed_agreement=n_agree / n,
        observed_agreement_cluster_bootstrap_interval=observed_interval,
        cohen_kappa=kappa,
        cohen_kappa_cluster_bootstrap_interval=kappa_interval,
        kappa_note=note,
        interval_note=interval_note,
    )


def _cluster_bootstrap_agreement_intervals(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    *,
    cluster_ids: Sequence[str],
    confidence: float,
    samples: int,
    seed: int,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Bootstrap whole articles, preserving within-article concern dependence."""
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(cluster_ids):
        if not cluster_id:
            raise ValidationPackError("Agreement cluster identifiers must be non-empty.")
        clusters[cluster_id].append(index)
    cluster_keys = sorted(clusters)
    if samples == 0 or len(cluster_keys) < 2:
        return None, None

    rng = random.Random(seed)
    observed_estimates: list[float] = []
    kappa_estimates: list[float] = []
    for _ in range(samples):
        sampled_clusters = [
            cluster_keys[rng.randrange(len(cluster_keys))]
            for _ in range(len(cluster_keys))
        ]
        indices = [
            index
            for cluster_id in sampled_clusters
            for index in clusters[cluster_id]
        ]
        observed_estimates.append(
            sum(labels_a[index] == labels_b[index] for index in indices)
            / len(indices)
        )
        estimate, _note = cohen_kappa(
            [labels_a[index] for index in indices],
            [labels_b[index] for index in indices],
        )
        if estimate is not None and math.isfinite(estimate):
            kappa_estimates.append(estimate)

    observed_estimates.sort()
    alpha = (1 - confidence) / 2
    observed_interval = (
        _percentile(observed_estimates, alpha),
        _percentile(observed_estimates, 1 - alpha),
    )
    if len(kappa_estimates) < 2:
        return observed_interval, None
    kappa_estimates.sort()
    return observed_interval, (
        _percentile(kappa_estimates, alpha),
        _percentile(kappa_estimates, 1 - alpha),
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    )


def _agreement_as_dict(estimate: AgreementEstimate) -> dict[str, Any]:
    return asdict(estimate)


def _align_rows(
    rows_a: Sequence[dict[str, str]],
    rows_b: Sequence[dict[str, str]],
    *,
    audit_type: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    indexed_a = {row["item_id"]: row for row in rows_a}
    indexed_b = {row["item_id"]: row for row in rows_b}
    if set(indexed_a) != set(indexed_b):
        only_a = sorted(set(indexed_a) - set(indexed_b))
        only_b = sorted(set(indexed_b) - set(indexed_a))
        raise ValidationPackError(
            f"Rater item sets differ for {audit_type}; "
            f"only rater 1={only_a[:3]}, only rater 2={only_b[:3]}."
        )
    item_ids = sorted(indexed_a)
    return (
        [indexed_a[item_id] for item_id in item_ids],
        [indexed_b[item_id] for item_id in item_ids],
    )


def _validate_key_coverage(
    answer_key: Mapping[str, Mapping[str, Any]],
    annotation_rows: Sequence[Mapping[str, Any]],
) -> None:
    annotation_ids = {str(row["item_id"]) for row in annotation_rows}
    missing = sorted(annotation_ids - set(answer_key))
    if missing:
        raise ValidationPackError(
            f"Coordinator answer key is missing item(s): {missing[:3]}"
        )
    unused = sorted(set(answer_key) - annotation_ids)
    if unused:
        raise ValidationPackError(
            f"Coordinator answer key contains unassigned item(s): {unused[:3]}"
        )


def _validate_stimulus_integrity(
    answer_key: Mapping[str, Mapping[str, Any]],
    annotation_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject altered rater stimuli before using any completed labels."""
    for row in annotation_rows:
        item_id = str(row["item_id"])
        key_row = answer_key[item_id]
        audit_type = _clean(row.get("audit_type"))
        key_audit_type = _clean(key_row.get("audit_type"))
        if key_audit_type != audit_type:
            raise ValidationPackError(
                f"Audit type does not match coordinator key for item {item_id}."
            )
        expected = _clean(key_row.get("stimulus_sha256"))
        if not expected:
            raise ValidationPackError(
                f"Coordinator answer key lacks stimulus_sha256 for item {item_id}."
            )
        observed = _stimulus_sha256(row, audit_type=audit_type)
        if observed != expected:
            raise ValidationPackError(
                "Annotation stimulus failed integrity check for item "
                f"{item_id}; source/context text may have been altered."
            )


def _validate_choice(
    row: Mapping[str, Any],
    row_number: int,
    field: str,
    choices: Sequence[str],
) -> str:
    value = _clean(row.get(field))
    if value not in choices:
        raise ValidationPackError(
            f"Row {row_number}: {field} must be one of "
            f"{', '.join(choices)}; found {value!r}."
        )
    return value


def _validate_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValidationPackError("confidence must be strictly between 0 and 1.")


def _stimulus_sha256(
    row: Mapping[str, Any],
    *,
    audit_type: str,
) -> str:
    fields = _STIMULUS_FIELDS.get(audit_type)
    if fields is None:
        raise ValidationPackError(f"Unknown audit type: {audit_type}")
    payload = {field: _exact_text(row.get(field)) for field in fields}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _deterministic_item_id(audit_type: str, source_id: str, seed: int) -> str:
    prefix = "cf" if audit_type == CONCERN_FIDELITY else "om"
    digest = hashlib.sha256(
        f"bioreview-validation|{seed}|{audit_type}|{source_id}".encode()
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _input_fingerprint(entries: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint identifiers, labels, and every text field shown to raters."""
    payload = []
    for entry in entries:
        concerns = sorted(
            (
                {
                    "concern_id": str(concern["concern_id"]),
                    "concern_text": str(concern["concern_text"]),
                    "category": str(concern["category"]),
                    "severity": str(concern["severity"]),
                    "author_stance": str(concern.get("author_stance", "")),
                }
                for concern in entry["concerns"]
            ),
            key=lambda concern: concern["concern_id"],
        )
        payload.append(
            {
                "id": str(entry["id"]),
                "source": str(entry["source"]),
                "benchmark_split": str(entry["benchmark_split"]),
                "title": str(entry["title"]),
                "decision_letter_raw": str(entry["decision_letter_raw"]),
                "concerns": concerns,
            }
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _count_strata(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    counts = Counter(tuple(str(row[field]) for field in fields) for row in rows)
    return [
        {
            **{field: value for field, value in zip(fields, key, strict=True)},
            "n": count,
        }
        for key, count in sorted(counts.items())
    ]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _exact_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValidationPackError(f"Annotation file does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationPackError(
                    f"{path}:{line_number}: invalid JSON ({exc.msg})."
                ) from exc
            if not isinstance(row, dict):
                raise ValidationPackError(
                    f"{path}:{line_number}: expected a JSON object."
                )
            rows.append(row)
    return rows

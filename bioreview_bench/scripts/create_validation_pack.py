"""Create and summarize system-label-blinded two-annotator validation packs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import click

from bioreview_bench.project_defaults import DEFAULT_BENCHMARK_SPLITS_DIR
from bioreview_bench.validate.validation_pack import (
    ValidationPackError,
    build_validation_pack,
    detect_suspicious_exact_identity,
    load_jsonl_entries,
    summarize_validation_pack,
    write_validation_pack,
)


@click.group()
def main() -> None:
    """Run independent validation of extraction fidelity and omissions."""


@main.command("create")
@click.option(
    "--split",
    "splits",
    type=click.Choice(["val", "test"]),
    multiple=True,
    default=("val", "test"),
    show_default=True,
    help="Benchmark split(s) to sample.",
)
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_BENCHMARK_SPLITS_DIR,
    show_default=True,
    help="Directory containing split JSONL files.",
)
@click.option(
    "--concern-sample-size",
    type=click.IntRange(min=1),
    default=300,
    show_default=True,
    help="Number of extracted concerns for fidelity/category/severity review.",
)
@click.option(
    "--omission-sample-size",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Number of articles for the separate omission audit.",
)
@click.option(
    "--annotator",
    "annotators",
    multiple=True,
    default=("annotator_1", "annotator_2"),
    show_default=True,
    help="Exactly two distinct annotator identifiers.",
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="New or empty output directory for the pack.",
)
def create_command(
    splits: tuple[str, ...],
    splits_dir: Path,
    concern_sample_size: int,
    omission_sample_size: int,
    annotators: tuple[str, ...],
    seed: int,
    output_dir: Path,
) -> None:
    """Create label-blinded rater files and a coordinator-only answer key."""
    try:
        entries = load_jsonl_entries(
            [splits_dir / f"{split}.jsonl" for split in splits]
        )
        pack = build_validation_pack(
            entries,
            concern_sample_size=concern_sample_size,
            omission_sample_size=omission_sample_size,
            annotators=annotators,
            seed=seed,
        )
        write_validation_pack(pack, output_dir)
    except ValidationPackError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Wrote validation pack: {output_dir}")
    click.echo(
        "Keep coordinator/answer_key.jsonl hidden until both annotator files "
        "are complete and locked."
    )
    click.echo(
        "Only system labels are hidden; source, title, and review context remain "
        "visible to raters."
    )


@main.command("summarize")
@click.option(
    "--pack-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    required=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional JSON output path; otherwise print to stdout.",
)
@click.option(
    "--confidence",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True),
    default=0.95,
    show_default=True,
)
@click.option(
    "--bootstrap-samples",
    type=click.IntRange(min=0),
    default=2_000,
    show_default=True,
    help=(
        "Article-cluster bootstrap replicates for descriptive agreement "
        "stability intervals (not population confidence intervals)."
    ),
)
@click.option("--seed", type=int, default=42, show_default=True)
def summarize_command(
    pack_dir: Path,
    output: Path | None,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> None:
    """Validate annotations and report descriptive agreement/stability."""
    try:
        summary = summarize_validation_pack(
            pack_dir,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    except ValidationPackError as exc:
        raise click.ClickException(str(exc)) from exc
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        click.echo(rendered, nl=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        click.echo(f"Wrote validation summary: {output}")


@main.command("audit-identity")
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    required=True,
    help="CSV or JSONL containing system and human label columns.",
)
@click.option("--min-rows", type=click.IntRange(min=1), default=20, show_default=True)
def audit_identity_command(input_path: Path, min_rows: int) -> None:
    """Warn when supposedly independent labels perfectly mirror system fields."""
    rows = _load_tabular_rows(input_path)
    try:
        check = detect_suspicious_exact_identity(rows, min_rows=min_rows)
    except ValidationPackError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(_dataclass_to_json(check), ensure_ascii=False, indent=2))
    if check.suspicious:
        raise click.exceptions.Exit(1)


def _load_tabular_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise click.ClickException(
                        f"{path}:{line_number}: invalid JSON ({exc.msg})."
                    ) from exc
                if not isinstance(row, dict):
                    raise click.ClickException(
                        f"{path}:{line_number}: expected a JSON object."
                    )
                rows.append(row)
        return rows
    raise click.ClickException("Input must use .csv, .jsonl, or .ndjson.")


def _dataclass_to_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            field: _dataclass_to_json(getattr(value, field))
            for field in value.__dataclass_fields__
        }
    if isinstance(value, tuple):
        return [_dataclass_to_json(item) for item in value]
    return value


if __name__ == "__main__":
    main()

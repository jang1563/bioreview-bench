"""Create a frozen stratified human-review subset from benchmark splits."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from bioreview_bench.validate.human_subset import (
    build_subset_manifest,
    load_entries_for_subset,
    sample_human_subset,
)

console = Console()


@click.command()
@click.option(
    "--split",
    "splits",
    type=click.Choice(["val", "test"]),
    multiple=True,
    default=("val", "test"),
    show_default=True,
    help="Benchmark split(s) to sample from.",
)
@click.option(
    "--n",
    type=int,
    default=100,
    show_default=True,
    help="Number of articles to include in the frozen subset.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducible sampling.",
)
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=Path("data/splits/v3"),
    show_default=True,
    help="Directory containing val.jsonl and test.jsonl.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/validation/human_subset_v1.jsonl"),
    show_default=True,
    help="Output JSONL path for the sampled subset.",
)
@click.option(
    "--manifest-output",
    type=click.Path(path_type=Path),
    default=Path("data/validation/human_subset_v1_manifest.json"),
    show_default=True,
    help="Output path for the subset manifest.",
)
def main(
    splits: tuple[str, ...],
    n: int,
    seed: int,
    splits_dir: Path,
    output: Path,
    manifest_output: Path,
) -> None:
    """Create a stratified subset for human agreement and upper-bound studies."""
    entries = load_entries_for_subset(splits_dir, splits)
    sampled = sample_human_subset(entries, n=n, seed=seed)
    manifest = build_subset_manifest(sampled)
    manifest["seed"] = seed

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in sampled:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    console.print(f"Wrote subset: {output}")
    console.print(f"Wrote manifest: {manifest_output}")
    console.print(f"Subset size: {manifest['n_articles']}")


if __name__ == "__main__":
    main()

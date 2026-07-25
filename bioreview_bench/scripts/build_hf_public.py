"""Build the sanitized public Hugging Face release folder."""

from __future__ import annotations

import json
from pathlib import Path

import click

from bioreview_bench.collect.hf_public import build_public_package
from bioreview_bench.project_defaults import DEFAULT_BENCHMARK_SPLITS_DIR


@click.command()
@click.option(
    "--repo-root",
    type=click.Path(path_type=Path),
    default=Path("."),
    show_default=True,
    help="Repository root containing public release artifacts.",
)
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_BENCHMARK_SPLITS_DIR,
    show_default=True,
    help="Private canonical split directory used as the sanitization source.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Empty local directory to populate; no remote operation is performed.",
)
def main(
    repo_root: Path,
    splits_dir: Path,
    output_dir: Path,
) -> None:
    """Create and validate a clean-history public HF upload package."""
    stats = build_public_package(
        repo_root=repo_root,
        splits_dir=splits_dir,
        output_dir=output_dir,
    )
    click.echo(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

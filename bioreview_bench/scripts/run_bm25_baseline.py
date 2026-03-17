"""Run the zero-cost BM25 baseline and emit benchmark-compatible JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from bioreview_bench.baseline.lexical import BM25ConcernRetriever

console = Console()

_DEFAULT_SPLITS_DIR = Path("data/splits/v3")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@click.command()
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="val",
    show_default=True,
    help="Target split to score.",
)
@click.option(
    "--corpus-split",
    "corpus_splits",
    type=click.Choice(["train", "val", "test"]),
    multiple=True,
    default=("train",),
    show_default=True,
    help="Split(s) used as the retrieval corpus.",
)
@click.option(
    "--splits-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_SPLITS_DIR,
    show_default=True,
    help="Directory containing canonical split JSONL files.",
)
@click.option(
    "--top-k-docs",
    type=int,
    default=8,
    show_default=True,
    help="Number of retrieved corpus articles to aggregate concerns from.",
)
@click.option(
    "--max-concerns",
    type=int,
    default=12,
    show_default=True,
    help="Maximum number of predicted concerns per article.",
)
@click.option(
    "--max-input-chars",
    type=int,
    default=40_000,
    show_default=True,
    help="Maximum article text length used for indexing and retrieval.",
)
@click.option(
    "--max-articles",
    type=int,
    default=None,
    help="Limit target articles for smoke tests.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSONL path (default: tool_outputs/bm25_{split}.jsonl).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print configuration and exit without writing output.",
)
def main(
    split: str,
    corpus_splits: tuple[str, ...],
    splits_dir: Path,
    top_k_docs: int,
    max_concerns: int,
    max_input_chars: int,
    max_articles: int | None,
    output: Path | None,
    dry_run: bool,
) -> None:
    """Run the lexical BM25 baseline."""
    target_path = splits_dir / f"{split}.jsonl"
    if not target_path.exists():
        console.print(f"[red]Error:[/red] Split file not found: {target_path}")
        sys.exit(1)

    corpus_entries: list[dict] = []
    for corpus_split in corpus_splits:
        corpus_path = splits_dir / f"{corpus_split}.jsonl"
        if not corpus_path.exists():
            console.print(f"[red]Error:[/red] Corpus split not found: {corpus_path}")
            sys.exit(1)
        corpus_entries.extend(_load_jsonl(corpus_path))

    target_entries = _load_jsonl(target_path)
    usable = [entry for entry in target_entries if entry.get("concerns")]
    if max_articles is not None:
        usable = usable[:max_articles]

    output = output or Path("tool_outputs") / f"bm25_{split}.jsonl"
    console.print(f"Corpus entries: {len(corpus_entries)} from {list(corpus_splits)}")
    console.print(f"Target entries: {len(usable)}")
    console.print(f"Output: {output}")

    if dry_run:
        console.print("[yellow]Dry run[/yellow] — no files written.")
        return

    retriever = BM25ConcernRetriever(
        corpus_entries,
        top_k_docs=top_k_docs,
        max_concerns=max_concerns,
        max_input_chars=max_input_chars,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh, Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Retrieving concerns", total=len(usable))
        total_concerns = 0

        for entry in usable:
            concerns = retriever.review_article(entry)
            fh.write(
                json.dumps(
                    {"article_id": entry.get("id", ""), "concerns": concerns},
                    ensure_ascii=False,
                )
                + "\n"
            )
            total_concerns += len(concerns)
            progress.advance(task)

    console.print(f"[green]Done.[/green] Wrote {len(usable)} rows and {total_concerns} concerns.")


if __name__ == "__main__":
    main()

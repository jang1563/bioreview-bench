"""Incremental dataset update pipeline CLI.

Orchestrates periodic collection across multiple sources with state tracking,
deduplication, and lockfile-based concurrency control.

Usage:
    # Dry-run: collect + parse only, no LLM, $0 cost
    uv run bioreview-update --source elife --dry-run -n 5

    # Full incremental update (append new articles)
    uv run bioreview-update --source elife -n 200

    # Update all sources
    uv run bioreview-update --source all -n 200

    # Update + re-split + push to HuggingFace Hub
    uv run bioreview-update --source elife -n 200 --update-splits --push-hf
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import fcntl
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console

from ..collect.registry import SOURCE_REGISTRY, get_source_config
from ..collect.state import (
    RunRecord,
    StateManager,
    _detect_trigger,
    make_run_id,
)

console = Console()
ROOT = Path(__file__).resolve().parents[2]  # peer-review-benchmark/

# Date buffer: when resuming incremental collection, look back this many days
# from last_article_date to catch articles with delayed indexing.
_DATE_BUFFER_DAYS = 3


def _acquire_lock(lock_path: Path) -> int | None:
    """Acquire an exclusive lockfile. Returns fd on success, None if locked."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError:
        os.close(fd)
        return None


def _release_lock(fd: int) -> None:
    """Release and close a lockfile."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


async def _run_source_update(
    source_name: str,
    max_new_articles: int,
    state_mgr: StateManager,
    model: str,
    dry_run: bool,
    data_dir: Path,
    manifest_dir: Path,
) -> dict:
    """Run incremental collection for a single source. Returns stats dict."""
    config = get_source_config(source_name)
    state = state_mgr.load()
    source_state = state.get_source(source_name)

    # Determine output path
    output_path = data_dir / "processed" / config.output_filename

    # Startup sync: add any IDs from JSONL that are missing from state
    # (recovery from interrupted runs). We only ADD, never REMOVE, because
    # state may track IDs from multiple files (e.g. elife_legacy_v1.jsonl).
    # Also sync last_article_date to prevent stale start_date after crash.
    if output_path.exists():
        jsonl_ids: set[str] = set()
        max_date_sync: str | None = None
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    aid = entry.get("id", "")
                    if aid:
                        jsonl_ids.add(aid)
                    pub = str(entry.get("published_date") or "")
                    if pub and (max_date_sync is None or pub > max_date_sync):
                        max_date_sync = pub

        missing_from_state = jsonl_ids - source_state.id_set
        date_stale = (
            max_date_sync
            and max_date_sync != source_state.last_article_date
        )
        if missing_from_state or date_stale:
            if missing_from_state:
                console.print(
                    f"  [yellow]State sync: +{len(missing_from_state)} IDs added from JSONL"
                )
                source_state.collected_ids = sorted(
                    set(source_state.collected_ids) | missing_from_state
                )
            if date_stale:
                console.print(
                    f"  [yellow]State sync: last_article_date "
                    f"{source_state.last_article_date} → {max_date_sync}"
                )
                source_state.last_article_date = max_date_sync
            state_mgr.save(state)
            state = state_mgr.load()
            source_state = state.get_source(source_name)

    # Calculate start_date with buffer
    if source_state.last_article_date:
        last_dt = datetime.strptime(source_state.last_article_date, "%Y-%m-%d")
        buffered = last_dt - timedelta(days=_DATE_BUFFER_DAYS)
        start_date = buffered.strftime("%Y-%m-%d")
    else:
        start_date = config.default_start_date

    known_ids = source_state.id_set
    run_started_at = datetime.now(_dt.UTC).isoformat()

    console.print(f"  [dim]start_date={start_date}, known_ids={len(known_ids)}[/dim]")

    # Currently only eLife uses _run() from collect_elife.py.
    # Other sources will be added as their CLIs are implemented.
    if source_name == "elife":
        from .collect_elife import _run

        manifest_path = manifest_dir / "em-v1.0.json"
        stats = await _run(
            subjects=config.default_subjects,
            start_date=start_date,
            end_date=None,
            order="desc",
            max_articles=max_new_articles,
            output=output_path,
            manifest_path=manifest_path,
            model=model,
            dry_run=dry_run,
            append=True,
            known_ids=known_ids,
        )
    else:
        console.print(f"  [yellow]Source '{source_name}' collection not yet implemented")
        stats = {"total_fetched": 0, "skipped": 0, "xml_ok": 0, "xml_fail": 0}

    # Update state with newly collected articles (add-only, same as startup sync)
    if stats["xml_ok"] > 0 and output_path.exists():
        state = state_mgr.load()
        source_state = state.get_source(source_name)

        jsonl_ids: set[str] = set()
        max_date = source_state.last_article_date
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                aid = entry.get("id", "")
                if aid:
                    jsonl_ids.add(aid)
                pub = str(entry.get("published_date") or "")
                if pub and (max_date is None or pub > max_date):
                    max_date = pub

        missing = jsonl_ids - source_state.id_set
        if missing:
            source_state.collected_ids = sorted(
                set(source_state.collected_ids) | missing
            )
            source_state.last_article_date = max_date

    # Record run
    run = RunRecord(
        run_id=make_run_id(source_name),
        source=source_name,
        started_at=run_started_at,
        completed_at=datetime.now(_dt.UTC).isoformat(),
        trigger=_detect_trigger(),
        new_articles=stats["xml_ok"],
        skipped_duplicates=stats.get("skipped", 0),
        cost_usd_est=stats["xml_ok"] * 0.002 if not dry_run else 0.0,
        dry_run=dry_run,
    )
    state.add_run(run)
    state_mgr.save(state)

    return stats


def _run_update_splits(data_dir: Path) -> None:
    """Re-run train/val/test split, preserving frozen test set if it exists."""
    import subprocess

    splits_dir = data_dir / "splits"
    frozen_path = splits_dir / "test_ids_frozen.json"
    script = ROOT / "scripts" / "create_splits.py"

    # Collect all processed JSONL files as inputs
    processed_dir = data_dir / "processed"
    input_files = sorted(processed_dir.glob("*.jsonl")) if processed_dir.exists() else []

    if not input_files:
        console.print("  [yellow]No JSONL files found in processed/")
        return

    cmd = [sys.executable, str(script)]
    for f in input_files:
        cmd.extend(["-i", str(f)])
    cmd.extend(["--output-dir", str(splits_dir)])

    if frozen_path.exists():
        cmd.extend(["--frozen-test", str(frozen_path)])
        console.print(f"  [dim]Using frozen test: {frozen_path.name}[/dim]")

    console.print(f"  [dim]Input files: {[f.name for f in input_files]}[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        console.print("  [green]Splits updated successfully")
        for line in result.stdout.strip().split("\n")[-4:]:
            console.print(f"  {line}")
    else:
        console.print(f"  [red]Split failed: {result.stderr}")


def _run_push_hf(data_dir: Path) -> None:
    """Push data to HuggingFace Hub."""
    from ..collect.hf_push import push_to_hub

    try:
        result = push_to_hub(data_dir=data_dir)
    except ImportError:
        console.print("  [red]huggingface_hub not installed. Run: uv sync --extra hub")
        return

    uploaded = result.get("uploaded", [])
    console.print(f"  [green]Uploaded {len(uploaded)} files to HuggingFace Hub")
    for path in uploaded:
        console.print(f"    {path}")


@click.command()
@click.option(
    "--source", "-s",
    type=click.Choice(["elife", "plos", "f1000", "all"]),
    default="elife",
    show_default=True,
    help="Source to collect from (or 'all' for all sources)",
)
@click.option(
    "--max-new-articles", "-n",
    default=200,
    show_default=True,
    help="Maximum new articles to collect per source",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Collection + parsing only, no LLM ($0 cost)",
)
@click.option(
    "--model",
    default=lambda: os.getenv("BIOREVIEW_MODEL_ID", "claude-haiku-4-5-20251001"),
    help="Anthropic model ID",
)
@click.option(
    "--data-dir",
    default=None,
    help="Data directory (default: <project>/data)",
)
@click.option(
    "--update-splits",
    is_flag=True,
    default=False,
    help="Re-run train/val/test split after collection (preserves frozen test set)",
)
@click.option(
    "--push-hf",
    is_flag=True,
    default=False,
    help="Push data to HuggingFace Hub after collection",
)
def main(
    source: str,
    max_new_articles: int,
    dry_run: bool,
    model: str,
    data_dir: str | None,
    update_splits: bool,
    push_hf: bool,
) -> None:
    """Incremental dataset update pipeline."""
    data_path = Path(data_dir) if data_dir else ROOT / "data"
    state_path = data_path / "update_state.json"
    manifest_dir = data_path / "manifests"
    lock_path = data_path / ".update.lock"

    # Determine sources to process
    if source == "all":
        sources = sorted(SOURCE_REGISTRY)
    else:
        sources = [source]

    console.print("[bold cyan]bioreview-bench Update Pipeline[/bold cyan]")
    console.print(f"  sources  : {sources}")
    console.print(f"  max/src  : {max_new_articles}")
    console.print(f"  dry-run  : {dry_run}")
    console.print(f"  state    : {state_path}")
    console.print()

    # Acquire lockfile
    lock_fd = _acquire_lock(lock_path)
    if lock_fd is None:
        console.print("[yellow]Another update is running. Exiting.")
        sys.exit(0)

    try:
        total_new = 0
        total_skipped = 0

        for src_name in sources:
            console.print(f"[bold]--- {src_name} ---[/bold]")
            state_mgr = StateManager(state_path)

            try:
                stats = asyncio.run(
                    _run_source_update(
                        source_name=src_name,
                        max_new_articles=max_new_articles,
                        state_mgr=state_mgr,
                        model=model,
                        dry_run=dry_run,
                        data_dir=data_path,
                        manifest_dir=manifest_dir,
                    )
                )
                total_new += stats.get("xml_ok", 0)
                total_skipped += stats.get("skipped", 0)
            except Exception as e:
                console.print(f"[red]Error collecting {src_name}: {e}")

        console.print()
        console.print("[bold]Collection complete[/bold]")
        console.print(f"  New articles: {total_new}")
        console.print(f"  Skipped:      {total_skipped}")

        # --- Post-collection: re-split ---
        if update_splits and total_new > 0 and not dry_run:
            console.print()
            console.print("[bold]Updating splits...[/bold]")
            _run_update_splits(data_path)

        # --- Post-collection: push to HuggingFace Hub ---
        if push_hf and not dry_run:
            console.print()
            console.print("[bold]Pushing to HuggingFace Hub...[/bold]")
            _run_push_hf(data_path)

        console.print()
        console.print("[bold green]All done.[/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.")
        sys.exit(0)
    finally:
        _release_lock(lock_fd)


if __name__ == "__main__":
    main()

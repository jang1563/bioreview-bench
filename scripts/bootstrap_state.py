"""One-time migration: create update_state.json from existing JSONL files.

Reads all existing collected JSONL files, extracts article IDs and
max published_date, and creates the initial state file.

Run once after implementing the update pipeline:
    python scripts/bootstrap_state.py

This is idempotent — running it again will re-create the state file
from the current JSONL contents.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreview_bench.collect.state import StateManager, UpdateState


def main() -> None:
    state_path = ROOT / "data" / "update_state.json"
    mgr = StateManager(state_path)

    # Define source → JSONL file mappings
    source_files: dict[str, list[Path]] = {
        "elife": [
            ROOT / "data" / "processed" / "elife_v1.1.jsonl",
            ROOT / "data" / "processed" / "elife_legacy_v1.jsonl",
        ],
        # Add other sources here as they become available:
        # "plos": [ROOT / "data" / "processed" / "plos_v1.jsonl"],
        # "f1000": [ROOT / "data" / "processed" / "f1000_v1.jsonl"],
    }

    state = UpdateState()

    print("Bootstrapping update state from existing JSONL files...\n")

    total_ids = 0
    for source, paths in source_files.items():
        existing_paths = [p for p in paths if p.exists()]
        if not existing_paths:
            print(f"  [{source}] No files found — skipping")
            continue

        source_state = mgr.initialize_from_jsonl(source, *existing_paths)
        state.sources[source] = source_state
        n_ids = len(source_state.collected_ids)
        total_ids += n_ids

        print(f"  [{source}]")
        for p in existing_paths:
            if p.exists():
                n_lines = sum(1 for line in p.open() if line.strip())
                print(f"    {p.name}: {n_lines} entries")
            else:
                print(f"    {p.name}: not found")
        print(f"    collected_ids: {n_ids}")
        print(f"    last_article_date: {source_state.last_article_date}")

    if total_ids == 0:
        print("\nERROR: No articles found in any JSONL file.")
        sys.exit(1)

    # Save state
    mgr.save(state)

    print(f"\nState file created: {state_path}")
    print(f"  Total sources: {len(state.sources)}")
    print(f"  Total article IDs: {total_ids}")

    # Verify by re-loading
    loaded = mgr.load()
    loaded_total = sum(len(s.collected_ids) for s in loaded.sources.values())
    assert loaded_total == total_ids, f"Verification failed: wrote {total_ids}, read back {loaded_total}"
    print(f"  Verification: OK (loaded back {loaded_total} IDs)")

    # Also create test_ids_frozen.json from current test.jsonl if it exists
    test_jsonl = ROOT / "data" / "splits" / "test.jsonl"
    freeze_path = ROOT / "data" / "splits" / "test_ids_frozen.json"
    if test_jsonl.exists() and not freeze_path.exists():
        test_ids = []
        with test_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    test_ids.append(entry["id"])

        freeze_data = {
            "frozen_at": "2026-02-28",
            "revision": "v1",
            "seed": 42,
            "n_test": len(test_ids),
            "test_ids": sorted(test_ids),
        }
        freeze_path.write_text(json.dumps(freeze_data, indent=2) + "\n")
        print(f"\nTest freeze file created: {freeze_path}")
        print(f"  Frozen test IDs: {len(test_ids)}")
    elif freeze_path.exists():
        print(f"\nTest freeze file already exists: {freeze_path} — skipping")
    else:
        print(f"\nNo test.jsonl found at {test_jsonl} — skipping freeze")


if __name__ == "__main__":
    main()

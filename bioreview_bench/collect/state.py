"""Update state management for incremental dataset collection.

Tracks per-source collection state (collected article IDs, last collection date)
and run history for the periodic update pipeline.

State file: data/update_state.json (git-tracked, deterministic output).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

_MAX_RUNS = 50  # Cap run history to prevent unbounded growth


@dataclass
class SourceState:
    """Per-source collection state."""

    last_article_date: str | None = None  # max(published_date) of collected articles
    collected_ids: list[str] = field(default_factory=list)  # sorted for deterministic diffs

    @property
    def id_set(self) -> set[str]:
        """Return collected_ids as a set for O(1) lookup."""
        return set(self.collected_ids)


@dataclass
class RunRecord:
    """Record of a single collection run."""

    run_id: str = ""
    source: str = ""
    started_at: str = ""
    completed_at: str | None = None
    trigger: str = "local"  # "local" | "github_actions_schedule" | "github_actions_manual"
    new_articles: int = 0
    skipped_duplicates: int = 0
    cost_usd_est: float = 0.0
    dry_run: bool = False


@dataclass
class UpdateState:
    """Top-level state containing per-source states and run history."""

    sources: dict[str, SourceState] = field(default_factory=dict)
    runs: list[RunRecord] = field(default_factory=list)
    dataset_version: str = "1.0"

    def get_source(self, source: str) -> SourceState:
        """Get or create a SourceState for the given source name."""
        if source not in self.sources:
            self.sources[source] = SourceState()
        return self.sources[source]

    def add_run(self, run: RunRecord) -> None:
        """Append a run record, capping history at _MAX_RUNS."""
        self.runs.append(run)
        if len(self.runs) > _MAX_RUNS:
            self.runs = self.runs[-_MAX_RUNS:]

    def bump_minor(self) -> str:
        """Bump minor version: 1.0 → 1.1, 1.1 → 1.2, etc."""
        major, minor = self.dataset_version.split(".")
        self.dataset_version = f"{major}.{int(minor) + 1}"
        return self.dataset_version

    def bump_major(self) -> str:
        """Bump major version: 1.x → 2.0, 2.x → 3.0, etc."""
        major, _minor = self.dataset_version.split(".")
        self.dataset_version = f"{int(major) + 1}.0"
        return self.dataset_version


def _detect_trigger() -> str:
    """Detect execution context from environment variables."""
    if os.getenv("GITHUB_ACTIONS"):
        event = os.getenv("GITHUB_EVENT_NAME", "")
        if event == "schedule":
            return "github_actions_schedule"
        return "github_actions_manual"
    return "local"


def make_run_id(source: str) -> str:
    """Generate a unique run ID like 'run-elife-2026-03-07-a1b2c3'."""
    ts = datetime.now(_dt.UTC).strftime("%Y-%m-%d")
    short_uuid = uuid.uuid4().hex[:6]
    return f"run-{source}-{ts}-{short_uuid}"


class StateManager:
    """Manages the update state file (data/update_state.json).

    Usage:
        mgr = StateManager(Path("data/update_state.json"))
        state = mgr.load()
        elife = state.get_source("elife")
        # ... update elife.collected_ids, elife.last_article_date ...
        mgr.save(state)
    """

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def load(self) -> UpdateState:
        """Load state from disk. Returns empty state if file doesn't exist."""
        if not self.state_path.exists():
            return UpdateState()

        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = UpdateState()
        state.dataset_version = data.get("dataset_version", "1.0")

        for source_name, source_data in data.get("sources", {}).items():
            state.sources[source_name] = SourceState(
                last_article_date=source_data.get("last_article_date"),
                collected_ids=source_data.get("collected_ids", []),
            )

        for run_data in data.get("runs", []):
            state.runs.append(RunRecord(
                run_id=run_data.get("run_id", ""),
                source=run_data.get("source", ""),
                started_at=run_data.get("started_at", ""),
                completed_at=run_data.get("completed_at"),
                trigger=run_data.get("trigger", "local"),
                new_articles=run_data.get("new_articles", 0),
                skipped_duplicates=run_data.get("skipped_duplicates", 0),
                cost_usd_est=run_data.get("cost_usd_est", 0.0),
                dry_run=run_data.get("dry_run", False),
            ))

        return state

    def save(self, state: UpdateState) -> None:
        """Save state to disk with deterministic JSON output (sorted keys/IDs)."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Build serializable dict with sorted collected_ids for deterministic diffs
        sources_dict = {}
        for name in sorted(state.sources):
            src = state.sources[name]
            sources_dict[name] = {
                "last_article_date": src.last_article_date,
                "collected_ids": sorted(src.collected_ids),
            }

        data = {
            "dataset_version": state.dataset_version,
            "sources": sources_dict,
            "runs": [asdict(r) for r in state.runs],
        }

        self.state_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def sync_from_jsonl(
        self,
        state: UpdateState,
        source: str,
        jsonl_path: Path,
    ) -> int:
        """Add-only sync: add IDs from JSONL that are missing from state.

        Reads all article IDs from the JSONL file and adds any that are
        missing from the in-memory state. Never removes IDs from state,
        because state may track IDs from multiple JSONL files (e.g.
        elife_v1.1.jsonl + elife_legacy_v1.jsonl).

        Also updates last_article_date if the JSONL contains a newer date.

        Returns:
            Number of IDs added to state.
        """
        if not jsonl_path.exists():
            return 0

        jsonl_ids: set[str] = set()
        max_date: str | None = None

        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                article_id = entry.get("id", "")
                if article_id:
                    jsonl_ids.add(article_id)
                pub_date = str(entry.get("published_date") or "")
                if pub_date and (max_date is None or pub_date > max_date):
                    max_date = pub_date

        src = state.get_source(source)
        missing = jsonl_ids - set(src.collected_ids)

        if missing:
            src.collected_ids = sorted(set(src.collected_ids) | missing)
        if max_date and (src.last_article_date is None or max_date > src.last_article_date):
            src.last_article_date = max_date

        return len(missing)

    def initialize_from_jsonl(
        self,
        source: str,
        *paths: Path,
    ) -> SourceState:
        """Bootstrap a SourceState from existing JSONL files (one-time migration).

        Reads all article IDs and finds the maximum published_date across
        the provided files.

        Args:
            source: Source name (e.g., "elife").
            *paths: JSONL file paths to read from.

        Returns:
            The created SourceState (also stored in self's next save).
        """
        all_ids: set[str] = set()
        max_date: str | None = None

        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    article_id = entry.get("id", "")
                    if article_id:
                        all_ids.add(article_id)
                    pub_date = str(entry.get("published_date") or "")
                    if pub_date and (max_date is None or pub_date > max_date):
                        max_date = pub_date

        return SourceState(
            last_article_date=max_date,
            collected_ids=sorted(all_ids),
        )

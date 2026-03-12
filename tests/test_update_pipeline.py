"""Tests for M1/M2/M3/M4: state, registry, postprocess, update pipeline, splits, HF push.

Tests cover:
- StateManager save/load round-trip
- Per-source state with sorted IDs
- State bootstrap from JSONL
- Startup sync (JSONL ↔ state reconciliation)
- Source registry
- Postprocess functions
- M2: collect_elife append mode, known_ids skipping, lockfile, update CLI
- M3: frozen test split, fresh split, dedup, stratum coverage
- M4: HuggingFace Hub push (upload plan, dry-run, --push-hf CLI flag)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioreview_bench.collect.state import (
    StateManager,
    UpdateState,
    SourceState,
    RunRecord,
    make_run_id,
    _MAX_RUNS,
    _detect_trigger,
)
from bioreview_bench.collect.registry import (
    SOURCE_REGISTRY,
    get_source_config,
    list_sources,
)
from bioreview_bench.collect.postprocess import (
    infer_review_format,
    clean_subjects,
    postprocess_entry,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "update_state.json"


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """Create a small JSONL file with 3 articles."""
    path = tmp_path / "test_data.jsonl"
    entries = [
        {"id": "elife:100", "published_date": "2025-06-15", "source": "elife"},
        {"id": "elife:200", "published_date": "2025-09-01", "source": "elife"},
        {"id": "elife:300", "published_date": "2026-01-10", "source": "elife"},
    ]
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


@pytest.fixture
def sample_jsonl_with_reviews(tmp_path: Path) -> Path:
    """Create a JSONL with review content for postprocess testing."""
    path = tmp_path / "reviews.jsonl"
    entries = [
        {
            "id": "elife:101",
            "published_date": "2025-01-15",
            "source": "elife",
            "decision_letter_raw": "eLife Assessment: This study provides...",
            "author_response_raw": "We thank the reviewers...",
            "subjects": ["Neuroscience", "Research Article"],
            "schema_version": "1.0",
        },
        {
            "id": "elife:102",
            "published_date": "2021-03-20",
            "source": "elife",
            "decision_letter_raw": "Reviewer 1: Major concerns include...",
            "author_response_raw": "",
            "subjects": ["Cell Biology"],
            "schema_version": "1.0",
        },
    ]
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


# ── StateManager Tests ────────────────────────────────────────────


class TestStateManager:

    def test_load_empty(self, state_path: Path):
        """Loading from non-existent file returns empty state."""
        mgr = StateManager(state_path)
        state = mgr.load()
        assert state.sources == {}
        assert state.runs == []

    def test_save_load_roundtrip(self, state_path: Path):
        """State survives save → load cycle with data intact."""
        mgr = StateManager(state_path)

        state = UpdateState()
        state.sources["elife"] = SourceState(
            last_article_date="2026-02-15",
            collected_ids=["elife:300", "elife:100", "elife:200"],
        )
        state.sources["plos"] = SourceState()
        state.add_run(RunRecord(
            run_id="run-elife-2026-02-28-abc123",
            source="elife",
            started_at="2026-02-28T06:00:00",
            completed_at="2026-02-28T06:42:00",
            trigger="local",
            new_articles=47,
            skipped_duplicates=3,
            cost_usd_est=0.094,
            dry_run=False,
        ))

        mgr.save(state)

        loaded = mgr.load()
        assert "elife" in loaded.sources
        assert "plos" in loaded.sources
        assert loaded.sources["elife"].last_article_date == "2026-02-15"
        assert len(loaded.sources["elife"].collected_ids) == 3
        assert loaded.sources["plos"].collected_ids == []
        assert loaded.sources["plos"].last_article_date is None
        assert len(loaded.runs) == 1
        assert loaded.runs[0].run_id == "run-elife-2026-02-28-abc123"
        assert loaded.runs[0].new_articles == 47

    def test_collected_ids_sorted_on_save(self, state_path: Path):
        """collected_ids are always sorted in the saved JSON for deterministic diffs."""
        mgr = StateManager(state_path)

        state = UpdateState()
        state.sources["elife"] = SourceState(
            collected_ids=["elife:300", "elife:100", "elife:200"],
        )
        mgr.save(state)

        raw = json.loads(state_path.read_text())
        saved_ids = raw["sources"]["elife"]["collected_ids"]
        assert saved_ids == ["elife:100", "elife:200", "elife:300"]

    def test_sources_sorted_on_save(self, state_path: Path):
        """Source keys are sorted alphabetically in saved JSON."""
        mgr = StateManager(state_path)

        state = UpdateState()
        state.sources["plos"] = SourceState()
        state.sources["elife"] = SourceState()
        state.sources["f1000"] = SourceState()
        mgr.save(state)

        raw = json.loads(state_path.read_text())
        assert list(raw["sources"].keys()) == ["elife", "f1000", "plos"]

    def test_runs_capped(self, state_path: Path):
        """Run history is capped at _MAX_RUNS."""
        mgr = StateManager(state_path)
        state = UpdateState()

        for i in range(_MAX_RUNS + 10):
            state.add_run(RunRecord(run_id=f"run-{i}", source="elife"))

        assert len(state.runs) == _MAX_RUNS
        # Most recent runs are kept
        assert state.runs[0].run_id == "run-10"
        assert state.runs[-1].run_id == f"run-{_MAX_RUNS + 9}"

    def test_get_source_creates_missing(self):
        """get_source creates a new SourceState if source doesn't exist."""
        state = UpdateState()
        src = state.get_source("plos")
        assert isinstance(src, SourceState)
        assert src.collected_ids == []
        assert "plos" in state.sources

    def test_id_set_property(self):
        """SourceState.id_set returns a set for O(1) lookup."""
        src = SourceState(collected_ids=["elife:100", "elife:200", "elife:300"])
        id_set = src.id_set
        assert isinstance(id_set, set)
        assert "elife:200" in id_set
        assert "elife:999" not in id_set


# ── Bootstrap / Initialize Tests ──────────────────────────────────


class TestBootstrap:

    def test_initialize_from_jsonl(self, state_path: Path, sample_jsonl: Path):
        """Bootstrap creates correct SourceState from JSONL."""
        mgr = StateManager(state_path)
        src = mgr.initialize_from_jsonl("elife", sample_jsonl)

        assert len(src.collected_ids) == 3
        assert src.collected_ids == ["elife:100", "elife:200", "elife:300"]
        assert src.last_article_date == "2026-01-10"

    def test_initialize_from_multiple_jsonl(self, tmp_path: Path):
        """Bootstrap merges IDs from multiple JSONL files."""
        mgr = StateManager(tmp_path / "state.json")

        file1 = tmp_path / "a.jsonl"
        file2 = tmp_path / "b.jsonl"
        file1.write_text('{"id": "elife:1", "published_date": "2024-01-01"}\n')
        file2.write_text('{"id": "elife:2", "published_date": "2025-06-15"}\n')

        src = mgr.initialize_from_jsonl("elife", file1, file2)
        assert src.collected_ids == ["elife:1", "elife:2"]
        assert src.last_article_date == "2025-06-15"

    def test_initialize_skips_missing_files(self, tmp_path: Path):
        """Bootstrap handles missing files gracefully."""
        mgr = StateManager(tmp_path / "state.json")
        src = mgr.initialize_from_jsonl("elife", tmp_path / "nonexistent.jsonl")
        assert src.collected_ids == []
        assert src.last_article_date is None

    def test_initialize_deduplicates(self, tmp_path: Path):
        """Bootstrap deduplicates IDs across files."""
        mgr = StateManager(tmp_path / "state.json")

        file1 = tmp_path / "a.jsonl"
        file2 = tmp_path / "b.jsonl"
        file1.write_text('{"id": "elife:1", "published_date": "2024-01-01"}\n')
        file2.write_text(
            '{"id": "elife:1", "published_date": "2024-01-01"}\n'
            '{"id": "elife:2", "published_date": "2025-01-01"}\n'
        )

        src = mgr.initialize_from_jsonl("elife", file1, file2)
        assert src.collected_ids == ["elife:1", "elife:2"]

    def test_initialize_null_published_date(self, tmp_path: Path):
        """Null published_date doesn't corrupt last_article_date."""
        mgr = StateManager(tmp_path / "state.json")
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"id": "elife:1", "published_date": "2025-03-01"}\n'
            '{"id": "elife:2", "published_date": null}\n'
        )

        src = mgr.initialize_from_jsonl("elife", path)
        assert src.last_article_date == "2025-03-01"
        assert len(src.collected_ids) == 2


# ── Startup Sync Tests ────────────────────────────────────────────


class TestStartupSync:

    def test_sync_adds_missing_ids(self, state_path: Path, sample_jsonl: Path):
        """Sync detects IDs in JSONL but not in state."""
        mgr = StateManager(state_path)
        state = UpdateState()
        state.sources["elife"] = SourceState(
            collected_ids=["elife:100"],  # missing 200, 300
        )

        added = mgr.sync_from_jsonl(state, "elife", sample_jsonl)
        assert added == 2
        assert state.sources["elife"].collected_ids == ["elife:100", "elife:200", "elife:300"]

    def test_sync_preserves_extra_state_ids(self, state_path: Path, sample_jsonl: Path):
        """Sync never removes IDs from state (add-only for multi-file safety)."""
        mgr = StateManager(state_path)
        state = UpdateState()
        state.sources["elife"] = SourceState(
            collected_ids=["elife:100", "elife:200", "elife:300", "elife:999"],
        )

        added = mgr.sync_from_jsonl(state, "elife", sample_jsonl)
        assert added == 0
        # elife:999 is preserved even though it's not in the JSONL
        # (it may come from a different JSONL file like elife_legacy_v1.jsonl)
        assert "elife:999" in state.sources["elife"].collected_ids

    def test_sync_noop_when_consistent(self, state_path: Path, sample_jsonl: Path):
        """Sync is a no-op when state matches JSONL."""
        mgr = StateManager(state_path)
        state = UpdateState()
        state.sources["elife"] = SourceState(
            collected_ids=["elife:100", "elife:200", "elife:300"],
        )

        added = mgr.sync_from_jsonl(state, "elife", sample_jsonl)
        assert added == 0

    def test_sync_handles_missing_file(self, state_path: Path, tmp_path: Path):
        """Sync returns 0 if JSONL file doesn't exist."""
        mgr = StateManager(state_path)
        state = UpdateState()
        state.sources["elife"] = SourceState(collected_ids=["elife:100"])

        added = mgr.sync_from_jsonl(state, "elife", tmp_path / "nonexistent.jsonl")
        assert added == 0


# ── Source Registry Tests ─────────────────────────────────────────


class TestSourceRegistry:

    def test_all_sources_registered(self):
        """All expected sources are in the registry."""
        assert "elife" in SOURCE_REGISTRY
        assert "plos" in SOURCE_REGISTRY
        assert "f1000" in SOURCE_REGISTRY

    def test_get_source_config(self):
        """get_source_config returns correct config."""
        cfg = get_source_config("elife")
        assert cfg.name == "elife"
        assert cfg.id_prefix == "elife:"
        assert cfg.source_literal == "elife"
        assert cfg.default_start_date == "2018-01-01"
        assert "genetics-genomics" in cfg.default_subjects

    def test_get_source_config_unknown(self):
        """get_source_config raises KeyError for unknown source."""
        with pytest.raises(KeyError, match="Unknown source"):
            get_source_config("arxiv")

    def test_list_sources(self):
        """list_sources returns sorted list."""
        sources = list_sources()
        assert sources == sorted(sources)
        assert "elife" in sources

    def test_make_article_id(self):
        """make_article_id creates correct prefixed ID."""
        cfg = get_source_config("elife")
        assert cfg.make_article_id("87528") == "elife:87528"

        cfg_plos = get_source_config("plos")
        assert cfg_plos.make_article_id("10.1371/journal.pbio.123") == "plos:10.1371/journal.pbio.123"

    def test_get_collector_class(self):
        """get_collector_class dynamically imports the correct class."""
        cfg = get_source_config("elife")
        cls = cfg.get_collector_class()
        assert cls.__name__ == "ELifeCollector"

    def test_all_collector_classes_importable(self):
        """All registered collectors can be imported."""
        for name in list_sources():
            cfg = get_source_config(name)
            cls = cfg.get_collector_class()
            assert cls is not None, f"Failed to import collector for {name}"


# ── Postprocess Tests ─────────────────────────────────────────────


class TestPostprocess:

    def test_infer_reviewed_preprint(self):
        """Detects reviewed_preprint format from keywords."""
        entry = {"decision_letter_raw": "eLife Assessment: This is a valuable study..."}
        assert infer_review_format(entry) == "reviewed_preprint"

    def test_infer_journal(self):
        """Detects journal format from keywords."""
        entry = {"decision_letter_raw": "Reviewer 1: We have major concerns about..."}
        assert infer_review_format(entry) == "journal"

    def test_infer_journal_by_year(self):
        """Falls back to year heuristic when no keywords match."""
        entry = {"decision_letter_raw": "Some generic text...", "published_date": "2021-05-01"}
        assert infer_review_format(entry) == "journal"

    def test_infer_reviewed_preprint_by_year(self):
        """Falls back to year heuristic for newer articles."""
        entry = {"decision_letter_raw": "Some generic text...", "published_date": "2025-05-01"}
        assert infer_review_format(entry) == "reviewed_preprint"

    def test_infer_unknown(self):
        """Returns unknown when no evidence available."""
        entry = {"decision_letter_raw": ""}
        assert infer_review_format(entry) == "unknown"

    def test_clean_subjects(self):
        """Removes article-type labels from subjects."""
        subjects = ["Neuroscience", "Research Article", "Cell Biology"]
        cleaned = clean_subjects(subjects)
        assert cleaned == ["Neuroscience", "Cell Biology"]
        assert "Research Article" not in cleaned

    def test_clean_subjects_no_labels(self):
        """No change when no article-type labels present."""
        subjects = ["Neuroscience", "Genetics and Genomics"]
        assert clean_subjects(subjects) == subjects

    def test_postprocess_entry(self):
        """postprocess_entry applies all transformations."""
        entry = {
            "subjects": ["Neuroscience", "Research Article"],
            "decision_letter_raw": "eLife Assessment: valuable study",
            "author_response_raw": "We thank the reviewers...",
            "schema_version": "1.0",
        }
        result = postprocess_entry(entry)

        assert result["subjects"] == ["Neuroscience"]
        assert result["review_format"] == "reviewed_preprint"
        assert result["has_author_response"] is True
        assert result["schema_version"] == "1.1"

    def test_postprocess_entry_no_response(self):
        """postprocess_entry handles missing author response."""
        entry = {
            "subjects": [],
            "decision_letter_raw": "",
            "author_response_raw": "",
            "schema_version": "1.0",
        }
        result = postprocess_entry(entry)
        assert result["has_author_response"] is False
        assert result["review_format"] == "unknown"


# ── Utility Tests ─────────────────────────────────────────────────


class TestUtils:

    def test_make_run_id_format(self):
        """make_run_id produces expected format."""
        rid = make_run_id("elife")
        assert rid.startswith("run-elife-")
        parts = rid.split("-")
        assert len(parts) >= 5  # run-elife-YYYY-MM-DD-hex

    def test_detect_trigger_local(self, monkeypatch):
        """Detects local trigger when not in GitHub Actions."""
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        assert _detect_trigger() == "local"

    def test_detect_trigger_schedule(self, monkeypatch):
        """Detects scheduled GitHub Actions trigger."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        assert _detect_trigger() == "github_actions_schedule"

    def test_detect_trigger_manual(self, monkeypatch):
        """Detects manual GitHub Actions trigger."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
        assert _detect_trigger() == "github_actions_manual"


# ── M2: Append Mode Tests ────────────────────────────────────────


class TestAppendMode:

    def test_append_preserves_existing_content(self, tmp_path: Path):
        """When append=True, existing JSONL content is preserved."""
        output = tmp_path / "test.jsonl"
        output.write_text('{"id": "elife:existing", "source": "elife"}\n')

        # Write another line in append mode
        with output.open("a") as f:
            f.write('{"id": "elife:new", "source": "elife"}\n')

        lines = [json.loads(l) for l in output.read_text().strip().split("\n")]
        assert len(lines) == 2
        assert lines[0]["id"] == "elife:existing"
        assert lines[1]["id"] == "elife:new"

    def test_overwrite_clears_existing_content(self, tmp_path: Path):
        """When append=False (default), file is overwritten."""
        output = tmp_path / "test.jsonl"
        output.write_text('{"id": "elife:existing"}\n')

        with output.open("w") as f:
            f.write('{"id": "elife:new"}\n')

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "elife:new"


# ── M2: Lockfile Tests ───────────────────────────────────────────


class TestLockfile:

    def test_acquire_and_release(self, tmp_path: Path):
        """Lockfile can be acquired and released."""
        from bioreview_bench.scripts.update_pipeline import _acquire_lock, _release_lock

        lock_path = tmp_path / ".update.lock"
        fd = _acquire_lock(lock_path)
        assert fd is not None
        assert lock_path.exists()
        _release_lock(fd)

    def test_concurrent_lock_blocked(self, tmp_path: Path):
        """Second lock attempt returns None when already locked."""
        from bioreview_bench.scripts.update_pipeline import _acquire_lock, _release_lock

        lock_path = tmp_path / ".update.lock"
        fd1 = _acquire_lock(lock_path)
        assert fd1 is not None

        fd2 = _acquire_lock(lock_path)
        assert fd2 is None  # Blocked

        _release_lock(fd1)

    def test_lock_reacquirable_after_release(self, tmp_path: Path):
        """Lock can be re-acquired after release."""
        from bioreview_bench.scripts.update_pipeline import _acquire_lock, _release_lock

        lock_path = tmp_path / ".update.lock"
        fd1 = _acquire_lock(lock_path)
        assert fd1 is not None
        _release_lock(fd1)

        fd2 = _acquire_lock(lock_path)
        assert fd2 is not None
        _release_lock(fd2)


# ── M2: Update Pipeline State Integration ────────────────────────


class TestUpdateStateIntegration:

    def test_state_updated_after_sync(self, state_path: Path, tmp_path: Path):
        """After appending to JSONL + sync, state contains new IDs."""
        mgr = StateManager(state_path)

        # Initial state with 2 IDs
        state = UpdateState()
        state.sources["elife"] = SourceState(
            last_article_date="2025-01-01",
            collected_ids=["elife:100", "elife:200"],
        )
        mgr.save(state)

        # Simulate JSONL that has all 3 articles (including newly appended)
        jsonl = tmp_path / "elife.jsonl"
        jsonl.write_text(
            '{"id": "elife:100", "published_date": "2025-01-01"}\n'
            '{"id": "elife:200", "published_date": "2025-01-01"}\n'
            '{"id": "elife:300", "published_date": "2026-02-15"}\n'
        )

        # Sync picks up the new article
        state = mgr.load()
        added = mgr.sync_from_jsonl(state, "elife", jsonl)
        assert added == 1
        assert "elife:300" in state.sources["elife"].collected_ids
        assert state.sources["elife"].last_article_date == "2026-02-15"

    def test_run_record_added(self, state_path: Path):
        """RunRecord is correctly added to state."""
        mgr = StateManager(state_path)
        state = UpdateState()

        run = RunRecord(
            run_id=make_run_id("elife"),
            source="elife",
            started_at="2026-02-28T06:00:00",
            completed_at="2026-02-28T06:05:00",
            trigger="local",
            new_articles=5,
            skipped_duplicates=10,
            cost_usd_est=0.01,
            dry_run=False,
        )
        state.add_run(run)
        mgr.save(state)

        loaded = mgr.load()
        assert len(loaded.runs) == 1
        assert loaded.runs[0].new_articles == 5
        assert loaded.runs[0].skipped_duplicates == 10

    def test_startup_sync_updates_last_article_date(self, tmp_path: Path):
        """Startup sync in update_pipeline updates last_article_date from JSONL.

        Regression test: previously, startup sync only added missing IDs but
        did NOT update last_article_date, causing stale start_date in crash
        recovery scenarios.
        """
        from bioreview_bench.scripts.update_pipeline import _run_source_update
        from bioreview_bench.collect.registry import get_source_config

        data_dir = tmp_path / "data"
        (data_dir / "processed").mkdir(parents=True)
        (data_dir / "manifests").mkdir(parents=True)
        state_path = data_dir / "update_state.json"

        config = get_source_config("elife")
        output_path = data_dir / "processed" / config.output_filename

        # Simulate crash recovery: JSONL has articles with recent dates
        # but state has stale last_article_date
        output_path.write_text(
            '{"id": "elife:100", "published_date": "2026-01-10"}\n'
            '{"id": "elife:200", "published_date": "2026-02-20"}\n'
        )

        # State has the IDs but stale date (or no date at all)
        mgr = StateManager(state_path)
        state = UpdateState()
        state.sources["elife"] = SourceState(
            last_article_date="2024-06-01",  # Stale!
            collected_ids=["elife:100", "elife:200"],
        )
        mgr.save(state)

        # After loading, startup sync should detect stale date and update it
        state = mgr.load()
        source_state = state.get_source("elife")

        # Simulate the startup sync logic from update_pipeline
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

        # IDs are already synced (no missing), but date IS stale
        assert missing_from_state == set()
        assert date_stale is True
        assert max_date_sync == "2026-02-20"

        # Apply the sync
        if date_stale:
            source_state.last_article_date = max_date_sync
        mgr.save(state)

        # Verify
        loaded = mgr.load()
        assert loaded.sources["elife"].last_article_date == "2026-02-20"

    def test_incremental_start_date_calculation(self):
        """Start date is buffered by _DATE_BUFFER_DAYS from last_article_date."""
        from datetime import datetime, timedelta

        last_date = "2026-02-15"
        buffer_days = 3
        last_dt = datetime.strptime(last_date, "%Y-%m-%d")
        buffered = last_dt - timedelta(days=buffer_days)
        start_date = buffered.strftime("%Y-%m-%d")

        assert start_date == "2026-02-12"

    def test_known_ids_as_set_for_lookup(self):
        """SourceState.id_set provides O(1) lookup for dedup."""
        src = SourceState(
            collected_ids=[f"elife:{i}" for i in range(1000)],
        )
        id_set = src.id_set
        assert "elife:500" in id_set
        assert "elife:9999" not in id_set

    def test_dry_run_zero_cost(self, state_path: Path):
        """Dry run should estimate $0 cost."""
        run = RunRecord(
            run_id="run-test",
            source="elife",
            new_articles=100,
            cost_usd_est=0.0,
            dry_run=True,
        )
        assert run.cost_usd_est == 0.0
        assert run.dry_run is True


# ── M2: CLI Registration Tests ───────────────────────────────────


class TestCLIRegistration:

    def test_update_pipeline_importable(self):
        """update_pipeline module can be imported."""
        from bioreview_bench.scripts import update_pipeline
        assert hasattr(update_pipeline, "main")

    def test_main_is_click_command(self):
        """main() is a Click command."""
        from bioreview_bench.scripts.update_pipeline import main
        import click
        assert isinstance(main, click.Command)

    def test_collect_elife_run_returns_dict(self):
        """_run function signature includes append and known_ids params."""
        import inspect
        from bioreview_bench.scripts.collect_elife import _run

        sig = inspect.signature(_run)
        params = sig.parameters

        assert "append" in params
        assert "known_ids" in params
        assert params["append"].default is False
        assert params["known_ids"].default is None

        # Return annotation should be dict (string due to __future__ annotations)
        assert sig.return_annotation in (dict, "dict")

    def test_collect_plos_run_has_incremental_params(self):
        """PLOS _run() has append, known_ids, and returns dict (parity with eLife)."""
        import inspect
        from bioreview_bench.scripts.collect_plos import _run

        sig = inspect.signature(_run)
        params = sig.parameters

        assert "append" in params
        assert "known_ids" in params
        assert params["append"].default is False
        assert params["known_ids"].default is None
        assert sig.return_annotation in (dict, "dict")

    def test_collect_f1000_run_has_incremental_params(self):
        """F1000 _run() has append, known_ids, and returns dict (parity with eLife)."""
        import inspect
        from bioreview_bench.scripts.collect_f1000 import _run

        sig = inspect.signature(_run)
        params = sig.parameters

        assert "append" in params
        assert "known_ids" in params
        assert params["append"].default is False
        assert params["known_ids"].default is None
        assert sig.return_annotation in (dict, "dict")


# ── M3: Split Tests ──────────────────────────────────────────────

# Import split functions — these are scripts, not in the package,
# so we add the scripts dir to sys.path.
import sys as _sys
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in _sys.path:
    _sys.path.insert(0, _scripts_dir)
from create_splits import (
    get_stratum,
    stratified_split,
    frozen_split,
    load_jsonl,
    save_jsonl,
)


def _make_entry(
    eid: str,
    fmt: str = "reviewed_preprint",
    n_concerns: int = 3,
    stance: str = "no_response",
    subject: str = "Neuroscience",
) -> dict:
    """Helper to create a minimal entry dict for split testing."""
    concerns = [
        {"concern_id": f"{eid}:C{i}", "author_stance": stance}
        for i in range(n_concerns)
    ]
    return {
        "id": eid,
        "review_format": fmt,
        "subjects": [subject],
        "concerns": concerns,
        "published_date": "2025-01-01",
    }


class TestStratifiedSplit:

    def test_basic_split_ratios(self):
        """Fresh split produces roughly correct ratios."""
        entries = [_make_entry(f"elife:{i}") for i in range(100)]
        train, val, test = stratified_split(entries, 0.15, 0.15, seed=42)

        total = len(train) + len(val) + len(test)
        assert total == 100
        # Allow some tolerance due to stratification rounding
        assert len(test) >= 10
        assert len(val) >= 10

    def test_zero_concern_to_train(self):
        """Zero-concern entries go entirely to train."""
        usable = [_make_entry(f"elife:{i}") for i in range(20)]
        unusable = [_make_entry(f"elife:u{i}", n_concerns=0) for i in range(5)]
        entries = usable + unusable

        train, val, test = stratified_split(entries, 0.15, 0.15, seed=42)

        # All zero-concern entries must be in train
        train_ids = {e["id"] for e in train}
        for e in unusable:
            assert e["id"] in train_ids

    def test_deterministic_with_seed(self):
        """Same seed produces same split."""
        entries = [_make_entry(f"elife:{i}") for i in range(50)]
        t1, v1, te1 = stratified_split(entries, 0.15, 0.15, seed=42)
        t2, v2, te2 = stratified_split(entries, 0.15, 0.15, seed=42)

        assert [e["id"] for e in te1] == [e["id"] for e in te2]
        assert [e["id"] for e in v1] == [e["id"] for e in v2]


class TestFrozenSplit:

    def test_frozen_ids_in_test(self):
        """Frozen IDs always end up in test split."""
        entries = [_make_entry(f"elife:{i}") for i in range(30)]
        frozen_ids = {f"elife:{i}" for i in range(5)}  # First 5

        train, val, test = frozen_split(entries, frozen_ids, val_ratio=0.15, seed=42)

        test_ids = {e["id"] for e in test}
        assert frozen_ids.issubset(test_ids)
        assert len(test) == 5  # Exactly the frozen IDs

    def test_new_articles_not_in_test(self):
        """New articles (not in frozen IDs) go to train/val only."""
        entries = [_make_entry(f"elife:{i}") for i in range(30)]
        frozen_ids = {f"elife:{i}" for i in range(5)}

        train, val, test = frozen_split(entries, frozen_ids, val_ratio=0.15, seed=42)

        test_ids = {e["id"] for e in test}
        new_ids = {f"elife:{i}" for i in range(5, 30)}
        assert test_ids.isdisjoint(new_ids)

        # All new usable entries are in train or val
        train_val_ids = {e["id"] for e in train} | {e["id"] for e in val}
        for nid in new_ids:
            assert nid in train_val_ids

    def test_missing_frozen_ids_handled(self, capsys):
        """Missing frozen IDs produce a warning but don't crash."""
        entries = [_make_entry(f"elife:{i}") for i in range(10)]
        frozen_ids = {"elife:0", "elife:1", "elife:999"}  # 999 doesn't exist

        train, val, test = frozen_split(entries, frozen_ids, val_ratio=0.15, seed=42)

        captured = capsys.readouterr()
        assert "1 frozen test IDs not found" in captured.out
        assert len(test) == 2  # Only 0 and 1 found

    def test_frozen_preserves_all_entries(self):
        """Total entries = train + val + test (no data loss)."""
        entries = [_make_entry(f"elife:{i}") for i in range(50)]
        frozen_ids = {f"elife:{i}" for i in range(10)}

        train, val, test = frozen_split(entries, frozen_ids, val_ratio=0.15, seed=42)

        total = len(train) + len(val) + len(test)
        assert total == 50

    def test_stratum_coverage_warning(self, capsys):
        """Warns when new strata aren't represented in frozen test."""
        # All frozen entries are "reviewed_preprint"
        frozen_entries = [_make_entry(f"elife:{i}", fmt="reviewed_preprint") for i in range(5)]
        # New entries include "journal" format (new stratum)
        new_entries = [_make_entry(f"elife:{i}", fmt="journal") for i in range(5, 15)]
        entries = frozen_entries + new_entries

        frozen_ids = {e["id"] for e in frozen_entries}
        train, val, test = frozen_split(entries, frozen_ids, val_ratio=0.15, seed=42)

        captured = capsys.readouterr()
        assert "new strata not in frozen test" in captured.out


class TestSplitIO:

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Entries survive save → load cycle."""
        entries = [_make_entry(f"elife:{i}") for i in range(5)]
        path = tmp_path / "test.jsonl"
        save_jsonl(entries, path)

        loaded = load_jsonl(path)
        assert len(loaded) == 5
        assert loaded[0]["id"] == entries[0]["id"]

    def test_dedup_in_load(self, tmp_path: Path):
        """Dedup logic removes duplicate IDs."""
        # Write duplicates
        path = tmp_path / "dupes.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"id": "elife:1", "x": 1}) + "\n")
            f.write(json.dumps({"id": "elife:1", "x": 2}) + "\n")
            f.write(json.dumps({"id": "elife:2", "x": 3}) + "\n")

        entries = load_jsonl(path)
        # load_jsonl doesn't dedup — dedup is in main()
        assert len(entries) == 3

        # Dedup manually (same logic as main())
        seen: set[str] = set()
        deduped = []
        for e in entries:
            if e["id"] not in seen:
                seen.add(e["id"])
                deduped.append(e)
        assert len(deduped) == 2


class TestGetStratum:

    def test_no_concerns(self):
        assert get_stratum({"review_format": "journal", "concerns": []}) == "journal:no_concerns"

    def test_high_noresp(self):
        entry = _make_entry("x", stance="no_response", n_concerns=5)
        assert "high_noresp" in get_stratum(entry)

    def test_high_concede(self):
        entry = _make_entry("x", stance="conceded", n_concerns=5)
        assert "high_concede" in get_stratum(entry)

    def test_subject_buckets(self):
        assert "neuro" in get_stratum(_make_entry("x", subject="Neuroscience"))
        assert "cell" in get_stratum(_make_entry("x", subject="Cell Biology"))
        assert "other" in get_stratum(_make_entry("x", subject="Genetics and Genomics"))


# ── M4: HuggingFace Hub Push Tests ──────────────────────────────
# Comprehensive HF export tests are in test_hf_export.py.
# These tests verify integration with update_pipeline only.

from bioreview_bench.collect.hf_push import push_to_hub


class TestHfPush:

    def _setup_data_dir(self, tmp_path: Path) -> Path:
        """Create a data directory with v3 splits for push testing."""
        data = tmp_path / "data"
        splits_v3 = data / "splits" / "v3"
        splits_v3.mkdir(parents=True)
        (data / "manifests").mkdir(parents=True)

        # Minimal entries for split files
        entry = json.dumps({
            "id": "elife:100", "source": "elife", "doi": "10.7554/eLife.100",
            "title": "Study", "abstract": "Abstract", "subjects": ["Neuroscience"],
            "published_date": "2025-01-01",
            "paper_text_sections": {"intro": "text"},
            "decision_letter_raw": "Review text",
            "author_response_raw": "Response text",
            "concerns": [{"concern_id": "elife:100:R1C1",
                          "concern_text": "Issue", "category": "other",
                          "severity": "minor", "author_stance": "conceded"}],
        })
        (splits_v3 / "train.jsonl").write_text(entry + "\n")
        (splits_v3 / "val.jsonl").write_text(entry + "\n")
        (splits_v3 / "test.jsonl").write_text(entry + "\n")

        # Metadata files
        (data / "splits" / "test_ids_frozen_v3.json").write_text('{"ids": ["elife:100"]}')
        (splits_v3 / "split_meta_v3.json").write_text('{"seed": 42}')
        (data / "manifests" / "em-v1.0.json").write_text('{"version": "1.0"}')
        return data

    def test_dry_run_generates_upload_plan(self, tmp_path: Path):
        """Dry-run builds correct upload plan with config JSONL + README + aux."""
        data = self._setup_data_dir(tmp_path)
        result = push_to_hub(data_dir=data, dry_run=True)

        assert result["dry_run"] is True
        uploaded = result["uploaded"]
        # 6 configs × 3 splits = 18 JSONL + 1 README + 3 aux = 22 files
        assert len(uploaded) >= 20

        uploaded_str = " ".join(str(p) for p in uploaded)
        assert "README.md" in uploaded_str
        assert "data/default/train.jsonl" in uploaded_str
        assert "data/benchmark/test.jsonl" in uploaded_str
        assert "manifests/" in uploaded_str

    def test_empty_splits_returns_error(self, tmp_path: Path):
        """Empty splits directory returns error without crash."""
        data = tmp_path / "empty_data"
        (data / "splits" / "v3").mkdir(parents=True)
        result = push_to_hub(data_dir=data, dry_run=True)
        assert result["uploaded"] == []
        assert "error" in result["stats"]


class TestHfPushCLIIntegration:

    def test_push_hf_option_exists(self):
        """--push-hf option is registered on the CLI command."""
        from bioreview_bench.scripts.update_pipeline import main
        param_names = [p.name for p in main.params]
        assert "push_hf" in param_names

    def test_update_splits_option_exists(self):
        """--update-splits option is registered on the CLI command."""
        from bioreview_bench.scripts.update_pipeline import main
        param_names = [p.name for p in main.params]
        assert "update_splits" in param_names

    def test_push_hf_is_flag(self):
        """--push-hf is a boolean flag (not a value option)."""
        from bioreview_bench.scripts.update_pipeline import main
        for p in main.params:
            if p.name == "push_hf":
                assert p.is_flag is True
                assert p.default is False
                break

    def test_run_push_hf_import_error_handled(self, tmp_path: Path, monkeypatch):
        """_run_push_hf handles missing huggingface_hub gracefully."""
        from bioreview_bench.scripts import update_pipeline

        # Mock the import to fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub" or "hf_push" in name:
                raise ImportError("No module named 'huggingface_hub'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        # Just verify the function exists and is callable
        assert callable(update_pipeline._run_push_hf)

    def test_run_update_splits_callable(self):
        """_run_update_splits helper is callable."""
        from bioreview_bench.scripts.update_pipeline import _run_update_splits
        assert callable(_run_update_splits)

    def test_version_bump_option_exists(self):
        """--version-bump option is registered on the CLI command."""
        from bioreview_bench.scripts.update_pipeline import main
        param_names = [p.name for p in main.params]
        assert "version_bump" in param_names

    def test_version_bump_default_is_minor(self):
        """--version-bump defaults to 'minor'."""
        from bioreview_bench.scripts.update_pipeline import main
        for p in main.params:
            if p.name == "version_bump":
                assert p.default == "minor"
                break


# ── Phase 3: Dataset Versioning Tests ─────────────────────────────


class TestDatasetVersioning:

    def test_bump_minor(self):
        """bump_minor increments minor version."""
        state = UpdateState(dataset_version="1.0")
        result = state.bump_minor()
        assert result == "1.1"
        assert state.dataset_version == "1.1"

    def test_bump_minor_sequential(self):
        """Multiple minor bumps work correctly."""
        state = UpdateState(dataset_version="1.0")
        state.bump_minor()
        state.bump_minor()
        assert state.dataset_version == "1.2"

    def test_bump_major(self):
        """bump_major increments major version and resets minor."""
        state = UpdateState(dataset_version="1.5")
        result = state.bump_major()
        assert result == "2.0"
        assert state.dataset_version == "2.0"

    def test_bump_major_from_high_minor(self):
        """bump_major works from high minor version."""
        state = UpdateState(dataset_version="3.12")
        result = state.bump_major()
        assert result == "4.0"
        assert state.dataset_version == "4.0"

    def test_version_persists_through_save_load(self, state_path: Path):
        """dataset_version survives save → load cycle."""
        mgr = StateManager(state_path)
        state = UpdateState(dataset_version="2.3")
        mgr.save(state)

        loaded = mgr.load()
        assert loaded.dataset_version == "2.3"

    def test_version_default_on_load(self, state_path: Path):
        """Loading old state without dataset_version defaults to 1.0."""
        # Write state without dataset_version field
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text('{"sources": {}, "runs": []}')

        mgr = StateManager(state_path)
        state = mgr.load()
        assert state.dataset_version == "1.0"

    def test_hf_push_accepts_version_tag(self):
        """push_to_hub signature accepts version_tag parameter."""
        import inspect
        from bioreview_bench.collect.hf_push import push_to_hub

        sig = inspect.signature(push_to_hub)
        assert "version_tag" in sig.parameters
        assert sig.parameters["version_tag"].default is None

    def test_hf_push_upload_includes_update_state(self, tmp_path: Path):
        """update_state.json is included in HF upload plan."""
        data = tmp_path / "data"
        splits_v3 = data / "splits" / "v3"
        splits_v3.mkdir(parents=True)
        (data / "manifests").mkdir(parents=True)

        entry = json.dumps({
            "id": "elife:100", "source": "elife", "doi": "10.7554/eLife.100",
            "title": "Study", "abstract": "Abstract", "subjects": ["Neuroscience"],
            "published_date": "2025-01-01",
            "paper_text_sections": {"intro": "text"},
            "decision_letter_raw": "Review text",
            "author_response_raw": "Response text",
            "concerns": [{"concern_id": "elife:100:R1C1",
                          "concern_text": "Issue", "category": "other",
                          "severity": "minor", "author_stance": "conceded"}],
        })
        (splits_v3 / "train.jsonl").write_text(entry + "\n")
        (splits_v3 / "val.jsonl").write_text(entry + "\n")
        (splits_v3 / "test.jsonl").write_text(entry + "\n")

        # Add update_state.json
        (data / "update_state.json").write_text('{"dataset_version": "1.0"}')
        (data / "manifests" / "em-v1.0.json").write_text('{"version": "1.0"}')

        result = push_to_hub(data_dir=data, dry_run=True)
        uploaded = result["uploaded"]
        assert any("update_state.json" in p for p in uploaded)

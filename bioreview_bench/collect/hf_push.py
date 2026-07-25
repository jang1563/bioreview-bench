"""Private Hugging Face raw-archive push — multi-config JSONL upload.

Generates per-config JSONL files in a staging directory, creates a
DatasetCard README, and uploads everything to HuggingFace Hub in a
single atomic commit. This legacy exporter contains publisher and peer-review
text, so it refuses the public, rights-minimized repository and requires a
private destination.

Usage::

    from bioreview_bench.collect.hf_push import push_to_hub
    push_to_hub(data_dir=Path("data"))

Requires: ``uv sync --extra hub``
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..project_defaults import DEFAULT_BENCHMARK_SPLIT_VERSION

log = logging.getLogger(__name__)

PUBLIC_REPO_ID = "jang1563/bioreview-bench"
PRIVATE_RAW_REPO_ID = "jang1563/bioreview-bench-raw-private"
_STAGING_DIR_NAME = ".hf_staging"
_OBSOLETE_REMOTE_PATHS = ("metadata/update_state.json",)


def _version_sort_key(name: str) -> tuple[int, ...]:
    parts = name.lstrip("v").split(".")
    return tuple(int(part) if part.isdigit() else 0 for part in parts)


def _resolve_splits_subdir(data_dir: Path, splits_subdir: str | None) -> tuple[Path, str]:
    """Resolve the requested split directory, preferring the newest version."""
    if splits_subdir is not None:
        splits_dir = data_dir / splits_subdir
        return splits_dir, Path(splits_subdir).name

    splits_root = data_dir / "splits"
    candidates = (
        sorted(
            [p for p in splits_root.iterdir() if p.is_dir() and p.name.startswith("v")],
            key=lambda path: _version_sort_key(path.name),
            reverse=True,
        )
        if splits_root.exists()
        else []
    )

    for candidate in candidates:
        if all((candidate / f"{split}.jsonl").exists() for split in ("train", "val", "test")):
            return candidate, candidate.name

    return splits_root / DEFAULT_BENCHMARK_SPLIT_VERSION, DEFAULT_BENCHMARK_SPLIT_VERSION


def _resolve_results_dir(data_dir: Path, version_name: str) -> Path | None:
    """Resolve the version-matched results directory, falling back to legacy root results."""
    results_root = data_dir.parent / "results"
    versioned_dir = results_root / version_name
    if versioned_dir.exists():
        return versioned_dir
    if results_root.exists():
        return results_root
    return None


def _load_leaderboard_snapshot(
    results_dir: Path | None,
    version_name: str,
) -> dict[str, Any] | None:
    """Load leaderboard data for the dataset card if versioned artifacts exist."""
    if results_dir is None:
        return None

    leaderboard_path = results_dir / "leaderboard.json"
    if not leaderboard_path.exists():
        return None

    try:
        snapshot: dict[str, Any] = {
            "version_name": version_name,
            "entries": json.loads(leaderboard_path.read_text(encoding="utf-8")),
        }
    except Exception as exc:
        log.warning("Failed to load leaderboard snapshot from %s: %s", leaderboard_path, exc)
        return None

    manifest_path = results_dir / "release_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to load release manifest from %s: %s", manifest_path, exc)
        else:
            snapshot["split"] = manifest.get("split", "test")
            snapshot["matching"] = manifest.get("matching", {})

    return snapshot


def push_to_hub(
    data_dir: Path,
    repo_id: str = PRIVATE_RAW_REPO_ID,
    dry_run: bool = False,
    splits_subdir: str | None = None,
    version_tag: str | None = None,
) -> dict[str, Any]:
    """Export full multi-config JSONL files to a private raw archive.

    Steps:
        1. Export all configs as JSONL into a staging directory.
        2. Generate the DatasetCard README.md.
        3. Upload via ``huggingface_hub`` in a single commit.
        4. Clean up the staging directory.

    Args:
        data_dir: Root data directory (parent of processed/, splits/, manifests/).
        repo_id: Private Hugging Face dataset repository ID. The public
            rights-minimized repository is always rejected.
        dry_run: If True, generate files locally but don't upload.
        splits_subdir: Path relative to data_dir for split JSONL files.

    Returns:
        Dict with ``uploaded`` file list, ``stats``, and ``dry_run`` flag.
    """
    if not dry_run and repo_id.strip().lower() == PUBLIC_REPO_ID.lower():
        raise ValueError(
            f"Refusing raw export to public repository {PUBLIC_REPO_ID}. "
            f"Use the private archive {PRIVATE_RAW_REPO_ID} or another verified "
            "private dataset repository."
        )

    from .hf_card import generate_dataset_card
    from .hf_export import export_all_configs

    splits_dir, version_name = _resolve_splits_subdir(data_dir, splits_subdir)
    results_dir = _resolve_results_dir(data_dir, version_name)
    staging_dir = data_dir / _STAGING_DIR_NAME

    # Clean previous staging
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    # Step 1: Export all configs
    log.info("Exporting configs from %s", splits_dir)
    stats = export_all_configs(splits_dir=splits_dir, output_dir=staging_dir / "data")

    if "error" in stats:
        log.error("Export failed: %s", stats["error"])
        return {"uploaded": [], "stats": stats, "dry_run": dry_run}

    # Step 2: Generate DatasetCard
    leaderboard = _load_leaderboard_snapshot(results_dir, version_name)
    readme_content = generate_dataset_card(stats, leaderboard=leaderboard)
    readme_path = staging_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    log.info("Generated DatasetCard: %d chars", len(readme_content))

    # Collect all files to upload
    upload_plan: list[tuple[Path, str]] = []  # (local_path, path_in_repo)

    # Config JSONL files: staging/data/{config}/{split}.jsonl → data/{config}/{split}.jsonl
    for jsonl in sorted(staging_dir.rglob("*.jsonl")):
        rel = jsonl.relative_to(staging_dir)
        upload_plan.append((jsonl, str(rel)))

    # README.md → README.md
    upload_plan.append((readme_path, "README.md"))

    # Auxiliary files from data dir
    _add_auxiliary_files(
        data_dir=data_dir,
        upload_plan=upload_plan,
        version_name=version_name,
        results_dir=results_dir,
    )

    log.info("Upload plan: %d files", len(upload_plan))

    if dry_run:
        file_list = [rp for _, rp in upload_plan]
        log.info("Dry run — staging dir preserved at %s", staging_dir)
        return {
            "uploaded": file_list,
            "delete_candidates": list(_OBSOLETE_REMOTE_PATHS),
            "stats": stats,
            "dry_run": True,
            "staging_dir": str(staging_dir),
        }

    # Step 3: Upload to HuggingFace Hub
    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF push. Install with: uv sync --extra hub"
        )

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    remote_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if remote_info.id != repo_id:
        raise RuntimeError(
            f"Refusing raw export because {repo_id} resolves to "
            f"{remote_info.id}; redirects are not accepted."
        )
    if remote_info.private is not True:
        raise RuntimeError(
            f"Refusing raw export to non-private repository {repo_id}. "
            "Make the destination private before retrying."
        )
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    deleted = [path for path in _OBSOLETE_REMOTE_PATHS if path in remote_files]

    timestamp = datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    total = stats.get("total_articles", 0)

    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=str(local_path),
        )
        for local_path, repo_path in upload_plan
    ]
    operations.extend(CommitOperationDelete(path_in_repo=path) for path in deleted)

    try:
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"bioreview-bench update: {total} articles ({timestamp})",
        )
    except Exception:
        log.error("Upload failed; staging dir preserved at %s", staging_dir)
        raise

    commit_oid = commit_info.oid

    # Tag the exact commit if requested. A requested tag is part of the release
    # contract: tag failure is surfaced and staging is preserved for diagnosis.
    if version_tag:
        try:
            api.create_tag(
                repo_id=repo_id,
                repo_type="dataset",
                tag=version_tag,
                tag_message=f"bioreview-bench {version_tag}: {total} articles",
                revision=commit_oid,
            )
            log.info("Created tag %s on %s", version_tag, repo_id)
        except Exception:
            log.error(
                "Commit %s uploaded but tag %s failed; staging preserved at %s",
                commit_oid,
                version_tag,
                staging_dir,
            )
            raise

    uploaded = [rp for _, rp in upload_plan]
    log.info("Pushed %d files to %s", len(uploaded), repo_id)

    # Step 4: Clean up
    shutil.rmtree(staging_dir)
    log.info("Cleaned staging dir")

    return {
        "uploaded": uploaded,
        "deleted": deleted,
        "commit_oid": commit_oid,
        "version_tag": version_tag,
        "stats": stats,
        "dry_run": False,
    }


def _add_auxiliary_files(
    data_dir: Path,
    upload_plan: list[tuple[Path, str]],
    version_name: str,
    results_dir: Path | None,
) -> None:
    """Add manifest and frozen-test metadata files to the upload plan."""
    initial_count = len(upload_plan)
    splits_root = data_dir / "splits"

    # Manifests
    manifests_dir = data_dir / "manifests"
    if manifests_dir.exists():
        for f in sorted(manifests_dir.glob("*.json")):
            upload_plan.append((f, f"manifests/{f.name}"))

    # Frozen split IDs for the selected split version.
    seen_metadata: set[str] = set()
    for frozen_name in (
        f"test_ids_frozen_{version_name}.json",
        f"val_ids_frozen_{version_name}.json",
    ):
        for frozen_path in (
            splits_root / frozen_name,
            splits_root / version_name / frozen_name,
        ):
            if frozen_path.exists() and frozen_name not in seen_metadata:
                upload_plan.append((frozen_path, f"metadata/{frozen_name}"))
                seen_metadata.add(frozen_name)

    # Split metadata
    split_meta = splits_root / version_name / f"split_meta_{version_name}.json"
    if split_meta.exists():
        upload_plan.append((split_meta, f"metadata/split_meta_{version_name}.json"))

    # NOTE: update_state.json is intentionally excluded from uploads.
    # It contains internal collection tracking state (all collected IDs)
    # with no value to dataset consumers.

    # Version-matched release artifacts. Keep both the immutable versioned
    # location and the root "current release" mirror synchronized.
    if results_dir is not None:
        root_results_dir = results_dir.parent if results_dir.name == version_name else results_dir
        versioned_leaderboard = results_dir / "leaderboard.json"
        root_leaderboard = root_results_dir / "leaderboard.json"
        root_mirror_is_current = (
            versioned_leaderboard.exists()
            and root_leaderboard.exists()
            and versioned_leaderboard.read_bytes() == root_leaderboard.read_bytes()
        )
        for lb_file in ["leaderboard.md", "leaderboard.json", "release_manifest.json"]:
            lb_path = results_dir / lb_file
            if lb_path.exists():
                upload_plan.append((lb_path, f"results/{version_name}/{lb_file}"))
                root_path = root_results_dir / lb_file
                upload_plan.append(
                    (
                        (root_path if root_mirror_is_current and root_path.exists() else lb_path),
                        f"results/{lb_file}",
                    )
                )

    added = len(upload_plan) - initial_count
    if added:
        log.info("Added %d auxiliary files", added)

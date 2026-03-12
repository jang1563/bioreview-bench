"""HuggingFace Hub dataset push — multi-config JSONL upload.

Generates per-config JSONL files in a staging directory, creates a
DatasetCard README, and uploads everything to HuggingFace Hub in a
single atomic commit.

Usage::

    from bioreview_bench.collect.hf_push import push_to_hub
    push_to_hub(data_dir=Path("data"))

Requires: ``uv sync --extra hub``
"""

from __future__ import annotations

import datetime as _dt
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_REPO_ID = "jang1563/bioreview-bench"
_STAGING_DIR_NAME = ".hf_staging"


def push_to_hub(
    data_dir: Path,
    repo_id: str = _DEFAULT_REPO_ID,
    dry_run: bool = False,
    splits_subdir: str = "splits/v3",
    version_tag: str | None = None,
) -> dict[str, Any]:
    """Export multi-config JSONL files and push to HuggingFace Hub.

    Steps:
        1. Export all configs as JSONL into a staging directory.
        2. Generate the DatasetCard README.md.
        3. Upload via ``huggingface_hub`` in a single commit.
        4. Clean up the staging directory.

    Args:
        data_dir: Root data directory (parent of processed/, splits/, manifests/).
        repo_id: HuggingFace dataset repository ID.
        dry_run: If True, generate files locally but don't upload.
        splits_subdir: Path relative to data_dir for split JSONL files.

    Returns:
        Dict with ``uploaded`` file list, ``stats``, and ``dry_run`` flag.
    """
    from .hf_card import generate_dataset_card
    from .hf_export import export_all_configs

    splits_dir = data_dir / splits_subdir
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
    readme_content = generate_dataset_card(stats)
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
    _add_auxiliary_files(data_dir, upload_plan, splits_subdir)

    log.info("Upload plan: %d files", len(upload_plan))

    if dry_run:
        file_list = [rp for _, rp in upload_plan]
        log.info("Dry run — staging dir preserved at %s", staging_dir)
        return {
            "uploaded": file_list,
            "stats": stats,
            "dry_run": True,
            "staging_dir": str(staging_dir),
        }

    # Step 3: Upload to HuggingFace Hub
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF push. "
            "Install with: uv sync --extra hub"
        )

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    timestamp = datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    total = stats.get("total_articles", 0)

    operations = [
        CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=str(local_path),
        )
        for local_path, repo_path in upload_plan
    ]

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"bioreview-bench update: {total} articles ({timestamp})",
        )
    except Exception:
        log.error("Upload failed; staging dir preserved at %s", staging_dir)
        raise

    # Tag the commit with a version if requested
    if version_tag:
        try:
            api.create_tag(
                repo_id=repo_id,
                repo_type="dataset",
                tag=version_tag,
                tag_message=f"bioreview-bench {version_tag}: {total} articles",
            )
            log.info("Created tag %s on %s", version_tag, repo_id)
        except Exception as e:
            log.warning("Failed to create tag %s: %s", version_tag, e)

    uploaded = [rp for _, rp in upload_plan]
    log.info("Pushed %d files to %s", len(uploaded), repo_id)

    # Step 4: Clean up
    shutil.rmtree(staging_dir)
    log.info("Cleaned staging dir")

    return {"uploaded": uploaded, "stats": stats, "dry_run": False}


def _add_auxiliary_files(
    data_dir: Path,
    upload_plan: list[tuple[Path, str]],
    splits_subdir: str,
) -> None:
    """Add manifest and frozen-test metadata files to the upload plan."""
    initial_count = len(upload_plan)
    version_name = Path(splits_subdir).name
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

    # Update state (collection tracking)
    update_state = data_dir / "update_state.json"
    if update_state.exists():
        upload_plan.append((update_state, "metadata/update_state.json"))

    # Leaderboard files (from results/ directory, sibling of data/)
    results_dir = data_dir.parent / "results"
    for lb_file in ["leaderboard.md", "leaderboard.json", "release_manifest.json"]:
        lb_path = results_dir / lb_file
        if lb_path.exists():
            upload_plan.append((lb_path, f"results/{lb_file}"))

    added = len(upload_plan) - initial_count
    if added:
        log.info("Added %d auxiliary files", added)

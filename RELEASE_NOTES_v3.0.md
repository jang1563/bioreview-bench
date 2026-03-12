# bioreview-bench v3.0 Release Notes

Tag: `v3.0-release`

Release commit:

```text
ec13324240b6db7f4ffc98863c066466699a0951
```

## Summary

This release finalizes the public `v3` benchmark snapshot, evaluation policy,
leaderboard generation rules, and release reproducibility artifacts.

The release freezes:

- dataset split version: `data/splits/v3`
- public benchmark split: `test`
- matching threshold: `0.65`
- matching algorithm: `hungarian`
- figure concerns: excluded from base metrics
- dataset-level aggregation: micro-averaged

## Included Public Results

The default public leaderboard includes these result files:

- `results/v3/haiku_test.json`
- `results/v3/gemini25flash_test_v2.json`
- `results/v3/gpt4omini_test.json`
- `results/v3/llama33_test.json`

Excluded from the default public leaderboard but retained in-repo:

- `results/v3/haiku_test_dedup.json`
- `results/v3/gemini25flash_test.json`
- `results/v3/auc_pr_comparison.json`

## Key Changes

- Fixed dataset evaluation so missing prediction rows are counted against recall.
- Fixed empty-ground-truth precision handling for the public evaluation API.
- Standardized evaluation CLI defaults on `data/splits/v3`.
- Standardized public documentation on `threshold=0.65` and Hungarian matching.
- Added filtered leaderboard generation that excludes `dedup_gt=true` runs and
  keeps only the strongest run per `(tool_name, tool_version)`.
- Backfilled `tool_version` in published `results/v3/*.json` files with exact
  model identifiers.
- Replaced naive UTC timestamps with timezone-aware UTC timestamps in
  `BenchmarkResult`.
- Disabled unused pytest `anyio` plugin auto-loading to stabilize test runs.

## Reproducibility Artifacts

This release adds:

- `results/release_manifest.json`
- `RELEASE_V3.md`
- `scripts/rebuild_release_artifacts.py`

Rebuild command:

```bash
./.venv/bin/python scripts/rebuild_release_artifacts.py \
  --results-dir results/v3 \
  --output-dir results \
  --split test
```

## Validation

Release smoke tests passed with:

```bash
./.venv/bin/python -m pytest -q \
  tests/test_release_artifacts.py \
  tests/test_leaderboard.py \
  tests/test_metrics.py \
  tests/test_runner_aggregation.py \
  tests/test_cli.py
```

Observed result:

```text
33 passed
```

## Reference Documents

- `RELEASE_V3.md`
- `results/release_manifest.json`
- `EVALUATION_PROTOCOL.md`
- `TASK_DEFINITION.md`

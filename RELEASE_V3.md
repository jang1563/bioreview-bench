# bioreview-bench v3 Release Reference

Date: 2026-03-10

This document defines the official public release reference for the current
`v3` benchmark snapshot and its published evaluation artifacts.

## Scope

This release reference covers:

- Dataset split version: `data/splits/v3`
- Public benchmark split: `test`
- Evaluation artifacts in `results/`
- Source evaluation result files in `results/v3/`

## Frozen Evaluation Settings

- Matching model family: SPECTER2
- Matching threshold: `0.65`
- Matching algorithm: `hungarian`
- Figure concerns: excluded from base metrics
- Dataset-level aggregation: micro-averaged

Authoritative details remain in:

- `EVALUATION_PROTOCOL.md`
- `TASK_DEFINITION.md`
- `results/release_manifest.json`

## Public Leaderboard Policy

The default public leaderboard is generated from `results/v3/*.json` with the
following filters:

- Only `split="test"` result files are eligible.
- Files with `dedup_gt=true` are excluded.
- For the same `(tool_name, tool_version)` pair, only the strongest run by
  `f1_micro` is retained.

The public leaderboard artifacts are:

- `results/leaderboard.md`
- `results/leaderboard.json`
- `results/release_manifest.json`

## Rebuild Command

Use this command to regenerate the public release artifacts:

```bash
./.venv/bin/python scripts/rebuild_release_artifacts.py \
  --results-dir results/v3 \
  --output-dir results \
  --split test
```

## Included Public Result Files

As of this release reference, the public ranking includes:

- `results/v3/haiku_test.json`
- `results/v3/gemini25flash_test_v2.json`
- `results/v3/gpt4omini_test.json`
- `results/v3/llama33_test.json`

Excluded but retained in-repo for analysis:

- `results/v3/haiku_test_dedup.json`
- `results/v3/gemini25flash_test.json`
- `results/v3/auc_pr_comparison.json`

## Publication Checklist

Before publishing an updated `v3` leaderboard:

1. Rebuild release artifacts with `scripts/rebuild_release_artifacts.py`.
2. Confirm `results/release_manifest.json` lists the intended included files.
3. Confirm `README.md` leaderboard summary matches `results/leaderboard.md`.
4. Run the release smoke tests.

Suggested smoke test command:

```bash
./.venv/bin/python -m pytest -q \
  tests/test_release_artifacts.py \
  tests/test_leaderboard.py \
  tests/test_metrics.py \
  tests/test_runner_aggregation.py \
  tests/test_cli.py
```

## Tagging Recommendation

If this repository is tagged for release, use a git tag that points to the same
commit as this document and the generated `results/release_manifest.json`.

Recommended tag shape:

```text
v3.0-release
```

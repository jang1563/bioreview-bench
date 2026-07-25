# bioreview-bench v4.1.3 Release Reference

Date: 2026-07-25

This document defines the official public release reference for the current
`v4.1.3` documentation/tooling release over the frozen v4 benchmark snapshot and
its published evaluation artifacts.

## Scope

This release reference covers:

- Dataset split metadata and frozen ID lists in `data/splits/v4` (full JSONL
  data remain restricted)
- Frozen internal evaluation split recorded by the score snapshot: `test`
- Evaluation artifacts in `results/v4/`
- Root release mirrors in `results/`
- Post-hoc non-eLife subset analysis in `results/v4_no_elife/`

## Frozen Evaluation Settings

- Historical wrapper: unadapted `allenai/specter2_base` with automatic mean
  pooling and no recorded SPECTER2 task adapter
- Historical model revision: unrecorded (`null` in manifests)
- Matching threshold: `0.65`
- Matching algorithm: `hungarian`
- Figure concerns: excluded from base metrics
- Dataset-level aggregation: micro-averaged
- Validation status: provisional; the historical wrapper and normalized labels
  have not completed independent human validation

Authoritative details remain in:

- `EVALUATION_PROTOCOL.md`
- `TASK_DEFINITION.md`
- `KNOWN_ISSUES.md`
- `results/v4/release_manifest.json`
- `results/v4/measurement_audit.{json,md}`
- `RELEASE_NOTES_v4.1.3.md`
- `RELEASE_NOTES_v4.1.2.md`
- `NEURIPS_2026_REVIEW_RESPONSE.md`

## Frozen Score-Artifact Policy

The frozen historical score snapshot is generated from `results/v4/*.json` with the
following filters:

- Only `split="test"` result files are eligible.
- Files with `dedup_gt=true` are excluded.
- Files whose method, embedding model/revision, effective threshold, algorithm,
  or figure policy differs from the frozen matcher signature are excluded.
- For the same `(tool_name, tool_version)` pair, only the strongest run by
  `f1_micro` is retained.

The frozen score artifacts are:

- `results/v4/leaderboard.md`
- `results/v4/leaderboard.json`
- `results/v4/release_manifest.json`

The aggregate post-release measurement artifacts are:

- `results/v4/measurement_audit.json`
- `results/v4/measurement_audit.md`

They document matcher sensitivity and the F1000 query-time family-filtering
intervention; they do not replace or rewrite the frozen score table. The
full-600 snapshot is primary, while the no-eLife-450 analysis is a post-hoc
source-subset sensitivity check. All Jaccard operating points are uncalibrated.
The v4.1.3 tooling also computes an aggregate, text-free threshold sweep and an
F1000 train/validation/test DOI-stem matrix when run against the authorized
local inputs. Exact reruns require those local full-schema splits and frozen
tool outputs.

The `results/` directory may mirror the same artifacts for repository-front
consumers.

The Hugging Face v4.1.3 public channel is a clean-history, rights-minimized index:
text-free article metadata for all splits and text-free category/severity/stance
rows for train/validation. Raw publisher content and test targets remain
private. It is not a public test set, scoring service, or open leaderboard.
Its exact package-tree contract includes the canonical Hugging Face
`.gitattributes`; every non-manifest file has a recorded byte count and SHA-256
digest, while the manifest correctly excludes its own self-hash.

## Rebuild Command

Use this command to regenerate the versioned public release artifacts:

```bash
uv run python scripts/rebuild_release_artifacts.py \
  --results-dir results/v4 \
  --output-dir results/v4 \
  --split test
```

## Included Public Result Files

As of this release reference, the frozen score snapshot includes:

- `results/v4/haiku_test_v4.json`
- `results/v4/gpt4omini_test_v4.json`
- `results/v4/gemini25flash_test_v4.json`
- `results/v4/bm25_test_v4.json`
- `results/v4/gemini_flash_lite_test_v4.json`
- `results/v4/llama33_test_v4.json`

Subset and paper-analysis outputs are retained separately:

- `results/v4_no_elife/paper_metrics.json`

## Publication Checklist

Before publishing updated `v4` score-snapshot metadata:

1. Rebuild release artifacts with `scripts/rebuild_release_artifacts.py`.
2. Confirm `results/v4/release_manifest.json` lists the intended included files.
3. Confirm `README.md` frozen score summary matches `results/v4/leaderboard.md`.
4. Run the release smoke tests.

Suggested smoke test command:

```bash
uv run pytest -q \
  tests/test_release_artifacts.py \
  tests/test_leaderboard.py \
  tests/test_metrics.py \
  tests/test_runner_aggregation.py \
  tests/test_cli.py
```

## Tagging Recommendation

If this repository is tagged for release, use a git tag that points to the same
commit as this document and the generated `results/v4/release_manifest.json`.

Release tag:

```text
v4.1.3
```

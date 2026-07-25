# Scripts

Root `scripts/` are a mix of maintenance utilities and research helpers.
The package-supported entrypoints that most users should start with live under
`bioreview_bench/scripts/` and are exposed as console commands such as
`bioreview-run`, `bioreview-baseline`, `bioreview-bm25`, `bioreview-update`,
and `bioreview-stats`.

## Public release maintenance

These are aligned with the current public benchmark release (`v4`):

- `check_release_artifacts.py`: verify checked-in docs and release artifacts match regenerated outputs
- `rebuild_release_artifacts.py`: rebuild `leaderboard.*` and `release_manifest.json`
- `rebuild_splits.py`: build multi-source splits with a write guard on canonical
  `data/splits/v4`; generic ratio runs require an explicit non-canonical output,
  while canonical writes require the exact frozen or balanced v4 configuration
- `category_breakdown.py`: generate per-category comparisons from current release result JSONs
- `compute_auc_pr.py`: sweep thresholds for current-model prediction files
- `manual_review.py`: sample current split entries for human inspection
- `generate_predictions.py`: generate prediction JSONL files from manuscript text
- `temporal_analysis.py`: analyze temporal behavior on the current release data

## Research analysis utilities

These are still useful, but they are experiment-oriented and assume specific
tool output files and result naming conventions:

- `cross_model_validation.py`
- `build_ensemble_gt.py`
- `ensemble_analysis.py`
- `source_analysis.py`

When using these, check the hard-coded model file map near the top of each
script before running a fresh experiment.

For an experimental 70/15/15 split, always choose a non-canonical output:

```bash
uv run python scripts/rebuild_splits.py \
  --input-dir data/processed \
  --output-dir data/splits/experiments/ratio-70-15-15
```

To reproduce the checked-in 5,387/953/600 v4 split, use the exact release
configuration:

```bash
uv run python scripts/rebuild_splits.py \
  --input-dir data/processed \
  --output-dir data/splits/v4 \
  --version v4 --seed 42 --val-ratio 0.15 --usable-only \
  --balanced-test \
  '{"elife":150,"plos":150,"f1000":150,"nature":100,"peerj":50}'
```

The canonical command aborts unless it reproduces the expected
5,387 train / 953 validation / 600 test article counts. Incremental updates use
the existing 600-ID frozen test artifact and abort if it is absent or malformed.

## Historical / one-off helpers

These are narrower workflow helpers and are not part of the public benchmark
release path:

- `create_splits.py`
- `backfill_concerns.py`
- `reprocess_editorial_decision.py`

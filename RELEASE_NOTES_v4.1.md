# bioreview-bench v4.1 Release Notes

Tag: `v4.1-release`
Date: 2026-07-24

## Summary

v4.1 is a post-review transparency and validity-hardening release over the
frozen v4 dataset split. It does not replace the 6,940-article split or current
baseline outputs. It corrects unsupported validation language, narrows the
scientific claims, makes matcher behavior explicit, and adds tooling for the
independent human studies that remain necessary.

The public review result and point-by-point revision record are in
`NEURIPS_2026_REVIEW_RESPONSE.md`.

## Important correction

The earlier claim that a 148-concern manual spot check validated the matching
threshold with Cohen's kappa = 1.000 is withdrawn. The legacy annotation
interface copied model labels into blank human-label fields, so that file is not
independent validation evidence.

As of v4.1:

- concern segmentation, category, severity, and stance remain LLM-derived
  silver annotations;
- the `allenai/specter2_base`/0.65 matcher remains a frozen but unvalidated
  operational setting; and
- leaderboard scores are provisional and matcher-dependent.

## Changes

- Added a public NeurIPS 2026 review record and revision response.
- Reframed the task as overlap with recorded review behavior rather than
  detection of objective scientific truth.
- Corrected the meaning of benchmark precision: matched-to-reference rate, not
  scientific validity.
- Defined Recall@Major and documented the v4 temporal boundary and v3/v4 split
  overlap.
- Added manuscript-version mismatch, possible AI-assisted review text, matcher
  validity, and BM25 topical-overlap confounding to the limitations.
- Added deterministic, system-label-blinded validation-pack tooling for
  independent multi-annotator audits. Source context remains visible; audit
  rates are descriptive unless population weights are supplied.
- Made embedding model/mode explicit and prevented silent metric changes when
  an embedding model cannot load.
- Hardened the Hugging Face publisher to synchronize root and versioned release
  artifacts, remove obsolete internal collection state, and fail closed on
  commit/tag errors.
- Published a clean-history Hugging Face index containing only text-free
  article metadata and train/validation categorical label rows. Raw publisher
  text, normalized concern text, reviewer names/emails, explicit identity
  fields, and test targets remain private. Stable concern IDs may retain
  source-local review ordinals that do not identify a person.
- Redirected the legacy full-data publisher and dependent workflows to the
  verified private raw archive; the public repository ID is now fail-closed.
- Kept the revised manuscript source local-only while aligning public
  package/citation metadata to v4.1.

## Frozen evaluation snapshot

- dataset split: `data/splits/v4`
- public test split: 600 articles / 8,647 normalized concerns
- base reference set after figure exclusion: 8,200 concerns
- matching: `allenai/specter2_base`, cosine threshold 0.65, Hungarian assignment
- ranking: micro-averaged F1

These settings are retained for reproducibility, not endorsed as a validated
semantic-equivalence standard.

## Evidence still required

- Completed system-label-blinded multi-annotator audit of extraction fidelity,
  omissions, category, severity, and stance.
- Completed human-calibrated matcher comparison with confidence intervals and
  rank-stability analysis.
- Source-held-out and topic-controlled BM25 experiments.
- Quantified manuscript-version mismatch.
- Updated model baselines.

# bioreview-bench v4.1.2 Release Notes

Tag: `v4.1.2`
Date: 2026-07-24

## Scope

v4.1.2 is a public-artifact and measurement-audit hotfix over the frozen v4
snapshot. It does not change the 6,940-article split, the six historical result
files, or any reported metric value.

## Public release contract

- The Hugging Face release is a rights-minimized, non-executable index and
  label-distribution artifact.
- Article/manuscript prose, normalized concern text, review/response prose,
  reviewer/data-row names, emails, and explicit identity fields, internal
  traces, and all test targets remain excluded. Project-maintainer contact
  metadata in `CITATION.cff` is outside this data-row identity exclusion.
- There is no public test-scoring or open leaderboard-submission service.
- Test membership, stable article IDs, and DOIs are public. Because source peer
  reviews are publicly retrievable, target withholding is a redistribution
  boundary rather than a secret/blind-test guarantee.
- The preserved six-system table is a frozen 600-article historical snapshot.
  The 450-article non-eLife analysis is a post-hoc source subset.

## Audit disclosures

- The post-release audit found related/versioned F1000 manuscript families
  crossing split boundaries: 42/150 F1000 test articles across 41 families.
  A same-family training record was BM25 rank 1 for all 42 affected queries.
  Query-time candidate filtering, with the original index statistics fixed,
  reduced one-to-one Jaccard BM25 F1 from 0.0690 to 0.0171 on the 450-article
  non-eLife set and from 0.5780 to 0.0361 on the 42 affected articles.
- Re-scoring the same six stored outputs with the lexical matcher changed the
  non-eLife rank order relative to the frozen embedding table
  (Kendall tau-a = -0.20). This is matcher sensitivity, not evidence that the
  lexical matcher is semantically valid.
- The historical matcher loaded the unadapted `allenai/specter2_base`
  checkpoint directly through `SentenceTransformer`. No SPECTER2 task adapter
  was recorded or applied; automatic mean pooling was used, and the resolved
  checkpoint revision was not recorded.
- Human validation is intentionally outside this measurement-audit release.
  The normalized labels and historical matcher remain unvalidated.
- Historical artifacts provide bootstrap confidence intervals for recall and
  precision only. F1 and major-concern recall remain point estimates.
- All six frozen result JSONs retain `"f1_macro": 0.0` as an unpopulated legacy
  sentinel despite non-zero per-category F1 values. Historical category-macro
  F1 is invalid and unreported; v4.1.2 fixes aggregation for future runs only.
- Nested per-category `"f1_macro": 0.0` fields are deprecated schema sentinels;
  `f1_micro` is the usable per-category F1 field.
- Frozen nested per-category `"aucpr": 0.0` values are also unpopulated schema
  sentinels; the single-threshold runs did not compute or report AUPRC.
- The realized split is 5,387 train / 953 validation / 600 test articles
  (77.6% / 13.7% / 8.6%), not 70% / 15% / 15%. Test selection used fixed
  per-source counts; remaining train/validation records were randomly allocated
  within `(source, editorial_decision, review_format)` strata. There was no
  explicit temporal stratification or chronological holdout.

## Packaging and provenance fixes

- Removed the text-bearing article-level error-analysis file from the public
  tree and rewrote ordinary GitHub branch/tag history to remove it together
  with local-only manuscript and session files. GitHub-managed caches or
  archived pull-request refs may persist; this is not a claim of cryptographic
  erasure.
- Aligned package, citation, release-manifest, generated score-snapshot, and
  Hugging Face card metadata to v4.1.2.
- Replaced the unavailable PyPI installation command with an install pinned to
  the GitHub `v4.1.2` tag.
- Added `KNOWN_ISSUES.md` and synchronized it into the public Hugging Face
  package.
- Added aggregate, text-free `results/v4/measurement_audit.{json,md}` artifacts
  and synchronized them into the public Hugging Face package.
- Updated generated cards and score-snapshot footers so future rebuilds retain
  the audited claim boundary.
- Added machine-readable manifest metadata identifying the six invalid legacy
  `f1_macro` sentinel fields and the future-run aggregation definition.

## Unchanged evidence requirements

- Independent system-label-blinded multi-annotator audit.
- Human-calibrated matcher comparison.
- Family-disjoint split and complete reruns.
- Source-held-out and topic-controlled BM25 experiments.
- Updated model baselines.

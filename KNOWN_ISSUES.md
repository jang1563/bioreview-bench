# BioReview-Bench v4.1.3 Known Issues

This file records release-critical limitations of the frozen v4 score artifacts
and the rights-minimized public index. It is part of the public release
contract, not a claim that the underlying studies have been completed.

## Public distribution is not an executable benchmark

The public Hugging Face repository contains text-free article metadata for all
splits and text-free categorical annotations for train and validation. It does
not contain manuscript/article prose, normalized concern text, review or author
response prose, or test targets. There is no public test-scoring service.

Consequently, the public artifact supports metadata and label-distribution
analysis but cannot independently reproduce or extend the frozen test scores.
The evaluation CLI requires an authorized local full-schema copy.

The index publishes test membership together with stable article IDs and DOIs.
Source peer reviews are publicly retrievable, so withholding target rows from
the Hugging Face package does not make the test set secret or blind. The frozen
test snapshot must not be used for new blind evaluation or as evidence of
hidden-test generalization.

## Git history cleanup boundary

The v4.1.2 publication rewrote the ordinary branch and tag history to remove
local-only manuscript files, internal session notes, and a text-bearing
article-level error-analysis artifact. A subsequent audit found that
GitHub-managed pull-request refs in that repository network still retained
deleted objects. For v4.1.3, that original network was made private and renamed
as a legacy containment repository. A first replacement candidate imported
selected historical refs, but a prepublication content audit found non-release
local-path and session-note fragments embedded inside an old tracked script;
that candidate was also contained privately. The canonical public repository
was therefore recreated as a single-root v4.1.3 snapshot with no earlier
commits, historical tags, or pull-request refs. The public replacement for
article-level diagnostics is the aggregate, text-free measurement audit.

This isolation does not claim cryptographic erasure from the private legacy
networks, cached object storage, or third-party clones. Both legacy
repositories must remain private unless GitHub confirms server-side
dereferencing and garbage collection. A verified local bundle preserves the
pre-rewrite research history outside the public release.

## Split-family and cross-release leakage risk

The frozen split guarantees disjoint article IDs, but the post-release audit
found that 42 of 150 F1000 test articles, spanning 41 version-independent DOI
families, have related/versioned training records. A same-family training
article was BM25 rank 1 for all 42 affected queries.

The train-only finding is not a complete split-family audit. The v4.1.3
F1000 DOI-stem scan finds 115 train–validation, 41 train–test, and 9
validation–test crossing families; 161 unique F1000 DOI-stem families cross at
least one partition boundary. Development data (train or validation) overlap
49/150 F1000 test articles across 48 families. These counts are an aggregate,
source-specific lower bound: the resolver does not detect fuzzy-title,
preprint-to-journal, or cross-source manuscript relations.

Holding the original BM25 corpus, postings, IDF, and average document length
fixed, query-time same-family candidate filtering reduced one-to-one Jaccard
F1 from 0.0577 to 0.0148 on the primary full-600 snapshot. The corresponding
post-hoc 450-article non-eLife result is 0.0690 to 0.0171, and the affected-42
result is 0.5780 to 0.0361. The intervention reconstructed all 42 targeted
frozen BM25 rows exactly before filtering; it did not reconstruct all 600
predictions. These are lexical sensitivity results at the uncalibrated
effective threshold 0.195, not corrected values for the historical embedding
table, a family-disjoint re-split, or a rerun of any LLM. Treat the existing
scores as a historical snapshot, not as contamination-free evidence of
out-of-family generalization. See
`results/v4/measurement_audit.{json,md}`.

Cross-release overlap is known and quantified: 439 of the 600 v4 test articles
were in v3 train, and all 600 appeared somewhere in the v3 split universe.
Systems trained or tuned on v3 data must not be compared directly with the
frozen v4 table.

## Withdrawn v3 pairwise p-values

The original `results/v3/pairwise_significance.json` used a paired bootstrap
that resampled observed article pairs and counted sign crossings without
constructing the null distribution. Those values were not valid two-sided
null p-values, and several were reported as exactly zero. v4.1.3 withdraws the
numeric artifact and replaces the executable path with an article-paired
label-swap randomization test that recomputes dataset-level micro-F1 and uses a
plus-one correction for Monte Carlo estimates. No corrected v3 pairwise
p-values are claimed in this release.

The realized frozen split is 5,387 train / 953 validation / 600 test articles
(77.6% / 13.7% / 8.6%), not 70% / 15% / 15%. The test set was sampled at fixed
per-source counts (150 eLife, 150 PLOS, 150 F1000Research, 100 Nature, and 50
PeerJ). The remaining records were randomly shuffled within
`(source, editorial_decision, review_format)` strata before train/validation
allocation. No explicit temporal stratification or chronological holdout was
used.

## Historical matcher provenance and validity

The frozen scores loaded the unadapted `allenai/specter2_base` checkpoint
directly through `SentenceTransformer`. No SPECTER2 task adapter was recorded
or applied; the wrapper created automatic mean pooling. The resolved checkpoint
revision was also not recorded. The stored model ID therefore must not be read
as a correctly specified SPECTER2 embedding recipe.

The historical wrapper used cosine threshold 0.65 and threshold-aware
Hungarian matching. Neither the wrapper nor threshold has completed independent
human-equivalence validation. Human validation is intentionally outside the
scope of this v4.1.3 measurement-audit release and remains future work. Scores
and small rank differences are therefore provisional and matcher-dependent.

## Matcher operating points are uncalibrated

The effective Jaccard threshold 0.195 is the historical heuristic
`0.65 × 0.3`, not a semantic calibration. v4.1.3 adds a deterministic,
aggregate threshold sweep with per-system match counts, F1, rank, and Kendall
tau-a. Rank trajectories across that sweep diagnose operating-point
sensitivity; they do not identify a correct matcher or validate either
threshold. The frozen embedding ranking is imported from aggregate artifacts
and its own 0.65 cutoff cannot be retrospectively validated from the public
index.

## 600-article snapshot versus 450-article analysis

The six-system score table is the primary frozen 600-article v4 snapshot. The
450-article non-eLife analysis is a post-hoc source-subset diagnostic. It is not
the official snapshot, a replacement test split, or a basis for reranking the
frozen table.

## Confidence intervals

The checked-in historical result files contain bootstrap intervals for recall
and precision only. F1 and major-concern recall are point estimates without
released confidence intervals.

## Historical category-macro F1 is invalid and unreported

All six frozen test-result JSON files store `"f1_macro": 0.0`, but their
per-category F1 values are non-zero. The overall field was left at a legacy
schema default rather than populated by the historical runner. It is therefore
an invalid sentinel, not a measured zero and not a reported historical metric.

v4.1.2 implemented the unweighted mean of represented-category F1 values for new
runs only. The frozen raw result JSONs and reported metric values remain
unchanged, and no corrected historical category-macro score is released.
Nested per-category `"f1_macro": 0.0` fields are also deprecated schema
sentinels; the usable per-category F1 field is `f1_micro`. Nested per-category
`"aucpr": 0.0` fields are likewise unpopulated sentinels: a single-threshold
run did not compute or report an AUPRC.

## Required interpretation

The normalized concerns and categorical labels are LLM-derived silver
annotations, not expert-adjudicated scientific truth. Reported precision means
matched-to-reference rate under the historical matcher, not scientific validity.
See `EVALUATION_PROTOCOL.md`, `LIMITATIONS_AND_ETHICS.md`, and
`NEURIPS_2026_REVIEW_RESPONSE.md` for the full claim boundary.

# NeurIPS 2026 Review Record and Revision Response

Date of this response: 2026-07-25
Submission: 1932
Public review forum: <https://openreview.net/forum?id=p0ZD90tQsu>

## Review result

This repository publishes the received review record rather than hiding an
unfavorable result. The three initial ratings were all **2/6 (Reject)**, with
reviewer confidences **4, 4, and 3**. The initial area-chair assessment described
a clear rejection consensus. This is the review-stage result, not the final
conference notification.

The reviews also consistently recognized value in the concern-level task, the
scale and multi-source biomedical corpus, and the inclusion of a strong BM25
baseline.

## Main validity concerns

### 1. The normalized targets lack independent human validation

Source platforms published the source prose as peer review, but an LLM
segmented it into concern units and assigned category, severity, and stance
labels. Possible AI assistance in individual recent reviews is unknown. The
released benchmark did not independently validate those transformations.

An earlier 148-row CSV does not solve this problem. Its `human_*` fields are
identical to the model labels, and the legacy interface copied the model label
when an annotator left a field blank. We therefore withdraw the associated
claim of Cohen's kappa = 1.000.

**v4.1 response:** documentation now calls these fields silver-standard, the
unsupported validation claim is removed, and system-label-blinded
multi-annotator audit tooling is included. Source identity and article context
remain visible to raters. The audit itself remains to be completed; adding
tooling is not presented as validation evidence.

### 2. The semantic matcher is not validated

The 0.65 cosine threshold for `allenai/specter2_base` was not calibrated against
independent human judgments of concern equivalence. Score distributions and
threshold sweeps show sensitivity, not semantic validity.

**v4.1.2 audit update:** the historical implementation loaded the unadapted
`allenai/specter2_base` checkpoint directly through `SentenceTransformer`.
No SPECTER2 task adapter was recorded or applied; the wrapper supplied
automatic mean pooling, and the resolved checkpoint revision was not recorded.
The stored model ID therefore does not establish that a correctly specified
SPECTER2 embedding recipe was used. The 0.65/Hungarian scores are retained only
as a provisional historical snapshot. Lexical matching remains a diagnostic
mode, not validation.

Human validation is intentionally not part of this measurement-audit release.
A powered, independently labeled matcher comparison remains future work; the
v4.1.2 release makes no claim that it has been completed.

The six frozen result JSONs also retain `f1_macro=0.0` as an unpopulated legacy
sentinel despite non-zero per-category F1. Historical category-macro F1 is
invalid and unreported; v4.1.2 fixes future aggregation only and does not
rewrite the frozen result files.

### 3. The BM25 result is open to alternative explanations

Strong lexical retrieval can reflect topic vocabulary, same-source structure,
normalization templates, or the matcher's own lexical sensitivity. It does not
by itself show that substantive reviewer concerns are predictable from article
content.

**v4.1.2 audit update:** causal language remains withdrawn. BM25 is a
competitive lexical control only. Source-held-out, topic-controlled, and
raw-versus-normalized experiments remain required follow-up evidence and are
not presented as completed controls in this release.

### 4. F1000 manuscript-family leakage

The frozen split is disjoint by article ID, but the v4.1.2 audit found
42 of 150 F1000 test articles, across 41 version-independent DOI families, have
related/versioned training records. A same-family training article was BM25
rank 1 for all 42 affected test queries. This is a leakage route that
article-ID checks do not detect.

**v4.1.3 hardening:** the validation-aware DOI-stem matrix additionally finds
115 train–validation, 41 train–test, and 9 validation–test crossing families.
Development data overlap 49/150 F1000 test articles across 48 families. This
resolver is a source-specific lower bound and does not detect fuzzy-title,
preprint-to-journal, or cross-source manuscript relations.

With the original BM25 corpus, postings, IDF, and average document length held
fixed, query-time same-family candidate filtering reduced one-to-one Jaccard F1
from 0.0577 to 0.0148 on the primary full-600 snapshot. The corresponding
post-hoc results are 0.0690 to 0.0171 on the 450-article non-eLife audit set and
0.5780 to 0.0361 on the 42 affected articles. All 42 targeted frozen BM25 rows
were reconstructed exactly before the intervention; this is not a claim that
all 600 predictions were reconstructed. The same six stored system outputs
also had Kendall tau-a = -0.20 between the frozen embedding and lexical rank
orders on the non-eLife subset at the uncalibrated effective Jaccard threshold
0.195. v4.1.3 adds a deterministic aggregate threshold sweep to expose
operating-point sensitivity; no swept cutoff is selected or validated. These
are measurement diagnostics, not corrected historical embedding scores, a
family-disjoint replacement split, or completed human validation. The frozen
results must not be interpreted as contamination-free out-of-family
generalization.

The realized split is 5,387 train / 953 validation / 600 test articles, not
70% / 15% / 15%. The 600 test articles were selected at fixed per-source
counts; remaining train/validation allocation used seeded random shuffling
within `(source, editorial_decision, review_format)` strata. No explicit
temporal stratification or chronological holdout was used.

The preserved six-system score table covers the original 600-article snapshot.
The 450-article non-eLife analysis is a post-hoc source subset, not the official
snapshot, a replacement leakage-controlled split, or a basis for reranking the
historical table.

## Additional reviewer requests

- Rewrite the paper for readability; define internal experiment labels and
  Recall@Major.
- Distinguish matching recorded reviewer behavior from identifying objective
  scientific flaws.
- Explain the operational taxonomy and its provisional status.
- Report quality control across sources and categories, not selected examples.
- Document model identifiers, provider access dates, split overlap, and temporal
  boundaries.
- Quantify mismatch between the manuscript version reviewed by peers and the
  later text supplied to benchmarked models.
- Disclose that recent published review prose may itself be AI-assisted.
- Keep source-specific and model-pair analyses clearly labeled as post-hoc
  diagnostics rather than substitutes for the frozen 600-article score
  snapshot.

## What v4.1, v4.1.2, and v4.1.3 change—and do not change

v4.1 improves claim boundaries, configuration provenance, validation tooling,
documentation, and the local manuscript. v4.1.2 is a public-artifact and
measurement audit: it discloses the F1000 family leakage, corrects the
historical matcher description, and narrows the public release to a
rights-minimized non-executable index plus a frozen score snapshot. It does
not rerun the systems. v4.1.3 withdraws the original v3 pairwise p-values,
whose sign-crossing bootstrap did not construct a valid null distribution,
and replaces the executable path with article-paired label swaps that
recompute dataset-level micro-F1. It also adds the validation-aware DOI-stem
matrix and deterministic aggregate threshold-sensitivity tooling and enforces
the exact public package tree. It does **not** rerun the six systems or claim
to have completed the two studies requested by all reviewers:

1. a system-label-blinded, multi-annotator audit of concern fidelity,
   omissions, category, severity, and stance; and
2. a human-calibrated comparison of semantic matchers.

Those studies, source-held-out/topic-controlled BM25 controls, a
family-disjoint re-split, and updated model baselines are evidence needed for a
substantially stronger future submission. None is represented as completed by
the v4.1.3 measurement-audit release.

# bioreview-bench Task Definition

> Version: 1.3
> Date: 2026-07-24

This document fully defines the evaluation task for the bioreview-bench benchmark.
Both tool developers and evaluators must follow this specification.
The public Hugging Face artifact is a rights-minimized, non-executable index.
Scoring requires an authorized local full-schema copy; the project does not
operate an open public test-scoring service or leaderboard.

---

## 1. Task Overview

**Task name**: Biomedical Peer Review Concern Detection

**Question**: How many concern units recorded by this benchmark from published
biomedical peer reviews does an AI review tool identify?

This is a behavioral overlap task. It does not determine whether a generated or
recorded concern is scientifically correct, and the benchmark reference units
are LLM-normalized silver annotations rather than expert-adjudicated truth.

---

## 2. Input Specification (Information Provided to AI Tools)

### 2.1 Permitted inputs

| Input | Format | Allowed |
|-------|--------|---------|
| Article title | plain text | Yes |
| Abstract | plain text | Yes |
| Body sections (Introduction, Methods, Results, Discussion) | plain text | Yes |
| Reference list | structured list | Yes |
| Journal name | plain text | Yes |
| Submission date / publication year | date | Yes |

### 2.2 Prohibited inputs (test-time leakage prevention)

| Prohibited information | Reason |
|------------------------|--------|
| Peer review text (decision letter) | Directly exposes scoring criteria |
| Author response letter | Exposes concern resolution information |
| Editor comments | Summarises reviewer concerns |
| Output from other AI tools | Prevents ensembling |
| Previous revision information | Indirectly reveals review content |
| Manuscript revision history | Same reason |

### 2.3 Standard input format (JSON)

```json
{
  "article_id": "elife:84798",
  "doi": "10.7554/eLife.84798",
  "journal": "eLife",
  "title": "Title of the paper",
  "abstract": "Abstract text...",
  "sections": {
    "introduction": "Introduction text...",
    "methods": "Methods text...",
    "results": "Results text...",
    "discussion": "Discussion text..."
  },
  "references": [
    {"authors": "...", "year": 2022, "title": "...", "journal": "..."}
  ]
}
```

---

## 3. Output Specification (Information Expected from AI Tools)

### 3.1 Standard output format

```json
[
  "The sample size in experiment 3 is insufficient for the claimed statistical power.",
  "Missing positive control in Figure 2B."
]
```

The CLI also accepts concern objects with `text` (or the legacy
`concern_text` alias). It normalizes either representation to concern-text
strings before scoring.

### 3.2 Field definitions

| Field | Type | Required | Valid values | Description |
|-------|------|----------|-------------|-------------|
| concern string, or object `text` / `concern_text` | string | Yes | Non-empty text | Specific description of the concern |
| object `category` | string | No | See category list below | Optional metadata; discarded and unscored by the runner |
| object `severity` | string | No | `major` \| `minor` \| `optional` | Optional metadata; discarded and unscored by the runner |

The frozen scoring contract is text-only. It does not evaluate tool-predicted
category or severity. Category breakdowns use reference-side attribution as
specified in `EVALUATION_PROTOCOL.md`: matched tool concerns inherit the matched
reference category, and unmatched tool concerns are assigned to the nearest
reference category within the article.

### 3.3 Permitted optional category metadata

| Category | Description | Example |
|----------|-------------|---------|
| `design_flaw` | Fundamental problems in experimental design | "No proper control for batch effect in multi-lab study" |
| `statistical_methodology` | Statistical method errors or gaps | "Multiple comparisons not corrected for" |
| `missing_experiment` | Key experiment needed to support claims is absent | "Rescue experiment needed to confirm causality" |
| `prior_art_novelty` | Missing prior work or overstated novelty claims | "Similar result was shown in Smith 2019" |
| `writing_clarity` | Unclear writing, missing definitions, logical gaps | "Figure 3 legend is incomplete" |
| `reagent_method_specificity` | Insufficient detail on materials/methods | "Antibody lot number and dilution not specified" |
| `interpretation` | Data interpretation errors or overclaiming | "Correlation presented as causation without mechanistic evidence" |
| `other` | Concerns not fitting the above categories | |

**Excluded category in the current base metric:**
- `figure_issue`: Concerns requiring visual figure inspection (included in the dataset but excluded from base metrics)

### 3.4 Concern count limits

- Minimum: 0 (an empty list is accepted and receives zero recall)
- Maximum: none (precision metric controls excessive flagging)
- Recommended: 3-15 per article (typical range for human reviewers)

---

## 4. Scoring Rules

### 4.1 Historical v4 scoring wrapper: threshold-aware bipartite matching

Threshold-aware bipartite matching between tool concerns and benchmark
reference concerns. The frozen implementation maximizes the number of eligible
one-to-one pairs, then total similarity:

```
1. Embed all tool concern texts with the historical direct
   `SentenceTransformer("allenai/specter2_base")` load
2. Embed all reference concern texts with the same wrapper
3. Compute N x M cosine similarity matrix
4. Hungarian bipartite matching (eligible-pair cardinality first, then total similarity; each concern matched at most once)
5. Matched pairs with similarity >= threshold = "matched"
6. Recall = matched / |recorded reference concerns|
7. Precision = matched / |tool concerns|
```

`SentenceTransformer` automatically supplied mean pooling. No SPECTER2 task
adapter or resolved checkpoint revision was recorded. The stored checkpoint
label therefore does not establish that a correctly specified, validated, or
currently recommended SPECTER2 method was used. A differently adapted or pooled
encoder is a new matcher and requires separate evaluation.

### 4.2 Threshold

- **Historical frozen setting**: 0.65 (cosine similarity from the wrapper above)
- **Status**: frozen but provisional; independent human equivalence validation
  has not been completed
- Threshold is fixed per release and published in `EVALUATION_PROTOCOL.md`

The historical 148-row annotation file is not valid independent evidence
because blank human fields were prefilled with model labels. Its previously
reported kappa is withdrawn. Scores must be described as matcher-dependent
until the system-label-blinded two-annotator validation study is complete.

### 4.3 Figure concern handling

- Human concerns with `category: figure_issue` are **excluded from base metrics**
- Figure-reference concerns are removed before matching. Tool concerns remain
  in the precision denominator; there is no separate automatic tool-side
  figure filter.

### 4.4 Severity weighting

The current base metric uses unweighted scoring (all recorded reference
concerns are treated equally).
Major-only recall is reported separately as a secondary metric.

---

## 5. Evaluation Metrics

| Metric | Description | Reporting level |
|--------|-------------|----------------|
| `recall` | Fraction of recorded reference concerns matched | **Primary** |
| `recall_major` | Fraction of non-figure reference concerns labelled `major` that are matched | **Primary** |
| `precision` | Fraction of tool concerns matched to a recorded reference concern | **Primary** |
| `f1` | Concern-level harmonic mean | **Primary** |
| `f1_macro` | Unweighted mean of represented-category F1 values | Secondary; future runs |
| Bootstrap 95% CI (n=1000) | Recall and precision only in the historical release | Required where reported |

The frozen historical score snapshot orders its recorded systems by
dataset-level micro-averaged F1 (`f1_micro`) as defined in
`EVALUATION_PROTOCOL.md`. It is not an open public leaderboard. Historical
micro-F1 and Recall@Major are point estimates; confidence intervals were
released only for recall and precision. The six frozen result JSONs did not
populate overall `f1_macro`: their stored `0.0` values are invalid legacy
sentinels, not measured zeros. v4.1.2 computes category-macro F1 for new runs
only and does not backfill a historical value.

`precision` is not a scientific-validity judgment. A correct concern absent
from the published review is counted as unmatched, while a matched reference
concern can still be contestable.

---

## 6. Data Split Policy

| Split | Frozen articles | Realized share | Purpose |
|-------|----------------:|---------------:|---------|
| train | 5,387 | 77.6% | Tool development and fine-tuning |
| validation | 953 | 13.7% | Historical threshold fixing and hyperparameter tuning |
| test | 600 | 8.6% | Frozen historical score snapshot |

**Split unit**: Article level (splitting by concern is prohibited — concerns from the same article must not appear in different splits to prevent leakage)

**Frozen construction**: Test selection used fixed per-source counts: 150
eLife, 150 PLOS, 150 F1000Research, 100 Nature, and 50 PeerJ. The remaining
records were randomly shuffled within
`(source, editorial_decision, review_format)` strata; approximately 15% of each
eligible stratum was assigned to validation and the rest to train. Subject
area, concern category, resolution, and publication time were not
stratification keys. There was no explicit chronological holdout.

**Known exception**: Article IDs are split-disjoint, but a post-release audit
found related/versioned training records for 42/150 F1000 test articles across
41 families. A same-family record is BM25 rank 1 for all 42 affected queries;
query-time family filtering reduces lexical-audit BM25 F1 from 0.0690 to
0.0171 on the 450-article non-eLife set. The frozen split is therefore not
family-disjoint, and its historical embedding scores must not be interpreted
as corrected or as contamination-free out-of-family generalization. See
`results/v4/measurement_audit.md`.

---

## 7. Authorized Local Result Record

```json
{
  "tool_name": "MyReviewTool",
  "tool_version": "1.2.3",
  "git_hash": "abc123",
  "split": "test",
  "predictions": {
    "elife:84798": [
      "The sample size is insufficient for the claimed statistical power."
    ],
    "elife:84799": [...]
  }
}
```

This is a provenance format for authorized local evaluation, not a public
leaderboard-submission interface. Future comparable records must also preserve
the exact matcher identifier, resolved revision, pooling/wrapper, adapter,
threshold, algorithm, and figure policy.

---

## 8. Prohibited Practices

- Using test split articles for validation purposes
- Using test-set reference concern texts as training data
- Resubmitting predictions for the same articles after viewing scores (cherry-picking)
- Using external peer review data for test articles
- Accessing peer review materials or author responses for test articles at inference time

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-24 | v1.3 — marked historical category-macro F1 invalid/unreported and corrected frozen split construction, proportions, and lack of temporal holdout |
| 2026-07-24 | v1.2 — documented the non-executable public boundary, historical matcher wrapper, future-run category-macro definition, released-CI scope, and F1000 family leakage |
| 2026-07-24 | v1.1 — clarified behavioral target and withdrew unsupported matcher-validation claim |
| 2026-03-01 | v1.0 — initial public release |

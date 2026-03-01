# bioreview-bench Task Definition

> Version: 1.0
> Date: 2026-03-01

This document fully defines the evaluation task for the bioreview-bench benchmark.
Both tool developers and evaluators must follow this specification.

---

## 1. Task Overview

**Task name**: Biomedical Peer Review Concern Detection

**Question**: How many of the substantive concerns raised by human peer reviewers
does an AI review tool successfully identify?

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
  {
    "text": "The sample size in experiment 3 is insufficient for the claimed statistical power.",
    "category": "statistical_methodology",
    "severity": "major"
  },
  {
    "text": "Missing positive control in Figure 2B.",
    "category": "missing_experiment",
    "severity": "minor"
  }
]
```

### 3.2 Field definitions

| Field | Type | Required | Valid values | Description |
|-------|------|----------|-------------|-------------|
| `text` | string | Yes | Non-empty, 10-500 characters | Specific description of the concern |
| `category` | string | Yes | See category list below | Concern type classification |
| `severity` | string | Yes | `major` \| `minor` \| `optional` | Concern severity |

### 3.3 Permitted categories

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

**Excluded category** (v1.0):
- `figure_issue`: Concerns requiring visual figure inspection (included in the dataset but excluded from base metrics)

### 3.4 Concern count limits

- Minimum: 1
- Maximum: none (precision metric controls excessive flagging)
- Recommended: 3-15 per article (typical range for human reviewers)

---

## 4. Scoring Rules

### 4.1 Scoring method: Bipartite Matching + SPECTER2

Maximum-weight bipartite matching between tool concerns and human concerns:

```
1. Embed all tool concern texts using SPECTER2
2. Embed all human concern texts using SPECTER2
3. Compute N x M cosine similarity matrix
4. Greedy bipartite matching (highest-similarity pairs first; each concern matched at most once)
5. Matched pairs with similarity >= threshold = "matched"
6. Recall = matched / |human concerns|
7. Precision = matched / |tool concerns|
```

### 4.2 Threshold

- **Default**: 0.65 (SPECTER2 cosine similarity)
- **Determination**: Validated on 20 held-out articles (148 concerns) with manual spot-check; Cohen's kappa = 1.000
- Threshold is fixed per release and published in `EVALUATION_PROTOCOL.md`

### 4.3 Figure concern handling

- Human concerns with `category: figure_issue` are **excluded from base metrics**
- Tool output on figure-related concerns is also excluded (no precision impact)

### 4.4 Severity weighting

v1.0 uses unweighted scoring (all concerns treated equally).
Major-only recall is reported separately as a secondary metric.

---

## 5. Evaluation Metrics

| Metric | Description | Reporting level |
|--------|-------------|----------------|
| `recall` | Overall human concern coverage | **Primary** |
| `recall_major` | Major-severity concern coverage | **Primary** |
| `precision` | Fraction of tool concerns that are valid | **Primary** |
| `f1` | Concern-level harmonic mean | **Primary** |
| `f1_macro` | Category-level macro F1 | Secondary |
| Bootstrap 95% CI (n=1000) for all primary metrics | | Required |

---

## 6. Data Split Policy

| Split | Ratio | Purpose |
|-------|-------|---------|
| train | 70% | Tool development and fine-tuning |
| validation | 15% | Threshold fixing, hyperparameter tuning |
| test | 15% | Final benchmark score (single evaluation only) |

**Split unit**: Article level (splitting by concern is prohibited — concerns from the same article must not appear in different splits to prevent leakage)

**Stratification**: By source journal and subject area

---

## 7. Submission Format (Leaderboard Submissions)

```json
{
  "tool_name": "MyReviewTool",
  "tool_version": "1.2.3",
  "git_hash": "abc123",
  "split": "test",
  "predictions": {
    "elife:84798": [
      {"text": "...", "category": "statistical_methodology", "severity": "major"}
    ],
    "elife:84799": [...]
  }
}
```

---

## 8. Prohibited Practices

- Using test split articles for validation purposes
- Using human concern texts as training data
- Resubmitting predictions for the same articles after viewing scores (cherry-picking)
- Using external peer review data for test articles
- Accessing peer review materials or author responses for test articles at inference time

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-01 | v1.0 — initial public release |

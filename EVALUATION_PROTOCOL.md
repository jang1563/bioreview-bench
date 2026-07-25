# bioreview-bench Evaluation Protocol

> Version: 1.6
> Date: 2026-07-25

This document fully defines the evaluation procedure for bioreview-bench.
Metric computation, threshold fixing, and statistical testing follow the specifications below.
Changes to this protocol after release require a major version bump.
The public Hugging Face artifact is a rights-minimized, non-executable index;
this protocol applies to authorized local full-schema copies. See
`KNOWN_ISSUES.md`.

---

## 1. Matching Model

**Historical checkpoint label**: `allenai/specter2_base`
**Historical wrapper**: direct `SentenceTransformer` load with automatically
created mean pooling; no SPECTER2 task adapter was recorded or applied
**Input**: Full concern text (max 512 tokens; longer texts truncated from the end)
**Output**: 768-dimensional vector
**Normalisation**: L2 normalisation followed by cosine similarity

```python
from sentence_transformers import SentenceTransformer

# Reconstruction of the historical wrapper, not a recommended SPECTER2 setup.
EMBED_MODEL = SentenceTransformer("allenai/specter2_base")

def embed(texts: list[str]) -> np.ndarray:
    embeddings = EMBED_MODEL.encode(texts, normalize_embeddings=True)
    return embeddings  # shape: (N, 768)
```

The historical runs loaded the base checkpoint without a SPECTER2 task adapter.
`SentenceTransformer` therefore supplied automatic mean pooling. The
stored model ID must not be read as evidence that the correctly adapted
SPECTER2 embedding recipe was used. Embedding load failures must not silently
change a run to lexical matching. Jaccard is available only as an explicitly
selected diagnostic mode and is not score-compatible with the frozen score
snapshot.

**Version pinning**: new runs should pass `--embedding-revision` with an exact
Hub commit and record the adapter and pooling configuration with the model
identifier. The historical v4 runs did not preserve the resolved commit, so
their manifests use `"embedding_revision": null`; v4.1.2 does not invent a
revision after the fact. A correctly adapted or differently pooled encoder is
a new matcher and requires re-evaluation.

---

## 2. Concern Matching Algorithm

### 2.1 Hungarian Bipartite Matching (default)

Tool concern set T = {t1, t2, ..., tm}
Reference concern set H = {h1, h2, ..., hn} (`figure_issue` concerns excluded)

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

def match_concerns(
    tool_concerns: list[str],
    reference_concerns: list[str],
    threshold: float,
) -> tuple[int, list[tuple[int, int, float]]]:
    """
    Returns:
        n_matched: Number of matched pairs (similarity >= threshold)
        matched_pairs: [(tool_idx, human_idx, similarity), ...]
    """
    if not tool_concerns or not reference_concerns:
        return 0, []

    t_emb = embed(tool_concerns)   # (M, 768)
    h_emb = embed(reference_concerns)  # (N, 768)

    sim_matrix = t_emb @ h_emb.T  # (M, N), cosine similarity

    # Frozen threshold-aware objective: maximize the number of eligible pairs,
    # then maximize total similarity among assignments with that cardinality.
    cost = 1.0 - sim_matrix
    cost[sim_matrix < threshold] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        sim = float(sim_matrix[r, c])
        if sim >= threshold:
            matched_pairs.append((r, c, sim))

    return len(matched_pairs), matched_pairs
```

The official public release uses `algorithm="hungarian"` throughout. The repository
retains a legacy greedy matcher only for ablation or compatibility checks.

**Rationale**: the frozen implementation prioritizes the number of
above-threshold one-to-one matches and uses total similarity as the secondary
objective. This is not the same as unconstrained maximum-weight assignment
followed by filtering. Complexity is O(n^3) but negligible for typical concern
counts (<= 50 per article).

### 2.2 Figure concern handling

```python
def filter_figure_concerns(
    human_concerns: list[ReviewerConcern],
) -> tuple[list[ReviewerConcern], list[ReviewerConcern]]:
    """Returns (non_figure, figure_only)"""
    non_figure = [c for c in human_concerns if not c.requires_figure_reading]
    figure_only = [c for c in human_concerns if c.requires_figure_reading]
    return non_figure, figure_only
```

Base metrics are computed on `non_figure` concerns only.
`figure_only` concerns are reported separately.

---

## 3. Threshold Fixing Procedure

### 3.1 Current status

The released historical threshold is **0.65** for cosine similarity from the
unadapted, auto-mean-pooled `allenai/specter2_base` wrapper described above. It
is a frozen operational setting, not a human-validated
semantic-equivalence boundary or a correctly specified SPECTER2 recipe.

An earlier 148-row internal CSV cannot support a validation claim: the legacy
annotation interface copied model labels into blank human-label fields, and the
file therefore does not constitute system-label-blinded, independent
annotation. The previously reported Cohen's kappa of 1.000 is withdrawn.

Human validation is intentionally outside the scope of the v4.1.3
measurement-audit release. Until the planned two-annotator study described in
`LIMITATIONS_AND_ETHICS.md` is completed in a future study, all published
frozen scores should be
reported as **provisional and matcher-dependent**. Threshold sweeps and lexical
controls test sensitivity but do not replace human validation of whether two
concerns are substantively equivalent.

The effective Jaccard threshold `0.195` used by the lexical audit is the
historical heuristic `0.65 × 0.3`, not a calibrated equivalence threshold.
v4.1.3 adds a deterministic aggregate threshold sweep over a declared grid.
The sweep reports per-system match counts, F1, rank, and Kendall tau-a without
emitting concern or article text. It is a sensitivity diagnostic only: no
threshold in the grid is selected or recommended from test performance. The
full 600-article test snapshot is the primary population; the 450-article
non-eLife subset is post-hoc.

### 3.2 Application to test split

- The frozen threshold is applied unchanged to the test split (no re-optimisation)
- Post-release threshold changes require a minor version bump and full re-evaluation
- Result comparisons are valid only when model, model revision, threshold,
  algorithm, figure policy, and matching mode are identical

### 3.3 Record format

```json
{
  "eval_version": "1.2",
  "matching_mode": "embedding",
  "embedding_model": "allenai/specter2_base",
  "embedding_revision": "record-resolved-commit-here",
  "threshold": 0.65,
  "threshold_status": "provisional_frozen_operational_setting",
  "threshold_locked_date": "2026-03-01",
  "matching_algorithm": "hungarian",
  "ranking_metric": "f1_micro"
}
```

---

## 4. Metric Computation

### 4.1 Article-level metrics

```python
def compute_article_metrics(
    article_id: str,
    tool_concerns: list[str],
    reference_concerns: list[ReviewerConcern],
    threshold: float,
) -> dict[str, str | float | int]:
    non_fig, _ = filter_figure_concerns(reference_concerns)
    n_matched, pairs = match_concerns(
        tool_concerns, [c.concern_text for c in non_fig], threshold
    )

    n_human = len(non_fig)
    n_tool = len(tool_concerns)

    recall = n_matched / n_human if n_human > 0 else 0.0
    precision = n_matched / n_tool if n_tool > 0 else 0.0
    f1 = (2 * recall * precision / (recall + precision)
           if (recall + precision) > 0 else 0.0)

    return {"article_id": article_id,
            "recall": recall, "precision": precision, "f1": f1,
            "n_matched": n_matched, "n_human": n_human, "n_tool": n_tool}
```

### 4.2 Dataset-level aggregation

```python
def aggregate_metrics(article_metrics: list[dict]) -> dict[str, float]:
    # Micro-average: weighted by concern count
    total_matched = sum(m["n_matched"] for m in article_metrics)
    total_human   = sum(m["n_human"]   for m in article_metrics)
    total_tool    = sum(m["n_tool"]    for m in article_metrics)

    recall    = total_matched / total_human if total_human > 0 else 0.0
    precision = total_matched / total_tool  if total_tool  > 0 else 0.0
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
```

The frozen v4 score snapshot and release manifest order systems by
micro-averaged F1 (`f1_micro`) computed from these dataset-level totals. This
historical ordering is not an open public leaderboard.

Here, "precision" means the fraction of generated concerns matched to a
recorded reference concern under the configured matcher. It does **not** measure
the scientific validity of unmatched or matched generated concerns.

### 4.3 Category-level macro F1

The evaluation runner accepts tool concern texts. If a submitted concern object
contains a tool-provided `category`, `_normalise_tool_concerns` discards that
field before scoring. Category metrics are therefore based on reference-side
category attribution, not submitted category predictions:

- matched tool concerns inherit the category of their matched reference
  concern;
- each unmatched tool concern is assigned to the category of its most similar
  non-figure reference concern in the same article, even when that maximum
  similarity is below the matching threshold;
- articles with no active non-figure reference concerns contribute no category
  counts.

```python
def article_category_counts(tool_texts, active_gt, similarity_matrix, matches):
    """Mirror ConcernMatcher.score_article category attribution."""
    gt_indices_by_category = defaultdict(list)
    for gt_idx, concern in enumerate(active_gt):
        category = concern.get("category", "other")
        gt_indices_by_category[category].append(gt_idx)

    n_tool_by_category = Counter()
    n_matched_by_category = Counter()
    matched_tool_indices = set()

    for match in matches:
        category = active_gt[match.gt_idx].get("category", "other")
        n_tool_by_category[category] += 1
        n_matched_by_category[category] += 1
        matched_tool_indices.add(match.tool_idx)

    # Each row contains similarities to all active reference concerns.
    for tool_idx, row in enumerate(similarity_matrix):
        if tool_idx in matched_tool_indices:
            continue
        nearest_gt_idx = argmax(row)
        category = active_gt[nearest_gt_idx].get("category", "other")
        n_tool_by_category[category] += 1

    return {
        category: {
            "n_gt": len(gt_indices),
            "n_tool": n_tool_by_category[category],
            "n_matched": n_matched_by_category[category],
        }
        for category, gt_indices in gt_indices_by_category.items()
    }


def aggregate_category_f1(article_counts):
    """Sum category counts across articles, then compute dataset-level F1."""
    totals = defaultdict(Counter)
    for article in article_counts:
        for category, counts in article.items():
            totals[category].update(counts)

    per_category = {}
    for category, counts in totals.items():
        recall = counts["n_matched"] / counts["n_gt"]
        precision = (
            counts["n_matched"] / counts["n_tool"]
            if counts["n_tool"] > 0
            else 0.0
        )
        f1 = (
            2 * recall * precision / (recall + precision)
            if recall + precision > 0
            else 0.0
        )
        per_category[category] = f1

    f1_macro = (
        sum(per_category.values()) / len(per_category)
        if per_category
        else 0.0
    )
    return {"f1_macro": f1_macro, "per_category": per_category}
```

The matcher first computes one all-tool-by-all-reference similarity matrix and
one threshold-aware bipartite assignment per article. Category attribution
happens after that assignment; the runner does not rematch independently within
submitted categories.

For new runs in v4.1.2 and later, `f1_macro` is exactly the unweighted
arithmetic mean of the dataset-level per-category F1 values for represented,
non-figure reference categories:

```text
f1_macro = sum(per_category[category].f1) / number_of_represented_categories
```

Each represented category receives equal weight regardless of its concern
count. If there are no represented categories, `f1_macro` is `0.0`. This is not
an average over articles or sources and must not be confused with a
macro-across-source diagnostic, the 450-article non-eLife subset, or any
source-held-out analysis.

**Historical artifact status:** all six frozen v4 result JSONs store
`"f1_macro": 0.0` because the historical runner left the overall field at its
legacy schema default. Their per-category F1 values are non-zero, so the stored
zero is an invalid sentinel rather than a measured score. v4.1.2 fixes future
aggregation only; the frozen raw result files are not backfilled and historical
category-macro F1 remains unreported.

### 4.4 Major-concern recall

`recall_major` (also written Recall@Major) is:

```text
number of matched non-figure reference concerns labelled severity="major"
---------------------------------------------------------------------------
total number of non-figure reference concerns labelled severity="major"
```

Severity labels are produced by the normalization pipeline and have not yet
received independent expert adjudication. Recall@Major therefore inherits both
normalization and matcher uncertainty.

---

## 5. Statistical Uncertainty and Paired Comparison

### 5.1 Article-level resampling

```python
import numpy as np

def bootstrap_ci(
    article_metrics: list[dict],
    metric_key: str = "recall",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Article-level bootstrap (resampling at the document level).
    Concern-level resampling is prohibited because it ignores
    within-article concern correlation.
    """
    rng = np.random.default_rng(seed)
    n = len(article_metrics)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        sample = rng.choice(n, size=n, replace=True)
        sampled = [article_metrics[i] for i in sample]

        total_matched = sum(m["n_matched"] for m in sampled)
        total_human   = sum(m["n_human"]   for m in sampled)
        score = total_matched / total_human if total_human > 0 else 0.0
        bootstrap_scores.append(score)

    alpha = 1 - ci_level
    lo = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    hi = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))
    return lo, hi
```

### 5.2 Paired label-swap randomization test (comparing micro-F1)

```python
from bioreview_bench.stats import paired_micro_f1_randomization_pvalue


def paired_micro_f1_test(
    article_metrics_a: list[dict],
    article_metrics_b: list[dict],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float | None:
    """
    Two-sided paired randomization test for a difference in dataset micro-F1.

    Returns None when there are no eligible article pairs.
    """
    if len(article_metrics_a) != len(article_metrics_b):
        raise ValueError("Tools must cover the same ordered article pairs")
    counts_a = []
    counts_b = []
    for a, b in zip(article_metrics_a, article_metrics_b):
        if a["article_id"] != b["article_id"]:
            raise ValueError("Article IDs must match at every paired position")
        if a["n_human"] != b["n_human"]:
            raise ValueError("Paired tools must use the same reference set")
        counts_a.append((a["n_matched"], a["n_human"], a["n_tool"]))
        counts_b.append((b["n_matched"], b["n_human"], b["n_tool"]))
    return paired_micro_f1_randomization_pvalue(
        counts_a,
        counts_b,
        n_resamples=n_resamples,
        seed=seed,
    )
```

The test assumes that complete tool-result labels are exchangeable within an
article under the sharp null. Each permutation swaps the article's
`(n_matched, n_human, n_tool)` tuple and recomputes both dataset-level
micro-F1 values; it does not treat an article-level recall difference as if it
were the nonlinear micro-F1 estimand. Article IDs and reference denominators
must match explicitly.

For at most 16 informative pairs, the implementation enumerates every label
assignment exactly. Larger samples use a seeded Monte Carlo distribution with
the plus-one correction, so an estimated p-value cannot be zero. Identical
pairs are uninformative, all-tie samples return `1.0`, and an empty paired set
returns `None`. The separate `paired_sign_flip_pvalue` helper is suitable only
when the declared estimand is an additive paired statistic such as mean
article-level recall. Confidence intervals and effect sizes remain primary; a
randomization p-value does not validate the matcher or reference concerns.
When many system pairs are explored, raw p-values must be labeled unadjusted
and must not receive significance stars; a prespecified confirmatory analysis
would additionally require an appropriate multiplicity procedure.

---

## 6. Data Split Policy

### 6.1 Split unit and realized sizes

| Split | Frozen articles | Realized share | Purpose |
|-------|----------------:|---------------:|---------|
| train | 5,387 | 77.6% | Tool development and fine-tuning |
| validation | 953 | 13.7% | Historical threshold fixing and hyperparameter tuning |
| test | 600 | 8.6% | Frozen historical score snapshot |

**Split unit**: **Article level** (concern-level splitting prohibited)
- All concerns from the same article must be in the same split
- Rationale: splitting concerns across splits would leak information

### 6.2 Frozen split construction

The v4 builder used balanced-test mode with seed 42:

1. It randomly sampled fixed test counts within each source: 150 eLife, 150
   PLOS, 150 F1000Research, 100 Nature, and 50 PeerJ.
2. It grouped the 6,340 remaining articles by
   `(source, editorial_decision, review_format)`.
3. Within each stratum it randomly shuffled records and assigned approximately
   15% to validation (with the builder's rounding/minimum rules); all remaining
   records went to train.

Subject area, concern category, resolution, and publication time were not
stratification keys.

### 6.3 Temporal considerations

- The frozen builder used no explicit temporal stratification, date balancing,
  or chronological holdout. Selection was random within the source selection
  and train/validation strata described above.
- The eLife format transition (2023) happens to be represented in both train
  and evaluation splits; that is a realized property, not a temporal design.
- The published frozen score snapshot uses the full v4 test split (600 articles).
- The v4 records have publication dates through 2026-03-03; the v4 test split
  ends at 2026-02-23. This is a snapshot boundary, not a guarantee that an
  evaluated model was trained before every article or review.
- Every future authorized evaluation must report the exact model identifier and
  provider access date. For closed models, training-data contamination cannot
  be ruled out.

### 6.4 v3/v4 split-universe overlap

The v4 splits (current) were rebalanced from the same article universe as the v3 splits (released 2026-03-01). Key overlap statistics:

| Comparison | Overlap | % |
|---|---:|---:|
| v4 test ∩ v3 train | 439 / 600 | 73.2% |
| v4 test ∩ v3 val | 81 / 600 | 13.5% |
| v4 test ∩ v3 test | 80 / 600 | 13.3% |
| v4 test ∩ v3 (any) | 600 / 600 | 100% |

**Implication**: Models fine-tuned on v3 train data have seen 73.2% of the v4
test articles during training. Such models **must not** be compared directly
against baselines in the v4 main table. Researchers using SFT models trained on
v3 data should either (a) report results on only the 80-article disjoint subset
(v4 test \ v3 any), or (b) re-train on v4 train and evaluate on v4 test.
Source-stratified analyses, including the 450-article no-eLife subset, must be
reported as post-hoc subset analyses and not as replacements for the frozen
600-article snapshot.

Article IDs are split-disjoint, but the v4.1.3 F1000 DOI-stem audit finds
related/versioned records crossing every split boundary: 115 families cross
train–validation, 41 cross train–test, and 9 cross validation–test. In total,
161 distinct F1000 DOI-stem families cross at least one boundary. Development
data (train or validation) overlap 49/150 F1000 test articles across 48
families. A same-family training article is BM25 rank 1 for all 42 affected
train–test queries.

These DOI-stem counts are a source-specific lower bound: they do not resolve
fuzzy-title, preprint-to-journal, or cross-source manuscript relations.
Holding the original BM25 corpus, postings, IDF, and average document length
fixed, query-time same-family candidate filtering reduces one-to-one Jaccard
BM25 F1 from 0.0577 to 0.0148 on the primary full-600 snapshot. The
corresponding post-hoc results are 0.0690 to 0.0171 on the 450-article non-eLife
subset and 0.5780 to 0.0361 on the 42 affected train–test articles. The
intervention reconstructed all 42 targeted frozen BM25 rows, not all 600
predictions. It does not modify the frozen embedding table, rebuild the split,
or rerun the LLMs. Together with the cross-release overlap above, it precludes
interpreting the frozen scores as contamination-free out-of-family
generalization. See `results/v4/measurement_audit.{json,md}`.

---

## 7. Reporting Standard

### 7.1 Required reporting items

```
All benchmark results must include the following:

Tool: MyReviewTool v1.2.3
Benchmark software: bioreview-bench v4.1.3
Score snapshot: frozen v4 test split

recall:        0.61 [95% CI: 0.55-0.67]
precision:     0.48 [95% CI: 0.43-0.54]
f1:            0.54 [point estimate; CI not released]
recall_major:  0.72 [point estimate; CI not released]
f1_macro:      unreported for the frozen v4 results

n_articles: 600 | n_human_concerns: 8,200 | n_excluded_figure: 447
Matched pairs (bipartite): ... / 8,200
Recall/precision bootstrap n=1000, seed=42
```

### 7.2 Authorized result-record format

The public index does not provide a test-scoring service or accept public
leaderboard submissions. Authorized local evaluations should retain this
record format for provenance:

```json
{
  "tool_name": "MyReviewTool",
  "tool_version": "1.2.3",
  "git_hash": "abc123",
  "benchmark_version": "4.1",
  "split": "test",
  "predictions": {
    "elife:84798": [
      "The sample size is insufficient for the claimed statistical power."
    ]
  }
}
```

Concern objects with `text` or `concern_text` are also accepted, but any
submitted category or severity fields are optional, discarded before scoring,
and not evaluated as predictions.

---

## 8. Limitations and Known Biases

1. **Unvalidated historical matcher**: The frozen runs used the unadapted
   `allenai/specter2_base` checkpoint with automatic mean pooling, not a
   recorded SPECTER2 task adapter. Neither that wrapper nor the 0.65 threshold
   has completed independent human-equivalence validation.
2. **Embedding-model bias**: The historical encoder wrapper was not trained as
   a concern-equivalence model. It may reward shared topic vocabulary and miss
   valid paraphrases.
3. **Threshold sensitivity**: small threshold changes can move absolute scores;
   report sensitivity analyses rather than treating 0.65 as a natural boundary.
4. **Figure concern exclusion**: Base recall may overestimate performance on figure-heavy articles.
5. **Bipartite matching limitation**: When multiple very similar concerns exist, only some are matched.
6. **Bootstrap assumption**: Article-level i.i.d. assumption does not account for same-author/lab clustering effects.
7. **Silver-standard normalized fields**: Reference concerns originate in
   published peer-review records, but concern segmentation, category, severity,
   and stance labels are LLM-derived and not exhaustively human-validated.
   Possible AI assistance in individual recent source reviews is unknown.
8. **F1000 family leakage**: DOI-stem families cross train–validation,
   train–test, and validation–test boundaries. Development data overlap 49/150
   F1000 test articles across 48 families. This source-specific resolver is a
   lower bound. Query-time family filtering sharply reduces BM25 under the
   lexical audit, but does not produce corrected historical embedding scores,
   reconstruct all 600 BM25 predictions, or create a family-disjoint split.

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-07-25 | 1.6 | Replaced the invalid paired-bootstrap p-value path with a tested article-paired label-swap test that recomputes micro-F1, added explicit article-ID alignment checks, and aligned the protocol with the v4.1.3 public hardening release |
| 2026-07-24 | 1.5 | Added the aggregate matcher-sensitivity and F1000 query-time family-filtering audit, including exact overlap and effect sizes |
| 2026-07-24 | 1.4 | Marked frozen `f1_macro` fields invalid/unreported, documented future aggregation and actual reference-side category attribution, and corrected frozen split construction and temporal claims |
| 2026-07-24 | 1.3 | Disclosed the non-executable public boundary, F1000 family leakage, historical unadapted auto-mean-pooled matcher, 600-versus-450 analysis boundary, recall/precision-only confidence intervals, and the future-run category-macro F1 definition |
| 2026-07-24 | 1.2 | Withdrew unsupported 148-row validation claim; documented provisional matcher status, exact model ID, Recall@Major, and temporal boundary |
| 2026-04-28 | 1.1 | Updated reporting standard for v4.0 public test split; metric rules unchanged |
| 2026-03-01 | 1.0 | Initial public release |

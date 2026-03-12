# bioreview-bench Evaluation Protocol

> Version: 1.0
> Date: 2026-03-01

This document fully defines the evaluation procedure for bioreview-bench.
Metric computation, threshold fixing, and statistical testing follow the specifications below.
Changes to this protocol after release require a major version bump.

---

## 1. Embedding Model

**Model**: `allenai/specter2` (SPECTER2 base)
**Input**: Full concern text (max 512 tokens; longer texts truncated from the end)
**Output**: 768-dimensional vector
**Normalisation**: L2 normalisation followed by cosine similarity

```python
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("allenai/specter2")

def embed(texts: list[str]) -> np.ndarray:
    embeddings = EMBED_MODEL.encode(texts, normalize_embeddings=True)
    return embeddings  # shape: (N, 768)
```

**Version pinning**: The `allenai/specter2` model commit hash is recorded in
`evaluation_manifest.json`. Re-evaluation is required if the model is updated.

---

## 2. Concern Matching Algorithm

### 2.1 Hungarian Bipartite Matching (default)

Tool concern set T = {t1, t2, ..., tm}
Human concern set H = {h1, h2, ..., hn} (figure_issue concerns excluded)

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

def match_concerns(
    tool_concerns: list[str],
    human_concerns: list[str],
    threshold: float,
) -> tuple[int, list[tuple[int, int, float]]]:
    """
    Returns:
        n_matched: Number of matched pairs (similarity >= threshold)
        matched_pairs: [(tool_idx, human_idx, similarity), ...]
    """
    if not tool_concerns or not human_concerns:
        return 0, []

    t_emb = embed(tool_concerns)   # (M, 768)
    h_emb = embed(human_concerns)  # (N, 768)

    sim_matrix = t_emb @ h_emb.T  # (M, N), cosine similarity

    # Maximum weight bipartite matching
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        sim = float(sim_matrix[r, c])
        if sim >= threshold:
            matched_pairs.append((r, c, sim))

    return len(matched_pairs), matched_pairs
```

The official public release uses `algorithm="hungarian"` throughout. The repository
retains a legacy greedy matcher only for ablation or compatibility checks.

**Rationale**: Optimal bipartite matching guarantees the maximum-weight assignment.
Complexity is O(n^3) but negligible for typical concern counts (<= 50 per article).

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

### 3.1 Determination method

1. Measure SPECTER2 cosine similarity distributions on the validation split (15%)
2. Manual spot-check of 148 concern pairs from 20 articles: genuine matches vs non-matches
3. **Default threshold: 0.65** — validated with Cohen's kappa = 1.000
4. Record the determined threshold in `evaluation_manifest.json` and fix it

### 3.2 Application to test split

- The validation threshold is applied unchanged to the test split (no re-optimisation)
- Post-release threshold changes require a minor version bump and full re-evaluation

### 3.3 Record format

```json
{
  "eval_version": "1.0",
  "specter2_commit": "abc123...",
  "threshold": 0.65,
  "threshold_method": "val_20article_148concern_spotcheck",
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
    tool_concerns: list[str],
    human_concerns: list[ReviewerConcern],
    threshold: float,
) -> dict[str, float]:
    non_fig, _ = filter_figure_concerns(human_concerns)
    n_matched, pairs = match_concerns(
        tool_concerns, [c.concern_text for c in non_fig], threshold
    )

    n_human = len(non_fig)
    n_tool = len(tool_concerns)

    recall = n_matched / n_human if n_human > 0 else 0.0
    precision = n_matched / n_tool if n_tool > 0 else 0.0
    f1 = (2 * recall * precision / (recall + precision)
           if (recall + precision) > 0 else 0.0)

    return {"recall": recall, "precision": precision, "f1": f1,
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

The official public leaderboard and release manifest rank systems by micro-averaged
F1 (`f1_micro`) computed from these dataset-level totals.

### 4.3 Category-level macro F1

```python
def compute_macro_f1_by_category(
    tool_concerns: list[dict],     # [{"text": ..., "category": ...}]
    human_concerns: list[ReviewerConcern],
    threshold: float,
) -> dict[str, float]:
    """Compute per-category recall/precision/F1, then macro-average."""
    categories = [c.value for c in ConcernCategory if c != ConcernCategory.FIGURE_ISSUE]
    cat_f1 = {}
    for cat in categories:
        t_cat = [c["text"] for c in tool_concerns if c.get("category") == cat]
        h_cat = [c for c in human_concerns
                 if c.category.value == cat and not c.requires_figure_reading]
        if not h_cat:
            continue
        n_matched, _ = match_concerns(t_cat, [c.concern_text for c in h_cat], threshold)
        r = n_matched / len(h_cat)
        p = n_matched / len(t_cat) if t_cat else 0.0
        f1 = 2*r*p/(r+p) if (r+p) > 0 else 0.0
        cat_f1[cat] = f1
    macro_f1 = sum(cat_f1.values()) / len(cat_f1) if cat_f1 else 0.0
    return {"f1_macro": macro_f1, "per_category": cat_f1}
```

---

## 5. Bootstrap Confidence Intervals

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

### 5.2 Paired significance test (comparing two tools)

```python
def paired_significance(
    article_metrics_a: list[dict],
    article_metrics_b: list[dict],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """
    Bootstrap paired difference test.
    H0: recall difference between tool A and tool B = 0
    Returns: p-value (two-sided)
    """
    assert len(article_metrics_a) == len(article_metrics_b)
    rng = np.random.default_rng(seed)
    n = len(article_metrics_a)

    recall_a = np.array([m["n_matched"] / m["n_human"]
                         for m in article_metrics_a if m["n_human"] > 0])
    recall_b = np.array([m["n_matched"] / m["n_human"]
                         for m in article_metrics_b if m["n_human"] > 0])
    observed_diff = recall_a.mean() - recall_b.mean()

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = recall_a[idx].mean() - recall_b[idx].mean()
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)
    p_value = float(np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff)))
    return p_value
```

---

## 6. Data Split Policy

### 6.1 Split unit and ratios

| Split | Ratio | Purpose |
|-------|-------|---------|
| train | 70% | Tool development, fine-tuning |
| validation | 15% | Threshold fixing, hyperparameter tuning |
| test | 15% | Final benchmark score (single evaluation) |

**Split unit**: **Article level** (concern-level splitting prohibited)
- All concerns from the same article must be in the same split
- Rationale: splitting concerns across splits would leak information

### 6.2 Stratification

Stratified by:
1. Source journal (eLife / PLOS / F1000Research)
2. Subject area
3. Resolution distribution (approximate proportion maintenance)

### 6.3 Temporal considerations

- Temporal bias prevention: avoid concentrating old articles in train and new ones in test
- eLife format transition (2023) is represented in both train and evaluation splits

---

## 7. Reporting Standard

### 7.1 Required reporting items

```
All benchmark results must include the following:

Tool: MyReviewTool v1.2.3
Benchmark: bioreview-bench v1.0 (test split)

recall:        0.61 [95% CI: 0.55-0.67]
precision:     0.48 [95% CI: 0.43-0.54]
f1:            0.54 [95% CI: 0.49-0.59]
recall_major:  0.72 [95% CI: 0.64-0.80]
f1_macro:      0.49 (8 categories)

n_articles: 149 | n_human_concerns: 1,515 | n_excluded_figure: ...
Matched pairs (bipartite): ... / 1,515
Bootstrap n=1000, seed=42
```

### 7.2 Leaderboard submission format

```json
{
  "tool_name": "MyReviewTool",
  "tool_version": "1.2.3",
  "git_hash": "abc123",
  "benchmark_version": "1.0",
  "split": "test",
  "predictions": {
    "elife:84798": [
      {"text": "...", "category": "statistical_methodology", "severity": "major"}
    ]
  }
}
```

---

## 8. Limitations and Known Biases

1. **SPECTER2 bias**: Specialised for biomedical text — review needed before extending to other domains.
2. **Threshold sensitivity**: Threshold 0.65 +/- 0.05 results in approximately +/- 3-5 percentage point recall variation.
3. **Figure concern exclusion**: v1.0 recall may overestimate performance on figure-heavy articles.
4. **Bipartite matching limitation**: When multiple very similar concerns exist, only some are matched.
5. **Bootstrap assumption**: Article-level i.i.d. assumption does not account for same-author/lab clustering effects.
6. **Silver-standard labels**: Category, severity, and stance labels are LLM-derived and not exhaustively human-validated.

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-03-01 | 1.0 | Initial public release |

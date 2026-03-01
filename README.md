# bioreview-bench

**A benchmark dataset and evaluation harness for AI biomedical peer review tools.**

[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-bioreview--bench-yellow)](https://huggingface.co/datasets/jang1563/bioreview-bench)
[![License: Apache-2.0](https://img.shields.io/badge/Code-Apache--2.0-blue)](LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/Data-CC--BY--4.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)

- **978 articles** from 3 journals (eLife, PLOS, F1000Research)
- **9,394 reviewer concerns** with category, severity, and author stance labels
- Outcome-anchored ground truth: each concern is annotated with how the authors responded
- Integrated evaluation harness with SPECTER2 semantic matching
- [GitHub](https://github.com/jang1563/bioreview-bench) | [HuggingFace](https://huggingface.co/datasets/jang1563/bioreview-bench) | [Demo](https://jang1563.github.io/bioreview-bench/)

---

## Overview

bioreview-bench evaluates whether AI tools can identify the substantive concerns that professional peer reviewers raise about biomedical research articles. Given the full text of a paper, an AI tool should produce a list of concerns; the benchmark measures how well these align with the concerns that human reviewers actually raised.

What makes bioreview-bench different:

- **Concern-level granularity.** Reviews are decomposed into individual concern units, not treated as monolithic blocks.
- **Author stance labels.** Each concern carries an outcome-anchored label (`conceded`, `rebutted`, `partial`, `unclear`, `no_response`) based on what the authors actually did in their revision.
- **Multi-source.** Three journals with different review cultures and editorial philosophies.
- **Evaluation harness.** Standardised metrics with SPECTER2 semantic matching and reproducible scoring.

---

## Quick Start

### Loading from HuggingFace

```python
from datasets import load_dataset

# Default config: all sources, all fields
dataset = load_dataset("jang1563/bioreview-bench")

train = dataset["train"]       # 680 articles, 6,444 concerns
val   = dataset["validation"]  # 149 articles, 1,435 concerns
test  = dataset["test"]        # 149 articles, 1,515 concerns

# Inspect an article
article = val[0]
print(article["id"], article["title"])
for c in article["concerns"]:
    print(f"  [{c['severity']}] {c['category']}: {c['concern_text'][:80]}...")
```

**Available HuggingFace configs:**

| Config | Description |
|--------|-------------|
| `default` | All sources, full article + concern records |
| `benchmark` | Test split with `concerns=[]` (no label leakage) |
| `concerns_flat` | One row per concern (9,394 rows) |
| `elife` | eLife articles only (730) |
| `plos` | PLOS articles only (163) |
| `f1000` | F1000Research articles only (85) |

```python
# Load a specific config
benchmark = load_dataset("jang1563/bioreview-bench", "benchmark")
flat = load_dataset("jang1563/bioreview-bench", "concerns_flat")
```

### Installing the Python package

```bash
# Base install (evaluation harness only)
pip install bioreview-bench

# With data collection tools
pip install bioreview-bench[collect]

# With SPECTER2 embedding support
pip install bioreview-bench[evaluate]
```

### Running evaluation

```bash
# Validate and score predictions
bioreview-run --predictions predictions.json
```

### Quick evaluation API

```python
from bioreview_bench.evaluate.metrics import quick_eval

result = quick_eval(
    tool_concerns=["No negative control for IP.", "Multiple testing not corrected."],
    gt_entry=article,  # a dataset row (dict)
)
print(f"Recall: {result.recall:.2f}, Precision: {result.precision:.2f}")
```

---

## Dataset Schema

### Article fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `elife:84798`) |
| `source` | string | Journal source (`elife`, `plos`, `f1000`) |
| `doi` | string | Digital Object Identifier |
| `title` | string | Article title |
| `abstract` | string | Article abstract |
| `subjects` | list[string] | Subject area(s) |
| `published_date` | string | Publication date (ISO format) |
| `review_format` | string | Review format (`reviewed_preprint`, `journal`) |
| `has_author_response` | bool | Whether author response letter exists |
| `concerns` | list[Concern] | List of reviewer concerns |

### Concern fields

| Field | Type | Description |
|-------|------|-------------|
| `concern_id` | string | Unique concern ID (e.g., `elife:84798:R1C1`) |
| `concern_text` | string | Full text of the concern |
| `category` | string | One of 9 categories (see below) |
| `severity` | string | `major`, `minor`, or `optional` |
| `author_stance` | string | `conceded`, `rebutted`, `partial`, `unclear`, `no_response` |

---

## Concern Categories

| Category | Description |
|----------|-------------|
| `design_flaw` | Fundamental flaws in experimental or study design |
| `statistical_methodology` | Errors or weaknesses in statistical analysis |
| `missing_experiment` | Key control, validation, or follow-up experiment absent |
| `figure_issue` | Problems with figures, panels, or image quality |
| `prior_art_novelty` | Insufficient novelty or failure to engage with prior work |
| `writing_clarity` | Ambiguous, unclear, or poorly structured writing |
| `reagent_method_specificity` | Insufficient detail on reagents, protocols, or methods |
| `interpretation` | Overclaiming, underclaiming, or unsupported interpretation |
| `other` | Concerns not captured by the above categories |

**Note:** `figure_issue` concerns are excluded from base evaluation metrics because they require visual inspection of figures.

---

## Dataset Statistics

### Split sizes

| Split | Articles | Concerns | Avg concerns/article |
|-------|----------|----------|---------------------|
| train | 680 | 6,444 | 9.5 |
| validation | 149 | 1,435 | 9.6 |
| test | 149 | 1,515 | 10.2 |
| **Total** | **978** | **9,394** | **9.6** |

### Source distribution

| Source | Articles | Notes |
|--------|----------|-------|
| eLife | 730 | 2019-2024; journal and reviewed_preprint formats |
| PLOS | 163 | PLOS ONE and PLOS Biology |
| F1000Research | 85 | Open peer review with named reviewers |

### Severity distribution

| Severity | Count | % |
|----------|-------|---|
| major | 6,677 | 71.1% |
| minor | 2,538 | 27.0% |
| optional | 179 | 1.9% |

### Author stance distribution

| Stance | Count | % |
|--------|-------|---|
| no_response | 6,615 | 70.4% |
| partial | 1,618 | 17.2% |
| conceded | 992 | 10.6% |
| rebutted | 144 | 1.5% |
| unclear | 25 | 0.3% |

---

## Evaluation Protocol

Concerns are matched using cosine similarity of SPECTER2 embeddings with greedy bipartite matching (threshold >= 0.65). See [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) for the full specification.

**Primary metrics:**

| Metric | Description |
|--------|-------------|
| `recall` | Fraction of human concerns detected by the tool |
| `precision` | Fraction of tool concerns that match a human concern |
| `f1` | Harmonic mean of recall and precision |
| `recall_major` | Recall restricted to major-severity concerns |

The matching threshold (0.65) was validated on 20 held-out articles (148 concerns) with Cohen's kappa = 1.000 between automated matching and human judgement.

---

## Leaderboard

Results on the **validation split** (149 articles, 1,435 concerns, figure concerns excluded). To submit results, open an issue or pull request.

| Rank | Tool | Version | Recall | Precision | F1 | Major Recall | Date |
|------|------|---------|--------|-----------|----|--------------|------|
| — | *(submit your results)* | — | — | — | — | — | — |

---

## Task Definition

See [TASK_DEFINITION.md](TASK_DEFINITION.md) for the complete task specification including input/output formats, scoring rules, and submission requirements.

---

## Related Work

| Dataset / Benchmark | Domain | Granularity | Author stance | Multi-journal | Eval harness |
|---------------------|--------|-------------|---------------|---------------|--------------|
| **bioreview-bench** | Biomedical | Concern-level | Yes | Yes (3) | Yes |
| PeerRead | General | Review-level | No | Yes | No |
| OpenEval | General | Claim-level | No | Yes | Partial |
| NLPeer | Multi-domain | Sentence-level | No | Yes | Partial |

---

## Citation

```bibtex
@dataset{bioreview-bench,
  author    = {Kim, JangKeun},
  title     = {bioreview-bench: A Benchmark Dataset for Evaluating AI Biomedical Peer Review Tools},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/jang1563/bioreview-bench},
  note      = {Version 1.0}
}
```

---

## License

This project uses a dual license:

- **Dataset** (JSONL data files on HuggingFace): [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). The underlying peer review content from eLife, PLOS, and F1000Research is published under CC-BY 4.0 by the respective publishers.
- **Code** (Python package, evaluation harness, scripts): [Apache-2.0](LICENSE).

Users who redistribute or build upon the dataset must provide appropriate attribution to both bioreview-bench and the original source articles (via DOIs included in the dataset).

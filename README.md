# bioreview-bench

**A benchmark dataset and evaluation harness for AI biomedical peer review tools.**

[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-bioreview--bench-yellow)](https://huggingface.co/datasets/jang1563/bioreview-bench)
[![License: Apache-2.0](https://img.shields.io/badge/Code-Apache--2.0-blue)](LICENSE)
[![License: Source-specific](https://img.shields.io/badge/Data-source--specific-lightgrey)](LICENSE_MATRIX.md)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)

- **6,696 articles** from 5 journals (eLife, PLOS, F1000Research, PeerJ, Nature)
- **96,866 reviewer concerns** with category, severity, and author stance labels
- Outcome-anchored ground truth: each concern is annotated with how the authors responded
- Integrated evaluation harness with SPECTER2 semantic matching
- [GitHub](https://github.com/jang1563/bioreview-bench) | [HuggingFace](https://huggingface.co/datasets/jang1563/bioreview-bench)

---

## Overview

bioreview-bench evaluates whether AI tools can identify the substantive concerns that professional peer reviewers raise about biomedical research articles. Given the full text of a paper, an AI tool should produce a list of concerns; the benchmark measures how well these align with the concerns that human reviewers actually raised.

What makes bioreview-bench different:

- **Concern-level granularity.** Reviews are decomposed into individual concern units, not treated as monolithic blocks.
- **Author stance labels.** Each concern carries an outcome-anchored label (`conceded`, `rebutted`, `partial`, `unclear`, `no_response`) based on what the authors actually did in their revision.
- **Multi-source.** Five journals with different review cultures and editorial philosophies.
- **Evaluation harness.** Standardised metrics with SPECTER2 semantic matching, bipartite concern matching, and bootstrap confidence intervals.

---

## Quick Start

### Loading from HuggingFace

```python
from datasets import load_dataset

# Default config: all sources, all fields
dataset = load_dataset("jang1563/bioreview-bench")

train = dataset["train"]       # 4,888 articles, 70,241 concerns
val   = dataset["validation"]  # 864 articles, 12,755 concerns
test  = dataset["test"]        # 944 articles, 13,870 concerns

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
| `concerns_flat` | One row per concern (96,866 rows) |
| `elife` | eLife articles only (1,810) |
| `plos` | PLOS articles only (1,737) |
| `f1000` | F1000Research articles only (2,679) |
| `peerj` | PeerJ articles only (244) |
| `nature` | Nature articles only (470) |

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
# Evaluate a prediction file (your tool produces {article_id, concerns:[...]} JSONL)
bioreview-run --tool-output predictions.jsonl --tool-name "MyTool" --split test

# Run the built-in baseline reviewer
bioreview-baseline --split val --model claude-haiku-4-5-20251001

# Or use another supported provider
bioreview-baseline --split val --provider google --model gemini-2.5-flash-lite

# Run the zero-cost lexical baseline
bioreview-bm25 --split val

# Regenerate split stats and verify docs stay in sync
bioreview-stats --check-docs

# Freeze a human-review pilot subset
bioreview-human-subset --n 100
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
| `source` | string | Journal source (`elife`, `plos`, `f1000`, `peerj`, `nature`) |
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
| `author_response_text` | string | Author's response to this concern |
| `evidence_of_change` | bool? | Whether author made revisions |
| `resolution_confidence` | float | LLM extraction confidence (0.0-1.0) |

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
| train | 4,888 | 70,241 | 14.4 |
| validation | 864 | 12,755 | 14.8 |
| test | 944 | 13,870 | 14.7 |
| **Total** | **6,696** | **96,866** | **14.5** |

### Source distribution

| Source | Articles | Notes |
|--------|----------|-------|
| F1000Research | 2,679 | Open peer review with named reviewers, 2013-present |
| eLife | 1,810 | 2019-2026; journal and reviewed_preprint formats |
| PLOS | 1,737 | PLOS ONE, PLOS Biology, and other PLOS journals |
| PeerJ | 244 | Open peer review, 2018-present |
| Nature | 470 | Nature Communications and Nature journals, PDF-based |

### Severity distribution

| Severity | Count | % |
|----------|-------|---|
| major | 61,093 | 63.1% |
| minor | 33,548 | 34.6% |
| optional | 2,225 | 2.3% |

### Author stance distribution

| Stance | Count | % |
|--------|-------|---|
| no_response | 87,833 | 90.7% |
| partial | 5,229 | 5.4% |
| conceded | 3,250 | 3.4% |
| rebutted | 491 | 0.5% |
| unclear | 63 | 0.1% |

### Category distribution

| Category | Count | % |
|----------|-------|---|
| writing_clarity | 32,885 | 33.9% |
| missing_experiment | 15,315 | 15.8% |
| interpretation | 14,808 | 15.3% |
| design_flaw | 10,020 | 10.3% |
| prior_art_novelty | 7,179 | 7.4% |
| reagent_method_specificity | 6,828 | 7.0% |
| statistical_methodology | 4,903 | 5.1% |
| figure_issue | 4,678 | 4.8% |
| other | 250 | 0.3% |

---

## Evaluation Protocol

Concerns are matched using cosine similarity of SPECTER2 embeddings with bipartite (Hungarian) matching (threshold = 0.65). See [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) for the full specification.

**Primary metrics:**

| Metric | Description |
|--------|-------------|
| `recall` | Fraction of human concerns detected by the tool |
| `precision` | Fraction of tool concerns that match a human concern |
| `f1` | Harmonic mean of recall and precision |
| `recall_major` | Recall restricted to major-severity concerns |

All metrics include bootstrap 95% confidence intervals (1,000 iterations, document-level resampling).

---

## Leaderboard

The current official public leaderboard snapshot covers **944 scored test articles** (after correction/erratum filtering).
The full test split contains 944 articles; the rankings below follow the frozen release
manifest in `results/release_manifest.json`. To submit results, open an issue or pull request.

Leaderboard inclusion policy:

- Default public rankings use only `split="test"` result files from `results/v3/`.
- Experimental `dedup_gt=true` runs are excluded from the default ranking.
- If multiple result files exist for the same `(tool_name, tool_version)`, only the strongest run by `f1_micro` is retained.
- `tool_version` should record the exact model or release identifier (for example, `claude-haiku-4-5-20251001`), not `unknown`.

| Rank | Tool | Version | Recall | 95% CI | Precision | 95% CI | F1 | Major Recall | Articles |
|------|------|---------|--------|--------|-----------|--------|----|--------------|----------|
| 1 | Haiku-4.5 | `claude-haiku-4-5-20251001` | 0.725 | [0.697, 0.753] | 0.675 | [0.653, 0.694] | 0.699 | 0.872 | 944 |
| 2 | GPT-4o-mini | `gpt-4o-mini` | 0.684 | [0.658, 0.712] | 0.703 | [0.682, 0.722] | 0.694 | 0.840 | 944 |
| 3 | Gemini-2.5-Flash | `gemini-2.5-flash` | 0.665 | [0.639, 0.693] | 0.709 | [0.688, 0.729] | 0.686 | 0.832 | 944 |
| 4 | BM25 | `bm25-specter2` | 0.637 | [0.611, 0.664] | 0.741 | [0.721, 0.760] | 0.685 | 0.794 | 944 |
| 5 | Gemini-2.5-Flash-Lite | `gemini-2.5-flash-lite` | 0.615 | [0.588, 0.644] | 0.708 | [0.685, 0.729] | 0.658 | 0.781 | 944 |
| 6 | Llama-3.3-70B | `llama-3.3-70b` | 0.554 | [0.530, 0.580] | 0.794 | [0.774, 0.811] | 0.653 | 0.753 | 944 |

> Ranking metric: micro-averaged F1 (`f1_micro`). 95% CI from 1,000 article-level bootstrap resamples.
> Matching: SPECTER2 cosine similarity, threshold=0.65, hungarian bipartite matching.
> Figure-issue concerns excluded from ground truth (require visual inspection).

Official release artifacts are rebuilt from raw result JSON files with:

```bash
./.venv/bin/python scripts/rebuild_release_artifacts.py \
  --results-dir results/v3 \
  --output-dir results \
  --split test
```

This regenerates `results/leaderboard.md`, `results/leaderboard.json`, and
`results/release_manifest.json`. The manifest freezes the included result files
and matching settings for the public release.

Release operators should also consult [RELEASE_V3.md](RELEASE_V3.md), which
defines the current public `v3` release reference, included result files, and
publication checklist.
For a GitHub-release-ready summary of the tagged snapshot, see
[RELEASE_NOTES_v3.0.md](RELEASE_NOTES_v3.0.md).

---

## Task Definition

See [TASK_DEFINITION.md](TASK_DEFINITION.md) for the complete task specification including input/output formats, scoring rules, and submission requirements.

**Input**: Full manuscript text (abstract + body sections). Peer review text and author response are NOT provided at test time.

**Output**: JSON list of concerns:
```json
{"article_id": "elife:12345", "concerns": ["concern text 1", "concern text 2", ...]}
```

---

## Related Work

| Dataset / Benchmark | Domain | Granularity | Author stance | Multi-journal | Eval harness |
|---------------------|--------|-------------|---------------|---------------|--------------|
| **bioreview-bench** | Biomedical | Concern-level | Yes | Yes (5) | Yes |
| PeerRead | General | Review-level | No | Yes | No |
| OpenEval | General | Claim-level | No | Yes | Partial |
| NLPeer | Multi-domain | Sentence-level | No | Yes | Partial |
| MOPRD | Multi-domain | Review-level | No | Yes | No |

---

## Citation

```bibtex
@misc{bioreview-bench,
  title   = {BioReview-Bench: A Benchmark for AI-Assisted Biomedical Peer Review},
  author  = {Kim, JangKeun},
  year    = {2026},
  url     = {https://huggingface.co/datasets/jang1563/bioreview-bench},
  note    = {Version 3.0 (v3.0-release)}
}
```

---

## License

This project uses a dual license:

- **Benchmark annotations and packaging metadata**: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Underlying article, review, and author-response content**: source-specific. Redistribution rights are not uniform across eLife, PLOS, F1000Research, PeerJ, Nature Portfolio, and other future sources; see [LICENSE_MATRIX.md](LICENSE_MATRIX.md).
- **Code** (Python package, evaluation harness, scripts): [Apache-2.0](LICENSE).

Users who redistribute or build upon the benchmark must provide appropriate attribution to both bioreview-bench and the original source articles (via DOIs included in the dataset), and must follow the per-source redistribution rules in [LICENSE_MATRIX.md](LICENSE_MATRIX.md).

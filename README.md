# bioreview-bench

**A benchmark dataset and evaluation harness for AI biomedical peer review tools.**

[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-public%20index-yellow)](https://huggingface.co/datasets/jang1563/bioreview-bench)
[![License: Apache-2.0](https://img.shields.io/badge/Code-Apache--2.0-blue)](LICENSE)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/Data-CC--BY--NC--4.0-lightgrey)](LICENSE)
[![License: Source-specific](https://img.shields.io/badge/Content-source--specific-lightgrey)](LICENSE_MATRIX.md)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)

- **6,940 articles** from 5 journals (eLife, PLOS, F1000Research, PeerJ, Nature)
- **101,869 reviewer concerns** with category, severity, and author stance labels
- Published peer-review substrate with LLM-normalized concern and response labels
- Integrated evaluation harness with configurable semantic or lexical matching
- [GitHub](https://github.com/jang1563/bioreview-bench) |
  [HuggingFace public index](https://huggingface.co/datasets/jang1563/bioreview-bench)

---

> **Validation status (v4.1.3):** BioReview-Bench is a silver-standard behavioral
> benchmark, not expert-adjudicated scientific truth. Concern segmentation,
> category, severity, and stance are LLM-derived. The historical unadapted
> `allenai/specter2_base`/auto-mean-pooling wrapper and 0.65 threshold have not
> completed independent human validation. A previous 148-row validation claim was
> withdrawn because the legacy interface prefilled blank human labels. The
> published frozen scores are provisional and matcher-dependent.
>
> **Known issues and release boundary:** the public Hugging Face repository is a
> rights-minimized, non-executable index—not a public test set or an open
> leaderboard. The six-system table below is a frozen historical 600-article
> snapshot; it is the primary reporting population. The separate 450-article
> non-eLife analysis is post-hoc. The F1000 DOI-stem audit is a source-specific
> lower bound: 115 families cross train–validation, 41 cross train–test, and 9
> cross validation–test. Development data overlap 49/150 F1000 test articles
> across 48 families. A same-family training article ranked first for all 42
> affected BM25 train–test queries. At the uncalibrated Jaccard cutoff 0.195,
> query-time family filtering reduced full-600 matched-record BM25 F1 from
> 0.0577 to 0.0148; the 450 non-eLife result, 0.0690 to 0.0171, is a post-hoc
> sensitivity result. These are operating-point-dependent Jaccard/Hungarian
> diagnostics, not corrected historical embedding scores.
> Historical scores used an unadapted `specter2_base` checkpoint with automatic
> mean pooling and no recorded revision. The public index reveals test
> membership, IDs, and DOIs; withholding target rows does not create a
> secret/blind test because source reviews are publicly retrievable. See the
> [measurement audit](results/v4/measurement_audit.md) and
> [known issues](KNOWN_ISSUES.md).
>
> See the public [NeurIPS 2026 review record and revision
> response](NEURIPS_2026_REVIEW_RESPONSE.md) and
> [v4.1.3 release notes](RELEASE_NOTES_v4.1.3.md).

---

## Overview

bioreview-bench evaluates whether AI tools can recover concern units recorded in
published biomedical peer reviews. Given article text, a tool produces a list of
concerns; the benchmark measures how well these align with an LLM-normalized
reference set. This is behavioral overlap with a review record, not a judgment
that a concern is scientifically correct.

What makes bioreview-bench different:

- **Concern-level granularity.** Reviews are decomposed by an LLM into
  individual silver-standard concern units rather than treated as monolithic
  blocks.
- **Author stance labels.** Each concern carries an LLM-derived response label
  (`conceded`, `rebutted`, `partial`, `unclear`, `no_response`) based on the
  available response record.
- **Multi-source.** Five journals with different review cultures and editorial philosophies.
- **Evaluation harness.** Standardised metrics with explicit matcher
  configuration, bipartite concern matching, and document-level bootstrap
  confidence intervals.

---

## Quick Start

### Loading the public HuggingFace index

The public repository is a rights-minimized, non-executable index release. It
contains text-free article metadata for all splits and text-free categorical
annotation rows for train/validation. Publisher text, normalized concern text,
review/response prose, reviewer/data-row names, emails, and explicit identity
fields, and test targets are excluded. Stable concern IDs may retain
source-local review ordinals such as `R1`; those ordinals do not identify a
person. This exclusion does not apply to project-maintainer contact metadata in
`CITATION.cff`.

```python
from datasets import load_dataset

index = load_dataset("jang1563/bioreview-bench", "index", revision="v4.1.3")
labels = load_dataset("jang1563/bioreview-bench", "annotations", revision="v4.1.3")

print(index["test"][0]["id"])       # metadata only; test targets withheld
print(labels["train"][0]["category"])
```

**Public configs:**

| Config | Rows | Splits | Description |
|--------|-----:|--------|-------------|
| `index` | 6,940 | train / validation / test | Text-free article ID/source/DOI/date/schema rows |
| `annotations` | 93,222 | train / validation | Text-free category, severity, and stance rows |

The `index` config exposes test membership plus stable IDs and DOIs. Since the
underlying source peer reviews are public, targets may be reconstructed outside
this package. The frozen test is therefore not secret or blind and should not
be used for new blind evaluation or hidden-test generalization claims.

The private raw archive is retained separately for authorized reproducibility.
It is not a public distribution channel.

### Installing the Python package

The package is not currently published on PyPI. Install the audited GitHub tag:

```bash
# Base install
python -m pip install \
  "bioreview-bench @ git+https://github.com/jang1563/bioreview-bench.git@v4.1.3"

# With data collection tools
python -m pip install \
  "bioreview-bench[collect] @ git+https://github.com/jang1563/bioreview-bench.git@v4.1.3"

# With the historical embedding-matcher dependencies
python -m pip install \
  "bioreview-bench[evaluate] @ git+https://github.com/jang1563/bioreview-bench.git@v4.1.3"
```

### Authorized local evaluation

The commands below require an authorized local full-schema copy at
`data/splits/v4/{train,val,test}.jsonl`. They cannot be run against the public
Hugging Face index, which contains neither manuscript/review prose nor test
targets.

```bash
# Evaluate a prediction file against an authorized local validation split
bioreview-run --tool-output predictions.jsonl --tool-name "MyTool" --split val

# Run the built-in baseline reviewer on an authorized local split
bioreview-baseline --split val --model claude-haiku-4-5-20251001

# Or use another supported provider
bioreview-baseline --split val --provider google --model gemini-2.5-flash-lite

# Run the zero-cost lexical baseline
bioreview-bm25 --split val

# Regenerate split stats and verify docs stay in sync
bioreview-stats --check-docs

# Create a system-label-blinded two-rater pack (annotation must still be completed)
bioreview-validation-pack create \
  --output-dir validation_pack \
  --concern-sample-size 300 \
  --omission-sample-size 100

# After both rater files are complete and locked
bioreview-validation-pack summarize --pack-dir validation_pack
```

The pack hides extracted category/severity labels from raters, but retains
source, title, and review context; it is not source-anonymous. Audit rates are
unweighted descriptions of the realized stratified sample. Agreement stability
intervals resample whole articles and are not population confidence intervals.

### Quick evaluation API

```python
import json
from pathlib import Path

from bioreview_bench.evaluate.metrics import quick_eval

with Path("data/splits/v4/val.jsonl").open(encoding="utf-8") as handle:
    article = json.loads(next(handle))  # authorized local full-schema row

result = quick_eval(
    tool_concerns=["No negative control for IP.", "Multiple testing not corrected."],
    gt_entry=article,
)
print(f"Recall: {result.recall:.2f}, Precision: {result.precision:.2f}")
```

---

## Private canonical schema

The schema below describes authorized local full-data rows, not the
rights-minimized public Hugging Face configs shown above.

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
| train | 5,387 | 79,121 | 14.7 |
| validation | 953 | 14,101 | 14.8 |
| test | 600 | 8,647 | 14.4 |
| **Total** | **6,940** | **101,869** | **14.7** |

These realized article proportions are 77.6% train, 13.7% validation, and 8.6%
test. The frozen test set used fixed source counts (150 eLife, 150 PLOS, 150
F1000Research, 100 Nature, and 50 PeerJ). After test selection, the remaining
records were randomly shuffled within
`(source, editorial_decision, review_format)` strata and approximately 15% of
each eligible stratum was assigned to validation. The builder used no explicit
temporal stratification or chronological holdout.

### Source distribution

| Source | Articles | Notes |
|--------|----------|-------|
| F1000Research | 2,679 | Open peer review with named reviewers, 2013-present |
| eLife | 1,810 | 2019-2026; journal and reviewed_preprint formats |
| PLOS | 1,737 | PLOS ONE, PLOS Biology, and other PLOS journals |
| Nature | 470 | Nature Communications and Nature journals, PDF-based |
| PeerJ | 244 | Open peer review, 2018-present |

### Severity distribution

| Severity | Count | % |
|----------|-------|---|
| major | 63,617 | 62.4% |
| minor | 35,869 | 35.2% |
| optional | 2,383 | 2.3% |

### Author stance distribution

| Stance | Count | % |
|--------|-------|---|
| no_response | 92,836 | 91.1% |
| partial | 5,229 | 5.1% |
| conceded | 3,250 | 3.2% |
| rebutted | 491 | 0.5% |
| unclear | 63 | 0.1% |

### Category distribution

| Category | Count | % |
|----------|-------|---|
| writing_clarity | 35,166 | 34.5% |
| missing_experiment | 15,715 | 15.4% |
| interpretation | 15,395 | 15.1% |
| design_flaw | 10,462 | 10.3% |
| prior_art_novelty | 7,550 | 7.4% |
| reagent_method_specificity | 7,282 | 7.1% |
| statistical_methodology | 5,090 | 5.0% |
| figure_issue | 4,950 | 4.9% |
| other | 259 | 0.3% |

---

## Evaluation Protocol

The frozen v4 scores used an unadapted `allenai/specter2_base` checkpoint loaded
through `SentenceTransformer`, which supplied automatic mean pooling; no
SPECTER2 task adapter or resolved revision was recorded. Matching then used
cosine similarity, threshold-aware Hungarian assignment, and a threshold of
0.65. This historical matcher is provisional and has not completed independent
human-equivalence validation. See
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) for the full specification and
[LIMITATIONS_AND_ETHICS.md](LIMITATIONS_AND_ETHICS.md) for the claim boundary.

**Primary metrics:**

| Metric | Description |
|--------|-------------|
| `recall` | Fraction of recorded reference concerns matched by the tool |
| `precision` | Fraction of tool concerns matched to a recorded reference concern |
| `f1` | Harmonic mean of recall and precision |
| `recall_major` | Recall restricted to major-severity concerns |
| `f1_macro` | Future-run secondary metric: unweighted mean of represented-category F1 values |

The frozen artifacts include article-level bootstrap 95% confidence intervals
for recall and precision only (1,000 iterations). F1 and `recall_major` are
reported as point estimates; no interval is claimed for those fields.

**Historical `f1_macro` warning:** all six frozen raw result JSONs contain
`"f1_macro": 0.0` as an unpopulated legacy sentinel even though their
per-category F1 values are non-zero. Historical category-macro F1 is invalid
and unreported. v4.1.2 implemented future-run aggregation only; the frozen raw
files and historical metric values were not backfilled.

---

## Frozen v4 score snapshot

The table below preserves the six historical runs on the frozen **600-article
v4 test snapshot**. It is published for transparency, not operated as an open
public leaderboard. The public index does not provide the inputs or targets
needed to reproduce or submit test scores.

The separately reported **450-article non-eLife analysis** is a post-hoc
source-subset diagnostic. It is not the official snapshot, a replacement test
set, or a basis for reranking the table below.

Both the historical embedding cutoff (0.65) and lexical Jaccard operating
points are uncalibrated. v4.1.3 adds a deterministic threshold-sweep mode that
reports aggregate F1, match counts, ranks, and Kendall tau-a across operating
points. The sweep demonstrates sensitivity; it does not select or validate a
semantic-equivalence threshold.

Frozen artifact construction policy:

- The preserved table uses only `split="test"` result files from `results/v4/`.
- Experimental `dedup_gt=true` runs are excluded from the default ranking.
- If multiple result files exist for the same `(tool_name, tool_version)`, only the strongest run by `f1_micro` is retained.
- `tool_version` should record the exact model or release identifier (for example, `claude-haiku-4-5-20251001`), not `unknown`.

| Rank | Tool | Version | Recall | 95% CI | Precision | 95% CI | F1 | Major Recall | Articles |
|------|------|---------|--------|--------|-----------|--------|----|--------------|----------|
| 1 | Haiku-4.5 | `claude-haiku-4-5-20251001` | 0.759 | [0.732, 0.790] | 0.692 | [0.667, 0.718] | 0.724 | 0.893 | 600 |
| 2 | Gemini-2.5-Flash | `gemini-2.5-flash` | 0.738 | [0.710, 0.768] | 0.703 | [0.679, 0.730] | 0.720 | 0.880 | 600 |
| 3 | GPT-4o-mini | `gpt-4o-mini` | 0.717 | [0.691, 0.748] | 0.721 | [0.698, 0.747] | 0.719 | 0.856 | 600 |
| 4 | BM25 | `bm25-specter2` | 0.668 | [0.642, 0.698] | 0.761 | [0.738, 0.786] | 0.711 | 0.810 | 600 |
| 5 | Llama-3.3-70B | `llama-3.3-70b` | 0.614 | [0.589, 0.643] | 0.785 | [0.764, 0.808] | 0.689 | 0.802 | 600 |
| 6 | Gemini-Flash-Lite | `gemini-2.5-flash-lite` | 0.643 | [0.614, 0.675] | 0.728 | [0.703, 0.754] | 0.683 | 0.800 | 600 |

> Ranking metric: micro-averaged F1 (`f1_micro`). The displayed 95% intervals
> apply only to recall and precision and use 1,000 article-level bootstrap resamples.
> Historical matching: unadapted `allenai/specter2_base` with automatic mean
> pooling, no recorded task adapter or checkpoint revision, cosine threshold
> 0.65, and threshold-aware cardinality-first Hungarian assignment. Scores are
> provisional and matcher-dependent.
> Figure-issue concerns excluded from the reference set (require visual inspection).
> The frozen raw `f1_macro=0.0` fields are invalid legacy sentinels, not measured
> zeros; historical category-macro F1 is unreported.
> Known split-family and cross-release overlap limitations are documented in
> [KNOWN_ISSUES.md](KNOWN_ISSUES.md), with aggregate evidence in the
> [v4 measurement audit](results/v4/measurement_audit.md).

Official release artifacts are rebuilt from raw result JSON files with:

```bash
uv run python scripts/rebuild_release_artifacts.py \
  --results-dir results/v4 \
  --output-dir results/v4 \
  --split test
```

This regenerates `results/v4/leaderboard.md`, `results/v4/leaderboard.json`, and
`results/v4/release_manifest.json`.

The deterministic measurement audit is checked in as
`results/v4/measurement_audit.{json,md}`. It rescored the same frozen outputs
under a lexical matcher and reconstructed only the 42 F1000 cross-split-family
BM25 queries, requiring all 42 rows to match the frozen predictions before
applying query-time candidate filtering. v4.1.3 tooling additionally accepts
the validation split, emits an aggregate F1000 DOI-stem split-family matrix,
and supports a deterministic Jaccard threshold sweep:

```bash
uv run python scripts/audit_v4_measurement.py \
  --train /authorized/data/splits/v4/train.jsonl \
  --validation /authorized/data/splits/v4/val.jsonl \
  --test /authorized/data/splits/v4/test.jsonl \
  --tool-output-dir /authorized/tool_outputs
```

The command requires authorized local full-schema inputs. Public outputs contain
aggregate counts and metrics only, never concern, review, or manuscript text.

---

## Task Definition

See [TASK_DEFINITION.md](TASK_DEFINITION.md) for the complete task
specification, including input/output formats, scoring rules, and authorized
result-record/reporting requirements.

The task definition describes the research protocol for authorized local
copies. It does not imply that the rights-minimized public index includes
manuscript text, normalized targets, or a public scoring service.

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
  note    = {Version 4.1.3 (v4.1.3)}
}
```

---

## License

This project uses a dual license:

- **Benchmark annotations and packaging metadata**: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- **Underlying article, review, and author-response content**: source-specific. Redistribution rights are not uniform across eLife, PLOS, F1000Research, PeerJ, Nature Portfolio, and other future sources; see [LICENSE_MATRIX.md](LICENSE_MATRIX.md).
- **Code** (Python package, evaluation harness, scripts): [Apache-2.0](LICENSE).

Users who redistribute or build upon the benchmark must provide appropriate attribution to both bioreview-bench and the original source articles (via DOIs included in the dataset), and must follow the per-source redistribution rules in [LICENSE_MATRIX.md](LICENSE_MATRIX.md).

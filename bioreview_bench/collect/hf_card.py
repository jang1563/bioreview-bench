"""Generate HuggingFace DatasetCard (README.md) for the bioreview-bench repo.

Produces YAML front matter with multi-config metadata plus a Markdown body
with usage examples, schema documentation, and licensing information.

The stats dict is produced by ``hf_export.export_all_configs()``.
"""

from __future__ import annotations

from typing import Any

_CONFIGS = ("default", "benchmark", "concerns_flat", "elife", "plos", "f1000")


def generate_dataset_card(stats: dict[str, Any]) -> str:
    """Return full README.md content for the HF dataset repo."""
    yaml = _build_yaml(stats)
    body = _build_body(stats)
    return f"---\n{yaml}---\n\n{body}"


# ── YAML front matter ──────────────────────────────────────────────

def _build_yaml(stats: dict[str, Any]) -> str:
    """Build YAML metadata block (without surrounding --- fences)."""
    lines: list[str] = []

    # Top-level metadata
    lines.extend([
        "language:",
        "  - en",
        "license: cc-by-4.0",
        "task_categories:",
        "  - text-classification",
        "  - text-generation",
        "tags:",
        "  - peer-review",
        "  - biomedical",
        "  - benchmark",
        "  - scientific-review",
        "  - elife",
        "  - rebuttal",
        "  - open-peer-review",
        'pretty_name: "BioReview-Bench"',
        "size_categories:",
        f"  - {_size_category(stats.get('total_articles', 0))}",
        "",
    ])

    # Configs
    lines.append("configs:")
    for cfg_name in _CONFIGS:
        is_default = cfg_name == "default"
        lines.append(f"  - config_name: {cfg_name}")
        if is_default:
            lines.append("    default: true")
        lines.append("    data_files:")
        for split in ("train", "validation", "test"):
            lines.append(f"      - split: {split}")
            lines.append(f'        path: "data/{cfg_name}/{split}.jsonl"')
    lines.append("")

    # dataset_info per config
    lines.append("dataset_info:")
    configs = stats.get("configs", {})
    for cfg_name in _CONFIGS:
        cfg = configs.get(cfg_name, {})
        splits = cfg.get("splits", {})
        lines.append(f"  - config_name: {cfg_name}")
        lines.append("    splits:")
        for split in ("train", "validation", "test"):
            n = splits.get(split, {}).get("num_rows", 0)
            lines.append(f"      - name: {split}")
            lines.append(f"        num_examples: {n}")
    lines.append("")

    return "\n".join(lines) + "\n"


def _size_category(n: int) -> str:
    if n < 1000:
        return "n<1K"
    if n < 10000:
        return "1K<n<10K"
    if n < 100000:
        return "10K<n<100K"
    return "100K<n<1M"


# ── Markdown body ──────────────────────────────────────────────────

def _build_body(stats: dict[str, Any]) -> str:
    """Build the Markdown content of the dataset card."""
    configs = stats.get("configs", {})
    total = stats.get("total_articles", 0)
    total_concerns = configs.get("default", {}).get("total_concerns", 0)

    sections = [
        _section_overview(total, total_concerns, configs),
        _section_configs(configs),
        _section_usage(),
        _section_schema(),
        _section_license(),
        _section_citation(),
    ]
    return "\n\n".join(sections) + "\n"


def _section_overview(
    total: int,
    total_concerns: int,
    configs: dict[str, Any],
) -> str:
    source_dist = {}
    for split_stats in configs.get("default", {}).get("splits", {}).values():
        for src, cnt in split_stats.get("source_distribution", {}).items():
            source_dist[src] = source_dist.get(src, 0) + cnt

    source_line = ", ".join(f"{src} ({cnt})" for src, cnt in sorted(source_dist.items()))

    return f"""# BioReview-Bench

A benchmark and training dataset for AI-assisted biomedical peer review.

- **{total:,} articles** with **{total_concerns:,} reviewer concerns**
- Sources: {source_line}
- Concern-level labels: 9 categories, 3 severity levels, 5 author stance types
- License: Data CC-BY-4.0 | Code Apache-2.0

## What makes this dataset unique

No other publicly available dataset provides **structured, concern-level
peer review data** for biomedical papers with:
- Categorised reviewer concerns (design flaw, statistical methodology, etc.)
- Severity labels (major / minor / optional)
- Author response tracking (conceded / rebutted / partial / unclear / no_response)
- Evidence-of-change flags"""


def _section_configs(configs: dict[str, Any]) -> str:
    rows: list[str] = []
    for cfg_name in _CONFIGS:
        cfg = configs.get(cfg_name, {})
        total_rows = cfg.get("total_rows", 0)
        total_c = cfg.get("total_concerns", 0)
        rows.append(f"| `{cfg_name}` | {total_rows:,} | {total_c:,} |")

    table = "\n".join(rows)

    return f"""## Configs

| Config | Total rows | Total concerns |
|--------|-----------|---------------|
{table}

- **`default`**: Full data — all fields, all sources. Use for analysis and research.
- **`benchmark`**: Task input format for AI review tool evaluation. Train/val include
  simplified concerns (text + category + severity). Test split has `concerns=[]` to
  prevent label leakage.
- **`concerns_flat`**: One row per concern with article context. Ideal for rebuttal
  generation training and stance classification. PLOS entries included (filter with
  `author_stance != "no_response"` for rebuttal tasks).
- **`elife`** / **`plos`** / **`f1000`**: Source-specific subsets of `default`."""


def _section_usage() -> str:
    return """## Quick start

```python
from datasets import load_dataset

# Full dataset (default config)
ds = load_dataset("jang1563/bioreview-bench")

# Benchmark evaluation — test split has no concerns (your tool generates them)
ds = load_dataset("jang1563/bioreview-bench", "benchmark")
for article in ds["test"]:
    text = article["paper_text_sections"]
    # ... run your review tool, then evaluate with bioreview_bench.evaluate.metrics

# Training a review generation model
ds = load_dataset("jang1563/bioreview-bench", "benchmark")
for article in ds["train"]:
    target_concerns = article["concerns"]  # [{concern_text, category, severity}]

# Rebuttal generation / stance classification
ds = load_dataset("jang1563/bioreview-bench", "concerns_flat")
for row in ds["train"]:
    concern = row["concern_text"]
    response = row["author_response_text"]
    stance = row["author_stance"]  # conceded / rebutted / partial / unclear / no_response

# Source-specific analysis
ds = load_dataset("jang1563/bioreview-bench", "elife")
```"""


def _section_schema() -> str:
    return """## Schema

### Article fields (default config)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Article ID (e.g. `elife:84798`) |
| `source` | string | Journal source (`elife`, `plos`, `f1000`) |
| `doi` | string | Article DOI |
| `title` | string | Article title |
| `abstract` | string | Abstract text |
| `subjects` | list[string] | Subject areas |
| `published_date` | string | ISO date |
| `paper_text_sections` | dict | Section name → text |
| `decision_letter_raw` | string | Raw peer review text |
| `author_response_raw` | string | Raw author response |
| `concerns` | list[object] | Extracted reviewer concerns |

### Concern fields

| Field | Type | Description |
|-------|------|-------------|
| `concern_id` | string | Unique ID (e.g. `elife:84798:R1C3`) |
| `concern_text` | string | Reviewer's concern (10-2000 chars) |
| `category` | string | One of 9 types (see below) |
| `severity` | string | `major` / `minor` / `optional` |
| `author_response_text` | string | Author's response to this concern |
| `author_stance` | string | `conceded` / `rebutted` / `partial` / `unclear` / `no_response` |
| `evidence_of_change` | bool? | Whether author made revisions |
| `resolution_confidence` | float | LLM confidence (0.0-1.0) |

### Concern categories

`design_flaw`, `statistical_methodology`, `missing_experiment`, `figure_issue`,
`prior_art_novelty`, `writing_clarity`, `reagent_method_specificity`,
`interpretation`, `other`"""


def _section_license() -> str:
    return """## License

- **Dataset** (JSONL data files): CC-BY-4.0. All source articles and reviews are
  published under CC-BY by their respective journals (eLife, PLOS, F1000Research).
- **Code** (Python package, evaluation harness): Apache-2.0.

See the [GitHub repository](https://github.com/jang1563/bioreview-bench) for
full license details."""


def _section_citation() -> str:
    return """## Citation

If you use this dataset, please cite:

```bibtex
@misc{bioreview-bench,
  title={BioReview-Bench: A Benchmark for AI-Assisted Biomedical Peer Review},
  author={Kim, JangKeun},
  year={2026},
  url={https://huggingface.co/datasets/jang1563/bioreview-bench}
}
```"""

# BioReview-Bench v4 Measurement Audit

**Status:** aggregate computational diagnostic. This audit does not replace the frozen benchmark, semantic-match validation, or human review.

## Fixed protocol

- Scope: 5387 train, 953 validation, and 600 test articles.
- Selected historical-heuristic Jaccard operating point: `0.195` (`0.65 × 0.3`); threshold-aware one-to-one Hungarian assignment; micro averaging.
- Matcher operating points are uncalibrated and have not received independent human-equivalence validation.
- Figure-dependent reference concerns are excluded.
- Frozen tool-output ID coverage: 6 / 6 models exactly cover all 600 test article IDs.
- BM25 family intervention: query-time candidate-list filtering after scoring; original corpus, postings, IDF, and average document length held fixed (no corpus reindexing).

## F1000 DOI-stem family split overlap (lower bound)

- 42 / 150 F1000 test articles (28.0%) share a version-independent DOI family with training.
- Overlapping families: 41; training rows in those families: 51.
- Same-family training article at BM25 rank 1: 42 / 42 (100.0%).
- Same-family training article within top 8: 42 / 42 (100.0%).

- This matrix covers explicit F1000Research DOI-version suffixes only; all counts are aggregate and text-free.
- Parseable DOI-family coverage (train / validation / test): 2149 / 380 / 150 articles.

| Partition relation | Crossing F1000 DOI-stem families |
|---|---:|
| train–validation | 115 |
| train–test | 41 |
| validation–test | 9 |
| development (train ∪ validation)–test | 48 |

- Unique F1000 DOI-stem families crossing any split boundary: 161.
- Development-overlapping F1000 test articles: 49 / 150 (32.7%).

## Frozen BM25 reconstruction

- Reconstruction scope: targeted F1000 test queries whose version-independent manuscript family crosses the train/test split.
- Exact targeted prediction rows: 42 / 42.
- Rows changed by query-time manuscript-family candidate filtering: 42.
- Defaults: `top_k_docs=8`, `max_concerns=12`, `max_input_chars=40000`, `k1=1.5`, `b=0.75`.

## BM25 current vs query-time family candidate filtering

| Evaluation subset | Frozen current | Query-time family-filtered | ΔF1 | Δmatches |
|---|---|---|---:|---:|
| full_test | R=0.0541, P=0.0617, F1=0.0577, matches=444 | R=0.0139, P=0.0158, F1=0.0148, matches=114 | -0.0429 | -330 |
| no_elife_450 | R=0.0600, P=0.0813, F1=0.0690, matches=439 | R=0.0149, P=0.0202, F1=0.0171, matches=109 | -0.0519 | -330 |
| f1000_test | R=0.1750, P=0.2189, F1=0.1945, matches=394 | R=0.0284, P=0.0356, F1=0.0316, matches=64 | -0.1629 | -330 |
| f1000_overlapping_families | R=0.4930, P=0.6984, F1=0.5780, matches=352 | R=0.0308, P=0.0437, F1=0.0361, matches=22 | -0.5419 | -330 |

## Jaccard operating-point sensitivity

All rows are aggregate, text-free diagnostics. No threshold is a validated semantic-equivalence operating point.

### full_600 (primary frozen test snapshot)

| Jaccard cutoff | Top system | Kendall tau-a vs frozen embedding | Total matches across systems |
|---:|---|---:|---:|
| 0.050 | GPT-4o-mini | +0.4667 | 32645 |
| 0.075 | GPT-4o-mini | +0.4667 | 27886 |
| 0.100 | GPT-4o-mini | +0.2000 | 19332 |
| 0.125 | GPT-4o-mini | +0.2000 | 11033 |
| 0.150 | GPT-4o-mini | +0.3333 | 5472 |
| 0.175 | Llama-3.3-70B | -0.0667 | 2694 |
| 0.195 | BM25 | -0.0667 | 1567 |
| 0.200 | BM25 | -0.0667 | 1472 |
| 0.225 | BM25 | +0.0667 | 837 |
| 0.250 | BM25 | -0.0667 | 595 |
| 0.275 | BM25 | +0.0667 | 460 |
| 0.300 | BM25 | +0.2000 | 416 |

### no_elife_450 (post-hoc sensitivity subset)

| Jaccard cutoff | Top system | Kendall tau-a vs frozen embedding | Total matches across systems |
|---:|---|---:|---:|
| 0.050 | Haiku-4.5 | +0.7333 | 27743 |
| 0.075 | GPT-4o-mini | +0.3333 | 23655 |
| 0.100 | GPT-4o-mini | +0.0667 | 16318 |
| 0.125 | GPT-4o-mini | +0.0667 | 9315 |
| 0.150 | GPT-4o-mini | +0.0667 | 4664 |
| 0.175 | BM25 | -0.2000 | 2339 |
| 0.195 | BM25 | -0.2000 | 1384 |
| 0.200 | BM25 | -0.2000 | 1306 |
| 0.225 | BM25 | -0.0667 | 766 |
| 0.250 | BM25 | -0.2000 | 560 |
| 0.275 | BM25 | -0.0667 | 448 |
| 0.300 | BM25 | +0.0667 | 406 |

## Frozen embedding vs one-to-one Jaccard at the selected 0.195 point: full_600 (primary frozen test snapshot)

| Rank | Frozen embedding | F1 | One-to-one Jaccard | F1 |
|---:|---|---:|---|---:|
| 1 | Haiku-4.5 | 0.7238 | BM25 | 0.0577 |
| 2 | Gemini-2.5-Flash | 0.7200 | Llama-3.3-70B | 0.0375 |
| 3 | GPT-4o-mini | 0.7192 | GPT-4o-mini | 0.0323 |
| 4 | BM25 | 0.7114 | Haiku-4.5 | 0.0288 |
| 5 | Llama-3.3-70B | 0.6892 | Gemini-2.5-Flash | 0.0224 |
| 6 | Gemini-Flash-Lite | 0.6828 | Gemini-Flash-Lite | 0.0193 |

- Kendall tau-a: **-0.0667**.

## Frozen embedding vs one-to-one Jaccard at the selected 0.195 point: no_elife_450 (post-hoc sensitivity subset)

| Rank | Frozen embedding | F1 | One-to-one Jaccard | F1 |
|---:|---|---:|---|---:|
| 1 | Haiku-4.5 | 0.7614 | BM25 | 0.0690 |
| 2 | Gemini-2.5-Flash | 0.7544 | Llama-3.3-70B | 0.0389 |
| 3 | GPT-4o-mini | 0.7490 | GPT-4o-mini | 0.0328 |
| 4 | BM25 | 0.7281 | Haiku-4.5 | 0.0289 |
| 5 | Gemini-Flash-Lite | 0.7005 | Gemini-2.5-Flash | 0.0233 |
| 6 | Llama-3.3-70B | 0.6958 | Gemini-Flash-Lite | 0.0200 |

- Kendall tau-a: **-0.2000**.

## Limitations

- The family matrix is an F1000 DOI-stem lower bound only; it does not detect fuzzy-title, preprint-to-journal, or cross-source families.
- Jaccard thresholds and the historical embedding threshold are uncalibrated operating points. The sweep is a lexical sensitivity analysis, not a human-validated semantic-equivalence measure.
- The BM25 family intervention is query-time candidate-list filtering after scoring. The original corpus, postings, IDF, and average document length are fixed; this is not a rebuilt leave-family-out index and does not alter any LLM prediction.
- The embedding ranking is read from frozen aggregate results and is not recomputed here; its historical matcher threshold and model revision remain independently unvalidated.
- A same-family high retrieval rank demonstrates a measurement confound, but does not prove that every copied concern is invalid.
- The Git repository does not redistribute benchmark JSONL or tool outputs; exact reruns require an authorized local copy of those artifacts.

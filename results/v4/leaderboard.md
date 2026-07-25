# bioreview-bench Frozen Score Snapshot (test split)

*Last updated: 2026-07-25. Ranked by F1.*

| Rank | Tool | Version | Recall | 95% CI | Precision | 95% CI | F1 | Major Recall | Articles | Date |
|------|------|---------|--------|--------|-----------|--------|----|--------------|----------|------|
| 1 | Haiku-4.5 | claude-haiku-4-5-20251001 | 0.759 | [0.732, 0.790] | 0.692 | [0.667, 0.718] | 0.724 | 0.893 | 600 | 2026-03-30 |
| 2 | Gemini-2.5-Flash | gemini-2.5-flash | 0.738 | [0.710, 0.768] | 0.703 | [0.679, 0.730] | 0.720 | 0.880 | 600 | 2026-03-30 |
| 3 | GPT-4o-mini | gpt-4o-mini | 0.717 | [0.691, 0.748] | 0.721 | [0.698, 0.747] | 0.719 | 0.856 | 600 | 2026-03-30 |
| 4 | BM25 | bm25-specter2 | 0.668 | [0.642, 0.698] | 0.761 | [0.738, 0.786] | 0.711 | 0.810 | 600 | 2026-03-30 |
| 5 | Llama-3.3-70B | llama-3.3-70b | 0.614 | [0.589, 0.643] | 0.785 | [0.764, 0.808] | 0.689 | 0.802 | 600 | 2026-03-30 |
| 6 | Gemini-Flash-Lite | gemini-2.5-flash-lite | 0.643 | [0.614, 0.675] | 0.728 | [0.703, 0.754] | 0.683 | 0.800 | 600 | 2026-03-30 |

> Frozen historical matcher wrapper: unadapted `allenai/specter2_base` with automatic mean pooling (checkpoint revision: unrecorded in historical runs), threshold=0.65; threshold-aware cardinality-first Hungarian matching.
> Figure-issue concerns excluded from ground truth (require visual inspection).
> Scores are matcher-dependent and provisional pending independent human validation of the matching threshold.
> The six frozen raw result files store `f1_macro=0.0` as an unpopulated legacy sentinel; historical category-macro F1 is invalid and unreported.
> Historical snapshot only; this is not an open public leaderboard. See [KNOWN_ISSUES.md](https://github.com/jang1563/bioreview-bench/blob/v4.1.3/KNOWN_ISSUES.md).
> [bioreview-bench v4.1.3](https://github.com/jang1563/bioreview-bench)

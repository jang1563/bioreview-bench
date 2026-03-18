# bioreview-bench Leaderboard (test split)

*Last updated: 2026-03-18. Ranked by F1.*

| Rank | Tool | Version | Recall | 95% CI | Precision | 95% CI | F1 | Major Recall | Articles | Date |
|------|------|---------|--------|--------|-----------|--------|----|--------------|----------|------|
| 1 | Haiku-4.5 | claude-haiku-4-5-20251001 | 0.725 | [0.697, 0.753] | 0.675 | [0.653, 0.694] | 0.699 | 0.872 | 944 | 2026-03-18 |
| 2 | GPT-4o-mini | gpt-4o-mini | 0.684 | [0.658, 0.712] | 0.703 | [0.682, 0.722] | 0.694 | 0.840 | 944 | 2026-03-18 |
| 3 | Gemini-2.5-Flash | gemini-2.5-flash | 0.665 | [0.639, 0.693] | 0.709 | [0.688, 0.729] | 0.686 | 0.832 | 944 | 2026-03-18 |
| 4 | BM25 | bm25-specter2 | 0.637 | [0.611, 0.664] | 0.741 | [0.721, 0.760] | 0.685 | 0.794 | 944 | 2026-03-18 |
| 5 | Gemini-Flash-Lite | gemini-2.5-flash-lite | 0.615 | [0.588, 0.644] | 0.708 | [0.685, 0.729] | 0.658 | 0.781 | 944 | 2026-03-18 |
| 6 | Llama-3.3-70B | llama-3.3-70b | 0.554 | [0.530, 0.580] | 0.794 | [0.774, 0.811] | 0.653 | 0.753 | 944 | 2026-03-18 |

> Matching: SPECTER2 cosine similarity, threshold=0.65, hungarian bipartite matching.
> Figure-issue concerns excluded from ground truth (require visual inspection).
> [bioreview-bench v1.0](https://github.com/jang1563/bioreview-bench)
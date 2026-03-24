# v4 Pre-Release Diagnostics

> Date: 2026-03-22

## 1. PeerJ Baseline Gap (F1 = 0.000)

**Finding:** All 37 missing Haiku baseline outputs are `peerj:*` article IDs.

- Test split: 981 articles (37 PeerJ)
- Haiku output: 944 articles (0 PeerJ)
- GPT-4o-mini, Gemini, BM25: all have 981 articles including PeerJ

**Root cause:** Operational error. The Haiku baseline was run before PeerJ articles
were added to the test split, or from a split file that excluded PeerJ.
No code bug — the baseline pipeline correctly processes PeerJ articles
(GPT and BM25 both produced PeerJ output).

**Fix:** Re-run Haiku baseline on the full test set. No code change needed.

---

## 2. eLife Over-Extraction (Precision = 38%)

**Finding:** 76% of eLife test articles have `decision_letter_raw` < 1,000 characters.
These contain only the eLife Assessment summary, not the individual referee reports.

Distribution of `decision_letter_raw` length (273 eLife test articles):

| DL Length | Count | % | Mean Concerns |
|-----------|-------|---|---------------|
| < 1,000 chars | 207 | 76% | 5.3 |
| 1,000–5,000 | 8 | 3% | — |
| > 5,000 chars | 58 | 21% | 8.0 |

- 235/273 (86%) have no "Reviewer #" or "Referee #" headers in the raw text
- 213/273 are `reviewed_preprint` format (post-2022)
- 59/273 are `journal` format (pre-2022) — these typically have full reviewer reports

**Root cause:** For the `reviewed_preprint` format, the JATS parser stores only
the eLife Assessment (a short editorial summary) in `decision_letter_raw`. The
individual referee-report sub-articles are parsed separately during collection
as `ParsedReview` objects, but not concatenated into `decision_letter_raw`.

This means the GT extractor sees only the summary (~500 chars), producing ~5
concerns. The baseline reviewer sees the full paper text and generates ~15
concerns, causing the 2.6x over-extraction ratio and 38% precision.

**Impact on v4:**
- The ensemble GT approach will partially mitigate this: GPT-4o-mini extraction
  from the same short Assessment will also produce ~5 concerns, so the intersection
  is consistent (both extractors agree on the Assessment-derived concerns).
- The balanced test set reduces eLife from 273 → 150 articles, decreasing its
  impact on aggregate metrics.
- A full fix requires re-processing eLife reviewed_preprint articles to concatenate
  referee-report sub-articles into `decision_letter_raw`. This is a data-cost item
  deferred to the API-cost phase.

**Potential future fix (deferred):**
In `bioreview_bench/parse/jats.py`, modify the reviewed_preprint parsing to
concatenate all referee-report sub-articles into `decision_letter_raw`:
```python
# Concatenate: eLife Assessment + all referee reports
decision_letter_raw = assessment_text + "\n\n" + "\n\n".join(
    f"Referee Report #{i+1}:\n{report.review_text}"
    for i, report in enumerate(referee_reports)
)
```
Then re-run concern extraction on affected articles (~1,600 eLife articles).

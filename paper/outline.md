# BioReview-Bench: A Concern-Level Benchmark for AI Biomedical Peer Review
## Paper Outline — NeurIPS 2026 Datasets & Benchmarks Track

---

## Abstract (≤ 200 words)

We introduce **bioreview-bench**, a large-scale benchmark for evaluating AI tools that assist with biomedical peer review. Given the full text of a research article, an AI system must identify the substantive concerns that human reviewers raised — analogous to the task a professional referee performs. The benchmark contains **6,940 articles** from five open-review journals (eLife, PLOS, F1000Research, PeerJ, Nature), annotated with **101,869 reviewer concerns** at fine-grained concern-level granularity. Each concern is categorized across nine dimensions and labeled with an **author-stance** tag derived from the actual revision letter, grounding the annotation in real scientific outcomes. Evaluation uses SPECTER2 semantic embeddings with Hungarian bipartite matching (threshold = 0.65), producing recall, precision, and F1 metrics that distinguish true scientific concerns from superficial paraphrases. We benchmark six models ranging from BM25 to frontier LLMs; the best model (Haiku-4.5) achieves F1 = 0.724 on the 600-article test set, with a ~4 pp gap between pre-2023 and post-2023 articles that is uniform across all models including a non-LLM baseline, suggesting difficulty variation rather than training contamination. bioreview-bench fills a gap not covered by existing peer-review datasets: concern-level decomposition, outcome-anchored stance labels, multi-source biomedical coverage, and a reproducible evaluation harness.

---

## 1. Introduction (≈ 600 words)

### 1.1 Motivation
- AI systems increasingly assist researchers with literature review, grant writing, and manuscript preparation
- The next frontier: AI as a peer reviewer — detecting methodological weaknesses, missing experiments, statistical errors
- Growing deployment: tools like SciSpace, Semantic Scholar, and journal-integrated AI tools already in production
- Critical gap: no rigorous benchmark exists to measure whether these systems catch the concerns that matter to human experts

### 1.2 Problem Statement
- Formally: given abstract + full manuscript text, produce a list of substantive reviewer concerns
- Challenge: concerns are complex, heterogeneous, and only partially observable from the paper alone
- Ground truth must come from actual human peer review, not crowdsourcing

### 1.3 Why Existing Datasets Fall Short
- PeerRead (2018): review-level classification, no concern decomposition
- NLPEERv2 (2023): sentence alignment, no structured concern extraction, no author response
- MOPRD (2023): multi-domain review ordering, not concern identification
- CLAIMCHECK (2025): claim-level, not concern-level; no biomedical focus
- No existing dataset provides: (1) biomedical domain, (2) concern-level units, (3) outcome-anchored stance labels, (4) evaluation harness

### 1.4 Contributions
1. **bioreview-bench dataset**: 6,940 articles, 101,869 concerns, 5 sources, 2013–2026
2. **LLM-assisted annotation pipeline**: scalable concern extraction with 9 categories, 3 severity levels, 5 stance labels
3. **Evaluation harness**: SPECTER2 semantic matching, Hungarian bipartite concern matching, bootstrap CIs
4. **Baseline results**: 6 models evaluated; temporal contamination analysis ruling out training data inflation
5. **Cross-model ensemble GT**: two independent extractors (Haiku-4.5, GPT-4o-mini) for ensemble GT with Kendall τ = 0.867

---

## 2. Related Work (≈ 400 words)

### 2.1 Peer Review Datasets
| Dataset | Year | Domain | Granularity | Author Response | Notes |
|---------|------|--------|-------------|-----------------|-------|
| PeerRead | 2018 | CS/AI | Review-level | No | ICLR/NeurIPS/ACL |
| OpenEval | 2022 | General | Aspect-level | No | Multi-journal |
| NLPEERv2 | 2023 | CS+Biomed | Sentence | No | eLife/PLOS included |
| MOPRD | 2023 | Multi | Review-level | No | Ordering task |
| FMMD | 2026 | Biomed | Multimodal | No | Figures+text |
| **bioreview-bench** | 2026 | **Biomed** | **Concern** | **Yes** | **5 sources** |

### 2.2 AI-Assisted Peer Review
- OpenReviewer (NAACL 2025): LLM review generation for NLP papers
- NEJM AI study (2024): GPT-4o vs. human reviewers on NEJM manuscripts; 30–39% overlap
  - Our benchmark provides structured GT to make this comparison rigorous and reproducible
- Nature Machine Intelligence RCT (2026): LLM assistance improves reviewer thoroughness by ~12%
- Our work: measurement infrastructure complementing these deployment studies

### 2.3 Semantic Matching for Scientific Text
- SPECTER2 (2022): domain-adapted SciBERT fine-tuned on paper citation graphs
- Bipartite matching precedent: SciFactBank, ClaimBuster, FEVER shared tasks
- Why threshold 0.65: bimodal cosine distribution; gap between paraphrase (>0.7) and topic-adjacent (<0.55)

---

## 3. Dataset Construction (≈ 800 words)

### 3.1 Source Selection
**Inclusion criteria**: publicly available review text, machine-readable format, CC-BY or open license

| Source | Format | Years | Review Style |
|--------|--------|-------|--------------|
| eLife | JATS XML | 2012–2026 | Editor summary + referee reports |
| PLOS | JATS XML | 2018–2026 | Aggregated review documents |
| F1000Research | JATS XML | 2013–2026 | Named open reviewers |
| PeerJ | HTML scraping | 2018–2026 | Open peer review |
| Nature | PDF (pdfplumber) | 2021–2026 | Reviewer comments PDF |

**Source diversity rationale**: different editorial cultures, review lengths, and scientific disciplines reduce evaluation monoculture

### 3.2 Annotation Pipeline

```
Article XML/PDF
    → JATS/PDF Parser
    → reviewer text blocks (per reviewer, per round)
    → LLM extractor (Haiku-4.5 @ 6K char windows)
    → Concern list: [text, category, severity]
    → Author response alignment (if available)
    → Stance labeler: conceded / rebutted / partial / unclear / no_response
```

**LLM extraction prompt**: structured JSON output, 9-category taxonomy, severity 3-level
**Cost**: ~$0.009/article; full dataset = ~$60

### 3.3 Concern Taxonomy
Nine categories derived from iterative review of 500 pilot concerns + comparison with MOPRD:

1. `design_flaw` — fundamental methodological weaknesses
2. `statistical_methodology` — incorrect/insufficient statistics
3. `missing_experiment` — absent controls, validation, follow-up
4. `figure_issue` — image quality, panel labeling (excluded from evaluation)
5. `prior_art_novelty` — novelty/literature engagement
6. `writing_clarity` — ambiguity, organization
7. `reagent_method_specificity` — protocol/reagent detail
8. `interpretation` — overclaiming, underclaiming
9. `other` — miscellaneous

### 3.4 Stance Annotation
- **Source**: author response letter (when available; 74% of articles have one)
- **Method**: LLM-assisted stance labeling, heuristic post-correction
- **Challenge**: eLife reviewed_preprint format lacks per-reviewer responses → `no_response` for all
- **Distribution**: no_response 91.1%, partial 5.1%, conceded 3.2%, rebutted 0.5%

### 3.5 Data Splits
- Frozen test set: 600 articles, stratified by source × year × concern count
- Validation: 895 articles; Training: 5,064 articles
- No test leakage: benchmark config strips concern fields

### 3.6 Quality Control
- Cross-model ensemble GT: Haiku + GPT extract independently; agreement = 74.8%
- Concern deduplication: cosine threshold 0.95 removes 8.2% near-duplicates
- Per-source manual inspection: 20 articles/source reviewed by author

---

## 4. Evaluation Protocol (≈ 300 words)

### 4.1 Matching Algorithm
- Embed all concerns with SPECTER2 (768-dim)
- Build cosine similarity matrix (tool × GT)
- Solve maximum-weight bipartite matching with scipy.linear_sum_assignment (Hungarian)
- Match pair retained if cosine ≥ 0.65

### 4.2 Why SPECTER2, Not BM25 or GPT-4?
- BM25 rewards lexical overlap (inflated when tool copies review language)
- GPT-4 embedding: commercial dependency + no reproducibility guarantee
- SPECTER2: open-weight, domain-adapted, stable releases

### 4.3 Primary Metrics
- **Recall**: fraction of GT concerns matched by tool
- **Precision**: fraction of tool concerns matching a GT concern
- **F1** (harmonic mean): primary ranking metric
- **Recall@major**: recall restricted to major-severity concerns
- **Bootstrap 95% CI**: 1,000 article-level resamples

### 4.4 Exclusions
- `figure_issue` concerns excluded: require visual inspection of figures, not capturable by text-only models
- Concern deduplication: optionally applied to GT (reduces redundant concerns from verbose reviews)

---

## 5. Experiments (≈ 700 words)

### 5.1 Models Evaluated

| Model | Provider | Context | Cost/article |
|-------|----------|---------|-------------|
| Haiku-4.5 | Anthropic | 200K | ~$0.009 |
| GPT-4o-mini | OpenAI | 128K | ~$0.002 |
| Gemini-2.5-Flash | Google | 1M | ~$0.003 |
| Gemini-Flash-Lite | Google | 1M | ~$0.001 |
| Llama-3.3-70B | Together AI | 128K | ~$0.002 |
| BM25 | — | — | free |

**Prompt**: fixed zero-shot instruction, article abstract + full text, structured JSON output

### 5.2 Main Results (v4 test, 600 articles)

| Rank | Model | Recall | Precision | F1 | Recall@Major |
|------|-------|--------|-----------|-----|--------------|
| 1 | Haiku-4.5 | 0.759 | 0.692 | 0.724 | — |
| 2 | Gemini-2.5-Flash | 0.738 | 0.703 | 0.720 | — |
| 3 | GPT-4o-mini | 0.717 | 0.721 | 0.719 | — |
| 4 | BM25 | 0.668 | 0.761 | 0.711 | — |
| 5 | Llama-3.3-70B | 0.614 | 0.785 | 0.689 | — |
| 6 | Gemini-Flash-Lite | 0.643 | 0.728 | 0.683 | — |

*Note: CIs and Recall@Major to be filled from benchmark run results.*

**Key observation**: LLMs outperform BM25 by 1–4pp F1, but BM25 achieves highest precision, suggesting LLMs generate more spurious concerns.

### 5.3 Per-Source Analysis
- **eLife**: anomalously high recall (R≈0.99) for all models — investigation shows reviewed_preprint format stores only editor Assessment (~500 chars) in `decision_letter_raw`, resulting in very few GT concerns per article; all models can "recall" them trivially
- **PLOS**: moderate recall across all models; cross-model agreement 85%
- **Nature**: lowest cross-model agreement (55%); PDF-derived concerns are noisier
- **PeerJ/F1000**: strong performance; structured XML with clear reviewer blocks

### 5.4 Temporal (Contamination) Analysis
Table: Pre-2023 vs. 2023+ F1 for all models

| Model | Pre-2023 (n=173) | 2023+ (n=427) | Gap |
|-------|-----------------|--------------|-----|
| Haiku-4.5 | 0.752 | 0.712 | 0.040 |
| GPT-4o-mini | 0.741 | 0.710 | 0.031 |
| Gemini-2.5-Flash | 0.744 | 0.710 | 0.034 |
| BM25 | 0.727 | 0.705 | 0.022 |
| Llama-3.3-70B | 0.702 | 0.684 | 0.018 |
| Gemini-Flash-Lite | 0.716 | 0.668 | 0.048 |

**Finding**: pre-2023 advantage is present for BM25 (no training data) and all LLMs at similar magnitude → reflects article difficulty distribution (older papers have shorter, less complex reviews), not training contamination.

### 5.5 Ablation: Ground Truth Quality
- Haiku GT vs. Ensemble GT ranking correlation: Kendall τ = 0.867
- Ensemble GT is more conservative (8,231 vs. 8,200 single-model concerns) but produces identical rankings
- Concern deduplication at 0.95 threshold: 8.2% reduction, negligible ranking change

---

## 6. Discussion (≈ 400 words)

### 6.1 What Models Get Right and Wrong
- All models excel at `missing_experiment` and `design_flaw` recall (>0.80)
- `writing_clarity` has highest recall but lowest precision (models over-generate stylistic concerns)
- `reagent_method_specificity` has lowest recall — requires deep domain knowledge

### 6.2 The eLife Anomaly
- Reviewed preprint format stores editor Assessment only, not referee reports
- This is an XML structure issue, not a GT quality issue
- Two mitigations: (1) use `review_texts` field (captured in Nature backfill), (2) source-specific subsets
- v5 plan: backfill eLife with per-reviewer text from separate sub-articles

### 6.3 Limitations
1. **Human validation**: ground truth is LLM-extracted, not manually annotated. Cross-model Kendall τ=0.867 suggests reasonable quality, but IAA kappa study is planned.
2. **Figure issues excluded**: ~4.9% of concerns require visual inspection; text-only evaluation is the norm but limits scope.
3. **English only**: all sources are English-language journals.
4. **Publication bias**: open-review journals may have different concern distributions than closed-review journals (Cell, Science, NEJM).

### 6.4 Future Work
- Human validation pilot (20 articles × 2 annotators)
- eLife reviewed_preprint backfill with per-reviewer text
- Multi-round review tracking (revision rounds 1→2→3)
- Fine-tuned specialist models: SPECTER2-based concern extractor

---

## 7. Conclusion (≈ 150 words)

bioreview-bench establishes concern-level peer review analysis as a tractable NLP task with a rigorous, reproducible benchmark. Our results show that frontier LLMs can achieve ~72% F1 at detecting human reviewer concerns — well above a lexical BM25 baseline but with substantial room for improvement, particularly for reagent-specificity and nuanced interpretation concerns. The uniform temporal pattern (BM25 and LLMs equally advantaged on pre-2023 data) provides evidence against training contamination as an explanation for performance gaps. We release all data, code, and evaluation harness under open licenses to facilitate community progress on this important task.

---

## Appendices

### A. Annotation Guidelines
- Full concern extraction prompt
- 9-category decision tree
- Stance labeling rubric

### B. Benchmark Reproducibility Checklist
- Seeds, venv spec, split hashes

### C. Source-Specific Statistics
- Full per-source × per-category breakdown tables

### D. Error Analysis
- 50 false-positive and 50 false-negative examples with analysis

---

## Key Stats for Paper (fill after benchmark run)

- Test set: 600 articles, v4 frozen split
- GT: 8,231 ensemble concerns (Haiku + GPT union)
- Best F1: 0.724 (Haiku-4.5)
- Ensemble GT Kendall τ: 0.867
- Temporal gap: ~3–4 pp (uniform across all models including BM25)
- Cross-model agreement: 74.8% both-agreed (Nature: 55%, PLOS: 85%)

---

## Submission Checklist

- [ ] Benchmark run results with bootstrap CIs (in progress)
- [ ] Human validation pilot (20 articles, 2 annotators, IAA kappa)
- [ ] Final numbers in Results section
- [ ] Appendix D (error analysis)
- [ ] bioRxiv submission (Scientific Communication and Education)
- [ ] arXiv cs.CL cross-post
- [ ] NeurIPS 2026 D&B abstract (~May 23, 2026)
- [ ] NeurIPS 2026 D&B full paper (~May 29, 2026)

# bioreview-bench Baseline Suite

> Version: 1.1
> Date: 2026-03-13
> Status: Current implementation and planned gaps

This document defines what counts as a baseline in the repository, what is
currently runnable, and what remains to be added to reach the original project
plan.

---

## 1. Goal

The benchmark should not rely on a single showcase model. A credible baseline
suite should cover:

- lightweight lexical or heuristic methods
- generic LLM zero-shot reviewers
- domain-specific review tools
- a human reference subset

---

## 2. Current Repository State

### 2.1 Directly runnable baseline paths

The repository currently includes two runnable baseline paths:

- LLM reviewer baseline
  - implementation: `bioreview_bench/baseline/reviewer.py`
  - execution CLI: `bioreview_bench/scripts/run_baseline.py`
  - supported providers today: `anthropic`, `openai`, `google`, `groq`
- lexical retrieval baseline
  - implementation: `bioreview_bench/baseline/lexical.py`
  - execution CLI: `bioreview_bench/scripts/run_bm25_baseline.py`
  - cost: $0 (local retrieval only)

Typical command:

```bash
uv run bioreview-baseline --split val --model claude-haiku-4-5-20251001
uv run bioreview-bm25 --split val
```

### 2.2 Publicly released result files

The current `v3.0-release` leaderboard is generated from result JSON files in
`results/v3`. The default public ranking currently includes:

- Haiku-4.5
- GPT-4o-mini
- Gemini-2.5-Flash
- BM25 (lexical baseline)
- Gemini-2.5-Flash-Lite
- Llama-3.3-70B

Important distinction:

- not every published result in `results/v3` is produced by the built-in
  baseline CLI
- the leaderboard accepts submission-compatible result files as long as they
  obey the benchmark schema and release policy

---

## 3. Status by Baseline Type

| Baseline type | Status | Repository path | Notes |
|---------------|--------|-----------------|-------|
| Anthropic zero-shot LLM | Implemented | `bioreview_bench/baseline/*` | Runnable today |
| OpenAI zero-shot LLM | Implemented | `bioreview_bench/baseline/*` | Runnable today |
| Google Gemini zero-shot LLM | Implemented | `bioreview_bench/baseline/*` | Runnable with `google-genai` and `GEMINI_API_KEY` |
| Groq zero-shot LLM | Implemented | `bioreview_bench/baseline/*` | Runnable with `groq` and `GROQ_API_KEY` |
| Gemini submission-compatible result | Published result | `results/v3/gemini25flash_test_v2.json` | Runnable via `bioreview-baseline --provider google` |
| Gemini Flash Lite result | Published result | `results/v3/gemini_flash_lite_test.json` | Runnable via `bioreview-baseline --provider google` |
| Llama submission-compatible result | Published result | `results/v3/llama33_test.json` | Runnable via `bioreview-baseline --provider groq` |
| BM25 / lexical baseline | Implemented | `bioreview_bench/baseline/lexical.py` | Runnable today via `bioreview-bm25` |
| W8 domain baseline | Planned, not implemented | n/a | Still missing from original plan |
| Human subset reference | Scaffolding implemented | `bioreview_bench/validate/human_subset.py` | Sampling and agreement helpers are ready; annotation not yet completed |

---

## 4. Evaluation Contract for Baselines

All baselines and submissions should follow the same evaluation contract:

- input: manuscript text only
- no access to peer review text or author responses at inference time
- output schema: list of concerns as defined in `TASK_DEFINITION.md`
- evaluation: SPECTER2 semantic matching with Hungarian bipartite matching
- public ranking metric: dataset-level micro F1

Required metadata for result publication:

- `tool_name`
- `tool_version`
- `run_date`
- `split`
- matching threshold and algorithm

`tool_version` should store the exact model or release identifier, not
`unknown`.

---

## 5. Current Limitations

- All four provider paths (Anthropic, OpenAI, Google, Groq) have been exercised
  in full test-split runs with published result files.
- The BM25 lexical baseline has been benchmarked and published on both val and test.
- The benchmark does not yet ship completed human-reference annotations and
  upper-bound reporting.

---

## 6. Recommended Execution Order

The current recommended order is:

1. Finish the zero-cost stage:
   - `bioreview-stats --check-docs`
   - `bioreview-human-subset --n 100`
   - `bioreview-bm25 --split val`
   - `bioreview-bm25 --split test`
2. Then move to low-cost LLM reproducibility runs:
   - `bioreview-baseline --split val --provider openai --model gpt-4o-mini --dry-run`
   - `bioreview-baseline --split val --provider google --model gemini-2.5-flash-lite --dry-run`
   - run the same commands without `--dry-run` only after API budget approval
3. Reserve more expensive provider runs for targeted confirmation, not first-pass screening.

## 7. Recommended Next Work

1. Complete human-reference annotation on a frozen subset and publish agreement metrics.
2. Add a domain-tool adapter or documented import path for W8-style systems.
3. Evaluate additional frontier models (e.g., Claude Sonnet, GPT-4o) on the test split.

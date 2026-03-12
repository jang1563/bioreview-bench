# bioreview-bench Baseline Suite

> Version: 1.0
> Date: 2026-03-11
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

### 2.1 Directly runnable baseline path

The repository currently includes a built-in LLM reviewer baseline:

- implementation: `bioreview_bench/baseline/reviewer.py`
- execution CLI: `bioreview_bench/scripts/run_baseline.py`
- supported providers today: `anthropic`, `openai`

Typical command:

```bash
uv run bioreview-baseline --split val --model claude-haiku-4-5-20251001
```

### 2.2 Publicly released result files

The current `v3.0-release` leaderboard is generated from result JSON files in
`results/v3`. The default public top ranking currently includes:

- Haiku-4.5
- Gemini-2.5-Flash
- GPT-4o-mini
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
| Gemini submission-compatible result | Published result | `results/v3/gemini25flash_test_v2.json` | Not directly runnable from current baseline CLI |
| Llama submission-compatible result | Published result | `results/v3/llama33_test.json` | Same limitation |
| BM25 / lexical baseline | Planned, not implemented | n/a | Still missing from original plan |
| W8 domain baseline | Planned, not implemented | n/a | Still missing from original plan |
| Human subset reference | Planned, not implemented | n/a | Still missing from original plan |

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

- The built-in baseline runner does not yet expose Gemini, Groq, or other
  provider adapters directly.
- The planned non-LLM lexical baseline is still missing.
- The benchmark does not yet ship a human-reference subset with agreement and
  upper-bound reporting.
- The original project plan called for four canonical baselines; today the repo
  has one runnable baseline pathway plus several published result files.

---

## 6. Recommended Next Work

1. Add a lexical baseline so the benchmark has a zero-cost floor.
2. Add a domain-tool adapter or documented import path for W8-style systems.
3. Define a human reference subset and reporting format.
4. Bring provider coverage in the runnable baseline CLI closer to the providers
   represented on the public leaderboard.

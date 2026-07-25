# bioreview-bench Baseline Suite

> Version: 1.5
> Date: 2026-07-25
> Status: Current implementation and planned gaps

This document defines what counts as a baseline in the repository, what is
currently runnable, and what remains to be added to reach the original project
plan.
The public Hugging Face artifact is a rights-minimized, non-executable index.
Baseline scoring requires an authorized local full-schema copy; the public
release does not provide an open test-scoring or leaderboard-submission
service.

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

Typical authorized-local validation commands:

```bash
uv run bioreview-baseline --split val --model claude-haiku-4-5-20251001
uv run bioreview-bm25 --split val
```

### 2.2 Frozen released result files

The v4.1.3 repository preserves a historical score snapshot generated from
result JSON files in `results/v4`. Its recorded ordering includes:

- Haiku-4.5
- GPT-4o-mini
- Gemini-2.5-Flash
- BM25 (lexical baseline)
- Gemini-2.5-Flash-Lite
- Llama-3.3-70B

Important distinction:

- not every published result in `results/v4` is produced by the built-in
  baseline CLI
- the files are frozen historical records, not entries in an open or currently
  accepting leaderboard

---

## 3. Status by Baseline Type

| Baseline type | Status | Repository path | Notes |
|---------------|--------|-----------------|-------|
| Anthropic zero-shot LLM | Implemented | `bioreview_bench/baseline/*` | Runnable today |
| OpenAI zero-shot LLM | Implemented | `bioreview_bench/baseline/*` | Runnable today |
| Google Gemini zero-shot LLM | Implemented | `bioreview_bench/baseline/*` | Runnable with `google-genai` and `GEMINI_API_KEY` |
| Groq zero-shot LLM | Implemented | `bioreview_bench/baseline/*` | Runnable with `groq` and `GROQ_API_KEY` |
| Gemini schema-compatible historical result | Published result | `results/v4/gemini25flash_test_v4.json` | Runnable via `bioreview-baseline --provider google` |
| Gemini Flash Lite result | Published result | `results/v4/gemini_flash_lite_test_v4.json` | Runnable via `bioreview-baseline --provider google` |
| Llama schema-compatible historical result | Published result | `results/v4/llama33_test_v4.json` | Runnable via `bioreview-baseline --provider groq` |
| BM25 / lexical baseline | Implemented | `bioreview_bench/baseline/lexical.py` | Runnable today via `bioreview-bm25` |
| W8 domain baseline | Planned, not implemented | n/a | Still missing from original plan |
| Human validation | Blinded pack tooling implemented | `bioreview_bench/validate/validation_pack.py` | Independent annotation study not yet completed |

---

## 4. Evaluation Contract for Authorized Local Baselines

Authorized local baselines and future result records should follow the same
evaluation contract:

- input: manuscript text only
- no access to peer review text or author responses at inference time
- output schema: list of concerns as defined in `TASK_DEFINITION.md`
- evaluation: explicitly configured matching mode; the frozen snapshot loaded
  `allenai/specter2_base` directly through `SentenceTransformer`, which
  automatically supplied mean pooling, and used threshold-aware Hungarian
  bipartite matching
- score-snapshot ordering metric: dataset-level micro F1

The historical wrapper had no recorded SPECTER2 task adapter or resolved
checkpoint revision. It is not a validated or currently recommended SPECTER2
method, and a correctly adapted or differently pooled encoder is a new matcher.

Required metadata for result publication:

- `tool_name`
- `tool_version`
- `run_date`
- `split`
- matching threshold and algorithm
- matching mode and exact embedding model identifier
- resolved embedding revision
- wrapper, pooling, and task-adapter configuration

`tool_version` should store the exact model or release identifier, not
`unknown`. Historical v4 records use a null embedding revision because the
resolved commit was not preserved; it must not be reconstructed after the fact.

---

## 5. Current Limitations

- All four provider paths (Anthropic, OpenAI, Google, Groq) have been exercised
  in full test-split runs with published result files.
- The BM25 lexical baseline has a published frozen test result; the harness also
  supports authorized local validation-split runs.
- The benchmark does not yet ship completed human-reference annotations and
  upper-bound reporting.
- The public Hugging Face artifact does not contain the full fields needed to
  execute scoring.
- The frozen auto-mean-pooled base-checkpoint wrapper and 0.65 threshold have
  not completed independent human-equivalence validation; scores are
  provisional and matcher-dependent.
- Related/versioned training records exist for 42/150 F1000 test articles. A
  same-family record is BM25 rank 1 in all 42 cases; query-time family
  filtering reduces lexical-audit BM25 F1 from 0.0690 to 0.0171 on the
  450-article non-eLife set. See `results/v4/measurement_audit.md`. This does
  not establish corrected historical embedding scores or contamination-free
  out-of-family generalization.

---

## 6. Recommended Execution Order

The current recommended order is:

1. Validate an authorized local full-schema copy:
   - `bioreview-stats --check-docs`
   - `bioreview-validation-pack create --output-dir validation_pack`
   - `bioreview-bm25 --split val`
2. Then move to low-cost LLM reproducibility runs:
   - `bioreview-baseline --split val --provider openai --model gpt-4o-mini --dry-run`
   - `bioreview-baseline --split val --provider google --model gemini-2.5-flash-lite --dry-run`
   - run the same commands without `--dry-run` only after API budget approval
3. Reserve more expensive provider runs for targeted confirmation, not first-pass screening.

Do not treat the frozen public test split as an open leaderboard target. A
future test snapshot should be run only on an authorized, frozen,
manuscript-family-disjoint split under its stated release policy.

## 7. Recommended Next Work

1. Construct and freeze manuscript-family-disjoint splits, then rerun baselines.
2. Complete human-reference annotation on a frozen subset and publish agreement metrics.
3. Add a domain-tool adapter or documented import path for W8-style systems.
4. Evaluate additional models only after the revised test protocol is frozen.

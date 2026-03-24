# bioreview-bench Adjudication Protocol

> Version: 1.0
> Date: 2026-03-22
> Status: Repository annotation policy

This document defines how annotation disagreements are resolved for
bioreview-bench. It applies to both human annotation and manual review of LLM
extraction errors.

---

## 1. Scope

Adjudication is required when annotators disagree on any of the following:

- whether a text span is a concern at all
- concern boundaries or split/merge decisions
- category label
- severity label
- `author_stance`
- `evidence_of_change`
- `requires_figure_reading`

The adjudication goal is not majority voting. It is to produce a single final
record that best matches the benchmark definitions in
`ANNOTATION_GUIDELINES.md`.

---

## 2. Roles

| Role | Responsibility |
|------|----------------|
| Primary annotator | Initial extraction and labeling |
| Secondary annotator | Independent second pass |
| Adjudicator | Final decision on disagreements |

Minimum rule:

- Primary and secondary annotators must work independently.
- The adjudicator may see both annotations, the source text, and the author
  response.

---

## 3. Required Inputs for Adjudication

The adjudicator should review:

- manuscript title and relevant manuscript section
- reviewer text containing the disputed concern
- author response text, if available
- both annotator outputs
- disagreement report produced from field-level comparison
- `ANNOTATION_GUIDELINES.md`
- `TASK_DEFINITION.md` and `EVALUATION_PROTOCOL.md` when scoring implications
  matter

---

## 4. Workflow

1. Primary and secondary annotations are completed independently.
2. A disagreement report is generated at concern and field level.
3. The adjudicator reviews the source materials and both candidate labels.
4. The adjudicator records one final decision plus a short rationale.
5. The final adjudicated record becomes the benchmark truth used for release.

Operational rules:

- Adjudication should happen from the original source text, not from the two
  annotator labels alone.
- If both annotations are poor, the adjudicator may replace both with a new
  concern formulation.
- If a disagreement exposes a recurring guideline ambiguity, update
  `ANNOTATION_GUIDELINES.md` after the adjudication batch is closed.

---

## 5. Decision Rules

### 5.1 Concern existence

Keep a concern only if it is a concrete scientific criticism, request, or
required clarification that the authors could address.

Exclude:

- praise or general sentiment
- vague overall judgments without a concrete issue
- typo-level edits that do not affect scientific interpretation
- reviewer self-promotion or citation requests without scientific substance

### 5.2 Concern boundaries

Split a concern into multiple records when:

- different parts would receive different categories
- different parts would receive different severities
- different parts receive separate author responses

Merge into one concern when:

- multiple sentences clearly restate the same underlying issue
- a request and its rationale are inseparable parts of one concern

### 5.3 Category precedence

Use these precedence rules for common collisions:

- `figure_issue` only when the claim truly requires visual inspection of the
  figure itself.
- `missing_experiment` when the paper needs an additional experiment to support
  a claim.
- `design_flaw` when the existing design or controls are fundamentally invalid.
- `reagent_method_specificity` for reproducibility-critical method details.
- `writing_clarity` for exposition issues that do not primarily concern missing
  reagent or protocol detail.
- `interpretation` for overclaiming or unsupported conclusions.
- `prior_art_novelty` for missing prior work, novelty inflation, or absent
  comparison to literature.

### 5.4 Severity

Use the question:

> If this concern were left unresolved, would the paper's main claims still be
> acceptable?

- `major`: no
- `minor`: yes, but substantive improvement is needed
- `optional`: useful but not necessary

### 5.5 Author stance and evidence of change

Adjudicate from the author response, not from the reviewer tone.

- `conceded`: the authors accept the point and describe a revision or new
  evidence
- `rebutted`: the authors clearly reject the point and defend the original claim
- `partial`: mixed acceptance/rebuttal, or acceptance without clear revision
- `unclear`: response exists but the final stance cannot be determined
- `no_response`: no response to the concern

For `evidence_of_change`:

- `true`: revision, experiment, analysis, or manuscript change is described
- `false`: authors explicitly decline to change or say it is out of scope
- `null`: response is too vague to determine whether a change occurred

---

## 6. Recording the Decision

Every adjudicated concern should retain:

- final concern text
- final labels for all benchmark fields
- adjudicator identifier
- adjudication date
- short rationale string

Recommended rationale format:

```json
{
  "reason": "Reclassified from writing_clarity to reagent_method_specificity because the issue is missing antibody catalog information."
}
```

---

## 7. Quality Control

Track these metrics per adjudication batch:

- percentage of concerns requiring adjudication
- most common disagreement types
- per-category disagreement rate
- per-field disagreement rate for `author_stance` and `evidence_of_change`

Escalate guideline review when:

- one category pair is repeatedly confused
- concern boundary disagreements exceed the category disagreements
- adjudicators are writing materially different rationales for similar cases

---

## 8. Release Policy

Public benchmark releases should use adjudicated records where available.
If a release contains a mixture of adjudicated and non-adjudicated records, the
release notes should say so explicitly.

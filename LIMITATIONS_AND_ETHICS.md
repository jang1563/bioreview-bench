# bioreview-bench Limitations and Ethics

> Version: 1.0
> Date: 2026-03-11
> Status: Repository disclosure document

This document records the main methodological, ethical, and deployment limits of
bioreview-bench. These limits should be disclosed in benchmark papers,
leaderboards, and downstream tool evaluations.

---

## 1. Silver-Label Limitation

The benchmark uses outcome-anchored silver labels, not objective truth.

Key consequence:

- an author concession is evidence about how the review process resolved a
  concern, not proof that the reviewer was objectively correct

Risks:

- authors may concede because of editorial pressure, not because the criticism
  is scientifically valid
- authors may rebut valid concerns due to limited time, scope, or incentive
- absent author response does not imply the concern was unimportant

Required disclosure:

- papers and leaderboard writeups should avoid language that equates benchmark
  recall with "scientific correctness"

---

## 2. Open-Review and Source Bias

The corpus is built from journals that expose peer-review materials publicly.
That induces selection effects.

Examples:

- open-review journals may differ in tone and concern style from closed-review
  venues
- journal-specific norms can change concern frequency and category balance
- optional transparent-review programs may overrepresent authors comfortable
  with public review

Required disclosure:

- benchmark claims should be framed as biomedical open-review coverage, not as a
  universal estimate of all peer review behavior

---

## 3. Figure and Modality Limits

The default benchmark excludes `figure_issue` concerns from base metrics because
the default task input is text-only.

Implications:

- the benchmark underestimates some real reviewer skill
- systems with strong visual reasoning are not fully rewarded in the default
  setup
- text-only systems are protected from precision penalties on purely visual
  issues

---

## 4. Provider and Reproducibility Bias

Public leaderboard performance depends partly on access to proprietary models and
commercial APIs.

Risks:

- users with different provider access may not be able to reproduce all results
- provider-side model updates can shift behavior even when model names remain
  stable
- cost differences affect which systems can practically be benchmarked at scale

Mitigations:

- require exact `tool_version`
- freeze release artifacts in `results/release_manifest.json`
- document rerun commands and release notes

---

## 5. Data Leakage and Contamination Risk

The benchmark prohibits access to peer review text and author responses at
inference time, but contamination cannot be ruled out completely.

Examples:

- model pretraining may already contain some source articles or public reviews
- users may accidentally consult public review pages while evaluating systems
- repeated leaderboard iteration can lead to benchmark-specific prompt tuning

Mitigations:

- frozen test split
- explicit prohibited-practice rules in `TASK_DEFINITION.md`
- release-manifest-backed public ranking policy

---

## 6. Licensing and Attribution Risk

Not all sources in the repository have identical redistribution rules.

Implications:

- no blanket statement of uniform CC-BY redistribution should be made for all
  source content
- source-specific packaging decisions must follow `LICENSE_MATRIX.md`
- broader public data releases should be conservative when source terms are
  optional or article-specific

---

## 7. Reviewer Identity and Privacy

Some sources, especially fully open-review platforms, expose reviewer names
publicly.

Ethical issue:

- a benchmark may technically be allowed to redistribute names while still
  making the social footprint of the data larger than necessary

Repository posture:

- reviewer-identity handling should be documented explicitly for every release
- downstream users should avoid repackaging reviewer identities unless the use
  case truly requires it

---

## 8. Misuse Risk

This benchmark is intended to evaluate assistive review tools, not to automate
editorial judgment without oversight.

Potential misuse:

- treating leaderboard rank as a substitute for expert peer review
- using concern recall as a proxy for publication worthiness
- deploying tools to generate authoritative rejection decisions without human
  accountability

Required disclosure:

- benchmark scores measure concern overlap with public peer review, not final
  editorial quality or truth

---

## 9. Practical Reporting Rules

Any benchmark report should state at least:

- release version or tag
- split and matching policy
- whether figure concerns were excluded
- whether the system is proprietary or open
- whether the result is directly reproducible from this repository
- material known limitations relevant to the evaluated system

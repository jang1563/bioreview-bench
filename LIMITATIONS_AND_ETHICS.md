# bioreview-bench Limitations and Ethics

> Version: 1.4
> Date: 2026-07-25
> Status: Repository disclosure document

This document records the main methodological, ethical, and deployment limits
of bioreview-bench. These limits should be disclosed in benchmark papers,
frozen score artifacts, and downstream tool evaluations. Release-critical
items are summarized in `KNOWN_ISSUES.md`.

---

## 1. Silver-Label Limitation

The benchmark uses response-derived silver labels, not objective truth.

Key consequence:

- an author concession is evidence about how the review process resolved a
  concern, not proof that the reviewer was objectively correct

Risks:

- authors may concede because of editorial pressure, not because the criticism
  is scientifically valid
- authors may rebut valid concerns due to limited time, scope, or incentive
- absent author response does not imply the concern was unimportant

Required disclosure:

- papers and score-snapshot writeups should avoid language that equates benchmark
  recall with "scientific correctness"
- author stance is an LLM interpretation of a response record, not expert
  adjudication of the underlying concern

---

## 2. Normalization and Human-Validation Gap

Source platforms publish the source prose as peer review; possible AI assistance
in individual recent reviews is unknown. An LLM segmented those records into
concern units and assigned category, severity, and stance labels. Those
normalized fields have not yet been independently expert-validated.

The historical 148-row annotation file is not evidence of perfect agreement:
the legacy interface copied model labels into blank human-label fields. The
reported kappa of 1.000 is withdrawn.

The validation work required before making calibrated accuracy claims is:

- a system-label-blinded, source/category/severity-stratified audit of
  extracted units (source and article context remain visible)
- a separate omission audit against raw review passages
- at least two independent annotators, followed by adjudication
- inter-annotator agreement reported separately for fidelity, category,
  severity, and stance, with article-cluster stability intervals
- unweighted audit rates described only for the realized sample unless
  inclusion weights are added for population inference

Until that study is complete, the dataset should be described as a large
silver-standard resource and not an expert-validated benchmark.

---

## 3. Open-Review and Source Bias

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

## 4. Manuscript-Version Mismatch

The benchmark input may contain the published or reviewed-preprint version
available from a source, while the recorded concerns may have been written
against an earlier submitted manuscript.

Consequences:

- some reviewer requests may already be satisfied in the benchmark input
- systems can receive credit for detecting an issue that is no longer present
- apparent false negatives can reflect version mismatch rather than review
  inability

Results should report the source's review format and avoid implying that every
input exactly reconstructs the manuscript seen by the reviewer.

---

## 5. Figure and Modality Limits

The default benchmark excludes `figure_issue` concerns from base metrics because
the default task input is text-only.

Implications:

- the benchmark underestimates some real reviewer skill
- systems with strong visual reasoning are not fully rewarded in the default
  setup
- text-only systems are protected from precision penalties on purely visual
  issues

---

## 6. Matcher and Metric Validity

The frozen scores used an unadapted `allenai/specter2_base` checkpoint loaded
directly through `SentenceTransformer`, which supplied automatic mean pooling.
No SPECTER2 task adapter or resolved checkpoint revision was recorded. Cosine
similarity at a frozen 0.65 threshold was followed by Hungarian assignment.
Neither this historical wrapper nor the threshold has completed independent
concern-equivalence validation. Human validation is intentionally outside the
scope of the v4.1.3 measurement-audit release.

Risks:

- shared biomedical topic words can create a match without shared criticism
- legitimate paraphrases can be missed
- rankings can change with the encoder, threshold, or matching policy
- "precision" measures overlap with the recorded reference set, not scientific
  validity of generated concerns

The published frozen scores are therefore provisional and matcher-dependent. Sensitivity
analyses, explicit lexical controls, and a human-labeled matcher test set should
accompany future claims.

---

## 7. Provider and Reproducibility Bias

The historical score snapshot depends partly on access to proprietary models
and commercial APIs.

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

## 8. Data Leakage, Contamination, and AI-Assisted Reviews

The benchmark prohibits access to peer review text and author responses at
inference time, but contamination cannot be ruled out completely.

Examples:

- model pretraining may already contain some source articles or public reviews
- users may accidentally consult public review pages while evaluating systems
- repeated score inspection can lead to benchmark-specific prompt tuning
- 42/150 F1000 test articles across 41 version-independent DOI families have
  related/versioned training records; a same-family record is BM25 rank 1 for
  all 42 affected queries
- the validation-aware F1000 DOI-stem scan finds 115 train–validation, 41
  train–test, and 9 validation–test crossing families; development data overlap
  49/150 F1000 test articles across 48 families
- 439 of 600 v4 test articles were in v3 train, and all 600 appeared somewhere
  in the v3 split universe
- reviews published in 2026 may themselves have been drafted or edited with AI;
  the benchmark does not currently identify or exclude such review text

Mitigations:

- frozen article-ID split, with the F1000 family-level limitation and the
  query-time candidate-filtering sensitivity analysis disclosed
- explicit treatment of the DOI-stem family matrix as a source-specific lower
  bound that misses fuzzy-title, preprint-to-journal, and cross-source relations
- explicit prohibited-practice rules in `TASK_DEFINITION.md`
- release-manifest-backed preservation of the historical score snapshot

---

## 9. BM25 and Topical-Overlap Confounding

Strong lexical retrieval performance does not establish that reviewer concerns
are predictable from manuscript content. It may reflect same-source vocabulary,
topic clustering, repeated normalization templates, or matcher preferences for
lexically similar text.

BM25 should be presented as a competitive lexical control. Claims about the
cause of its performance require source-held-out, topic-controlled, and
raw-versus-normalized analyses.

The full 600-article snapshot is the primary reporting population. The
450-article non-eLife analysis is a post-hoc source-subset sensitivity check.
The effective Jaccard threshold 0.195 and all v4.1.3 sweep points are
uncalibrated; rank changes across them diagnose matcher sensitivity but do not
identify a correct semantic-equivalence boundary. Query-time family filtering
does not create a family-disjoint corpus or reconstruct all 600 BM25
predictions.

---

## 10. Licensing and Attribution Risk

Not all sources in the repository have identical redistribution rules.

Implications:

- no blanket statement of uniform CC-BY redistribution should be made for all
  source content
- source-specific packaging decisions must follow `LICENSE_MATRIX.md`
- broader public data releases should be conservative when source terms are
  optional or article-specific

---

## 11. Reviewer Identity and Privacy

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

## 12. Misuse Risk

This benchmark is intended to evaluate assistive review tools, not to automate
editorial judgment without oversight.

Potential misuse:

- treating the frozen score ordering as a substitute for expert peer review
- using concern recall as a proxy for publication worthiness
- deploying tools to generate authoritative rejection decisions without human
  accountability

Required disclosure:

- benchmark scores measure concern overlap with public peer review, not final
  editorial quality or truth

---

## 13. Practical Reporting Rules

Any benchmark report should state at least:

- release version or tag
- split and matching policy
- embedding model identifier/revision and matcher mode
- whether figure concerns were excluded
- whether the system is proprietary or open
- whether the result is directly reproducible from this repository
- material known limitations relevant to the evaluated system
- a statement that normalized annotations and the matcher have not yet
  completed independent expert validation

# bioreview-bench Data Curation Report

> Version: 1.4
> Date: 2026-07-25
> Status: Repository process summary aligned with `v4.1.3`

This report summarizes how data enters the repository, how it is normalized into
benchmark-ready records, and what controls are in place before release.

---

## 1. Scope

The repository currently supports a multi-source biomedical peer-review corpus
with frozen `data/splits/v4` evaluation splits and release artifacts rooted in:

- `data/splits/v4`
- `results/v4`
- `results/release_manifest.json`

The v4.1.3 repository preserves a frozen six-system score snapshot on 600 test
articles (150 eLife, 150 PLOS, 150 F1000Research, 100 Nature Portfolio, and 50
PeerJ). The release manifest freezes the exact historical artifact set and its
known metric limitations. This is not an open public leaderboard or test-scoring
service.

The Hugging Face release is a rights-minimized, non-executable index. It exposes
split membership and stable article IDs/DOIs, so the test membership is not
secret; source reviews are publicly retrievable and the snapshot is not suitable
for new blind evaluation.

---

## 2. Pipeline Overview

The curation flow has five stages:

1. Source collection
2. Article/review parsing
3. Concern extraction and normalization
4. Schema validation and split generation
5. Release artifact generation

Representative repository entry points:

- collection/update orchestration: `bioreview_bench/scripts/update_pipeline.py`
- source collectors: `bioreview_bench/collect/*.py`
- JATS/PDF parsing: `bioreview_bench/parse/jats.py`,
  `bioreview_bench/parse/pdf.py`
- concern extraction: `bioreview_bench/parse/concern_extractor.py`
- schema validation: `bioreview_bench/validate/schema_validator.py`
- evaluation and release artifacts: `bioreview_bench/evaluate/*.py`,
  `scripts/rebuild_release_artifacts.py`

---

## 3. Included Sources

Current repository coverage includes:

- eLife
- PLOS
- F1000Research
- PeerJ
- Nature Portfolio

These sources differ materially in review publication policy, article license,
and packaging constraints. Source-specific redistribution decisions are tracked
in `LICENSE_MATRIX.md`.

---

## 4. Inclusion and Exclusion Rules

### 4.1 Include

- publicly accessible article records from supported sources
- review packages that can be parsed into article/review/response structure
- articles with enough manuscript text to serve as benchmark input
- concern records that satisfy the benchmark definition in
  `TASK_DEFINITION.md` and the decision rules in `ADJUDICATION_PROTOCOL.md`

### 4.2 Exclude or down-scope

- malformed records that cannot be parsed into usable article objects
- duplicate article identifiers
- review content that is not clearly linked to a specific article version
- source content whose redistribution status is ambiguous for public release
- `figure_issue` concerns from base scoring, while retaining them in the corpus
  when extracted

Operational note:

- benchmark integrity and licensing are separate filters. A field may be lawful
  to redistribute but still withheld from the test release to avoid label
  leakage.

---

## 5. Quality Controls

The repository currently applies the following controls:

- Pydantic model validation for benchmark records
- schema-level checks in `bioreview_bench/validate/schema_validator.py`
- post-processing and dedup helpers in `bioreview_bench/collect/postprocess.py`
- frozen evaluation split handling in `data/splits/v4`
- release-manifest freezing for the historical score-snapshot artifacts
- regression tests for metrics, score-snapshot filtering, CLI behavior, and release
  artifact generation
- blinded two-annotator validation-pack tooling with separate extraction-fidelity
  and omission audits

Recent release-hardening work also fixed:

- missing-prediction rows being dropped from dataset scoring
- empty/empty article precision mismatch
- stale score-snapshot policy metadata
- inconsistent `v2` vs `v3` split defaults in evaluation tooling

---

## 6. Split and Release Policy

`data/splits/v4` is the current evaluation reference for repository CLIs.

The realized split is 5,387 train / 953 validation / 600 test articles (77.6% /
13.7% / 8.6%). The builder selected fixed per-source test counts with seed 42.
It then randomly shuffled the remaining articles within
`(source, editorial_decision, review_format)` strata, assigning approximately
15% of each eligible stratum to validation and the remainder to train. It did
not stratify by subject area, concern category, resolution, or publication time
and did not create a chronological holdout.

The split is article-ID-disjoint but not manuscript-family-disjoint. The v4.1.3
F1000 DOI-stem scan finds 115 train–validation, 41 train–test, and 9
validation–test crossing families. Development data overlap 49/150 F1000 test
articles across 48 families. These aggregate counts are a source-specific lower
bound: the scan does not resolve fuzzy-title, preprint-to-journal, or
cross-source manuscript relations. See
`results/v4/measurement_audit.{json,md}`; this audit does not rebuild the split.

Frozen score-snapshot construction policy:

- use only `results/v4/*.json` with `split="test"`
- exclude `dedup_gt=true` runs from the default ranking
- keep only the strongest run per `(tool_name, tool_version)`
- rank by micro-averaged F1
- record exact threshold and matching algorithm in the release manifest
- treat all six raw `"f1_macro": 0.0` fields as invalid legacy sentinels;
  historical category-macro F1 is unreported
- treat the full 600-article score snapshot as primary and the 450-article
  non-eLife result as a post-hoc sensitivity subset
- describe the effective Jaccard cutoff 0.195 and all threshold-sweep points as
  uncalibrated matcher diagnostics

Release operators should treat these files as the frozen artifact contract:

- `RELEASE_V4.md`
- `results/release_manifest.json`
- `results/leaderboard.md`
- `results/leaderboard.json`
- `results/v4/measurement_audit.json`
- `results/v4/measurement_audit.md`

---

## 7. Known Gaps

The current repository is operational but not fully closed on curation
documentation. Remaining gaps include:

- source-specific legal review for broader full-text redistribution
- explicit reviewer-name policy for fully open-review sources
- completion of independent human validation for normalized concerns, labels,
  omissions, and semantic matching
- completion of planned benchmark baselines beyond the currently runnable
  zero-shot LLM baseline path
- a new family-disjoint, genuinely blind evaluation design if future claims
  require hidden-test generalization

---

## 8. Recommended Next Additions

- publish per-source inclusion/exclusion counts
- publish hard-failure reasons from the collection pipeline
- add a small end-to-end fixture covering `collect -> split -> evaluate ->
  release artifacts`
- keep release notes synchronized with manifest-backed artifact changes

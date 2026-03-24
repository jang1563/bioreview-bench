# bioreview-bench Data Curation Report

> Version: 1.0
> Date: 2026-03-22
> Status: Repository process summary aligned with `v3.0-release`

This report summarizes how data enters the repository, how it is normalized into
benchmark-ready records, and what controls are in place before release.

---

## 1. Scope

The repository currently supports a multi-source biomedical peer-review corpus
with frozen `data/splits/v3` evaluation splits and release artifacts rooted in:

- `data/splits/v3`
- `results/v3`
- `results/release_manifest.json`

The public `v3.0-release` leaderboard ranks 944 scored test articles (981 test
articles minus 37 PeerJ articles not evaluated by all models). The full test
split contains 981 entries; the release manifest freezes the exact scored
artifact set used for the public ranking.

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
  `ANNOTATION_GUIDELINES.md`

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
- frozen evaluation split handling in `data/splits/v3`
- release-manifest freezing for public leaderboard artifacts
- regression tests for metrics, leaderboard filtering, CLI behavior, and release
  artifact generation

Recent release-hardening work also fixed:

- missing-prediction rows being dropped from dataset scoring
- empty/empty article precision mismatch
- stale leaderboard policy metadata
- inconsistent `v2` vs `v3` split defaults in evaluation tooling

---

## 6. Split and Release Policy

`data/splits/v3` is the current evaluation reference for repository CLIs.

Public leaderboard policy:

- use only `results/v3/*.json` with `split="test"`
- exclude `dedup_gt=true` runs from the default ranking
- keep only the strongest run per `(tool_name, tool_version)`
- rank by micro-averaged F1
- record exact threshold and matching algorithm in the release manifest

Release operators should treat these files as the public contract:

- `RELEASE_V3.md`
- `results/release_manifest.json`
- `results/leaderboard.md`
- `results/leaderboard.json`

---

## 7. Known Gaps

The current repository is operational but not fully closed on curation
documentation. Remaining gaps include:

- source-specific legal review for broader full-text redistribution
- explicit reviewer-name policy for fully open-review sources
- deeper empirical reporting on disagreement rates and annotation drift
- completion of planned benchmark baselines beyond the currently runnable
  zero-shot LLM baseline path

---

## 8. Recommended Next Additions

- publish per-source inclusion/exclusion counts
- publish hard-failure reasons from the collection pipeline
- add a small end-to-end fixture covering `collect -> split -> evaluate ->
  release artifacts`
- keep release notes synchronized with manifest-backed artifact changes

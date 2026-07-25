# BioReview-Bench v4.1.3

Date: 2026-07-25

v4.1.3 is a public-release hardening update over the frozen v4 benchmark
snapshot. It improves statistical correctness, aggregate leakage diagnostics,
and package validation. It does not rerun the six systems, alter benchmark
labels or split membership, complete human validation, or publish local
manuscript and restricted research data.

## Statistical correction

- Replaced the executable legacy paired-bootstrap significance path with a
  tested article-paired label-swap randomization test. Every permutation swaps
  complete per-article sufficient statistics and recomputes the nonlinear
  dataset-level micro-F1 difference.
- The analysis now aligns pairs by explicit article ID and rejects mismatched
  reference denominators. Small informative samples are enumerated exactly;
  larger samples use a seeded Monte Carlo estimate with a plus-one correction.
- The v3 pairwise-significance JSON is withdrawn rather than silently retained:
  its old sign-crossing bootstrap values were not valid null p-values.
- Future regenerated pairwise outputs are explicitly
  `exploratory_unadjusted`; the script no longer prints significance stars for
  the 15 unadjusted comparisons.
- A separate paired sign-flip helper remains available for explicitly additive
  paired estimands. Empty paired samples return no p-value (`None`), all-tie
  samples return `1.0`, and malformed inputs fail closed.

This changes protocol guidance and reusable code. It does not create new
inferential results for the frozen score table.

## Aggregate measurement-audit hardening

- Added a deterministic, aggregate, text-free Jaccard threshold sweep. It
  reports per-system match counts, F1, rank, and Kendall tau-a at declared
  operating points.
- The sweep is a sensitivity analysis. The historical effective Jaccard cutoff
  `0.195` (`0.65 × 0.3`) and all swept cutoffs remain uncalibrated; none is
  selected as a valid semantic-equivalence boundary.
- The 600-article test snapshot is the primary reporting population. The
  450-article non-eLife result remains a post-hoc source-subset diagnostic.
- Added a validation-aware F1000 DOI-stem split matrix. It finds 115
  train–validation, 41 train–test, and 9 validation–test crossing families;
  161 distinct families cross at least one boundary. Development data overlap
  49/150 F1000 test articles across 48 families.
- These DOI-stem counts are a source-specific lower bound. They do not resolve
  fuzzy-title, preprint-to-journal, or cross-source manuscript relations.
- Query-time family filtering is explicitly described as a sensitivity
  intervention that leaves the original BM25 corpus statistics fixed. It
  reconstructed all 42 targeted train–test rows, not all 600 BM25 predictions.

Exact audit regeneration still requires the authorized local full-schema
splits and frozen tool outputs; those text-bearing inputs are not part of the
public repository.

## Public-package integrity

- Replaced the public GitHub repository network with a single-root v4.1.3
  snapshot. No earlier commits, historical tags, or pull-request refs are
  reachable from the canonical repository. The former network and an
  intermediate selected-history candidate remain private as legacy
  containment repositories.
- Added canonical Hugging Face `.gitattributes` content to the exact package
  allowlist and release manifest.
- Package validation now rejects missing, extra, modified, or non-canonical
  `.gitattributes` content before publication.
- The v4.1.3 release notes are included in the public package contract.

## Claim boundary

- The historical matcher and its 0.65 threshold have not completed independent
  human-equivalence validation.
- Concern segmentation, category, severity, and stance remain silver-standard
  LLM-normalized fields.
- The frozen scores are historical, provisional, matcher-dependent, and not
  evidence of contamination-free out-of-family generalization.
- The public Hugging Face artifact remains a rights-minimized,
  non-executable index. It is not a public test set, scoring service, or open
  leaderboard.
- Paper source, private benchmark inputs, raw review/publisher prose, internal
  notes, and article-level error analysis remain outside the public release.

## Verification

Release validation covers the paired test, threshold sweep and split matrix,
exact package-tree/manifest enforcement, deterministic public builds, release
metadata, and the full repository test suite.

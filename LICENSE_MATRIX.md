# bioreview-bench License Matrix

> Version: 1.2
> Date: 2026-03-22
> Status: Release-policy matrix aligned with `v3.0-release`

This document records the repository's operational redistribution policy by
source. It is not legal advice. When publisher terms are ambiguous, the project
defaults to a conservative release posture.

---

## 1. Current Release Scope

The official `v3.0-release` snapshot publishes:

- code, tests, and evaluation scripts
- benchmark documentation and release notes
- leaderboard artifacts and result JSON metadata
- structured benchmark packaging guidance

The release does **not** grant rights beyond the original publisher licenses.
Redistributing article text, peer review text, or author responses requires a
source-by-source check against the rules below.

Conservative default:

- If review or response publication is optional, treat redistribution as
  article-specific rather than source-wide.
- If article text is not clearly open access, do not redistribute the full text.
- If reviewer identities are public, preserve attribution intentionally and
  document whether names are retained or removed in downstream exports.

---

## 2. Summary Matrix

| Source | Article text | Review text | Author response | Metadata | Repository stance |
|--------|--------------|-------------|-----------------|----------|-------------------|
| eLife | Yes | Yes | Yes | Yes | Release-safe for current benchmark packaging |
| PLOS | Yes | Conditional | Conditional | Yes | Allow only where peer-review history is explicitly published |
| PeerJ | Yes | Conditional | Conditional | Yes | Allow only where the article exposes peer-review history |
| Nature Portfolio | Conditional | Conditional | Conditional | Yes | Conservative: OA + transparent peer review only |
| EMBO Press | Yes | Conditional | Conditional | Yes | Allow when review process files are present |
| F1000Research | Yes | Yes | Yes | Yes | Release-safe, with reviewer-name policy required |

---

## 3. Source-Specific Notes

### 3.1 eLife

Official basis:

- https://elifesciences.org/about/peer-review

Operational interpretation:

- eLife publishes articles under open-access terms and publicly exposes the peer
  review package alongside the article record.
- For repository purposes, article text, decision letters, author responses, and
  metadata can be included together when present on the public article page.

Repository rule:

- `eLife` remains the least restrictive source in the benchmark.
- Full-text benchmark packaging is allowed for current use, subject to normal
  attribution and DOI retention.

### 3.2 PLOS

Official basis:

- https://plos.org/open-science/open-access/
- https://journals.plos.org/plosone/s/peer-review-terms

Operational interpretation:

- PLOS research articles are open access.
- Peer review publication is not universal across all journals or all articles.
  Review text and author responses should therefore be treated as conditional.

Repository rule:

- Article text and metadata can be redistributed where the article license
  permits it.
- Review and author-response text should only be packaged when the article page
  explicitly publishes the peer-review history.

### 3.3 PeerJ

Official basis:

- https://peerj.com/about/policies-and-procedures/#copyright-and-licensing
- https://peerj.com/about/faq/#peer-review-history

Operational interpretation:

- PeerJ articles are published under open-access terms.
- Publication of the full peer-review history is optional and article-specific.

Repository rule:

- Treat article text as redistributable with attribution.
- Treat review and author-response text as opt-in only. If the article does not
  explicitly publish peer-review history, do not redistribute those materials.

### 3.4 Nature Portfolio

Official basis:

- https://www.nature.com/nature-portfolio/open-access
- https://www.nature.com/nature-portfolio/open-access#peer-review

Operational interpretation:

- Nature Portfolio mixes open-access and subscription content.
- Transparent peer review is article-specific and not equivalent to blanket
  permission across the source.

Repository rule:

- Only redistribute article text when the article itself is open access under a
  compatible license.
- Only redistribute review/response text when a transparent peer-review package
  is publicly available for that article.
- Subscription article text and ambiguous review files should be excluded from
  public benchmark exports.

### 3.5 EMBO Press

Official basis:

- https://www.embopress.org/page/journal/14602075/open-science
- https://www.embopress.org/page/journal/14602075/review-process-files

Operational interpretation:

- EMBO Press states that research papers are published open access under
  Creative Commons Attribution.
- Review process files are available for many research papers, but availability
  is still article-level rather than guaranteed across all records.

Repository rule:

- Article text may be redistributed with attribution.
- Review and author-response text may be redistributed only when the article
  exposes a public review process file.

### 3.6 F1000Research

Official basis:

- https://f1000research.com/about/policies
- https://f1000research.com/for-referees

Operational interpretation:

- F1000Research is fully open access and publishes referee reports as part of
  the platform workflow.
- Reviewer identities are public by design.

Repository rule:

- Article text, review text, author responses, and metadata are all acceptable
  for benchmark packaging.
- Downstream exports must make an explicit choice about retaining or masking
  reviewer names; the project should document that choice at release time.

---

## 4. Repository Distribution Policy

Default public benchmark artifacts should prefer the smallest legally necessary
surface area:

- code and documentation: always public
- leaderboard and evaluation artifacts: always public
- structured concern annotations and metadata: public, with attribution
- article input fields needed for benchmark execution: public only where source
  terms permit them
- raw review text and author responses: public only when source rules clearly
  permit redistribution for that specific article

Test-split leakage policy is separate from license policy:

- test article input text may be public if licensing allows it
- test human concern labels, review text, and author-response text can still be
  withheld for benchmark integrity even when redistribution would be lawful

---

## 5. Open Items Before a Broader Data Release

- Confirm the exact packaging footprint of the HuggingFace dataset card against
  this matrix.
- Decide and document the reviewer-name policy for F1000Research-derived data.
- Re-check optional-review sources before any large full-text export refresh.
- Keep README and dataset card language conservative: do not describe all source
  content as uniformly CC-BY redistributable.

---

## 6. Changelog

| Date | Change |
|------|--------|
| 2026-03-22 | Aligned license references to CC-BY-NC 4.0 for benchmark annotations (matching LICENSE file) |
| 2026-03-11 | Reworked matrix for `v3.0-release`, added PeerJ, and aligned repository policy with source-specific redistribution rules |
| 2026-02-28 | Initial draft created |

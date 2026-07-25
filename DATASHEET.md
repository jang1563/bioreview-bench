# Datasheet for bioreview-bench

**Dataset name:** bioreview-bench
**Version:** 4.1.3
**Schema version:** 1.1
**Date of this document:** 2026-07-25
**GitHub:** [github.com/jang1563/bioreview-bench](https://github.com/jang1563/bioreview-bench)
**HuggingFace public rights-minimized index:**
[huggingface.co/datasets/jang1563/bioreview-bench](https://huggingface.co/datasets/jang1563/bioreview-bench)
**License:** Benchmark annotations and packaging metadata: CC-BY-NC 4.0 | Underlying source content: source-specific | Code: Apache-2.0
**Known issues:** [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

This document follows the Datasheets for Datasets framework (Gebru et al., 2018) and provides structured documentation of bioreview-bench for dataset consumers, downstream researchers, and auditors.

---

## 1. Motivation

**For what purpose was the dataset created?**

bioreview-bench was created to support reproducible evaluation of AI tools that
assist with biomedical peer review. A growing number of large language model
(LLM) applications claim to identify methodological concerns, statistical
errors, and missing experiments in research manuscripts. bioreview-bench
combines biomedical open-review records, concern-level normalization,
response-derived stance labels, and an integrated evaluation harness. The
reference concerns are a behavioral record of published peer review, not
objective scientific ground truth.

The dataset supports the following primary use cases:
- Benchmarking AI peer review assistants on recall, precision, F1, and major-concern recall.
- Training and fine-tuning models to identify reviewer concerns in biomedical articles.
- Studying the relationship between reviewer concern categories, severity, and author response patterns.
- Developing and validating semantic matching methods for free-text concern alignment.

**Who created the dataset, and on behalf of which entity?**

The dataset was created by the bioreview-bench project team. See the GitHub repository at [github.com/jang1563/bioreview-bench](https://github.com/jang1563/bioreview-bench) for the current list of contributors.

**Who funded the creation of the dataset?**

Funding information will be disclosed in the associated publication. See the GitHub repository for up-to-date acknowledgements.

---

## 2. Composition

**What do the instances that comprise the dataset represent?**

Each instance represents one published biomedical research article together with a structured list of the substantive concerns raised by peer reviewers during the journal review process. Concerns are extracted from open peer review materials (decision letters, author response letters, and eLife Assessments) and normalised into a common schema. Each concern is annotated with a category, a severity level, and an author stance label indicating how the authors responded to that concern in their revision.

**How many instances are there in total, and per split?**

The current repository snapshot contains 6,940 articles (instances) and 101,869
concern records across all splits.

| Split      | Articles | Concerns |
|------------|----------|----------|
| train      | 5,387    | 79,121   |
| validation | 953      | 14,101   |
| test       | 600      | 8,647    |
| **Total**  | **6,940**| **101,869**|

The realized article shares are 77.6% train, 13.7% validation, and 8.6% test.
These are not a 70% / 15% / 15% split.

**Does the dataset contain all possible instances, or is it a sample?**

The dataset is a sample. The current repository snapshot spans eLife, PLOS,
F1000Research, PeerJ, and Nature Portfolio articles that met source-specific
collection and packaging criteria. Not all articles published by these journals
in the collection period are included; articles without usable public review
materials, adequate text access, or release-compatible packaging status may be
excluded.

**What data does each instance consist of?**

Each instance contains:

| Field      | Content                                                                     |
|------------|-----------------------------------------------------------------------------|
| `id`       | Unique article identifier (e.g., `elife:84798`)                            |
| `source`   | Journal source (e.g., `elife`)                                              |
| `doi`      | Digital Object Identifier of the published article                          |
| `title`    | Article title                                                               |
| `abstract` | Article abstract                                                            |
| `concerns` | List of `ReviewerConcern` objects (see below)                               |

Each `ReviewerConcern` object contains:

| Field           | Content                                                                    |
|-----------------|----------------------------------------------------------------------------|
| `concern_id`    | Unique concern identifier (e.g., `elife:84798:c01`)                       |
| `concern_text`  | Full text of the concern as extracted from peer review materials           |
| `category`      | Concern category (one of nine values; see below)                           |
| `severity`      | Concern severity (`major`, `minor`, or `optional`)                        |
| `author_stance` | Author response label (`conceded`, `rebutted`, `partial`, `unclear`, `no_response`) |

For records that pass source-specific packaging checks, dataset records may
include article text sections such as methods, results, discussion, and figure
captions in `paper_text_sections` / `paper_text_v1_sections`, together with raw
or lightly processed public review materials. These fields are included to make
benchmark instances inspectable and executable during review; their reuse remains
governed by the original publisher terms documented in `LICENSE_MATRIX.md`.

**Is there a label or target associated with each instance?**

Yes. Labels operate at two levels:

1. **Concern-level labels.** Each concern record carries three labels:
   - `category`: The type of concern (design_flaw, statistical_methodology, missing_experiment, figure_issue, prior_art_novelty, writing_clarity, reagent_method_specificity, interpretation, other).
   - `severity`: An LLM-assigned operational label (`major`, `minor`,
     `optional`), not an independently adjudicated reviewer rating.
   - `author_stance`: An LLM-derived label describing how the authors appeared
     to respond (`conceded`, `rebutted`, `partial`, `unclear`,
     `no_response`). It is not a scientific-validity label.

2. **Article-level task references.** For the benchmark task, the normalized set
   of concern texts for an article constitutes the recorded reference set
   against which an AI tool is evaluated. This set may contain normalization
   errors and omissions.

The benchmark target is grounded in materials published by source platforms as
peer review and author response. Possible AI assistance in individual recent
reviews is unknown. The normalized concern units, category labels, severity
labels, and author-stance labels are silver-standard fields derived from LLM
extraction plus rule-based post-processing; they have not been independently
re-adjudicated for every record. See the Preprocessing section for details and
limitations.

**Is any information missing from individual instances?**

Some instances may lack an author response letter if the article was accepted without revision or if the revision materials were not publicly archived at the time of data collection. In such cases, `author_stance` is set to `no_response` for all concerns in that article. A small number of eLife reviewed_preprint articles published near the 2023 format transition may have incomplete eLife Assessment fields; these are flagged in the metadata.

Some records may omit body sections or review/response text when packaging
eligibility is unavailable or incomplete. Users should treat the DOI, publisher
article page, API response, or transparent-review package as the upstream source
of record for any missing or source-governed text.

**Are relationships between individual instances made explicit?**

Concerns within an article are linked by the shared article `id` and by
sequential `concern_id` values. No cross-article family relationship is
encoded. Consequently, although article IDs are split-disjoint, the
v4.1.3 DOI-stem audit found related/versioned F1000 manuscript families across
all partition pairs: 115 train–validation, 41 train–test, and 9
validation–test crossing families. Development data overlap 49/150 F1000 test
articles across 48 families. These counts are a source-specific lower bound
that does not resolve fuzzy-title, preprint-to-journal, or cross-source
relations. At the uncalibrated effective Jaccard cutoff 0.195, query-time
same-family filtering reduced lexical-audit BM25 F1 from 0.0577 to 0.0148 on
the primary full-600 snapshot; the 0.0690 to 0.0171 result on 450 non-eLife
articles is post-hoc. This diagnostic does not replace the frozen split or
historical embedding table. Full aggregate evidence is in
`results/v4/measurement_audit.{json,md}` and `KNOWN_ISSUES.md`.

**Are there recommended data splits?**

The authorized canonical dataset provides train, validation, and test splits.
The frozen test snapshot was intended for final evaluation only; hyperparameter
tuning and model selection should use validation. The splits preserve
approximate category and severity proportions, but do not guarantee
split-disjoint manuscript families. The rights-minimized public index does not
contain the prose or targets needed to execute this protocol.

**Are there any errors, sources of noise, or redundancies in the dataset?**

Several sources of noise should be noted:

- **LLM extraction errors.** Concern extraction and categorisation used LLM-based processing. LLMs may merge distinct concerns, split a single concern into multiple records, or misclassify categories. The silver-standard labels are not gold standard.
- **Severity labelling noise.** Severity labels (major, minor, optional) are derived from reviewer language (e.g., "essential", "important", "minor point") using heuristic rules. Reviewer language is not always consistent across journals or individuals.
- **Author stance ambiguity.** Distinguishing `conceded` from `partial`, or `rebutted` from `unclear`, can require nuanced reading of author response letters. LLM-based stance assignment may not resolve all ambiguous cases correctly.
- **eLife format heterogeneity.** Pre-2023 (journal format) and post-2023 (reviewed_preprint format) articles have structurally different review materials. Normalisation to a common schema introduces the risk of format-specific extraction errors.
- **Concern granularity variation.** Reviewers vary in how atomically they express concerns. Some concern records may represent a composite of what could reasonably be considered two or more distinct issues.

**Is the dataset self-contained?**

The concern records and metadata are self-contained within the dataset. Access to
full article body text depends on the exported config and the source-specific
release policy. Users should not assume that every source article can be
redistributed under identical terms.

**Does the dataset contain data that might be considered confidential?**

No unpublished manuscript content, private reviewer identities, or confidential
correspondence are intended to be included. However, source materials are not
uniformly licensed across all journals. Some review and response materials are
article-specific or optional-publication artifacts. Reviewer names, where they
appeared in publicly released source materials (particularly F1000Research), may
appear within extracted concern text but were not specifically sought or
catalogued.

**Does the dataset contain data that might be considered offensive or harmful?**

The dataset consists of scientific peer review commentary on biomedical research. The content is technical and professional in nature. It is unlikely to contain offensive content. However, peer review commentary occasionally includes strong critical language, and individual records may reflect reviewer opinions that authors considered unfair or incorrect. Such records are retained as they represent authentic peer review discourse relevant to the benchmark task.

---

## 3. Collection Process

**How was the data associated with each instance acquired?**

Article metadata (title, abstract, DOI) and peer review materials were
retrieved from source-specific public APIs, article pages, or public peer review
archives. The repository treats redistribution rights as source-specific rather
than uniform; see `LICENSE_MATRIX.md` for the current packaging policy.

Concern records were derived from peer review materials using the following pipeline:

1. **Retrieval.** Peer review materials were fetched programmatically from source-specific APIs and pages: eLife API (JATS XML sub-articles), PLOS JATS XML (aggregated-review-documents), F1000Research JATS XML (reviewer-report), PeerJ HTML scraping (/reviews/ pages), and Nature HTML scraping followed by PDF extraction (pdfplumber).
2. **Segmentation.** Review text was segmented into candidate concern units using a combination of sentence boundary detection and LLM-based paragraph-level analysis.
3. **Extraction.** An LLM was prompted to identify and extract substantive reviewer concerns from each segment, discarding complimentary remarks, general framing sentences, and concerns that were purely editorial.
4. **Categorisation.** Each extracted concern was assigned a category and severity label by the LLM, with rule-based post-processing to enforce label validity and handle edge cases.
5. **Author stance assignment.** The corresponding section of the author response letter was retrieved for each concern, and an LLM was prompted to assign an author stance label based on the content of the response. Stance assignment was anchored to manuscript revision evidence where available.
6. **Deduplication and filtering.** Near-duplicate concerns (cosine similarity >= 0.95 between SPECTER2 embeddings) within a single article were collapsed. Concerns falling outside the defined category set after post-processing were assigned `other`.

**What mechanisms or procedures were used to collect data?**

Data collection used public APIs and public article/review pages from the
supported sources, Python scripting for data retrieval and normalisation, and
LLM API calls for concern extraction, categorisation, and stance labelling. All
collection scripts are available in the `scripts/` directory of the GitHub
repository.

**If the dataset is a sample, what was the sampling strategy?**

Articles were included if they met all of the following criteria:
- Published in one of the currently supported benchmark sources.
- Peer review materials were publicly available at the time of collection.
- The article could be normalized into the benchmark schema.
- The article satisfied the source-specific packaging policy for the intended
  release footprint.

The source-collection universe was not created by random subsampling from all
articles published by the journals. Split assignment did use seeded random
sampling: fixed per-source counts were selected for test, and the remaining
records were randomly allocated within the documented train/validation strata.

**Who was involved in the data collection process?**

Automated data collection was performed using scripts developed by the project
team. LLM-based extraction and labelling used commercial LLM APIs. As of this
release, no completed independent human study validates the extracted concern
units, category/severity labels, or semantic matching threshold. A historical
148-row file is retained for provenance but is not accepted as validation
evidence because the legacy interface prefilled blank human fields with model
labels.

**Over what timeframe was the data collected?**

Source articles were published between 2013 and 2026, with the majority from 2019-2025. Data collection (API retrieval, LLM processing, and schema normalisation) was conducted in 2025-2026.

**Were any ethical review processes conducted?**

The dataset is derived from publicly available scientific publications and peer
review commentary. No human subjects research was conducted; no individual
participants were recruited or surveyed. The project team assessed that formal
ethics review was not required under applicable institutional guidelines for
research using publicly available published data.

**Did you collect the data from the individuals in question directly?**

No. Data were collected from publicly available journal archives and APIs, not directly from peer reviewers or authors.

**Were the individuals notified about the data collection?**

No. The peer reviewers and authors whose published communications appear in the
dataset were not individually notified. The repository relies on already public
source materials and applies a source-specific redistribution policy.

**Did the individuals in question consent to the collection and use of their data?**

Reviewers and authors participated in source journals whose public-review
workflows publish some or all peer review materials. Consent and redistribution
conditions are therefore mediated by the source journal's publication policy,
not by a single uniform benchmark-wide license.

**If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent?**

Consent, where applicable, was provided at the level of source-journal
publication rather than at the level of benchmark inclusion. No
dataset-specific consent revocation mechanism exists. Individuals who believe
their content has been included incorrectly or harmfully may contact the
dataset maintainers via the GitHub repository.

**Has an analysis of the potential impact of the dataset and its use on data subjects been conducted?**

The dataset contains commentary by peer reviewers (who are typically anonymous in the published eLife format) and responses by article authors (who are identified by name in the published article). The potential impact on data subjects was considered as follows:

- **Reviewers:** identity and review-publication policies vary by source.
  eLife, PLOS, PeerJ, Nature Portfolio, and F1000Research do not expose review
  materials under one uniform rule, and F1000Research may expose reviewer names
  publicly.
- **Authors:** Authors are identified by DOI and article title. The dataset
  records concerns raised about their work and their responses. This
  information is already public, but repackaging it can still make critique
  easier to search and aggregate. Downstream users should avoid using the
  dataset to profile individual authors or make claims about researcher quality.
- **AI-generated analysis:** LLM-based concern extraction may introduce errors that misrepresent the original peer review. Downstream users should treat concern records as silver-standard data and consult the original published review materials for authoritative content.

---

## 4. Preprocessing, Cleaning, and Labeling

**Was any preprocessing, cleaning, or labeling of the data done?**

Yes. Substantial preprocessing, cleaning, and labelling were applied:

- **Text normalisation.** HTML and XML tags were stripped from retrieved review texts. Unicode normalisation (NFC) was applied. Figures and table references embedded in review text were retained as-is.
- **Concern extraction.** LLM-based segmentation and extraction identified individual concern units from multi-topic review paragraphs. This step introduced the possibility of under-segmentation (merging distinct concerns) and over-segmentation (splitting a single concern).
- **Category and severity labelling.** LLM-based classification assigned category and severity labels. Rule-based post-processing enforced schema validity (e.g., rejecting out-of-vocabulary labels) and resolved a small number of extraction failures by assigning `other` / `minor` as defaults.
- **Author stance labelling.** LLM-based analysis of the author response letter produced stance labels. Stance was anchored to evidence of manuscript revision: a concern was labelled `conceded` only if the response letter described a corresponding change to the manuscript. Polite acknowledgements without revision were not sufficient for `conceded`. This anchoring was applied heuristically via prompt engineering; it was not verified by manual inspection of every revised manuscript.
- **Deduplication.** Near-duplicate concerns within an article (SPECTER2 cosine similarity >= 0.95) were collapsed to a single record.
- **Split assignment.** With seed 42, the builder sampled a fixed 600-article
  test set by source (150 eLife, 150 PLOS, 150 F1000Research, 100 Nature, and 50
  PeerJ). It grouped the remaining records by
  `(source, editorial_decision, review_format)`, randomly shuffled within each
  stratum, assigned approximately 15% of each eligible stratum to validation,
  and assigned the rest to train. Subject area, concern category, resolution,
  and publication time were not stratification keys. No explicit temporal
  balancing or chronological holdout was used.

**Known limitations of the preprocessing pipeline:**

- Concern segmentation, category labels, and severity labels are silver standard,
  not gold standard. Users relying on these fields for training or analysis
  should be aware that a non-trivial fraction may be incorrect.
- Author stance labels reflect LLM interpretation of published author response
  letters, which may not always accurately capture the nuance of the authors'
  actual position.
- The matching threshold (`allenai/specter2_base` cosine >= 0.65) is a
  provisional frozen operating point. It has not completed independent human
  validation, and all published frozen scores are matcher-dependent.
- The effective lexical-audit Jaccard threshold 0.195 and every point in the
  deterministic v4.1.3 threshold sweep are uncalibrated sensitivity settings,
  not validated semantic-equivalence boundaries.
- All six frozen result JSONs store overall `f1_macro=0.0` as an unpopulated
  legacy sentinel despite non-zero per-category F1 values. Historical
  category-macro F1 is invalid and unreported; v4.1.2 computes it only for new
  runs.
- The benchmark does not yet quantify extraction omissions against raw review
  text with a powered, system-label-blinded human audit. Source and article
  context would remain visible to raters.
- LLM-based processing may reflect the biases of the underlying LLM (e.g., systematic tendencies to classify certain concern types differently across subfields of biomedicine).

**Was the "raw" data saved in addition to the preprocessed data?**

Yes, for records that passed source-specific packaging checks, the review
artifact includes article-level raw or lightly processed fields such as
`paper_text_sections`, `decision_letter_raw`, and, when available,
`author_response_raw`. These fields are included so benchmark instances are
inspectable and runnable during review; they are not a blanket grant of rights
beyond the original publisher terms. For conditional sources, release eligibility
is handled at the article level and documented in `LICENSE_MATRIX.md`. The
publisher article page, API response, or transparent-review package remains the
upstream source of record.

**Is the software used to preprocess, clean, or label the data available?**

Yes. Collection and preprocessing scripts are available in the `scripts/`
directory of the GitHub repository at
[github.com/jang1563/bioreview-bench](https://github.com/jang1563/bioreview-bench).
The package is not currently on PyPI; install the audited GitHub `v4.1.3` tag as
shown in `README.md`.

---

## 5. Uses

**Has the dataset been used for any tasks already?**

At the time of this document, bioreview-bench had been used internally to
produce a frozen six-system score snapshot on 600 articles. The separately
reported 450-article non-eLife analysis is post-hoc. These artifacts are
published for transparency; they are not an open public leaderboard.

**Is there a repository that links to any or all papers or systems that use the dataset?**

The GitHub repository at [github.com/jang1563/bioreview-bench](https://github.com/jang1563/bioreview-bench) maintains a list of known publications and systems that use the dataset. Users are encouraged to open an issue or pull request to register their work.

**What (other) tasks could the dataset be used for?**

Beyond the primary benchmark task (concern detection), the dataset is suitable for:

- **Concern classification.** Training models to assign category and severity labels to free-text reviewer concerns.
- **Author stance prediction.** Training models to predict how authors will respond to a given concern, given the article text and the concern text.
- **Review quality analysis.** Studying the distribution of concern types, severity, and author stances across biomedical subfields, article types, or time periods.
- **Semantic matching development.** Developing and validating embedding-based or cross-encoder matching methods for scientific text.
- **Summarisation evaluation.** Evaluating whether AI-generated review summaries preserve the substantive concerns present in the original reviews.
- **Longitudinal analysis.** Studying trends in reviewer expectations and author responses across the 2019-2024 period.

**Is there anything about the composition or collection that might impact future uses?**

Yes. Several factors should be considered:

- **Source composition.** The current repository snapshot contains five sources
  (F1000Research 2,679, eLife 1,810, PLOS 1,737, PeerJ 244, Nature 470). Each
  journal has distinct editorial philosophy and reviewer culture. Models trained
  on bioreview-bench may not generalise to journals with significantly different
  practices.
- **Silver-standard normalized fields.** The source platforms publish the
  underlying prose as peer review, but concern segmentation, category, severity,
  and author stance are not exhaustively human-validated. Possible AI assistance
  in recent source reviews is not identified. Models trained on these fields may
  learn the biases of the extraction pipeline in addition to source signal.
- **Temporal distribution.** The 2013-2026 collection period includes the COVID-19 pandemic (2020-2022), during which peer review practices and research topics changed in ways that may affect model generalisability to other periods.
- **eLife format change.** The shift from the journal format to the reviewed_preprint format in 2023 introduced structural differences in review materials. Models trained on the combined dataset may perform differently on the two subsets.
- **LLM contamination risk.** Articles and review materials in the dataset may appear in the training corpora of LLMs used to benchmark against bioreview-bench. Test set contamination cannot be ruled out for LLMs with training data cutoffs after 2013.

**Are there tasks for which the dataset should not be used?**

The dataset should not be used for:

- **Identifying or evaluating specific researchers.** The dataset should not be used to build systems that profile individual researchers, reviewers, or authors based on their review behaviour or response patterns.
- **Automated editorial decision-making.** The dataset was created to evaluate concern-detection tools, not to support automated accept/reject decisions. The author stance labels reflect one dimension of review outcome and should not be interpreted as a proxy for article quality.
- **Training content moderation systems.** Peer review commentary is a specialised scientific discourse that should not be treated as equivalent to general-domain content requiring moderation.

---

## 6. Distribution

**Will the dataset be distributed to third parties beyond the team/entity?**

The named GitHub repository is the public code, documentation, and
evaluation-results channel; manuscript source and PDF remain local-only. The
canonical Hugging Face repository is a clean-history, rights-minimized public
index. It contains text-free article metadata for all splits and text-free
project-created category/severity/stance rows for train and validation.

**How will the dataset be distributed?**

Public code and release artifacts are distributed through GitHub. The public
Hugging Face index excludes titles, abstracts, article body text, normalized
concern text, raw reviews/decision letters, author responses, reviewer
and data-row names, emails, and explicit identity fields, internal traces, and
test targets. Stable concern IDs may retain source-local review ordinals such
as `R1`; these ordinals do not identify a person. Project-maintainer contact
metadata in `CITATION.cff` is outside the reviewer/data-row identity exclusion.
A separate private raw archive is retained only for authorized reproducibility.

The public index exposes train/validation/test membership together with stable
article IDs and DOIs. Because source peer reviews are publicly retrievable,
withholding test target rows from the package does not make the test secret or
blind. This snapshot should not be used for new blind evaluation or claims of
hidden-test generalization.

**When will the dataset be distributed?**

The GitHub v4.1.3 release and the rights-minimized Hugging Face index are public
as of 2026-07-25. Any broader text-bearing distribution requires article-level
rights review and a separate release decision.

**Will the dataset be distributed under a copyright or other IP license?**

The project uses a dual-license-plus-source-policy model. Benchmark annotations
and packaging metadata are released under CC-BY-NC 4.0. Code (Python package,
evaluation harness, scripts) is released under Apache-2.0. Underlying article,
review, and author-response materials remain subject to source-specific terms.

For source-specific redistribution rules, see `LICENSE_MATRIX.md`. Users should
not assume that all source content in the benchmark can be republished under one
uniform content license.

**Have any third parties imposed IP-based or other restrictions on the data?**

Potential restrictions vary by source and by article. Optional-publication
review histories, non-OA article text, or source-specific terms may constrain
redistribution of some materials. The repository therefore uses a conservative
packaging policy rather than assuming no third-party restrictions.

**Do any export controls or other regulatory restrictions apply?**

No export controls are known to apply to this dataset. The content is
scientific peer review commentary on published biomedical research and does not
knowingly include controlled technology, dual-use research of concern, or
private sensitive data. Public author names, DOIs, article titles, and, for
some sources, public reviewer names may still constitute personal data under
some data-protection regimes; users are responsible for complying with
applicable local requirements.

---

## 7. Maintenance

**Who will be supporting, hosting, and maintaining the dataset?**

The project team maintains a public rights-minimized index and a separate
private raw archive on Hugging Face. GitHub is the public venue for code,
documentation, release metadata, issue tracking, version management, and
community contributions.

**How can the owner, curator, or manager of the dataset be contacted?**

The preferred contact mechanism is via GitHub Issues at [github.com/jang1563/bioreview-bench/issues](https://github.com/jang1563/bioreview-bench/issues). For other enquiries, contact information for the project maintainer is available in the GitHub repository.

**Is there an erratum?**

Yes. `KNOWN_ISSUES.md`, `RELEASE_NOTES_v4.1.2.md`, and
`RELEASE_NOTES_v4.1.3.md` record the current
measurement and release corrections, including the invalid historical
`f1_macro` sentinel, split-construction corrections, matcher provenance, and
family leakage. The frozen raw result files were not rewritten, and no
independent human-validation study was added.

**Will the dataset be updated?**

Yes. Planned updates include:

- **Future:** Addition of further journal sources (e.g., Review Commons) and additional article batches.
- **Ongoing:** Correction of identified labelling errors, schema refinements, and expansion as journals continue to publish open peer review materials.

Updates will be versioned in GitHub. Dataset snapshots will be distributed only
where their fields and source-specific terms permit. Release notes will be
published in the GitHub repository.

**If the dataset relates to people, are there applicable limits on the retention of the data?**

The dataset contains scientific publications and peer review commentary, which
are public records. No private sensitive personal data is knowingly collected or
stored. Public author names appear as part of bibliographic metadata, and public
reviewer names may appear for sources where reviewer identity is published by
the source journal. No retention limits are anticipated, but the project team
will respond to requests from individuals who believe their data has been
handled incorrectly.

**Will older versions of the dataset continue to be supported, hosted, and maintained?**

GitHub release history and the clean public Hugging Face `v4.1.3` tag remain
available; `v4.1.2` and `v4.1` are retained as historical public-index tags.
Raw historical
Hugging Face snapshots remain private for authorized reproducibility and are
not implied to be publicly redistributable.

**If others want to extend, augment, build on, or contribute to the dataset, is there a mechanism for them to do so?**

Yes. Contributions are welcome via the GitHub repository. The preferred mechanisms are:

- **Bug reports and data corrections:** GitHub Issues.
- **New journal sources or additional articles:** GitHub Pull Requests with accompanying data in the required schema format. Contributions should include extraction scripts or a clear description of the collection procedure.
- **Evaluation methodology:** Open an issue or pull request to propose protocol
  changes. There is currently no public test-scoring or leaderboard-submission
  service.
- **Schema proposals:** Open a GitHub Issue to propose schema extensions or modifications before submitting a pull request.

Data and documentation contributions must be compatible with the benchmark's
CC-BY-NC 4.0 data license and source-specific redistribution policy. Code
contributions must be compatible with Apache-2.0. Contributors are expected to
ensure that any content they contribute is either their own original work,
already openly licensed under compatible terms, or derived from public-domain
sources.

---

## References

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daume III, H., & Crawford, K. (2018). Datasheets for Datasets. *arXiv preprint arXiv:1803.09010*. https://arxiv.org/abs/1803.09010

"""JATS XML parser.

Namespace version differences neutralized via local-name() XPath.
Based on AI_Scientist_team xml_parser.py; SQLModel dependency removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lxml import etree


# ── XPath helpers (namespace-agnostic) ───────────────────────────────────────

def _xpath(node: Any, expr: str) -> list[Any]:
    """local-name()-based XPath; neutralizes namespace version differences."""
    return node.xpath(expr)  # type: ignore[no-any-return]


def _text(nodes: list[Any]) -> str:
    """Extract text from a node list with whitespace normalization."""
    parts = []
    for n in nodes:
        if hasattr(n, "itertext"):
            parts.append("".join(n.itertext()).strip())
        elif isinstance(n, str):
            parts.append(n.strip())
    return " ".join(p for p in parts if p)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ParsedReview:
    """Review text + author response for a single reviewer."""
    reviewer_num: int
    review_text: str
    author_response_text: str = ""


@dataclass
class ParsedArticle:
    """Full article information extracted from JATS XML."""
    article_id: str
    doi: str
    title: str
    abstract: str
    subjects: list[str] = field(default_factory=list)
    # Body sections
    sections: dict[str, str] = field(default_factory=dict)
    references: list[dict] = field(default_factory=list)
    # Review / author response
    decision_letter_raw: str = ""
    author_response_raw: str = ""
    reviews: list[ParsedReview] = field(default_factory=list)
    published_date: str = ""
    editorial_decision: str = "unknown"


class JATSParser:
    """Multi-source JATS XML → ParsedArticle.

    Supported sub-article formats by source:

    eLife new format (Reviewed Preprint, 2023+):
    - sub-article[@article-type="editor-report"]             → eLife Assessment
    - sub-article[@article-type="referee-report"]            → individual reviewer reports
    - sub-article[@article-type="author-comment"]            → author response

    eLife old format (Journal, ~2022):
    - sub-article[@article-type="decision-letter"]           → editor decision + all reviewer comments
    - sub-article[@article-type="reply"]                     → author response

    eLife very old format (pre-2016, GitHub XML):
    - sub-article[@article-type="article-commentary"]        → decision letter (equiv. to decision-letter)
    - sub-article[@article-type="reply"]                     → author response

    PLOS Biology/Genetics/etc.:
    - sub-article[@article-type="editor-report"]             → editor decision letters (multiple rounds)
    - sub-article[@article-type="aggregated-review-documents"] → combined editor letter + all reviewer comments
    - sub-article[@article-type="author-comment"]            → author response (may be attachment-only)

    F1000Research / Wellcome Open Research / Gates Open Research:
    - sub-article[@article-type="reviewer-report"]           → individual reviewer report
    - sub-article[@article-type="response"]                  → author response (per-reviewer or combined)
    """

    # Section title → standard key mapping
    _SECTION_MAP: dict[str, str] = {
        "introduction": "introduction",
        "intro": "introduction",
        "background": "introduction",
        "materials and methods": "methods",
        "methods": "methods",
        "materials & methods": "methods",
        "experimental procedures": "methods",
        "results": "results",
        "discussion": "discussion",
        "conclusion": "discussion",
        "conclusions": "discussion",
    }

    def parse(self, xml_bytes: bytes, article_id: str = "") -> ParsedArticle:
        """XML bytes → ParsedArticle."""
        try:
            root = etree.fromstring(xml_bytes)
        except etree.XMLSyntaxError as e:
            raise ValueError(f"XML parse error for {article_id}: {e}") from e

        article = ParsedArticle(
            article_id=article_id,
            doi=self._extract_doi(root),
            title=self._extract_title(root),
            abstract=self._extract_abstract(root),
            subjects=self._extract_subjects(root),
            sections=self._extract_sections(root),
            references=self._extract_references(root),
            published_date=self._extract_date(root),
        )

        # Parse sub-articles (open peer review content)
        sub_articles = _xpath(
            root,
            ".//*[local-name()='sub-article']"
        )
        reviews_by_num: dict[int, ParsedReview] = {}
        # Track whether we've already captured R1 content (for multi-round PLOS)
        _plos_review_captured = False
        _plos_response_captured = False
        # Counter for F1000 reviewer-report numbering
        _f1000_reviewer_counter = 0

        for sub in sub_articles:
            article_type = sub.get("article-type", "")

            # ── eLife new format / PLOS editor-report ──────────────────────
            if article_type == "editor-report":
                text = self._extract_body_text(sub)
                # For PLOS: first editor-report before aggregated-review-documents
                # is triage only; skip it. Use only the one that contains decision.
                # For eLife new format: this IS the assessment.
                if not article.decision_letter_raw:
                    article.decision_letter_raw = text
                    article.editorial_decision = self._infer_decision(text)

            # ── eLife new format: individual reviewer reports ───────────────
            elif article_type == "referee-report":
                reviewer_num = self._extract_reviewer_num(sub)
                review_text = self._extract_body_text(sub)
                if reviewer_num not in reviews_by_num:
                    reviews_by_num[reviewer_num] = ParsedReview(
                        reviewer_num=reviewer_num,
                        review_text=review_text,
                    )
                else:
                    reviews_by_num[reviewer_num].review_text += "\n\n" + review_text

            # ── eLife / PLOS author response ────────────────────────────────
            elif article_type == "author-comment":
                body_text = self._extract_body_text(sub)
                # Skip if body is just an attachment filename reference
                if self._is_attachment_only(body_text):
                    continue
                # For PLOS multi-round: only capture first author response (R1)
                if not _plos_response_captured:
                    article.author_response_raw = body_text
                    self._assign_author_responses(sub, reviews_by_num)
                    _plos_response_captured = True

            # ── PLOS: aggregated decision letter + all reviewer comments ────
            elif article_type == "aggregated-review-documents":
                if _plos_review_captured:
                    # Only use first round (R1) for v1.0
                    continue
                full_text = self._extract_body_text(sub)
                article.decision_letter_raw = full_text
                article.editorial_decision = self._infer_decision(full_text)
                # Split individual reviewer comments within the aggregated text
                self._split_legacy_decision_letter(full_text, reviews_by_num)
                _plos_review_captured = True

            # ── F1000: individual reviewer reports ──────────────────────────
            elif article_type == "reviewer-report":
                _f1000_reviewer_counter += 1
                review_text = self._extract_body_text(sub)
                num = _f1000_reviewer_counter
                if num not in reviews_by_num:
                    reviews_by_num[num] = ParsedReview(
                        reviewer_num=num,
                        review_text=review_text,
                    )
                # Accumulate all reviewer reports into decision_letter_raw
                sep = "\n\n" if article.decision_letter_raw else ""
                article.decision_letter_raw += sep + f"Reviewer {num}:\n{review_text}"
                if not article.editorial_decision or article.editorial_decision == "unknown":
                    article.editorial_decision = self._infer_decision(review_text)

            # ── F1000: author response (per-reviewer or combined) ───────────
            elif article_type == "response":
                body_text = self._extract_body_text(sub)
                if article.author_response_raw:
                    article.author_response_raw += "\n\n" + body_text
                else:
                    article.author_response_raw = body_text

            # ── eLife old format (Journal, ~2022) ───────────────────────────
            elif article_type == "decision-letter":
                full_text = self._extract_body_text(sub)
                article.decision_letter_raw = full_text
                article.editorial_decision = self._infer_decision(full_text)
                self._split_legacy_decision_letter(full_text, reviews_by_num)

            # ── eLife very old format (pre-2016, GitHub XML) ─────────────────
            elif article_type == "article-commentary":
                if not article.decision_letter_raw:
                    full_text = self._extract_body_text(sub)
                    article.decision_letter_raw = full_text
                    article.editorial_decision = self._infer_decision(full_text)
                    self._split_legacy_decision_letter(full_text, reviews_by_num)

            elif article_type == "reply":
                article.author_response_raw = self._extract_body_text(sub)
                self._assign_author_responses(sub, reviews_by_num)

        article.reviews = sorted(reviews_by_num.values(), key=lambda r: r.reviewer_num)
        return article

    # ── Metadata extraction ───────────────────────────────────────────────────

    def _extract_doi(self, root: Any) -> str:
        # Extract DOI from main article front-matter only (exclude sub-articles)
        nodes = _xpath(
            root,
            "./*[local-name()='front']//*[local-name()='article-id'][@pub-id-type='doi']"
        )
        return _text(nodes[:1])

    def _extract_title(self, root: Any) -> str:
        nodes = _xpath(root, ".//*[local-name()='article-title']")
        return _text(nodes[:1])

    def _extract_abstract(self, root: Any) -> str:
        # Extract abstract from main front only
        nodes = _xpath(root, "./*[local-name()='front']//*[local-name()='abstract'][not(@abstract-type)]")
        if not nodes:
            nodes = _xpath(root, "./*[local-name()='front']//*[local-name()='abstract']")
        return _text(nodes[:1])

    def _extract_subjects(self, root: Any) -> list[str]:
        # Extract scientific subject areas only (subj-group-type="heading")
        # Exclude display-channel (article types like "Research Article")
        nodes = _xpath(
            root,
            "./*[local-name()='front']//*[local-name()='subj-group']"
            "[@subj-group-type='heading']/*[local-name()='subject']"
        )
        if nodes:
            return [_text([n]) for n in nodes if _text([n])]
        # Fallback: if no heading type, return all subjects except display-channel
        all_nodes = _xpath(
            root,
            "./*[local-name()='front']//*[local-name()='subject']"
        )
        results = []
        for n in all_nodes:
            parent = n.getparent()
            if parent is not None and parent.get("subj-group-type", "") == "display-channel":
                continue
            t = _text([n])
            if t:
                results.append(t)
        return results

    def _extract_date(self, root: Any) -> str:
        """Return date in YYYY-MM-DD format, or empty string if unavailable."""
        # Search pub-date only in main article front-matter (exclude sub-articles)
        pub_date_nodes = _xpath(
            root,
            "./*[local-name()='front']//*[local-name()='pub-date'][@pub-type='epub']"
        )
        if not pub_date_nodes:
            pub_date_nodes = _xpath(
                root,
                "./*[local-name()='front']//*[local-name()='pub-date']"
            )
        # eLife new format: date-type="publication"
        if not pub_date_nodes:
            pub_date_nodes = _xpath(
                root,
                "./*[local-name()='front']//*[local-name()='pub-date'][@date-type='publication']"
            )

        if pub_date_nodes:
            pub = pub_date_nodes[0]
            year = _text(_xpath(pub, "./*[local-name()='year']"))
            month = _text(_xpath(pub, "./*[local-name()='month']")) or "01"
            day = _text(_xpath(pub, "./*[local-name()='day']")) or "01"
            if year:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Fallback: only year available in front
        year_nodes = _xpath(root, "./*[local-name()='front']//*[local-name()='year']")
        year = _text(year_nodes[:1])
        if year:
            return f"{year}-01-01"
        return ""

    # ── Body section extraction ───────────────────────────────────────────────

    def _extract_sections(self, root: Any) -> dict[str, str]:
        body_nodes = _xpath(root, ".//*[local-name()='body']")
        if not body_nodes:
            return {}

        body = body_nodes[0]
        sections: dict[str, str] = {}

        for sec in _xpath(body, "./*[local-name()='sec']"):
            title_nodes = _xpath(sec, "./*[local-name()='title']")
            title_raw = _text(title_nodes).lower().strip().rstrip(".")

            # Map to standard section key
            key = None
            for pattern, standard_key in self._SECTION_MAP.items():
                if pattern in title_raw:
                    key = standard_key
                    break
            if key is None:
                key = re.sub(r"[^a-z0-9_]", "_", title_raw)[:40] or "section"

            text = self._extract_body_text(sec)
            if text:
                if key in sections:
                    sections[key] += "\n\n" + text
                else:
                    sections[key] = text

        # If no sections found, use full body text
        if not sections:
            full_text = self._extract_body_text(body)
            if full_text:
                sections["body"] = full_text

        return sections

    def _extract_body_text(self, node: Any) -> str:
        """Extract all text from a node (strips tags)."""
        texts = []
        for elem in node.iter():
            tag = elem.tag
            if isinstance(tag, str):
                local = tag.split("}")[-1] if "}" in tag else tag
                # Include only heading and body text elements
                if local in ("title", "p", "td", "th", "li"):
                    t = (elem.text or "").strip()
                    if t:
                        texts.append(t)
            # Tail text
            t = (elem.tail or "").strip()
            if t:
                texts.append(t)
        return " ".join(texts)

    # ── Reference extraction ──────────────────────────────────────────────────

    def _extract_references(self, root: Any) -> list[dict]:
        refs = []
        for ref in _xpath(root, ".//*[local-name()='ref']"):
            ref_id = ref.get("id", "")
            authors = _xpath(ref, ".//*[local-name()='name']")
            author_list = [_text([a]) for a in authors[:3]]

            year_nodes = _xpath(ref, ".//*[local-name()='year']")
            year = _text(year_nodes[:1])

            title_nodes = _xpath(ref, ".//*[local-name()='article-title']")
            title = _text(title_nodes[:1])

            journal_nodes = _xpath(ref, ".//*[local-name()='source']")
            journal = _text(journal_nodes[:1])

            refs.append({
                "id": ref_id,
                "authors": author_list,
                "year": year,
                "title": title,
                "journal": journal,
            })
        return refs

    # ── Reviewer number extraction ────────────────────────────────────────────

    def _extract_reviewer_num(self, sub: Any) -> int:
        """Extract reviewer number from a referee-report sub-article."""
        contrib_nodes = _xpath(sub, ".//*[local-name()='contrib']")
        for contrib in contrib_nodes:
            role_nodes = _xpath(contrib, "./*[local-name()='role']")
            role = _text(role_nodes).lower()
            # Match "Reviewer #2" pattern
            m = re.search(r"reviewer\s*#?\s*(\d+)", role, re.I)
            if m:
                return int(m.group(1))

        # Try to extract from sub-article id ("sa2" → 2)
        sub_id = sub.get("id", "")
        m = re.search(r"sa(\d+)", sub_id)
        if m:
            return int(m.group(1))

        return 1  # fallback

    # ── Author response assignment ────────────────────────────────────────────

    def _assign_author_responses(
        self,
        author_comment_sub: Any,
        reviews_by_num: dict[int, ParsedReview],
    ) -> None:
        """Split author response into per-reviewer sections and assign to each ParsedReview."""
        full_text = self._extract_body_text(author_comment_sub)

        # Split on "Reviewer #1" or "Reviewer 1" patterns
        pattern = re.compile(r"reviewer\s*#?\s*(\d+)", re.IGNORECASE)
        matches = list(pattern.finditer(full_text))

        if not matches:
            # Cannot split — assign full response to all reviewers
            for review in reviews_by_num.values():
                review.author_response_text = full_text
            return

        # Assign text between each match to the corresponding reviewer
        for i, match in enumerate(matches):
            reviewer_num = int(match.group(1))
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            response_text = full_text[start:end].strip()

            if reviewer_num in reviews_by_num:
                reviews_by_num[reviewer_num].author_response_text = response_text

    # ── Legacy decision letter splitting ──────────────────────────────────────

    def _split_legacy_decision_letter(
        self,
        full_text: str,
        reviews_by_num: dict[int, ParsedReview],
    ) -> None:
        """Split individual reviewer comments from an old-format decision letter.

        In old eLife format, all reviewer comments are embedded in the decision letter.
        Split on "Reviewer #N:" or "Reviewer N:" patterns.
        If unsplittable, register the full text as reviewer 1's comment.
        """
        pattern = re.compile(r"reviewer\s*#?\s*(\d+)\s*[:\.]", re.IGNORECASE)
        matches = list(pattern.finditer(full_text))

        if not matches:
            # No reviewer delimiter found — register full text as reviewer 1
            if 1 not in reviews_by_num and full_text.strip():
                reviews_by_num[1] = ParsedReview(
                    reviewer_num=1,
                    review_text=full_text.strip(),
                )
            return

        for i, match in enumerate(matches):
            reviewer_num = int(match.group(1))
            start = match.end()  # text after "Reviewer #N:"
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            review_text = full_text[start:end].strip()

            if not review_text:
                continue
            if reviewer_num not in reviews_by_num:
                reviews_by_num[reviewer_num] = ParsedReview(
                    reviewer_num=reviewer_num,
                    review_text=review_text,
                )
            else:
                reviews_by_num[reviewer_num].review_text += "\n\n" + review_text

    # ── Attachment-only detection ─────────────────────────────────────────────

    @staticmethod
    def _is_attachment_only(text: str) -> bool:
        """Return True if the body text is just an uploaded attachment filename.

        PLOS author-comment sub-articles sometimes contain only a filename like:
            "Attachment Submitted filename: Response_to_Reviewers.docx"
        These carry no parseable text content for concern extraction.
        """
        stripped = text.strip()
        if len(stripped) > 300:
            return False
        return bool(re.search(
            r"submitted\s+filename|attachment\s*:|\.(docx|pdf|doc|txt)\b",
            stripped,
            re.IGNORECASE,
        ))

    # ── Editorial decision inference ──────────────────────────────────────────

    def _infer_decision(self, decision_text: str) -> str:
        t = decision_text.lower()

        # Major revision keywords (check first — most common decision)
        major_keywords = [
            "major revision", "major revisions",
            "essential revision", "essential revisions",
            "substantive revision", "substantive revisions",
            "substantial revision", "substantial revisions",
            "significant revision", "significant revisions",
        ]
        if any(kw in t for kw in major_keywords):
            return "major_revision"

        # Minor revision keywords
        minor_keywords = [
            "minor revision", "minor revisions",
            "optional revision", "optional revisions",
        ]
        if any(kw in t for kw in minor_keywords):
            return "minor_revision"

        # Accept (no revision mentioned)
        if "accept" in t and "revision" not in t:
            return "accept"

        if "reject" in t:
            return "reject"
        return "unknown"

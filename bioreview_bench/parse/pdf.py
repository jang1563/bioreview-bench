"""Nature peer review PDF parser using pdfplumber.

Nature transparent peer review PDFs follow a consistent structure:
  - Page 1: Cover page (title, CC BY license notice, reviewer counts)
  - Page 2+: "REVIEWER COMMENTS" header, then individual reviewer sections
    headed "Reviewer #N (Remarks to the Author):"
  - Optional later section: Author point-by-point response, preceded by
    phrases such as "Reviewer comments are highlighted in brown",
    "AUTHOR RESPONSE", or "Response to Reviewers".

The parser splits the PDF text into decision_letter_raw (all reviewer
comments) and author_response_raw (author rebuttal, if present).
"""

from __future__ import annotations

import io
import re


class NaturePDFParser:
    """Parse Nature peer review PDFs into structured text fields.

    Usage::

        parser = NaturePDFParser()
        result = parser.parse(pdf_bytes)
        decision_letter_raw = result["decision_letter_raw"]
        author_response_raw = result["author_response_raw"]
    """

    # Patterns that indicate the start of the author response section.
    _AUTHOR_RESPONSE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"reviewer\s+comments\s+are\s+highlighted", re.IGNORECASE),
        re.compile(r"REVISION\s+NOTES?", re.IGNORECASE),
        re.compile(r"AUTHOR\s+RESPONSE", re.IGNORECASE),
        re.compile(r"POINT[- ]BY[- ]POINT\s+RESPONSES?", re.IGNORECASE),
        re.compile(r"Response\s+to\s+Reviewers?", re.IGNORECASE),
        re.compile(r"Authors['']?\s+Response", re.IGNORECASE),
    ]

    # "REVIEWER COMMENTS" marks the end of the cover page (page 1).
    _REVIEWER_COMMENTS_HEADER = re.compile(r"REVIEWER\s+COMMENTS", re.IGNORECASE)

    def parse(self, pdf_bytes: bytes) -> dict[str, str]:
        """Parse PDF bytes and return decision_letter_raw / author_response_raw.

        Args:
            pdf_bytes: Raw bytes of the Nature peer review PDF.

        Returns:
            dict with keys:
              - ``decision_letter_raw``: Reviewer comment text.
              - ``author_response_raw``: Author rebuttal text, or ``""`` if absent.
        """
        try:
            import pdfplumber  # lazy import so non-Nature paths don't require it

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required to parse Nature PDFs. "
                "Install it with: uv add pdfplumber"
            ) from exc
        except Exception as exc:
            import sys
            print(f"[warn] NaturePDFParser: failed to extract text from PDF ({len(pdf_bytes)} bytes): {exc}", file=sys.stderr)
            return {"decision_letter_raw": "", "author_response_raw": ""}

        return self._split_sections(full_text)

    def _split_sections(self, text: str) -> dict[str, str]:
        """Split full PDF text into reviewer comments and author response.

        Strategy:
          1. Find the "REVIEWER COMMENTS" header to skip the cover page.
          2. Scan forward for the earliest author response pattern.
          3. Split at that boundary.

        Args:
            text: Concatenated text from all PDF pages.

        Returns:
            dict with ``decision_letter_raw`` and ``author_response_raw``.
        """
        # Skip the cover page: start from the "REVIEWER COMMENTS" header.
        m_header = self._REVIEWER_COMMENTS_HEADER.search(text)
        content_start = m_header.start() if m_header else 0

        # Search for author-response markers starting after the header line.
        # +20 is enough to skip the "REVIEWER COMMENTS" header itself so it
        # isn't mistakenly treated as an author-response boundary.
        header_end = (m_header.end() if m_header else content_start) + 1
        search_from = header_end
        author_response_start = -1

        for pattern in self._AUTHOR_RESPONSE_PATTERNS:
            m = pattern.search(text, search_from)
            if m:
                if author_response_start < 0 or m.start() < author_response_start:
                    author_response_start = m.start()

        if author_response_start > content_start:
            decision_letter_raw = text[content_start:author_response_start].strip()
            author_response_raw = text[author_response_start:].strip()
        else:
            decision_letter_raw = text[content_start:].strip()
            author_response_raw = ""

        return {
            "decision_letter_raw": decision_letter_raw,
            "author_response_raw": author_response_raw,
        }

"""Nature peer review PDF parser using pdfplumber.

Nature transparent peer review PDFs come in several structural variants:

  Variant A (standard, single-round):
    - Cover page (CC BY license, title)
    - "REVIEWER COMMENTS" header
    - "Reviewer #N (Remarks to the Author):" sections
    - Separate author response section ("Response to Reviewers", etc.)

  Variant B (interleaved, multi-round):
    - Cover page with "Reviewers' Comments:" header
    - First-round reviewer comments (Reviewer #1, #2, #3)
    - Author responses re-stating reviewer comment then replying (Reviewer #1 again...)
    - Possibly additional review rounds

  Variant C (no explicit header, starts directly with review content):
    - "reviewer comments based on the revised version..."
    - May have non-sequential reviewer numbering

The parser handles all three variants and produces:
  - decision_letter_raw: reviewer comments (cover page stripped)
  - author_response_raw: author rebuttal if separately identifiable
  - review_texts: list of individual reviewer comment blocks
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
        review_texts = result["review_texts"]  # list[str], one per reviewer
    """

    # ── Section start headers ──────────────────────────────────────────────
    # Any of these marks the end of the cover page / start of review content.
    _REVIEWER_SECTION_HEADERS: list[re.Pattern[str]] = [
        re.compile(r"REVIEWER\s+COMMENTS", re.IGNORECASE),
        re.compile(r"Reviewers'\s+Comments:", re.IGNORECASE),
        re.compile(r"Reviewers\s+Comments:", re.IGNORECASE),
        re.compile(r"reviewer\s+comments\s+based\s+on", re.IGNORECASE),
        re.compile(r"The\s+following\s+(?:is|are)\s+the\s+(?:editorial|peer\s*review)", re.IGNORECASE),
    ]

    # ── Author response section markers ────────────────────────────────────
    _AUTHOR_RESPONSE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"reviewer\s+comments\s+are\s+highlighted", re.IGNORECASE),
        re.compile(r"REVISION\s+NOTES?", re.IGNORECASE),
        re.compile(r"AUTHOR\s+RESPONSE", re.IGNORECASE),
        re.compile(r"POINT[- ]BY[- ]POINT\s+RESPONSES?", re.IGNORECASE),
        re.compile(r"Response\s+to\s+Reviewers?", re.IGNORECASE),
        re.compile(r"Authors?'\s*Response", re.IGNORECASE),
    ]

    # ── Per-reviewer section headers ────────────────────────────────────────
    # Captures "Reviewer #1", "Reviewer 2 (Remarks to the Author):", "Referee #3:", etc.
    #
    # IMPORTANT: must start at the beginning of a line (after optional whitespace)
    # to avoid matching inline references like "(Referee #1);" in editor letters.
    # The re.MULTILINE flag makes ^ match at the start of each line.
    _REVIEWER_HEADER = re.compile(
        r"(?m)^[ \t]*(?:Reviewer|Referee)\s*#?\s*(\d+)\s*(?:\([^)]*\))?\s*:?",
        re.IGNORECASE,
    )

    # ── Cover page detection ────────────────────────────────────────────────
    # These phrases appear in Nature cover pages but NOT in review content.
    _COVER_PAGE_MARKERS = re.compile(
        r"Creative\s+Commons\s+Attribution|Open\s+Access\s+This\s+file\s+is\s+licensed",
        re.IGNORECASE,
    )

    def parse(self, pdf_bytes: bytes) -> dict[str, object]:
        """Parse PDF bytes and return decision_letter_raw / author_response_raw / review_texts.

        Args:
            pdf_bytes: Raw bytes of the Nature peer review PDF.

        Returns:
            dict with keys:
              - ``decision_letter_raw``: All reviewer comment text (concatenated).
              - ``author_response_raw``: Author rebuttal text, or ``""`` if absent.
              - ``review_texts``: list[str] — individual reviewer comment blocks.
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
            print(
                f"[warn] NaturePDFParser: failed to extract text from PDF "
                f"({len(pdf_bytes)} bytes): {exc}",
                file=sys.stderr,
            )
            return {"decision_letter_raw": "", "author_response_raw": "", "review_texts": []}

        return self._split_sections(full_text)

    def _find_content_start(self, text: str) -> int:
        """Find where actual review content begins (skip cover page).

        Tries, in order:
        1. Explicit section headers ("REVIEWER COMMENTS", "Reviewers' Comments:", etc.)
        2. If text starts with a cover page (CC license marker), jump to first
           "Reviewer #N" occurrence.
        3. Fall back to position 0 (use full text).

        Returns:
            Character position where review content starts.
        """
        # 1. Try explicit section headers
        earliest: int | None = None
        for pattern in self._REVIEWER_SECTION_HEADERS:
            m = pattern.search(text)
            if m and (earliest is None or m.start() < earliest):
                earliest = m.start()
        if earliest is not None:
            return earliest

        # 2. Cover page detected in first 2000 chars → jump to first reviewer header
        if self._COVER_PAGE_MARKERS.search(text[:2000]):
            m_rev = self._REVIEWER_HEADER.search(text)
            if m_rev:
                return m_rev.start()

        # 3. No cover page markers and no section header → content starts at 0
        return 0

    def _split_sections(self, text: str) -> dict[str, object]:
        """Split full PDF text into reviewer comments and author response.

        Strategy:
          1. Find content_start (skip cover page).
          2. Try named author-response section markers for a hard split.
          3. If no named marker found, detect the interleaved format:
             find the first repeated reviewer number — that position is the
             boundary between first-round comments and the author response.
          4. Split per-reviewer blocks from the decision_letter portion only.

        Args:
            text: Concatenated text from all PDF pages.

        Returns:
            dict with ``decision_letter_raw``, ``author_response_raw``, ``review_texts``.
        """
        content_start = self._find_content_start(text)

        # ── Try explicit author-response markers (Variant A) ──────────────
        search_from = content_start + 1
        author_response_start: int = -1

        for pattern in self._AUTHOR_RESPONSE_PATTERNS:
            m = pattern.search(text, search_from)
            if m:
                if author_response_start < 0 or m.start() < author_response_start:
                    author_response_start = m.start()

        if author_response_start > content_start:
            decision_letter_raw = text[content_start:author_response_start].strip()
            author_response_raw = text[author_response_start:].strip()
            # Use _split_interleaved on the DL to correctly handle multi-round
            # formats (e.g. NatCellBio) where the DL contains multiple review
            # rounds and _split_reviewers would return one huge block.
            _, _, review_texts = self._split_interleaved(decision_letter_raw, 0)
            if not review_texts:
                review_texts = self._split_reviewers(decision_letter_raw)
            return {
                "decision_letter_raw": decision_letter_raw,
                "author_response_raw": author_response_raw,
                "review_texts": review_texts,
            }

        # ── Interleaved format: detect author response via repeated reviewer # ──
        # Walk through reviewer headers; the first "repeat" of a reviewer number
        # signals the start of the author response (restated reviewer comments).
        decision_end, author_response_raw, review_texts = self._split_interleaved(
            text, content_start
        )

        decision_letter_raw = text[content_start:decision_end].strip()
        return {
            "decision_letter_raw": decision_letter_raw,
            "author_response_raw": author_response_raw,
            "review_texts": review_texts,
        }

    # Minimum character length for a reviewer block to be considered substantive.
    # Blocks shorter than this are likely just header lines ("Reviewer #1:") with
    # no actual content — they arise when reviewer headers appear close together.
    _MIN_BLOCK_CHARS = 150

    def _split_interleaved(
        self, text: str, content_start: int
    ) -> tuple[int, str, list[str]]:
        """Handle interleaved (Variant B/C) PDFs.

        Detects the first repeated reviewer number as the author-response boundary.
        Returns (decision_end_pos, author_response_raw, per_reviewer_blocks).
        """
        seen_nums: set[str] = set()
        first_round_boundaries: list[int] = []
        author_response_start: int | None = None

        for m in self._REVIEWER_HEADER.finditer(text, content_start):
            num = m.group(1)
            if num in seen_nums:
                # First repeat → start of author response
                author_response_start = m.start()
                break
            seen_nums.add(num)
            first_round_boundaries.append(m.start())

        decision_end = author_response_start if author_response_start is not None else len(text)
        author_response_raw = text[author_response_start:].strip() if author_response_start else ""

        # Build per-reviewer blocks within the first-round portion
        review_texts: list[str] = []
        for i, start in enumerate(first_round_boundaries):
            end = (
                first_round_boundaries[i + 1]
                if i + 1 < len(first_round_boundaries)
                else decision_end
            )
            block = text[start:end].strip()
            # Drop header-only noise blocks (too short to contain substantive content)
            if block and len(block) >= self._MIN_BLOCK_CHARS:
                review_texts.append(block)

        # If no reviewer headers found at all, return full content as single block
        if not review_texts:
            full_content = text[content_start:decision_end].strip()
            if full_content:
                review_texts = [full_content]

        return decision_end, author_response_raw, review_texts

    def _split_reviewers(self, text: str) -> list[str]:
        """Split reviewer comment block into individual reviewer sections.

        Used for Variant A where decision_letter_raw is already clean (no
        author-response interleaving).  Falls back to returning the full text
        as a single-element list if fewer than two reviewer headers are found.

        Args:
            text: The reviewer-comments portion of the PDF (decision_letter_raw).

        Returns:
            List of per-reviewer comment strings. Empty strings are dropped.
        """
        seen_nums: set[str] = set()
        boundaries: list[int] = []

        for m in self._REVIEWER_HEADER.finditer(text):
            num = m.group(1)
            if num in seen_nums:
                continue
            seen_nums.add(num)
            boundaries.append(m.start())

        if len(boundaries) < 2:
            stripped = text.strip()
            return [stripped] if stripped else []

        review_texts: list[str] = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            block = text[start:end].strip()
            if block and len(block) >= self._MIN_BLOCK_CHARS:
                review_texts.append(block)

        return review_texts

"""Nature portfolio transparent peer review article collector.

Uses the CrossRef API (https://api.crossref.org/works) for article metadata.
No API key required. Uses the "polite pool" (mailto parameter) for better rate limits.

NOTE — BETA STATUS:
    This collector is in BETA. Before using it, ensure the following:

    1. Peer review content for Nature journals is distributed as PDF files, not
       clean JATS XML.  Full extraction of structured reviewer comments requires
       a PDF parsing library such as ``pdfplumber``, which is out of scope for
       v1.0.  ``fetch_peer_review_pdf`` returns the raw PDF bytes; downstream
       parsing is the caller's responsibility.

    2. Per-article manual verification of peer review availability is recommended
       because not every article in a peer-review-enabled journal has published
       reviewer reports (e.g. articles published before the journal adopted
       transparent peer review).

Journals with transparent peer review (available since ~2022):
    - Nature
    - Nature Communications
    - Nature Methods
    - Nature Genetics
    - Nature Cell Biology
    - Communications Biology
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import date, datetime
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

CROSSREF_API_BASE = "https://api.crossref.org/works"

# Nature-portfolio journals that publish transparent peer reviews
JOURNALS_WITH_REVIEWS = {
    "Nature",
    "Nature Communications",
    "Nature Methods",
    "Nature Genetics",
    "Nature Cell Biology",
    "Communications Biology",
}

# Default list of journals to query
DEFAULT_JOURNALS: list[str] = list(JOURNALS_WITH_REVIEWS)

# Default mailto for CrossRef polite pool (improves rate limits)
_DEFAULT_MAILTO = "bioreview-bench@research.example.com"


class NatureArticleMeta:
    """Article metadata extracted from the CrossRef API response."""

    __slots__ = (
        "article_id",
        "doi",
        "title",
        "abstract",
        "journal",
        "subjects",
        "published",
        "has_reviews",
    )

    def __init__(
        self,
        article_id: str,
        doi: str,
        title: str,
        abstract: str,
        journal: str,
        subjects: list[str],
        published: str,
        has_reviews: bool,
    ) -> None:
        self.article_id = article_id
        self.doi = doi
        self.title = title
        self.abstract = abstract
        self.journal = journal
        self.subjects = subjects
        self.published = published
        self.has_reviews = has_reviews


class NatureCollector:
    """Nature portfolio article metadata collector via CrossRef API.

    No API key required. Uses CrossRef polite pool for better rate limits.

    Peer review content for Nature journals is PDF-based, not JATS XML.
    ``fetch_xml`` always returns None.  Use ``fetch_peer_review_pdf`` to
    attempt download of the reviewer report PDF.

    Usage:
        async with NatureCollector() as collector:
            async for meta, xml_bytes in collector.iter_articles(
                journals=["Nature Communications"], max_articles=10
            ):
                # xml_bytes is always None for Nature articles
                pdf = await collector.fetch_peer_review_pdf(meta.doi)
    """

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        timeout: float = 30.0,
        mailto: str = _DEFAULT_MAILTO,
    ) -> None:
        self._delay = rate_limit_delay
        self._timeout = timeout
        self._mailto = mailto
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def __aenter__(self) -> NatureCollector:
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"Accept": "application/json", "User-Agent": f"bioreview-bench/1.0 (mailto:{self._mailto})"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request_time = time.monotonic()

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "NatureCollector client is not initialized. Use 'async with NatureCollector()'."
            )
        return self._client

    @staticmethod
    def _extract_published_date(value: str) -> date | None:
        text = (value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            pass
        if len(text) == 4 and text.isdigit():
            try:
                return date(int(text), 1, 1)
            except ValueError:
                return None
        return None

    @classmethod
    def _is_on_or_after(cls, value: str, start_date: date | None) -> bool:
        if start_date is None:
            return True
        published = cls._extract_published_date(value)
        if published is None:
            return True
        return published >= start_date

    @classmethod
    def _is_on_or_before(cls, value: str, end_date: date | None) -> bool:
        if end_date is None:
            return True
        published = cls._extract_published_date(value)
        if published is None:
            return True
        return published <= end_date

    @staticmethod
    def _infer_has_reviews(journal_name: str) -> bool:
        """Return True if the journal publishes transparent peer reviews."""
        return journal_name in JOURNALS_WITH_REVIEWS

    @staticmethod
    def _article_id_from_doi(doi: str) -> str:
        """Derive a Nature article identifier from a DOI.

        For example, "10.1038/s41586-023-12345-6" becomes "s41586-023-12345-6".
        """
        return doi.split("/")[-1] if "/" in doi else doi

    @staticmethod
    def _clean_abstract(abstract: str) -> str:
        """Remove JATS XML tags from CrossRef abstract text."""
        return re.sub(r"<[^>]+>", "", abstract).strip()

    @staticmethod
    def _date_parts_to_str(date_parts: list) -> str:
        """Convert CrossRef date-parts [[YYYY, MM, DD]] to ISO string."""
        if not date_parts or not date_parts[0]:
            return ""
        parts = date_parts[0]
        if len(parts) >= 3:
            return f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
        if len(parts) == 2:
            return f"{parts[0]:04d}-{parts[1]:02d}-01"
        return f"{parts[0]:04d}-01-01"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _get(
        self,
        url: str,
        params: dict[str, object] | None = None,
    ) -> httpx.Response:
        client = self._require_client()
        await self._throttle()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp

    async def list_reviewed_articles(
        self,
        subjects: list[str] | None = None,
        start_date: str = "2022-01-01",
        end_date: str | None = None,
        order: str = "desc",
        page_size: int = 100,
        max_articles: int | None = None,
        journals: list[str] | None = None,
    ) -> list[NatureArticleMeta]:
        """Retrieve article metadata from the CrossRef API.

        Args:
            subjects: Optional list of subject area strings to filter on
                (post-hoc, case-insensitive substring match).
            start_date: Collection start date (inclusive). Default "2022-01-01"
                because transparent peer review at Nature began ~2022.
            end_date: Collection end date (inclusive). None means unlimited.
            order: "desc" (newest first) or "asc" (oldest first).
            page_size: Number of records per API page (max 1000).
            max_articles: Stop after collecting this many articles.
            journals: Nature journal names to include. Defaults to all journals
                with transparent peer review.
        """
        if journals is None:
            journals = DEFAULT_JOURNALS

        results: list[NatureArticleMeta] = []
        start_date_obj = self._extract_published_date(start_date)
        end_date_obj = self._extract_published_date(end_date) if end_date else None

        for journal_name in journals:
            offset = 0

            while True:
                # CrossRef filter syntax for journal + date range
                filter_parts = [f"container-title:{journal_name}"]
                filter_parts.append(f"from-pub-date:{start_date}")
                if end_date:
                    filter_parts.append(f"until-pub-date:{end_date}")
                filter_parts.append("type:journal-article")

                api_params: dict[str, object] = {
                    "filter": ",".join(filter_parts),
                    "rows": min(page_size, 1000),
                    "offset": offset,
                    "sort": "published",
                    "order": order,
                    "mailto": self._mailto,
                    "select": "DOI,title,abstract,published,container-title,subject",
                }

                try:
                    resp = await self._get(CROSSREF_API_BASE, params=api_params)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (404, 400):
                        break
                    raise

                data = resp.json()
                message = data.get("message", {})
                items = message.get("items", [])
                total = message.get("total-results", 0)

                if not items:
                    break

                for item in items:
                    doi = item.get("DOI", "")
                    if not doi:
                        continue

                    title_list = item.get("title", [])
                    title = title_list[0] if title_list else ""

                    abstract_raw = item.get("abstract", "")
                    abstract = self._clean_abstract(abstract_raw)

                    # CrossRef container-title is a list
                    ct_list = item.get("container-title", [])
                    journal = ct_list[0] if ct_list else journal_name

                    # Published date from date-parts
                    pub_dp = item.get("published", {}).get("date-parts", [])
                    published = self._date_parts_to_str(pub_dp)

                    subject_raw = item.get("subject", [])
                    subj_list: list[str] = subject_raw if isinstance(subject_raw, list) else [subject_raw]

                    # Belt-and-suspenders date check
                    if not self._is_on_or_after(published, start_date_obj):
                        continue
                    if not self._is_on_or_before(published, end_date_obj):
                        if order == "asc":
                            # Exceeded end_date in ascending order; no more needed
                            return results
                        continue

                    # Post-hoc subject filtering
                    if subjects:
                        subj_lower = [s.lower() for s in subj_list]
                        if not any(
                            any(req.lower() in s for s in subj_lower)
                            for req in subjects
                        ):
                            continue

                    article_id = self._article_id_from_doi(doi)
                    has_reviews = self._infer_has_reviews(journal)

                    results.append(
                        NatureArticleMeta(
                            article_id=article_id,
                            doi=doi,
                            title=title,
                            abstract=abstract,
                            journal=journal,
                            subjects=subj_list,
                            published=published,
                            has_reviews=has_reviews,
                        )
                    )

                    if max_articles and len(results) >= max_articles:
                        return results

                offset += len(items)
                if offset >= total:
                    break

        return results

    async def fetch_xml(self, doi: str) -> bytes | None:
        """Return None: JATS XML is not available for Nature articles.

        Nature peer review content is distributed as PDF files.  Use
        ``fetch_peer_review_pdf`` to retrieve the reviewer report PDF.

        Args:
            doi: Article DOI (ignored; included for API symmetry with other
                 collectors).

        Returns:
            Always None.
        """
        print(
            f"[warn] JATS XML is not available for Nature articles (doi={doi}). "
            "Use fetch_peer_review_pdf() to retrieve the peer review PDF."
        )
        return None

    # Patterns to find the peer review PDF link in the Nature article HTML page
    _PDF_HREF_PATTERNS: list[re.Pattern[str]] = [
        # Primary: static-content.springer.com esm MOESM PDF links
        re.compile(
            r'href="(https://static-content\.springer\.com/esm/[^"]+MOESM[^"]*\.pdf)"',
            re.IGNORECASE,
        ),
        # Fallback: any springer esm link with "peer-review" in the path
        re.compile(
            r'href="(https://static-content\.springer\.com/esm/[^"]*peer[- _]review[^"]*\.pdf)"',
            re.IGNORECASE,
        ),
    ]
    # Link labels that indicate the peer review supplementary file
    _PDF_LABEL_PATTERN = re.compile(
        r"peer\s+review\s+file|transparent\s+peer\s+review",
        re.IGNORECASE,
    )

    async def _find_peer_review_pdf_url(self, doi: str) -> str | None:
        """Scrape the Nature article HTML page to find the peer review PDF URL.

        Nature peer review PDFs are hosted on Springer's static content server
        with URLs of the form:
            https://static-content.springer.com/esm/art%3A{doi_encoded}/
                MediaObjects/{nums}_MOESM{N}_ESM.pdf

        The exact URL must be discovered by scraping the article HTML page
        because the filename suffix is not deterministic.

        Args:
            doi: Article DOI (e.g. "10.1038/s41586-023-12345-6").

        Returns:
            Absolute PDF URL, or ``None`` if not found.
        """
        client = self._require_client()
        article_id = self._article_id_from_doi(doi)
        article_url = f"https://www.nature.com/articles/{article_id}"

        try:
            await self._throttle()
            resp = await client.get(
                article_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/121.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Referer": "https://www.nature.com/",
                },
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            print(f"[warn] Could not fetch article page for doi={doi}: {e}")
            return None

        # First pass: look for MOESM PDF links near a peer-review label.
        # Nature HTML typically has: <a href="...MOESM...pdf" ...>Peer Review File</a>
        # Strategy: find all MOESM pdf hrefs, then check surrounding context.
        candidates: list[str] = []
        for pattern in self._PDF_HREF_PATTERNS:
            for m in pattern.finditer(html):
                url = m.group(1)
                # Check if surrounding 200 chars mention peer review
                context_start = max(0, m.start() - 200)
                context_end = min(len(html), m.end() + 200)
                context = html[context_start:context_end]
                if self._PDF_LABEL_PATTERN.search(context):
                    return url  # Best match: href + label
                candidates.append(url)

        # Second pass: if no label match, return first MOESM candidate.
        if candidates:
            return candidates[0]

        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def fetch_peer_review_pdf(self, doi: str) -> bytes | None:
        """Download the peer review PDF for a Nature article.

        Fetches the article HTML page to discover the actual PDF URL (since
        the URL contains a non-deterministic MOESM number), then downloads
        the PDF.

        Returns None if:
        - The article page cannot be fetched.
        - No peer review PDF link is found on the page.
        - The PDF download fails.

        Args:
            doi: Article DOI (e.g. "10.1038/s41586-023-12345-6").

        Returns:
            Raw PDF bytes, or None if the peer review PDF could not be fetched.
        """
        client = self._require_client()

        pdf_url = await self._find_peer_review_pdf_url(doi)
        if pdf_url is None:
            print(f"[warn] No peer review PDF link found for doi={doi}")
            return None

        try:
            await self._throttle()
            resp = await client.get(
                pdf_url,
                headers={
                    "Accept": "application/pdf,*/*",
                    "Referer": f"https://www.nature.com/articles/{self._article_id_from_doi(doi)}",
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/121.0.0.0 Safari/537.36"
                    ),
                },
            )
            if resp.status_code == 404:
                print(f"[warn] Peer review PDF not found at {pdf_url}")
                return None
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as e:
            print(f"[warn] Failed to download peer review PDF for doi={doi}: {e}")
            return None

    async def iter_articles(
        self,
        subjects: list[str] | None = None,
        start_date: str = "2022-01-01",
        end_date: str | None = None,
        order: str = "desc",
        max_articles: int = 10,
        dry_run: bool = False,
        journals: list[str] | None = None,
    ) -> AsyncIterator[tuple[NatureArticleMeta, bytes | None]]:
        """Yield (meta, xml_bytes) pairs for Nature articles in order.

        ``xml_bytes`` is always None because Nature peer review content is
        PDF-based, not JATS XML.  Use ``fetch_peer_review_pdf`` separately to
        retrieve peer review PDFs.

        Args:
            subjects: Optional subject area filter strings.
            start_date: Collection start date (inclusive). Default "2022-01-01".
            end_date: Collection end date (inclusive). None means unlimited.
            order: "desc" (newest first) or "asc" (oldest first).
            max_articles: Maximum number of articles to yield.
            dry_run: If True, return metadata only (xml_bytes=None in all cases).
            journals: Nature journal names to include. Defaults to all journals
                with transparent peer review.
        """
        metas = await self.list_reviewed_articles(
            subjects=subjects,
            start_date=start_date,
            end_date=end_date,
            order=order,
            max_articles=max_articles,
            journals=journals,
        )

        for meta in metas:
            # XML is never available for Nature articles; yield None unconditionally
            yield meta, None

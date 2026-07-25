"""PLOS Open Access journals article collector.

Uses the PLOS Search API (https://api.plos.org/search) + JATS XML download.
No API key required for low-volume access.
Rate limit: recommended 1 req/s.

Journals with confirmed transparent peer review embedded as sub-articles:
    - PLoS Biology        (PLoSBiology)   ← confirmed
    - PLoS Genetics       (PLoSGenetics)  ← confirmed (articles ≤ ~6 months old
                                             may not have reviews embedded yet)
    - PLoS Computational Biology (PLoSCompBiol)
    - PLoS Medicine       (PLoSMedicine)

Journals with uncertain/no peer review in XML:
    - PLoS Pathogens      (PLoSPathogens) ← empirically tested, no sub-articles
                                             found; included but filtered at
                                             download time

Note: PLoS ONE does NOT publish peer reviews and is excluded by default.

Post-download filtering: iter_articles() checks each downloaded XML for the
``<sub-article article-type="aggregated-review-documents">`` element.  Articles
without this element are skipped (yielding no output) rather than producing
empty entries in the output dataset.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

PLOS_SEARCH_API = "https://api.plos.org/search"

# Journals with transparent peer review embedded in JATS XML as sub-articles
JOURNALS_WITH_REVIEWS = {
    "PLoSBiology",
    "PLoSGenetics",
    "PLoSCompBiol",
    "PLoSPathogens",
    "PLoSMedicine",
}

# Mapping from journal key to the URL slug used in the JATS XML download URL
JOURNAL_SLUG_MAP: dict[str, str] = {
    "PLoSBiology": "plosbiology",
    "PLoSGenetics": "plosgenetics",
    "PLoSCompBiol": "ploscompbiol",
    "PLoSPathogens": "plospathogens",
    "PLoSMedicine": "plosmedicine",
}

# Default list of journals to collect (transparent peer review journals only)
DEFAULT_JOURNALS: list[str] = list(JOURNALS_WITH_REVIEWS)


class PLOSArticleMeta:
    """Article metadata extracted from the PLOS Search API response."""

    __slots__ = (
        "article_id",
        "doi",
        "title",
        "abstract",
        "journal",
        "journal_key",
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
        journal_key: str,
        subjects: list[str],
        published: str,
        has_reviews: bool,
    ) -> None:
        self.article_id = article_id
        self.doi = doi
        self.title = title
        self.abstract = abstract
        self.journal = journal
        self.journal_key = journal_key
        self.subjects = subjects
        self.published = published
        self.has_reviews = has_reviews


class PLOSCollector:
    """PLOS Open Access article metadata + JATS XML collector.

    Usage:
        async with PLOSCollector() as collector:
            async for meta, xml_bytes in collector.iter_articles(
                journals=["PLoSBiology", "PLoSGenetics"], max_articles=10
            ):
                ...
    """

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        timeout: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        self._delay = rate_limit_delay
        self._timeout = timeout
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def __aenter__(self) -> PLOSCollector:
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"Accept": "application/json"},
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
                "PLOSCollector client is not initialized. Use 'async with PLOSCollector()'."
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
            # Conservatively include items with incomplete metadata
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
    def _infer_has_reviews(journal_key: str) -> bool:
        """Return True if the journal publishes transparent peer reviews in JATS XML."""
        return journal_key in JOURNALS_WITH_REVIEWS

    @staticmethod
    def _journal_key_from_name(journal_name: str) -> str:
        """Derive a journal key from the human-readable journal name returned by the API.

        The PLOS Search API returns ``journal`` as a full name such as
        "PLoS Biology".  This helper strips spaces so it can be compared
        against the key-style strings used internally (e.g. "PLoSBiology").
        """
        return journal_name.replace(" ", "").replace(".", "")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _get(
        self,
        url: str,
        accept: str = "application/json",
        params: dict[str, object] | None = None,
    ) -> httpx.Response:
        client = self._require_client()
        await self._throttle()
        resp = await client.get(url, params=params, headers={"Accept": accept})
        resp.raise_for_status()
        return resp

    async def list_reviewed_articles(
        self,
        subjects: list[str] | None = None,
        start_date: str = "2018-01-01",
        end_date: str | None = None,
        order: str = "desc",
        page_size: int = 100,
        max_articles: int | None = None,
        journals: list[str] | None = None,
    ) -> list[PLOSArticleMeta]:
        """Retrieve article metadata from the PLOS Search API.

        Args:
            subjects: Optional list of subject area strings to filter on
                (post-hoc, case-insensitive substring match against the API
                ``subject`` field).
            start_date: Collection start date (inclusive). Default "2018-01-01".
            end_date: Collection end date (inclusive). None means unlimited.
            order: "desc" (newest first, default) or "asc" (oldest first).
            page_size: Number of results per API page (``rows`` param).
            max_articles: Stop after collecting this many articles.
            journals: List of PLOS journal keys to include.  Defaults to all
                journals with transparent peer review.
        """
        if journals is None:
            journals = DEFAULT_JOURNALS

        results: list[PLOSArticleMeta] = []
        start_date_obj = self._extract_published_date(start_date)
        end_date_obj = self._extract_published_date(end_date) if end_date else None

        # Build the journal filter query fragment
        journal_fq = " OR ".join(f"journal_key:{jk}" for jk in journals)

        # Build the date range filter
        end_date_str = f"{end_date}T23:59:59Z" if end_date else "NOW"
        date_fq = f"publication_date:[{start_date}T00:00:00Z TO {end_date_str}]"

        sort_field = "publication_date"
        sort_order = "desc" if order == "desc" else "asc"
        sort = f"{sort_field} {sort_order}"

        offset = 0

        while True:
            api_params: dict[str, object] = {
                "q": "*:*",
                "fq": [journal_fq, date_fq],
                "fl": "id,title,abstract,publication_date,journal,journal_key,subject,article_type",
                "wt": "json",
                "sort": sort,
                "start": offset,
                "rows": page_size,
            }
            if self._api_key:
                api_params["api_key"] = self._api_key

            try:
                resp = await self._get(PLOS_SEARCH_API, params=api_params)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    break
                raise

            data = resp.json()
            response_block = data.get("response", {})
            docs = response_block.get("docs", [])
            num_found = response_block.get("numFound", 0)

            if not docs:
                break

            for doc in docs:
                doi = str(doc.get("id", ""))
                # Skip sub-items: PLOS API returns DOI/title, DOI/abstract, etc.
                # Keep only root DOIs (no path suffix after the journal.xxx.XXXXXXX part)
                doi_parts = doi.split("/")
                if len(doi_parts) > 2:
                    # e.g. "10.1371/journal.pbio.3002133/abstract" has 3 parts → skip
                    continue

                # Skip non-research-article types (corrections, discussions, reviews)
                article_type = str(doc.get("article_type", "")).lower()
                if article_type and article_type not in ("research article", "research-article", ""):
                    continue

                title = doc.get("title", "")
                abstract_raw = doc.get("abstract", "")
                # The abstract field may be a list or a plain string
                if isinstance(abstract_raw, list):
                    abstract = " ".join(abstract_raw)
                else:
                    abstract = str(abstract_raw)

                journal = doc.get("journal", "")
                # Prefer the explicit journal_key field; fall back to derivation
                journal_key_raw = doc.get("journal_key", "")
                if journal_key_raw:
                    journal_key = journal_key_raw
                else:
                    journal_key = self._journal_key_from_name(journal)

                subject_raw = doc.get("subject", [])
                subj_list: list[str] = subject_raw if isinstance(subject_raw, list) else [subject_raw]

                published = doc.get("publication_date", "")

                # Date boundary checks (belt-and-suspenders; the API fq already
                # filters, but the check guards against off-by-one edge cases)
                if not self._is_on_or_after(published, start_date_obj):
                    continue
                if not self._is_on_or_before(published, end_date_obj):
                    if order == "asc":
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

                has_reviews = self._infer_has_reviews(journal_key)

                results.append(
                    PLOSArticleMeta(
                        article_id=doi,
                        doi=doi,
                        title=title,
                        abstract=abstract,
                        journal=journal,
                        journal_key=journal_key,
                        subjects=subj_list,
                        published=published,
                        has_reviews=has_reviews,
                    )
                )

                if max_articles and len(results) >= max_articles:
                    return results

            offset += len(docs)
            if offset >= num_found:
                break

        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def fetch_xml(self, doi: str, journal_key: str) -> bytes:
        """Download the JATS XML manuscript for a PLOS article.

        Args:
            doi: The article DOI (e.g. "10.1371/journal.pbio.3001234").
            journal_key: The PLOS journal key (e.g. "PLoSBiology") used to
                resolve the correct URL slug.

        Returns:
            Raw bytes of the JATS XML file.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
            KeyError: If ``journal_key`` is not present in ``JOURNAL_SLUG_MAP``.
        """
        client = self._require_client()
        await self._throttle()

        slug = JOURNAL_SLUG_MAP.get(journal_key)
        if slug is None:
            raise KeyError(
                f"Unknown journal_key '{journal_key}'. "
                f"Valid keys: {list(JOURNAL_SLUG_MAP)}"
            )

        url = (
            f"https://journals.plos.org/{slug}/article/file"
            f"?id={doi}&type=manuscript"
        )
        resp = await client.get(
            url,
            headers={"Accept": "application/xml, text/xml"},
        )
        resp.raise_for_status()
        return resp.content

    async def iter_articles(
        self,
        subjects: list[str] | None = None,
        start_date: str = "2018-01-01",
        end_date: str | None = None,
        order: str = "desc",
        max_articles: int = 10,
        dry_run: bool = False,
        journals: list[str] | None = None,
    ) -> AsyncIterator[tuple[PLOSArticleMeta, bytes | None]]:
        """Yield (meta, xml_bytes) pairs for PLOS articles in order.

        Args:
            subjects: Optional subject area filter strings.
            start_date: Collection start date (inclusive). Default "2018-01-01".
            end_date: Collection end date (inclusive). None means unlimited.
            order: "desc" (newest first, default) or "asc" (oldest first).
            max_articles: Maximum number of articles to yield.
            dry_run: If True, return metadata only without downloading XML.
            journals: PLOS journal keys to include. Defaults to all journals
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
            if dry_run:
                yield meta, None
            else:
                try:
                    xml_bytes = await self.fetch_xml(meta.doi, meta.journal_key)
                except (httpx.HTTPError, KeyError) as e:
                    print(f"[warn] Failed to fetch XML for {meta.doi}: {e}")
                    yield meta, None
                    continue

                # Only yield articles that have peer review sub-articles embedded.
                # Articles without <sub-article article-type="aggregated-review-documents">
                # will produce empty parsed.reviews and are not useful for the benchmark.
                if b"aggregated-review-documents" not in xml_bytes:
                    print(
                        f"[skip] No peer-review sub-articles in XML for {meta.doi} "
                        f"({meta.journal}); skipping."
                    )
                    continue

                yield meta, xml_bytes

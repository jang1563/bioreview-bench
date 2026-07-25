"""F1000Research / Wellcome Open Research / Gates Open Research collector.

All three platforms run on the same infrastructure and publish JATS XML
with reviewer-report + response sub-articles.  Article discovery uses
the CrossRef API (no auth required) filtered by container-title.

JATS XML endpoint:
  F1000Research:       https://f1000research.com/articles/{N}/{V}/xml
  Wellcome Open Res:   https://wellcomeopenresearch.org/articles/{N}/{V}/xml
  Gates Open Res:      https://gatesopenresearch.org/articles/{N}/{V}/xml

DOI pattern:
  10.12688/f1000research.{N}.{V}
  10.12688/wellcomeopenres.{N}.{V}
  10.12688/gatesopenres.{N}.{V}
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

# Mapping from CrossRef container-title to XML host
_JOURNAL_TO_HOST: dict[str, str] = {
    "F1000Research": "f1000research.com",
    "Wellcome Open Research": "wellcomeopenresearch.org",
    "Gates Open Research": "gatesopenresearch.org",
}

DEFAULT_JOURNALS: list[str] = list(_JOURNAL_TO_HOST)
_DEFAULT_MAILTO = "bioreview-bench@research.example.com"

# DOI sub-type prefix → host
_DOI_PREFIX_TO_HOST: dict[str, str] = {
    "f1000research": "f1000research.com",
    "wellcomeopenres": "wellcomeopenresearch.org",
    "gatesopenres": "gatesopenresearch.org",
}

# Regex to parse 10.12688/{subtype}.{article_num}.{version}
_DOI_PATTERN = re.compile(
    r"10\.12688/([^.]+)\.(\d+)\.(\d+)",
    re.IGNORECASE,
)


class F1000ArticleMeta:
    """Minimal metadata for one F1000-family article."""

    __slots__ = (
        "article_id",
        "doi",
        "title",
        "abstract",
        "journal",
        "subjects",
        "published",
        "xml_url",
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
        xml_url: str,
    ) -> None:
        self.article_id = article_id
        self.doi = doi
        self.title = title
        self.abstract = abstract
        self.journal = journal
        self.subjects = subjects
        self.published = published
        self.xml_url = xml_url


def _doi_to_xml_url(doi: str) -> str | None:
    """Convert an F1000-family DOI to its JATS XML download URL.

    Examples:
        "10.12688/f1000research.157738.2"
            → "https://f1000research.com/articles/157738/2/xml"
        "10.12688/wellcomeopenres.18000.1"
            → "https://wellcomeopenresearch.org/articles/18000/1/xml"
    """
    m = _DOI_PATTERN.search(doi.lower())
    if not m:
        return None
    subtype, article_num, version = m.group(1), m.group(2), m.group(3)
    host = _DOI_PREFIX_TO_HOST.get(subtype)
    if not host:
        return None
    return f"https://{host}/articles/{article_num}/{version}/xml"


class F1000Collector:
    """Collect JATS XML from F1000Research, Wellcome Open Research, Gates Open Research.

    Uses CrossRef for article discovery and direct JATS XML URLs for content.
    Peer review is embedded in the XML as ``reviewer-report`` and ``response``
    sub-articles (handled by JATSParser).

    Usage::

        async with F1000Collector() as collector:
            async for meta, xml_bytes in collector.iter_articles(
                journals=["F1000Research"],
                max_articles=50,
            ):
                if xml_bytes is not None:
                    # pass xml_bytes to JATSParser
                    ...
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

    async def __aenter__(self) -> F1000Collector:
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": f"bioreview-bench/1.0 (mailto:{self._mailto})",
            },
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "F1000Collector must be used as an async context manager."
            )
        return self._client

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request_time = time.monotonic()

    @staticmethod
    def _clean_abstract(text: str) -> str:
        """Strip JATS/HTML tags from CrossRef abstract."""
        return re.sub(r"<[^>]+>", "", text).strip()

    @staticmethod
    def _date_parts_to_str(date_parts: list) -> str:
        if not date_parts or not date_parts[0]:
            return ""
        parts = date_parts[0]
        if len(parts) >= 3:
            return f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
        if len(parts) == 2:
            return f"{parts[0]:04d}-{parts[1]:02d}-01"
        return f"{parts[0]:04d}-01-01"

    @staticmethod
    def _extract_date(value: str) -> date | None:
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _get_json(self, url: str, params: dict | None = None) -> dict:
        client = self._require_client()
        await self._throttle()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _get_bytes(self, url: str) -> bytes | None:
        client = self._require_client()
        await self._throttle()
        # Use text/xml Accept header; some platforms return HTML without it
        resp = await client.get(url, headers={"Accept": "text/xml, application/xml, */*"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content

    async def list_articles(
        self,
        start_date: str = "2013-01-01",
        end_date: str | None = None,
        order: str = "desc",
        page_size: int = 100,
        max_articles: int | None = None,
        journals: list[str] | None = None,
        subjects: list[str] | None = None,
    ) -> list[F1000ArticleMeta]:
        """Retrieve article metadata from CrossRef.

        Args:
            start_date: Earliest publication date (YYYY-MM-DD).
            end_date: Latest publication date, or None for no limit.
            order: ``"desc"`` (newest first) or ``"asc"``.
            page_size: Records per CrossRef page (max 1000).
            max_articles: Total cap on articles returned.
            journals: F1000-family journal names; defaults to all three.
            subjects: Optional subject keywords for post-hoc filtering.

        Returns:
            List of :class:`F1000ArticleMeta` objects.
        """
        if journals is None:
            journals = DEFAULT_JOURNALS

        results: list[F1000ArticleMeta] = []
        start_date_obj = self._extract_date(start_date)
        end_date_obj = self._extract_date(end_date) if end_date else None

        for journal_name in journals:
            offset = 0

            while True:
                filter_parts = [f"container-title:{journal_name}"]
                filter_parts.append(f"from-pub-date:{start_date}")
                if end_date:
                    filter_parts.append(f"until-pub-date:{end_date}")
                filter_parts.append("type:journal-article")

                params: dict[str, object] = {
                    "filter": ",".join(filter_parts),
                    "rows": min(page_size, 1000),
                    "offset": offset,
                    "sort": "published",
                    "order": order,
                    "mailto": self._mailto,
                    "select": "DOI,title,abstract,published,container-title,subject",
                }

                try:
                    data = await self._get_json(CROSSREF_API_BASE, params)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (400, 404):
                        break
                    raise

                message = data.get("message", {})
                items = message.get("items", [])
                total = message.get("total-results", 0)

                if not items:
                    break

                for item in items:
                    doi = item.get("DOI", "")
                    if not doi:
                        continue

                    # Only include articles with parseable F1000-format DOI
                    xml_url = _doi_to_xml_url(doi)
                    if not xml_url:
                        continue

                    title_list = item.get("title", [])
                    title = title_list[0] if title_list else ""

                    abstract = self._clean_abstract(item.get("abstract", ""))

                    ct_list = item.get("container-title", [])
                    journal = ct_list[0] if ct_list else journal_name

                    pub_dp = item.get("published", {}).get("date-parts", [])
                    published = self._date_parts_to_str(pub_dp)

                    # Date range check
                    pub_date_obj = self._extract_date(published)
                    if start_date_obj and pub_date_obj and pub_date_obj < start_date_obj:
                        continue
                    if end_date_obj and pub_date_obj and pub_date_obj > end_date_obj:
                        if order == "asc":
                            return results
                        continue

                    subj_raw = item.get("subject", [])
                    subj_list: list[str] = (
                        subj_raw if isinstance(subj_raw, list) else [subj_raw]
                    )

                    # Optional subject filtering
                    if subjects:
                        subj_lower = [s.lower() for s in subj_list]
                        if not any(
                            any(req.lower() in s for s in subj_lower)
                            for req in subjects
                        ):
                            continue

                    article_id = doi.replace("/", "_").replace(":", "_")

                    results.append(
                        F1000ArticleMeta(
                            article_id=article_id,
                            doi=doi,
                            title=title,
                            abstract=abstract,
                            journal=journal,
                            subjects=subj_list,
                            published=published,
                            xml_url=xml_url,
                        )
                    )

                    if max_articles and len(results) >= max_articles:
                        return results

                offset += len(items)
                if offset >= total:
                    break

        return results

    async def fetch_xml(self, meta: F1000ArticleMeta) -> bytes | None:
        """Download the JATS XML for the given article.

        Args:
            meta: Article metadata with pre-computed ``xml_url``.

        Returns:
            Raw XML bytes, or ``None`` if the article has no XML
            (e.g., no peer review published yet, or 404).
        """
        try:
            xml_bytes = await self._get_bytes(meta.xml_url)
        except httpx.HTTPStatusError:
            return None
        except Exception:
            return None

        if xml_bytes is None:
            return None

        # Quick sanity check: must contain at least one reviewer-report
        if b"reviewer-report" not in xml_bytes:
            return None

        return xml_bytes

    async def iter_articles(
        self,
        start_date: str = "2013-01-01",
        end_date: str | None = None,
        order: str = "desc",
        max_articles: int = 100,
        dry_run: bool = False,
        journals: list[str] | None = None,
        subjects: list[str] | None = None,
    ) -> AsyncIterator[tuple[F1000ArticleMeta, bytes | None]]:
        """Yield ``(meta, xml_bytes)`` pairs for F1000-family articles.

        ``xml_bytes`` is ``None`` when:
        - ``dry_run=True``
        - The article has no published peer review (XML missing or no
          reviewer-report sub-articles).

        Args:
            start_date: Earliest publication date.
            end_date: Latest publication date.
            order: ``"desc"`` or ``"asc"``.
            max_articles: Maximum number of articles to yield.
            dry_run: Return metadata only without downloading XMLs.
            journals: F1000-family journal names.
            subjects: Optional subject keywords for post-hoc filtering.
        """
        metas = await self.list_articles(
            start_date=start_date,
            end_date=end_date,
            order=order,
            max_articles=max_articles,
            journals=journals,
            subjects=subjects,
        )

        for meta in metas:
            if dry_run:
                yield meta, None
            else:
                xml_bytes = await self.fetch_xml(meta)
                yield meta, xml_bytes

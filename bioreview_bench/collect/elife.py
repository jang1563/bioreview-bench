"""eLife article collector.

Uses eLife API (https://api.elifesciences.org) + JATS XML download.
Rate limit: recommended 1 req/s.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

ELIFE_API_BASE = "https://api.elifesciences.org"
ELIFE_XML_BASE = "https://elifesciences.org/articles"
ELIFE_GITHUB_XML_BASE = (
    "https://raw.githubusercontent.com/elifesciences/elife-article-xml/master/articles"
)

# eLife subject area → API parameter mapping
SUBJECT_MAP = {
    "genetics-genomics": "genetics-and-genomics",
    "cell-biology": "cell-biology",
    "neuroscience": "neuroscience",
    "biochemistry": "biochemistry-and-chemical-biology",
    "biochemistry-chemical-biology": "biochemistry-and-chemical-biology",
    "microbiology": "microbiology-and-infectious-disease",
    "microbiology-infectious-disease": "microbiology-and-infectious-disease",
    "immunology": "immunology-and-inflammation",
    "immunology-inflammation": "immunology-and-inflammation",
    "computational": "computational-and-systems-biology",
    "computational-systems-biology": "computational-and-systems-biology",
}


class ELifeArticleMeta:
    """Article metadata extracted from the eLife API response."""

    __slots__ = ("article_id", "doi", "title", "subjects", "published", "has_reviews")

    def __init__(
        self,
        article_id: str,
        doi: str,
        title: str,
        subjects: list[str],
        published: str,
        has_reviews: bool,
    ) -> None:
        self.article_id = article_id
        self.doi = doi
        self.title = title
        self.subjects = subjects
        self.published = published
        self.has_reviews = has_reviews


class ELifeCollector:
    """eLife article metadata + XML collector.

    Usage:
        async with ELifeCollector() as collector:
            async for article_id, xml_bytes in collector.iter_articles(
                subjects=["genetics-genomics"], max_articles=10
            ):
                ...
    """

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        self._delay = rate_limit_delay
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def __aenter__(self) -> ELifeCollector:
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
            raise RuntimeError("ELifeCollector client is not initialized. Use 'async with ELifeCollector()'.")
        return self._client

    @staticmethod
    def _normalize_subjects(subjects: list[str] | None) -> list[str]:
        if not subjects:
            return []
        mapped = [SUBJECT_MAP.get(s, s) for s in subjects]
        # Deduplicate while preserving input order
        return list(dict.fromkeys(mapped))

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
    def _infer_has_reviews(item: dict) -> bool:
        for key in (
            "has_reviews",
            "reviews_available",
            "has_reviewer_reports",
            "open_peer_review",
        ):
            if key in item:
                return bool(item[key])
        if "decision_letter" in item or "review" in item:
            return True
        status = str(item.get("status", "")).lower()
        return "reviewed" in status

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
    ) -> list[ELifeArticleMeta]:
        """Retrieve article list from the reviewed-preprints or articles endpoint.

        Args:
            start_date: Collection start date (inclusive). Default "2018-01-01".
            end_date: Collection end date (inclusive). None means unlimited.
            order: "desc" (newest first, default) or "asc" (oldest first).
                   Use "asc" for efficient collection of old-format articles.
        """
        results: list[ELifeArticleMeta] = []
        page = 1
        mapped_subjects = self._normalize_subjects(subjects)
        start_date_obj = self._extract_published_date(start_date)
        end_date_obj = self._extract_published_date(end_date) if end_date else None

        # eLife API: /articles?type=research-article
        # NOTE: eLife API does NOT support server-side subject filtering.
        # Subject filtering is applied post-hoc on the returned metadata.
        while True:
            api_params: dict[str, object] = {
                "page": page,
                "per-page": page_size,
                "type": "research-article",
                "order": order,
            }

            try:
                resp = await self._get(
                    f"{ELIFE_API_BASE}/articles",
                    accept="application/vnd.elife.article-list+json;version=1",
                    params=api_params,
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    break
                raise

            data = resp.json()
            items = data.get("items", [])
            if not items:
                break

            all_out_of_range = True
            for item in items:
                article_id = str(item.get("id", ""))
                doi = item.get("doi", f"10.7554/eLife.{article_id}")
                title = item.get("title", "")
                subj_list = [
                    s.get("id", "") for s in item.get("subjects", [])
                ]
                published = item.get("published", "")

                if not self._is_on_or_after(published, start_date_obj):
                    # asc order: haven't reached start_date yet — continue
                    continue
                if not self._is_on_or_before(published, end_date_obj):
                    # Beyond end_date
                    if order == "asc":
                        # asc: exceeded end_date — no more articles needed
                        return results
                    continue

                all_out_of_range = False

                # Post-hoc subject filtering (when mapped_subjects is specified)
                if mapped_subjects and not any(s in mapped_subjects for s in subj_list):
                    continue

                has_reviews = self._infer_has_reviews(item)

                results.append(ELifeArticleMeta(
                    article_id=article_id,
                    doi=doi,
                    title=title,
                    subjects=subj_list,
                    published=published,
                    has_reviews=has_reviews,
                ))

                if max_articles and len(results) >= max_articles:
                    return results

            # desc order + end_date: stop if entire page is before end_date
            if order == "desc" and end_date_obj and all_out_of_range:
                break

            total = data.get("total", 0)
            if page * page_size >= total:
                break
            page += 1

        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def fetch_xml(self, article_id: str) -> bytes | None:
        """Download JATS XML. eLife XML is publicly accessible via direct URL.

        Falls back to GitHub (elifesciences/elife-article-xml) when the primary
        URL returns HTML (old articles pre-2016 redirect to the article page).
        Returns None only when neither source has the XML.
        """
        client = self._require_client()
        await self._throttle()

        url = f"{ELIFE_XML_BASE}/{article_id}.xml"
        resp = await client.get(
            url,
            headers={"Accept": "application/xml, text/xml"},
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type:
            return resp.content

        # Primary URL returned HTML — try GitHub repo as fallback.
        # Old articles have IDs like "8505" → pad to 5 digits for filename.
        try:
            padded = f"{int(article_id):05d}"
        except ValueError:
            return None

        for version in ("v2", "v1", "v3", "v4"):
            gh_url = f"{ELIFE_GITHUB_XML_BASE}/elife-{padded}-{version}.xml"
            gh_resp = await client.get(gh_url)
            if gh_resp.status_code == 200:
                ct = gh_resp.headers.get("content-type", "")
                if "html" not in ct:
                    return gh_resp.content

        return None

    async def iter_articles(
        self,
        subjects: list[str] | None = None,
        start_date: str = "2018-01-01",
        end_date: str | None = None,
        order: str = "desc",
        max_articles: int = 10,
        dry_run: bool = False,
    ) -> AsyncIterator[tuple[ELifeArticleMeta, bytes | None]]:
        """Yield (meta, xml_bytes) pairs in order.

        dry_run=True: return metadata only without downloading XML.
        """
        metas = await self.list_reviewed_articles(
            subjects=subjects,
            start_date=start_date,
            end_date=end_date,
            order=order,
            max_articles=max_articles,
        )

        for meta in metas:
            if dry_run:
                yield meta, None
            else:
                try:
                    xml_bytes = await self.fetch_xml(meta.article_id)
                    yield meta, xml_bytes
                except httpx.HTTPError as e:
                    print(f"[warn] Failed to fetch XML for {meta.article_id}: {e}")
                    yield meta, None

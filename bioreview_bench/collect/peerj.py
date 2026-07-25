"""PeerJ open peer review collector.

PeerJ publishes peer reviews at https://peerj.com/articles/{id}/reviews/
There is NO JATS XML sub-article structure for peer reviews — reviews only
exist as HTML on the /reviews/ page.

Discovery:
  CrossRef API filtered by DOI prefix 10.7717 and container-title "PeerJ"

Review extraction:
  HTML parsing of /reviews/ page:
  - Find version-0-1 section (initial submission round)
  - Extract reviewer report bodies (itemprop="reviewBody")
  - Exclude editor decision blocks (class "publication-decision")

Author response:
  Rebuttal letters are downloadable Office documents (/articles/{id}v0.2/rebuttal).
  We fetch and parse the initial rebuttal so concern stance labels are based on
  real author text rather than an empty placeholder.
"""

from __future__ import annotations

import asyncio
import io
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from typing import AsyncIterator

import httpx
from lxml import etree, html
from tenacity import retry, stop_after_attempt, wait_exponential

CROSSREF_API_BASE = "https://api.crossref.org/works"
PEERJ_BASE = "https://peerj.com"

# Only biology/medicine PeerJ, not computer science
_TARGET_JOURNALS = ("PeerJ",)

_DEFAULT_MAILTO = "bioreview-bench@research.example.com"

# DOI pattern: 10.7717/peerj.{N}
_DOI_PATTERN = re.compile(r"10\.7717/peerj\.(\d+)$", re.IGNORECASE)


class PeerJArticleMeta:
    __slots__ = ("article_id", "doi", "title", "abstract", "subjects", "published")

    def __init__(
        self,
        article_id: str,
        doi: str,
        title: str,
        abstract: str,
        subjects: list[str],
        published: str,
    ) -> None:
        self.article_id = article_id
        self.doi = doi
        self.title = title
        self.abstract = abstract
        self.subjects = subjects
        self.published = published


@dataclass(slots=True)
class PeerJReviewData:
    review_texts: list[str]
    editorial_decision: str
    author_response_raw: str
    paper_text_sections: dict[str, str]

    @property
    def has_author_response(self) -> bool:
        return bool(self.author_response_raw.strip())


class _TextExtractor(HTMLParser):
    """Strip HTML tags, decode entities, return plain text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple]) -> None:
        attr_dict = dict(attrs)
        # Skip script/style content
        if tag in ("script", "style"):
            self._skip = True
            self._skip_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self._skip:
            if tag in ("script", "style"):
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._parts)).strip()


def _strip_html(html_fragment: str) -> str:
    parser = _TextExtractor()
    parser.feed(html_fragment)
    return parser.get_text()


def _normalize_section_key(title: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return key or "section"


def _extract_docx_text(docx_bytes: bytes) -> str:
    """Extract plain text paragraphs from a DOCX response letter."""
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
            document_xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile):
        return ""

    root = etree.fromstring(document_xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []

    for para in root.xpath(".//w:p", namespaces=ns):
        text = "".join(para.xpath(".//w:t/text()", namespaces=ns))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def _extract_sections_from_article_html(article_html: str) -> dict[str, str]:
    """Extract abstract and top-level manuscript sections from a PeerJ article page."""
    try:
        document = html.fromstring(article_html)
    except etree.ParserError:
        return {}

    article_nodes = document.xpath("//article")
    if not article_nodes:
        return {}
    article_node = article_nodes[0]

    sections: dict[str, str] = {}

    abstract_nodes = article_node.xpath(".//div[contains(@class, 'abstract')]")
    if abstract_nodes:
        abstract_text = re.sub(r"\s+", " ", " ".join(abstract_nodes[0].xpath(".//text()"))).strip()
        if abstract_text:
            sections["abstract"] = abstract_text

    top_level_sections = article_node.xpath(
        ".//section[contains(concat(' ', normalize-space(@class), ' '), ' sec ')]"
        "[not(ancestor::section[contains(concat(' ', normalize-space(@class), ' '), ' sec ')])]"
    )
    for sec in top_level_sections:
        heading = sec.xpath("./h2[contains(@class, 'heading')][1]/text()")
        title = re.sub(r"\s+", " ", heading[0]).strip() if heading else "Section"
        body_text = " ".join(
            text.strip()
            for text in sec.xpath(".//p//text() | .//li//text() | .//td//text() | .//th//text()")
            if text and text.strip()
        )
        body_text = re.sub(r"\s+", " ", body_text).strip()
        if not body_text:
            continue

        key = _normalize_section_key(title)
        if key in sections:
            sections[key] += "\n\n" + body_text
        else:
            sections[key] = body_text

    if not sections:
        paragraphs = article_node.xpath(".//p//text()")
        body_text = re.sub(r"\s+", " ", " ".join(t for t in paragraphs if t and t.strip())).strip()
        if body_text:
            sections["body"] = body_text

    return sections


def _extract_reviews_from_html(html: str) -> tuple[list[str], str, str | None]:
    """Parse PeerJ /reviews/ HTML.

    Returns:
        review_texts: list of reviewer report texts (one per reviewer, first round)
        editorial_decision: accept / major_revision / minor_revision / reject / unknown
        rebuttal_path: initial-round rebuttal link if present
    """
    # ── 1. Find version-0-1 section (initial round) ──────────────────────────
    # Split at version boundaries and take the first submission block
    # Strategy: find everything between id="version-0-1" and the next version id
    v01_match = re.search(r'id="version-0-1"', html)
    if not v01_match:
        return [], "unknown", None

    v01_start = v01_match.start()
    # PeerJ renders versions newest-first, so any next version-0-N section
    # after v01_start would be older (not applicable for v01 = original).
    # Search for next version boundary only after v01_start.
    next_ver_match = re.search(r'id="version-0-\d+"', html[v01_start + 1:])
    v01_block = html[v01_start: v01_start + 1 + next_ver_match.start() if next_ver_match else len(html)]

    # ── 2. Editorial decision from version-0-1 ───────────────────────────────
    dec_match = re.search(
        r'article-recommendation-([\w-]+)"', v01_block
    )
    if dec_match:
        raw_dec = dec_match.group(1).lower()
        decision_map = {
            "accept": "accept",
            "minor": "minor_revision",
            "major": "major_revision",
            "reject": "reject",
        }
        editorial_decision = next(
            (v for k, v in decision_map.items() if raw_dec.startswith(k)),
            "unknown",
        )
    else:
        editorial_decision = "unknown"

    # ── 3. Reviewer report bodies (exclude editor decision blocks) ────────────
    # Editor decision divs have both "publication-review well" AND "publication-decision"
    # Reviewer divs have "publication-review well" WITHOUT "publication-decision"
    # and have id="version-0-1-review-N"
    reviewer_block_pattern = re.compile(
        r'<div[^>]+class="publication-review well"[^>]+id="version-0-1-review-\d+"[^>]*>(.*?)'
        r'(?=<div[^>]+class="publication-review|$)',
        re.DOTALL,
    )
    review_texts: list[str] = []
    for m in reviewer_block_pattern.finditer(v01_block):
        block = m.group(1)
        # Extract reviewBody content
        body_match = re.search(
            r'itemprop="reviewBody">(.*?)(?:</div>\s*</div>|$)', block, re.DOTALL
        )
        if body_match:
            text = _strip_html(body_match.group(1))
            if text and len(text) >= 20:
                review_texts.append(text)

    # ── 4. Author response: keep the version-0-1 rebuttal link if present ─────
    rebuttal_match = re.search(r'href="([^"]+/articles/\S+/rebuttal|/articles/\S+/rebuttal)"', v01_block)
    rebuttal_path = rebuttal_match.group(1) if rebuttal_match else None

    return review_texts, editorial_decision, rebuttal_path


class PeerJCollector:
    """Async collector for PeerJ open peer reviews via HTML scraping."""

    @staticmethod
    def _clean_abstract(text: str) -> str:
        """Strip JATS/HTML tags from CrossRef abstract."""
        return re.sub(r"<[^>]+>", "", text).strip()

    def __init__(
        self,
        mailto: str = _DEFAULT_MAILTO,
        request_delay: float = 1.0,
    ) -> None:
        self._mailto = mailto
        self._delay = request_delay
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def __aenter__(self) -> "PeerJCollector":
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": f"bioreview-bench/1.0 (mailto:{self._mailto})",
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=30.0,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request_time = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _get(
        self, url: str, *, params: dict | None = None, headers: dict | None = None,
    ) -> httpx.Response:
        assert self._client
        await self._throttle()
        return await self._client.get(url, params=params, headers=headers)

    async def list_articles(
        self,
        start_date: str = "2013-01-01",
        end_date: str | None = None,
        max_articles: int = 500,
    ) -> list[PeerJArticleMeta]:
        """Discover PeerJ articles via CrossRef API."""
        assert self._client
        articles: list[PeerJArticleMeta] = []
        cursor = "*"
        rows = 200

        filters = [
            "prefix:10.7717",
            "type:journal-article",
            f"from-pub-date:{start_date}",
        ]
        if end_date:
            filters.append(f"until-pub-date:{end_date}")

        while len(articles) < max_articles:
            params = {
                "filter": ",".join(filters),
                "rows": rows,
                "cursor": cursor,
                "select": "DOI,title,abstract,subject,published,container-title",
                "mailto": self._mailto,
            }
            resp = await self._get(
                CROSSREF_API_BASE,
                params=params,
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                break

            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                break

            for item in items:
                doi = item.get("DOI", "")
                container = item.get("container-title", [""])[0]

                # Only "PeerJ" (not PeerJ Computer Science etc.)
                if container not in _TARGET_JOURNALS:
                    continue

                # Only match standard peerj DOIs (10.7717/peerj.N)
                m = _DOI_PATTERN.match(doi)
                if not m:
                    continue

                article_id = m.group(1)
                title = " ".join(item.get("title", [""])) or ""
                abstract = self._clean_abstract(item.get("abstract", "") or "")
                subjects = item.get("subject", []) or []
                pub = item.get("published", {})
                dp = pub.get("date-parts", [[2020]])[0]
                if len(dp) >= 3:
                    published = f"{dp[0]:04d}-{dp[1]:02d}-{dp[2]:02d}"
                elif len(dp) == 2:
                    published = f"{dp[0]:04d}-{dp[1]:02d}-01"
                else:
                    published = f"{dp[0]:04d}-01-01"

                articles.append(
                    PeerJArticleMeta(
                        article_id=article_id,
                        doi=doi,
                        title=title,
                        abstract=abstract,
                        subjects=subjects,
                        published=published,
                    )
                )

                if len(articles) >= max_articles:
                    break

            next_cursor = data.get("message", {}).get("next-cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        return articles

    async def fetch_article_sections(self, article_id: str) -> dict[str, str]:
        """Fetch the article HTML page and extract manuscript sections."""
        url = f"{PEERJ_BASE}/articles/{article_id}/"
        try:
            resp = await self._get(url)
        except Exception as e:
            print(f"[warn] PeerJ article fetch failed for article {article_id}: {e}")
            return {}

        if resp.status_code != 200:
            return {}
        return _extract_sections_from_article_html(resp.text)

    async def fetch_author_response(self, rebuttal_path: str | None) -> str:
        """Fetch and parse the initial author rebuttal letter."""
        if not rebuttal_path:
            return ""

        url = rebuttal_path if rebuttal_path.startswith("http") else f"{PEERJ_BASE}{rebuttal_path}"
        try:
            resp = await self._get(url)
        except Exception as e:
            print(f"[warn] PeerJ rebuttal fetch failed for {url}: {e}")
            return ""

        if resp.status_code != 200:
            return ""

        disposition = resp.headers.get("content-disposition", "").lower()
        content_type = resp.headers.get("content-type", "").lower()
        if "docx" in disposition or resp.content.startswith(b"PK\x03\x04"):
            return _extract_docx_text(resp.content)
        if content_type.startswith("text/"):
            return _strip_html(resp.text)
        return ""

    async def fetch_reviews(
        self, article_id: str, dry_run: bool = False
    ) -> PeerJReviewData | None:
        """Fetch review text, manuscript sections, and the initial author response."""
        if dry_run:
            return None

        url = f"{PEERJ_BASE}/articles/{article_id}/reviews/"
        try:
            resp = await self._get(url)
        except Exception as e:
            print(f"[warn] PeerJ fetch failed for article {article_id}: {e}")
            return None

        if resp.status_code != 200:
            return None

        review_texts, decision, rebuttal_path = _extract_reviews_from_html(resp.text)

        if not review_texts:
            return None

        paper_text_sections, author_response_raw = await asyncio.gather(
            self.fetch_article_sections(article_id),
            self.fetch_author_response(rebuttal_path),
        )

        return PeerJReviewData(
            review_texts=review_texts,
            editorial_decision=decision,
            author_response_raw=author_response_raw,
            paper_text_sections=paper_text_sections,
        )

    async def iter_articles(
        self,
        start_date: str = "2013-01-01",
        end_date: str | None = None,
        max_articles: int = 500,
        dry_run: bool = False,
    ) -> AsyncIterator[tuple[PeerJArticleMeta, PeerJReviewData | None]]:
        """Yield (meta, review_data) for each PeerJ article with open reviews.

        review_data contains review text, manuscript sections, and author response,
        or None for dry-run / no-reviews articles.
        """
        metas = await self.list_articles(
            start_date=start_date,
            end_date=end_date,
            max_articles=max_articles,
        )

        for meta in metas:
            if dry_run:
                yield meta, None
                continue

            review_data = await self.fetch_reviews(meta.article_id)
            yield meta, review_data

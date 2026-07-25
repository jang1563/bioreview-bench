from __future__ import annotations

import io
import zipfile

from bioreview_bench.collect.peerj import (
    _extract_docx_text,
    _extract_reviews_from_html,
    _extract_sections_from_article_html,
)


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    body = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("[Content_Types].xml", "")
    return buf.getvalue()


def test_extract_reviews_from_html_returns_initial_rebuttal_link() -> None:
    html = """
    <div class="publication-review-version" id="version-0-2">
      <div class="publication-review-files">
        <a href="/articles/123v0.2/rebuttal" class="btn">Download author's response letter</a>
      </div>
    </div>
    <div class="publication-review-version" id="version-0-1">
      <div class="publication-review well publication-decision article-recommendation-major" id="version-0-1-decision">
        <div itemprop="reviewBody">Decision text</div>
      </div>
      <div class="publication-review well" id="version-0-1-review-1">
        <div itemprop="reviewBody">The control experiment is missing and should be added.</div>
      </div>
      <div class="publication-review-files">
        <a href="/articles/123v0.1/rebuttal" class="btn">Download author's response letter</a>
      </div>
    </div>
    """

    review_texts, decision, rebuttal_path = _extract_reviews_from_html(html)

    assert review_texts == ["The control experiment is missing and should be added."]
    assert decision == "major_revision"
    assert rebuttal_path == "/articles/123v0.1/rebuttal"


def test_extract_sections_from_article_html_reads_abstract_and_sections() -> None:
    html = """
    <article>
      <div class="abstract" itemprop="description">
        <p>This is the abstract.</p>
      </div>
      <section class="sec" id="intro">
        <h2 class="heading">Introduction</h2>
        <p>The <i>introduction</i> text.</p>
      </section>
      <section class="sec" id="results">
        <h2 class="heading">Results</h2>
        <p>Result paragraph one.</p>
        <p>Result paragraph two.</p>
      </section>
    </article>
    """

    sections = _extract_sections_from_article_html(html)

    assert sections["abstract"] == "This is the abstract."
    assert sections["introduction"] == "The introduction text."
    assert sections["results"] == "Result paragraph one. Result paragraph two."


def test_extract_docx_text_reads_paragraphs() -> None:
    docx_bytes = _make_docx_bytes(
        ["Response to reviewer 1", "We added the missing control experiment."]
    )

    text = _extract_docx_text(docx_bytes)

    assert "Response to reviewer 1" in text
    assert "We added the missing control experiment." in text

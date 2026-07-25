from __future__ import annotations

from lxml import etree

from bioreview_bench.parse.jats import JATSParser


def test_extract_body_text_preserves_inline_tag_text() -> None:
    parser = JATSParser()
    node = etree.fromstring(
        """
        <sec>
          <p>The <italic>very</italic> important result.</p>
        </sec>
        """
    )

    text = parser._extract_body_text(node)

    assert text == "The very important result."

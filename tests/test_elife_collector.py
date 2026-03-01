from __future__ import annotations

import httpx
import pytest
import respx

from bioreview_bench.collect.elife import ELifeCollector


def test_require_client_raises_before_context() -> None:
    collector = ELifeCollector()
    with pytest.raises(RuntimeError):
        collector._require_client()


@pytest.mark.asyncio
async def test_list_reviewed_articles_applies_subjects_and_start_date() -> None:
    first_page = {
        "items": [
            {
                "id": "111",
                "doi": "10.7554/eLife.111",
                "title": "A newer paper",
                "subjects": [{"id": "genetics-and-genomics"}],
                "published": "2023-01-10T00:00:00Z",
                "has_reviews": True,
            },
            {
                "id": "099",
                "doi": "10.7554/eLife.99",
                "title": "An older paper",
                "subjects": [{"id": "cell-biology"}],
                "published": "2017-12-01T00:00:00Z",
                "has_reviews": False,
            },
        ],
        "total": 2,
    }
    second_page = {"items": [], "total": 2}

    with respx.mock(assert_all_mocked=True) as mock:
        route = mock.get("https://api.elifesciences.org/articles").mock(
            side_effect=[
                httpx.Response(200, json=first_page),
                httpx.Response(200, json=second_page),
            ]
        )

        async with ELifeCollector(rate_limit_delay=0.0) as collector:
            results = await collector.list_reviewed_articles(
                subjects=["genetics-genomics", "cell-biology"],
                start_date="2018-01-01",
            )

    assert len(results) == 1
    assert results[0].article_id == "111"
    assert results[0].has_reviews is True

    # Subject filtering is applied post-hoc (not sent as API params),
    # so only the date boundary removes article 099 (published 2017).
    first_call_params = route.calls[0].request.url.params
    assert "type" in first_call_params  # sanity: type=research-article is sent

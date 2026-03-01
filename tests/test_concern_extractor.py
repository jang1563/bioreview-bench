from __future__ import annotations

from bioreview_bench.parse.concern_extractor import ConcernExtractor
from bioreview_bench.parse.jats import ParsedReview


def test_coerce_evidence_of_change_strict_bool() -> None:
    assert ConcernExtractor._coerce_evidence_of_change("true") is True
    assert ConcernExtractor._coerce_evidence_of_change("false") is False
    assert ConcernExtractor._coerce_evidence_of_change("unknown") is None
    assert ConcernExtractor._coerce_evidence_of_change(1) is True
    assert ConcernExtractor._coerce_evidence_of_change(0) is False


def test_article_token_from_doi_elife_revision_suffix() -> None:
    token = ConcernExtractor._article_token_from_doi("10.7554/eLife.84798.3")
    assert token == "84798"


def test_process_review_builds_stable_concern_id(monkeypatch) -> None:
    extractor = ConcernExtractor(manifest_id="em-test")

    monkeypatch.setattr(
        extractor,
        "_extract_concerns_from_review",
        lambda _review_text: [
            {
                "text": "The control condition is missing for experiment 2.",
                "category": "design_flaw",
                "severity": "major",
            }
        ],
    )
    monkeypatch.setattr(
        extractor,
        "_classify_resolutions",
        lambda _concerns, _response, reviewer_num=1: [
            {"author_stance": "conceded", "evidence_of_change": "false"}
        ],
    )

    review = ParsedReview(
        reviewer_num=2,
        review_text="Dummy review",
        author_response_text="We agree but cannot add experiments in this revision.",
    )
    concerns = extractor.process_review(
        review,
        article_doi="10.7554/eLife.84798.3",
        article_source="elife",
    )

    assert len(concerns) == 1
    assert concerns[0].concern_id == "elife:84798:R2C1"
    assert concerns[0].evidence_of_change is False

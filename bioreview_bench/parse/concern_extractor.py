"""LLM-based concern extractor.

Two-stage silver label extraction: author_stance + evidence_of_change.
Reproducibility guaranteed via ExtractionManifest and trace_id.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.concern import AuthorStance, ConcernCategory, Resolution, ReviewerConcern
from ..parse.jats import ParsedReview

# ── Prompts ───────────────────────────────────────────────────────────────────

CONCERN_EXTRACTION_SYSTEM = """\
You are a biomedical peer review analyst. Extract specific scientific concerns from reviewer reports.

CATEGORY DEFINITIONS:
- design_flaw: Fundamental experimental design problems that threaten conclusion validity (missing controls, uncontrolled confounders, inappropriate comparisons)
- statistical_methodology: Statistical analysis errors (multiple comparison issues, wrong test, missing effect sizes, p-hacking)
- missing_experiment: Key experiments absent that are needed to support the main claim
- figure_issue: ONLY concerns that REQUIRE directly viewing the figure to assess (image quality, band manipulation suspicion, labeling errors). Do NOT use for concerns about figure legends or interpretations that can be assessed from text.
- prior_art_novelty: Missing citations, overstated novelty, inadequate comparison to prior work
- writing_clarity: Unclear descriptions, undefined terms, missing methods details (reproducibility level)
- reagent_method_specificity: Missing reagent catalog numbers, software versions, analysis parameters
- interpretation: Over-interpretation of data, correlation/causation confusion, unsupported generalizations
- other: Legitimate scientific concern not fitting above categories (use sparingly, <10%)

EXTRACTION RULES:
1. Extract only specific, actionable concerns — not general praise or vague criticism
2. Each concern must relate to a specific aspect of the paper
3. Separate compound concerns into individual items when clearly distinct
4. Do NOT extract: reviewer self-citations, purely stylistic preferences, typos without scientific impact
5. figure_issue: ONLY if the concern literally cannot be assessed without viewing the figure
6. If the report contains ONLY a paper summary, praise, or general enthusiasm with NO specific weaknesses or recommendations, return an empty array: []
7. Do NOT invent concerns about what the reviewer failed to mention — extract only what is explicitly raised

OUTPUT FORMAT: Return a JSON array only, no other text:
[{"text": "...", "category": "...", "severity": "major|minor|optional"}]
"""

RESOLUTION_SYSTEM = """\
You are analyzing an author response document to classify how authors addressed specific peer review concerns.

IMPORTANT CONTEXT:
- The author response may begin with a preamble ("The following is the authors' response...") followed by a paper summary — this preamble is NOT a response to any concern.
- The actual responses to specific concerns follow the preamble, often numbered or labeled by reviewer.
- If the response document does not specifically address a concern, classify it as "no_response".

STANCE DEFINITIONS:
- conceded: Author explicitly agrees with the concern AND describes specific changes made to the manuscript
- rebutted: Author explicitly disagrees and provides clear scientific justification why the concern is unwarranted
- partial: Author partially agrees, or agrees but makes no concrete changes, or only addresses part of the concern
- unclear: Response is vague, tangential, or ambiguous about this specific concern
- no_response: This specific concern is not addressed anywhere in the response

EVIDENCE_OF_CHANGE:
- true: Author explicitly describes revisions made (added figure, revised text, new experiment, new analysis)
- false: Author acknowledges the concern but explicitly states no changes were made
- null: Cannot determine from the response text

CONFIDENCE (0.0–1.0):
- 0.9–1.0: Response explicitly and unambiguously addresses this concern
- 0.7–0.9: Response likely addresses this concern but with some ambiguity
- 0.5–0.7: Response may be tangentially related but doesn't clearly address this concern
- 0.3–0.5: Response doesn't clearly address this concern; stance inferred
- 0.1–0.3: No relevant response found; defaulting to no_response

OUTPUT FORMAT: JSON array matching input concerns order:
[{"author_stance": "...", "evidence_of_change": true|false|null, "confidence": 0.0}]
"""


def _prompt_hash(system: str, user_template: str) -> str:
    combined = system + "\n---\n" + user_template
    return "sha256:" + hashlib.sha256(combined.encode()).hexdigest()[:16]


CONCERN_PROMPT_HASH = _prompt_hash(CONCERN_EXTRACTION_SYSTEM, "{review_text}")
RESOLUTION_PROMPT_HASH = _prompt_hash(RESOLUTION_SYSTEM, "{concerns_json}\n{response_text}")


class ConcernExtractor:
    """Reviewer concern extraction + resolution classification.

    Args:
        model: Anthropic model ID
        manifest_id: ExtractionManifest ID (reproducibility tracking)
        max_tokens: Maximum output tokens for the LLM
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        manifest_id: str = "em-v1.0",
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._manifest_id = manifest_id
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, system: str, user: str) -> str:
        """Call the LLM with temperature=0 for deterministic output."""
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text  # type: ignore[index]

    @staticmethod
    def _parse_json(text: str) -> list[dict]:
        """Parse JSON from LLM output, handling markdown fences and trailing text."""
        # 1. Try to extract first JSON block inside a markdown fence
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                result = json.loads(candidate)
                return result if isinstance(result, list) else []
            except json.JSONDecodeError:
                pass

        # 2. Extract JSON array directly from full text (first [ ... ] match)
        bracket_match = re.search(r"(\[.*?\])", text, flags=re.DOTALL)
        if bracket_match:
            try:
                result = json.loads(bracket_match.group(1))
                return result if isinstance(result, list) else []
            except json.JSONDecodeError:
                pass

        # 3. Parse from first [ to last ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                result = json.loads(text[start : end + 1])
                return result if isinstance(result, list) else []
            except json.JSONDecodeError:
                pass

        return []

    def _extract_concerns_from_review(
        self,
        review_text: str,
    ) -> list[dict[str, Any]]:
        """Single review text → list of raw concern dicts."""
        user = f"Reviewer report:\n\n{review_text[:6000]}"  # token limit
        raw = self._call_llm(CONCERN_EXTRACTION_SYSTEM, user)
        parsed = self._parse_json(raw)

        # Validate categories
        valid_cats = {c.value for c in ConcernCategory}
        result = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            cat = item.get("category", "other")
            if cat not in valid_cats:
                cat = "other"
            sev = item.get("severity", "minor")
            if sev not in ("major", "minor", "optional"):
                sev = "minor"
            text = str(item.get("text", "")).strip()
            if len(text) >= 10:
                result.append({"text": text, "category": cat, "severity": sev})
        return result

    def _classify_resolutions(
        self,
        concerns: list[dict[str, Any]],
        author_response: str,
        reviewer_num: int = 1,
    ) -> list[dict[str, Any]]:
        """Classify resolutions for concerns + author response (with confidence)."""
        if not author_response.strip() or not concerns:
            return [
                {"author_stance": "no_response", "evidence_of_change": None, "confidence": 0.1}
                for _ in concerns
            ]

        concerns_json = json.dumps(
            [{"text": c["text"]} for c in concerns], ensure_ascii=False
        )
        user = (
            f"These concerns are from Reviewer {reviewer_num}.\n\n"
            f"Concerns to classify:\n{concerns_json}\n\n"
            f"Author response document:\n\n{author_response[:8000]}"
        )
        raw = self._call_llm(RESOLUTION_SYSTEM, user)
        parsed = self._parse_json(raw)

        # Align count with input concerns (LLM may omit some)
        results = []
        valid_stances = {s.value for s in AuthorStance}
        for i, concern in enumerate(concerns):
            if i < len(parsed) and isinstance(parsed[i], dict):
                stance = parsed[i].get("author_stance", "unclear")
                if stance not in valid_stances:
                    stance = "unclear"
                eoc = self._coerce_evidence_of_change(
                    parsed[i].get("evidence_of_change", None)
                )
                raw_conf = parsed[i].get("confidence", None)
                confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.6
                confidence = max(0.0, min(1.0, confidence))  # clamp
            else:
                stance = "unclear"
                eoc = None
                confidence = 0.3
            results.append({"author_stance": stance, "evidence_of_change": eoc, "confidence": confidence})
        return results

    @staticmethod
    def _coerce_evidence_of_change(value: Any) -> bool | None:
        """Normalize LLM output value to strict bool or None."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
            return None
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
            return None
        return None

    @staticmethod
    def _article_token_from_doi(article_doi: str) -> str:
        """Generate a stable token from a DOI for use in concern_id."""
        doi = article_doi.strip()
        if not doi:
            return "unknown"

        # eLife DOI: 10.7554/eLife.84798(.N) -> 84798
        elife_match = re.search(r"elife\.(\d+)", doi, flags=re.IGNORECASE)
        if elife_match:
            return elife_match.group(1)

        tail = doi.split("/", 1)[-1] if "/" in doi else doi
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-_.")
        return safe[:64] if safe else "unknown"

    @staticmethod
    def _heuristic_fix_stance(
        concern_text: str,
        author_response: str,
        stance: str,
    ) -> str:
        """Apply simple heuristics to correct obvious classification errors."""
        resp_lower = author_response.lower()
        # Clear acceptance patterns
        agree_patterns = ["we agree", "we have added", "we have included",
                          "we have revised", "as suggested", "following this suggestion"]
        # Clear rebuttal patterns
        disagree_patterns = ["we disagree", "we believe this is", "we respectfully",
                             "however, we think", "this is not the case"]

        if stance == "unclear":
            for p in agree_patterns:
                if p in resp_lower:
                    return "conceded"
            for p in disagree_patterns:
                if p in resp_lower:
                    return "rebutted"
        return stance

    def process_review(
        self,
        review: ParsedReview,
        article_doi: str,
        article_source: str = "elife",
    ) -> list[ReviewerConcern]:
        """Single review → list of ReviewerConcern objects."""
        # 1. Extract concerns
        raw_concerns = self._extract_concerns_from_review(review.review_text)
        if not raw_concerns:
            return []

        # 2. Classify resolutions (pass reviewer_num for context)
        resolutions = self._classify_resolutions(
            raw_concerns, review.author_response_text, reviewer_num=review.reviewer_num
        )
        article_token = self._article_token_from_doi(article_doi)

        # 3. Build ReviewerConcern objects
        concerns = []
        for i, (rc, res) in enumerate(zip(raw_concerns, resolutions)):
            trace_id = str(uuid.uuid4())
            stance_str = res.get("author_stance", "unclear")

            # Apply heuristic correction
            stance_str = self._heuristic_fix_stance(
                rc["text"],
                review.author_response_text,
                stance_str,
            )

            evidence = res.get("evidence_of_change", None)
            cat = ConcernCategory(rc["category"])

            concern = ReviewerConcern(
                concern_id=f"{article_source}:{article_token}:R{review.reviewer_num}C{i+1}",
                reviewer_num=review.reviewer_num,
                concern_text=rc["text"],
                category=cat,
                severity=rc["severity"],
                author_response_text=review.author_response_text or None,
                author_stance=AuthorStance(stance_str),
                evidence_of_change=evidence,
                resolution_confidence=res.get("confidence", 0.6),
                resolution=Resolution(stance_str),
                requires_figure_reading=(cat == ConcernCategory.FIGURE_ISSUE),
                extraction_trace_id=trace_id,
                extraction_manifest_id=self._manifest_id,
                source=article_source,
                article_doi=article_doi,
            )
            concerns.append(concern)

        return concerns

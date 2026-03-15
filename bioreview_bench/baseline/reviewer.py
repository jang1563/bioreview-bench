"""Baseline LLM reviewer — generates reviewer-like concerns from paper text.

The BaselineReviewer reads a paper (title, abstract, sections) and generates
a list of scientific concerns, mimicking the peer review process.

Usage::

    from bioreview_bench.baseline import BaselineReviewer

    reviewer = BaselineReviewer(model="claude-haiku-4-5-20251001")
    concerns = reviewer.review_article(entry_dict)
    # => ["The study lacks a negative control...", "Statistical analysis uses..."]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

# ── Section priority for truncation ──────────────────────────────────────────

_SECTION_PRIORITY = [
    "methods", "materials and methods", "materials & methods", "method", "methodology",
    "results", "result",
    "introduction", "intro", "background",
    "discussion",
    "conclusion", "conclusions",
    "supplementary", "supplemental", "supporting",
]

_CONCERN_CATEGORIES = (
    "design_flaw",
    "statistical_methodology",
    "missing_experiment",
    "prior_art_novelty",
    "writing_clarity",
    "reagent_method_specificity",
    "interpretation",
    "other",
)
_CATEGORY_RE = "|".join(_CONCERN_CATEGORIES)

# ── System prompt ────────────────────────────────────────────────────────────

REVIEWER_SYSTEM = """\
You are an expert peer reviewer for biomedical research papers published in \
high-impact journals such as eLife, Nature, and Science. Your task is to \
carefully read the manuscript and identify specific scientific concerns, \
weaknesses, and issues that a rigorous reviewer would raise.

CONCERN CATEGORIES (for your reference — output only concern text):
- design_flaw: Fundamental experimental design problems (missing controls, confounders, inappropriate comparisons)
- statistical_methodology: Statistical errors (wrong test, missing corrections, effect sizes, p-hacking)
- missing_experiment: Key experiments absent that are needed to support the main claim
- prior_art_novelty: Missing citations, overstated novelty, inadequate comparison to prior work
- writing_clarity: Unclear descriptions, undefined terms, missing methods details
- reagent_method_specificity: Missing reagent IDs, software versions, analysis parameters
- interpretation: Over-interpretation, correlation/causation confusion, unsupported generalizations
- other: Legitimate scientific concern not fitting above categories

RULES:
1. Generate 10-15 specific, actionable scientific concerns
2. Each concern should be a clear, self-contained statement (1-3 sentences)
3. Cover diverse concern types: experimental design, methodology, statistics, interpretation, writing clarity, and reagent/method specificity
4. Do NOT generate concerns about figures — you cannot see them
5. Do NOT include general praise, vague criticism, or stylistic preferences
6. Each concern must be specific enough to be addressed by the authors
7. Prioritize major issues over minor ones
8. Do NOT repeat essentially the same concern for multiple figures, sections, or experiments — each concern must address a distinct scientific issue

OUTPUT FORMAT: Return a JSON array of concern strings, nothing else:
["The study lacks a negative control for the X assay, making it impossible to ...", \
"The statistical analysis uses t-tests for multiple comparisons without correction ..."]"""


class BaselineReviewer:
    """LLM-based baseline reviewer that generates concerns from paper text.

    Args:
        model: Model identifier (e.g. "claude-haiku-4-5-20251001", "gpt-4o-mini").
        provider: "anthropic", "openai", "google", or "groq".
        max_tokens: Maximum output tokens for the LLM.
        max_input_chars: Maximum chars for paper text (truncated if exceeded).
        temperature: LLM sampling temperature.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        provider: str = "anthropic",
        max_tokens: int = 4096,
        max_input_chars: int = 80_000,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.max_input_chars = max_input_chars
        self.temperature = temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the API client."""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI()
        elif self.provider == "google":
            from google import genai

            self._client = genai.Client()
        elif self.provider == "groq":
            from groq import Groq

            self._client = Groq()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return self._client

    def review_article(self, entry: dict) -> list[str]:
        """Generate reviewer concerns from a paper entry dict.

        Args:
            entry: Paper dict with keys: title, abstract,
                   paper_text_sections/sections, etc.

        Returns:
            List of concern strings.
        """
        paper_input = self._format_paper_input(entry)
        if not paper_input.strip():
            log.warning(
                "Empty paper input for article %s",
                entry.get("id", entry.get("article_id", "?")),
            )
            return []

        raw = self._call_llm(REVIEWER_SYSTEM, paper_input)
        return self._parse_concerns(raw)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, system: str, user: str) -> str:
        """Call the LLM with retry logic."""
        client = self._get_client()

        if self.provider == "anthropic":
            msg = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text  # type: ignore[union-attr]

        if self.provider == "openai":
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

        if self.provider == "google":
            try:
                from google.genai import types
                config = types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except ImportError:
                config = {
                    "system_instruction": system,
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
            resp = client.models.generate_content(
                model=self.model,
                contents=user,
                config=config,
            )
            return resp.text or ""

        if self.provider == "groq":
            resp = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _format_paper_input(self, entry: dict) -> str:
        """Format paper entry into LLM input, truncating to max_input_chars.

        Priority: title > abstract > methods > results > intro > discussion > rest.
        """
        parts: list[str] = []
        budget = self.max_input_chars

        # Title (always included)
        title = entry.get("title", "")
        if title:
            header = f"# {title}\n\n"
            parts.append(header)
            budget -= len(header)

        # Abstract (always included)
        abstract = entry.get("abstract", "")
        if abstract and budget > 0:
            block = f"## Abstract\n{abstract}\n\n"
            if len(block) > budget:
                block = block[:budget]
            parts.append(block)
            budget -= len(block)

        # Sections — use paper_text_sections or sections
        sections = entry.get("paper_text_sections") or entry.get("sections") or {}
        if not sections or budget <= 0:
            return "".join(parts)

        # Sort sections by priority
        ordered = self._prioritize_sections(sections)

        for name, text in ordered:
            if budget <= 0:
                break
            block = f"## {name.title()}\n{text}\n\n"
            if len(block) > budget:
                block = block[:budget]
                parts.append(block)
                budget = 0
            else:
                parts.append(block)
                budget -= len(block)

        return "".join(parts)

    @staticmethod
    def _prioritize_sections(sections: dict[str, str]) -> list[tuple[str, str]]:
        """Sort sections by review priority."""
        ordered: list[tuple[str, str]] = []
        seen: set[str] = set()

        for priority_name in _SECTION_PRIORITY:
            for name, text in sections.items():
                if name.lower() in seen:
                    continue
                if priority_name in name.lower():
                    ordered.append((name, text))
                    seen.add(name.lower())

        # Remaining sections not matching any priority
        for name, text in sections.items():
            if name.lower() not in seen:
                ordered.append((name, text))
                seen.add(name.lower())

        return ordered

    @staticmethod
    def _parse_concerns(text: str) -> list[str]:
        """Parse JSON array of concern strings from LLM output.

        Handles: clean JSON, markdown fences, trailing text, numbered lists.
        Reuses fence -> bracket -> range pattern from concern_extractor.
        """
        # 1. Try markdown fence
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL
        )
        if fence_match:
            result = _try_parse_string_array(fence_match.group(1).strip())
            if result is not None:
                return result

        # 2. First [ ... ] match (non-greedy)
        bracket_match = re.search(r"(\[.*?\])", text, flags=re.DOTALL)
        if bracket_match:
            result = _try_parse_string_array(bracket_match.group(1))
            if result is not None:
                return result

        # 3. First [ to last ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            result = _try_parse_string_array(text[start : end + 1])
            if result is not None:
                return result

        result = _try_parse_concern_list(text)
        if result is not None:
            return result

        return []


def _try_parse_string_array(text: str) -> list[str] | None:
    """Try to parse text as a JSON array of strings."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    results: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            s = item.strip()
            if s:
                results.append(s)
        elif isinstance(item, dict):
            # Accept {"text": "..."} format too
            s = str(item.get("text", item.get("concern_text", ""))).strip()
            if s:
                results.append(s)

    return results


def _try_parse_concern_list(text: str) -> list[str] | None:
    """Try to parse numbered or bulleted concern lists from plain text."""
    results: list[str] = []
    current: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        item_match = re.match(r"^(?:[-*]|\d+[.)])\s+(.*)$", line)
        if item_match:
            if current:
                concern = _normalize_concern_item(" ".join(current))
                if concern:
                    results.append(concern)
            current = [item_match.group(1).strip()]
            continue

        if current:
            current.append(line)

    if current:
        concern = _normalize_concern_item(" ".join(current))
        if concern:
            results.append(concern)

    return results or None


def _normalize_concern_item(text: str) -> str:
    """Strip leading category labels and collapse whitespace."""
    text = " ".join(text.split())
    if not text:
        return ""

    text = re.sub(
        rf"^\*\*({_CATEGORY_RE})(?::)?\*\*[:\s-]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"^({_CATEGORY_RE})[:\s-]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()

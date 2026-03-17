"""Zero-cost lexical baseline using BM25 article retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "with", "we", "our", "these", "those", "than", "then",
    "which", "while", "into", "using", "used", "use", "also", "can",
}


class BM25ConcernRetriever:
    """Retrieve concerns from similar training articles with BM25."""

    def __init__(
        self,
        corpus_entries: list[dict[str, Any]],
        *,
        top_k_docs: int = 8,
        max_concerns: int = 12,
        max_input_chars: int = 40_000,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.top_k_docs = top_k_docs
        self.max_concerns = max_concerns
        self.max_input_chars = max_input_chars
        self.k1 = k1
        self.b = b

        self._doc_ids: list[str] = []
        self._doc_lengths: list[int] = []
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_concerns: list[list[str]] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._idf: dict[str, float] = {}
        self._avg_doc_len = 0.0

        self._build(corpus_entries)

    def review_article(self, entry: dict[str, Any]) -> list[str]:
        """Return concern strings for an input article."""
        query_tokens = _tokenize(_article_text(entry, self.max_input_chars))
        if not query_tokens:
            return []

        query_terms = tuple(dict.fromkeys(query_tokens))
        scores = self._score(query_terms)
        article_id = str(entry.get("id", entry.get("article_id", "")))

        concern_scores: defaultdict[str, float] = defaultdict(float)
        for rank, (doc_idx, score) in enumerate(scores[: self.top_k_docs], start=1):
            if score <= 0:
                continue
            if self._doc_ids[doc_idx] == article_id:
                continue
            weight = score / rank
            for concern in self._doc_concerns[doc_idx]:
                normalized = _normalize_concern(concern)
                if normalized:
                    concern_scores[normalized] += weight

        ranked = sorted(
            concern_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [concern for concern, _score in ranked[: self.max_concerns]]

    def _build(self, corpus_entries: list[dict[str, Any]]) -> None:
        doc_freqs: Counter[str] = Counter()

        for entry in corpus_entries:
            article_id = str(entry.get("id", entry.get("article_id", "")))
            concerns = [
                str(concern.get("concern_text", "")).strip()
                for concern in entry.get("concerns", [])
                if str(concern.get("concern_text", "")).strip()
            ]
            if not article_id or not concerns:
                continue

            tokens = _tokenize(_article_text(entry, self.max_input_chars))
            if not tokens:
                continue

            term_freq = Counter(tokens)
            self._doc_ids.append(article_id)
            self._doc_term_freqs.append(term_freq)
            self._doc_lengths.append(sum(term_freq.values()))
            self._doc_concerns.append(concerns)
            doc_freqs.update(term_freq.keys())
            doc_idx = len(self._doc_ids) - 1
            for term, tf in term_freq.items():
                self._postings[term].append((doc_idx, tf))

        if not self._doc_lengths:
            return

        n_docs = len(self._doc_lengths)
        self._avg_doc_len = sum(self._doc_lengths) / n_docs
        self._idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freqs.items()
        }

    def _score(self, query_terms: tuple[str, ...]) -> list[tuple[int, float]]:
        scores: defaultdict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for doc_idx, tf in self._postings.get(term, []):
                doc_len = self._doc_lengths[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
                scores[doc_idx] += idf * ((tf * (self.k1 + 1)) / denom)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked


def _article_text(entry: dict[str, Any], max_input_chars: int) -> str:
    parts = [
        str(entry.get("title", "")),
        str(entry.get("abstract", "")),
    ]
    sections = entry.get("paper_text_sections") or entry.get("sections") or {}
    for _name, text in sections.items():
        if not text:
            continue
        parts.append(str(text))
        if sum(len(part) for part in parts) >= max_input_chars:
            break
    return "\n".join(parts)[:max_input_chars]


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        token = raw.lower()
        if len(token) < 3:
            continue
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _normalize_concern(text: str) -> str:
    return " ".join(text.split())

"""Agreement utilities for manual review and human-reference workflows."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

VALID_STANCES = ("conceded", "rebutted", "partial", "unclear", "no_response")


@dataclass(frozen=True)
class CategoryAgreement:
    category: str
    n_rows: int
    category_agreement: float
    stance_agreement: float


@dataclass(frozen=True)
class AgreementSummary:
    n_rows: int
    category_agreement: float
    stance_agreement: float
    kappa: float
    quality_label: str
    per_category: tuple[CategoryAgreement, ...]
    stance_disagreements: tuple[dict[str, str], ...]


def compute_label_agreement(rows: list[dict[str, Any]]) -> AgreementSummary:
    """Compute exact-match agreement summary for manual validation rows."""
    if not rows:
        return AgreementSummary(
            n_rows=0,
            category_agreement=0.0,
            stance_agreement=0.0,
            kappa=0.0,
            quality_label="No data",
            per_category=(),
            stance_disagreements=(),
        )

    cat_agree = sum(1 for row in rows if row.get("llm_category") == row.get("human_category"))
    stance_agree = sum(1 for row in rows if row.get("llm_stance") == row.get("human_stance"))
    n_rows = len(rows)

    llm_dist = Counter(str(row.get("llm_stance", "")) for row in rows)
    human_dist = Counter(str(row.get("human_stance", "")) for row in rows)
    p_agree = stance_agree / n_rows
    p_chance = sum(
        (llm_dist.get(stance, 0) / n_rows) * (human_dist.get(stance, 0) / n_rows)
        for stance in VALID_STANCES
    )
    kappa = (p_agree - p_chance) / (1 - p_chance) if p_chance < 1 else 0.0

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = str(row.get("human_category") or row.get("llm_category") or "unknown")
        grouped[category].append(row)

    per_category = tuple(
        CategoryAgreement(
            category=category,
            n_rows=len(group_rows),
            category_agreement=(
                sum(1 for row in group_rows if row.get("llm_category") == row.get("human_category"))
                / len(group_rows)
            ),
            stance_agreement=(
                sum(1 for row in group_rows if row.get("llm_stance") == row.get("human_stance"))
                / len(group_rows)
            ),
        )
        for category, group_rows in sorted(grouped.items())
    )

    stance_disagreements = tuple(
        {
            "concern_id": str(row.get("concern_id", "")),
            "llm_stance": str(row.get("llm_stance", "")),
            "human_stance": str(row.get("human_stance", "")),
            "concern_text": str(row.get("concern_text", "")),
        }
        for row in rows
        if row.get("llm_stance") != row.get("human_stance")
    )

    return AgreementSummary(
        n_rows=n_rows,
        category_agreement=cat_agree / n_rows,
        stance_agreement=stance_agree / n_rows,
        kappa=kappa,
        quality_label=quality_label(kappa),
        per_category=per_category,
        stance_disagreements=stance_disagreements,
    )


def quality_label(kappa: float) -> str:
    if kappa < 0.2:
        return "Poor"
    if kappa < 0.4:
        return "Fair"
    if kappa < 0.6:
        return "Moderate"
    if kappa < 0.8:
        return "Substantial"
    return "Almost perfect"

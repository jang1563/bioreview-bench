"""Baseline concern predictor: paper text → predicted reviewer concerns.

Given ONLY a manuscript's text (title, abstract, and body sections),
predict specific scientific concerns that a peer reviewer would raise.

Output is compatible with ``scripts/run_benchmark.py`` for evaluation.

Usage:
    # Predict on val split (quick test, 10 articles)
    python scripts/generate_predictions.py \
        --split val --model claude-haiku-4-5-20251001 \
        --splits-dir data/splits/v3 \
        --max-articles 10 \
        --output tool_outputs/haiku_val_10.jsonl

    # Full val prediction
    python scripts/generate_predictions.py \
        --split val --model claude-haiku-4-5-20251001 \
        --splits-dir data/splits/v3 \
        --output tool_outputs/haiku_val.jsonl

    # Test prediction with Sonnet
    python scripts/generate_predictions.py \
        --split test --model claude-sonnet-4-6-20250514 \
        --splits-dir data/splits/v3 \
        --output tool_outputs/sonnet_test.jsonl
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import click

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Prediction prompt ─────────────────────────────────────────────────

PREDICTION_SYSTEM = """\
You are an expert biomedical peer reviewer. Given ONLY a manuscript's text \
(title, abstract, and body sections), identify specific scientific concerns \
that a peer reviewer would likely raise.

CATEGORY DEFINITIONS:
- design_flaw: Fundamental experimental design problems (missing controls, confounders)
- statistical_methodology: Statistical analysis errors (wrong test, multiple comparisons)
- missing_experiment: Key experiments absent to support the main claim
- prior_art_novelty: Missing citations, overstated novelty
- writing_clarity: Unclear descriptions, missing method details
- reagent_method_specificity: Missing reagent catalog numbers, software versions
- interpretation: Over-interpretation, correlation/causation confusion
- other: Legitimate concern not fitting above (use sparingly, <10%)

RULES:
1. Base concerns ONLY on what is written — do not hallucinate issues
2. Each concern must be specific and actionable
3. Assign category and severity (major/minor/optional)
4. Generate as many or as few concerns as warranted by the paper's quality. \
Short or well-designed papers may have 3-5 concerns; longer papers with \
methodology issues may have 15-25. Do NOT pad to fill a quota.
5. Focus on methodological and scientific rigor
6. Do NOT raise figure_issue concerns (you cannot see figures)
7. Text sections may be truncated (marked with "[…truncated]"). \
Do NOT raise concerns about missing or incomplete text due to truncation.

OUTPUT: JSON array only, no other text:
[{"text": "...", "category": "...", "severity": "major|minor|optional"}]
"""

# ── Section truncation limits (chars) ─────────────────────────────────

_SECTION_LIMITS: dict[str, int] = {
    "introduction": 4000,
    "main": 4000,
    "methods": 5000,
    "results": 5000,
    "discussion": 3000,
}
_DEFAULT_SECTION_LIMIT = 2000


# ── Helpers ───────────────────────────────────────────────────────────

def _truncate_sections(entry: dict) -> str:
    """Build a truncated manuscript text from paper_text_sections."""
    parts: list[str] = []

    title = entry.get("title", "")
    abstract = entry.get("abstract", "")
    if title:
        parts.append(f"TITLE: {title}")
    if abstract:
        parts.append(f"ABSTRACT:\n{abstract}")

    sections = entry.get("paper_text_sections", {})
    if sections:
        for sec_key in ["introduction", "main", "results", "methods", "discussion"]:
            text = sections.get(sec_key, "")
            if text:
                limit = _SECTION_LIMITS.get(sec_key, _DEFAULT_SECTION_LIMIT)
                truncated = text[:limit]
                if len(text) > limit:
                    truncated += "\n[…truncated]"
                label = sec_key.upper()
                parts.append(f"{label}:\n{truncated}")

        # Include any remaining sections up to a total budget
        used_keys = {"introduction", "main", "results", "methods", "discussion"}
        for sec_key, text in sections.items():
            if sec_key in used_keys or not text:
                continue
            # Skip non-scientific sections
            if sec_key in (
                "acknowledgements", "author_contributions", "competing_interests",
                "footnotes", "references", "associated_data", "peer_review",
                "online_content", "supplementary_information",
            ):
                continue
            truncated = text[:_DEFAULT_SECTION_LIMIT]
            if len(text) > _DEFAULT_SECTION_LIMIT:
                truncated += "\n[…truncated]"
            parts.append(f"{sec_key.upper()}:\n{truncated}")

    if not sections and not abstract:
        # Fallback: title only
        return f"TITLE: {title}\n\n(No abstract or full text available)"

    return "\n\n".join(parts)


# Regex to detect truncation-artifact concerns.
# Context-aware: requires document-related words near "truncat*" to avoid
# filtering real science (e.g. "truncating mutations", "truncated protein").
_TRUNCATION_RE = re.compile(
    r"(?:section|text|paper|manuscript|introduction|methods|results|discussion)"
    r".{0,30}(?:truncat|incomplete|cut off|not shown)"
    r"|(?:appears?|is|seems?)\s+(?:to be\s+)?truncat"
    r"|(?:due to|because of)\s+truncat"
    r"|mid-sentence"
    r"|remainder.*not shown"
    r"|appears to (?:end|stop) abruptly"
    r"|text is incomplete",
    re.I,
)


def _parse_json(raw: str) -> list[dict]:
    """Parse JSON array from LLM output (3-stage fallback)."""
    # Stage 1: JSON fence
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Stage 2: Direct bracket match
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Stage 3: Try the whole string
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return []


def load_split(splits_dir: Path, split: str) -> list[dict]:
    """Load a split JSONL file."""
    path = splits_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_existing_ids(output_path: Path) -> set[str]:
    """Load article IDs already in the output file (for resume)."""
    if not output_path.exists():
        return set()
    ids = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["article_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


async def predict_concerns(
    entry: dict,
    model: str,
    max_tokens: int = 2048,
    semaphore: asyncio.Semaphore | None = None,
) -> dict:
    """Predict reviewer concerns for a single article."""
    import anthropic

    article_id = entry["id"]
    manuscript_text = _truncate_sections(entry)

    async def _call():
        client = anthropic.AsyncAnthropic()
        msg = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=PREDICTION_SYSTEM,
            messages=[{"role": "user", "content": manuscript_text}],
        )
        return msg.content[0].text

    if semaphore:
        async with semaphore:
            raw = await _call()
    else:
        raw = await _call()

    parsed = _parse_json(raw)

    # Extract concern texts
    concerns = []
    for item in parsed:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if len(text) >= 10:
                concerns.append(text)
        elif isinstance(item, str) and len(item.strip()) >= 10:
            concerns.append(item.strip())

    # Filter out truncation-artifact concerns (A2)
    concerns = [c for c in concerns if not _TRUNCATION_RE.search(c)]

    return {"article_id": article_id, "concerns": concerns}


async def run_predictions(
    entries: list[dict],
    model: str,
    output_path: Path,
    concurrency: int = 5,
    resume: bool = True,
) -> dict:
    """Run predictions on all entries and write results to JSONL."""
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    console = Console()

    # Resume support
    existing_ids = load_existing_ids(output_path) if resume else set()
    remaining = [e for e in entries if e["id"] not in existing_ids]

    if existing_ids:
        console.print(f"  [dim]Resume: skipping {len(existing_ids)} already predicted[/dim]")

    stats = {
        "total": len(entries),
        "skipped": len(existing_ids),
        "predicted": 0,
        "empty": 0,
        "total_concerns": 0,
        "errors": 0,
    }

    semaphore = asyncio.Semaphore(concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()

    with (
        output_path.open("a" if resume else "w", encoding="utf-8") as fout,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress,
    ):
        ptask = progress.add_task("Predicting concerns...", total=len(remaining))

        async def _process(entry: dict) -> None:
            try:
                result = await predict_concerns(entry, model=model, semaphore=semaphore)
                async with lock:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    stats["predicted"] += 1
                    n_concerns = len(result["concerns"])
                    stats["total_concerns"] += n_concerns
                    if n_concerns == 0:
                        stats["empty"] += 1
                    progress.advance(ptask)
            except Exception as e:
                async with lock:
                    console.print(f"[red]Error {entry['id']}: {e}")
                    stats["errors"] += 1
                    fout.write(json.dumps({"article_id": entry["id"], "concerns": []}) + "\n")
                    fout.flush()
                    progress.advance(ptask)

        await asyncio.gather(*[_process(e) for e in remaining])

    return stats


@click.command()
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="val",
    show_default=True,
    help="Dataset split to predict on.",
)
@click.option(
    "--splits-dir",
    type=click.Path(exists=True, path_type=Path),
    default=_REPO_ROOT / "data" / "splits" / "v3",
    show_default=True,
    help="Directory containing split JSONL files.",
)
@click.option(
    "--model",
    default="claude-haiku-4-5-20251001",
    show_default=True,
    help="Anthropic model ID for prediction.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file path.",
)
@click.option(
    "--max-articles", "-n",
    type=int,
    default=None,
    help="Limit to first N articles (for quick testing).",
)
@click.option(
    "--concurrency",
    type=int,
    default=5,
    show_default=True,
    help="Number of concurrent API requests.",
)
@click.option(
    "--no-resume",
    is_flag=True,
    default=False,
    help="Start fresh (overwrite existing output).",
)
def main(
    split: str,
    splits_dir: Path,
    model: str,
    output: Path,
    max_articles: int | None,
    concurrency: int,
    no_resume: bool,
) -> None:
    """Baseline concern predictor: paper text → predicted reviewer concerns."""
    from rich.console import Console

    console = Console()
    console.print("[bold cyan]bioreview-bench Concern Predictor[/bold cyan]")
    console.print(f"  split      : {split}")
    console.print(f"  splits-dir : {splits_dir}")
    console.print(f"  model      : {model}")
    console.print(f"  output     : {output}")
    console.print(f"  max-articles: {max_articles or 'all'}")
    console.print(f"  concurrency: {concurrency}")
    console.print(f"  resume     : {not no_resume}")
    console.print()

    entries = load_split(splits_dir, split)
    if max_articles:
        entries = entries[:max_articles]

    console.print(f"Loaded {len(entries)} articles from {split} split")
    console.print()

    stats = asyncio.run(
        run_predictions(
            entries=entries,
            model=model,
            output_path=output,
            concurrency=concurrency,
            resume=not no_resume,
        )
    )

    console.print()
    console.print("[bold]Prediction complete[/bold]")
    console.print(f"  Total:    {stats['total']}")
    console.print(f"  Skipped:  {stats['skipped']}")
    console.print(f"  Predicted: {stats['predicted']}")
    console.print(f"  Empty:    {stats['empty']}")
    console.print(f"  Errors:   {stats['errors']}")
    console.print(f"  Concerns: {stats['total_concerns']}")
    if stats["predicted"] > 0:
        console.print(
            f"  Avg/article: {stats['total_concerns'] / stats['predicted']:.1f}"
        )
    console.print(f"  Output:   {output}")


if __name__ == "__main__":
    main()

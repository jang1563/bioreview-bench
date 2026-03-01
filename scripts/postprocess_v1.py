"""Post-processing script for elife_v1.jsonl.

Applies the following fixes to the Phase 1 collected data:
1. Remove article-type labels from subjects field (e.g. "Research Article")
2. Add review_format field (journal / reviewed_preprint / unknown)
3. Add has_author_response field
4. Upgrade schema_version to "1.1"

Input:  data/processed/elife_v1.jsonl      (600 articles, schema 1.0)
Output: data/processed/elife_v1.1.jsonl    (600 articles, schema 1.1)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreview_bench.collect.postprocess import clean_subjects, infer_review_format
from bioreview_bench.models.entry import _ARTICLE_TYPE_LABELS  # noqa: F401 (re-export for backcompat)


def postprocess(input_path: Path, output_path: Path) -> None:
    entries = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {input_path}")

    stats = {
        "subjects_cleaned": 0,
        "format_journal": 0,
        "format_reviewed_preprint": 0,
        "format_unknown": 0,
        "has_author_response_true": 0,
        "schema_upgraded": 0,
    }

    processed = []
    for entry in entries:
        # 1. Clean subjects field
        raw_subjects = entry.get("subjects", [])
        cleaned = clean_subjects(raw_subjects)
        if len(cleaned) != len(raw_subjects):
            stats["subjects_cleaned"] += 1
        entry["subjects"] = cleaned

        # 2. Infer review_format
        fmt = infer_review_format(entry)
        entry["review_format"] = fmt
        stats[f"format_{fmt}"] += 1

        # 3. Set has_author_response
        has_resp = bool((entry.get("author_response_raw") or "").strip())
        entry["has_author_response"] = has_resp
        if has_resp:
            stats["has_author_response_true"] += 1

        # 4. Upgrade schema_version
        if entry.get("schema_version") != "1.1":
            entry["schema_version"] = "1.1"
            stats["schema_upgraded"] += 1

        processed.append(entry)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n=== Post-processing complete ===")
    print(f"Output: {output_path} ({len(processed)} entries)")
    print(f"\nChanges applied:")
    print(f"  subjects cleaned:          {stats['subjects_cleaned']}")
    print(f"  schema upgraded to 1.1:    {stats['schema_upgraded']}")
    print(f"  has_author_response=True:  {stats['has_author_response_true']}")
    print(f"\nreview_format distribution:")
    print(f"  journal:            {stats['format_journal']}")
    print(f"  reviewed_preprint:  {stats['format_reviewed_preprint']}")
    print(f"  unknown:            {stats['format_unknown']}")

    # Print cleaned subjects sample
    from collections import Counter
    all_subjects: list[str] = []
    for e in processed:
        all_subjects.extend(e.get("subjects", []))
    subject_counts = Counter(all_subjects)
    print(f"\nTop 15 subjects after cleaning:")
    for subj, cnt in subject_counts.most_common(15):
        print(f"  {cnt:4d}  {subj}")


if __name__ == "__main__":
    input_path = ROOT / "data" / "processed" / "elife_v1.jsonl"
    output_path = ROOT / "data" / "processed" / "elife_v1.1.jsonl"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    postprocess(input_path, output_path)

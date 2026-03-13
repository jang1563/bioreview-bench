from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from bioreview_bench.baseline.lexical import BM25ConcernRetriever
from bioreview_bench.scripts.run_bm25_baseline import main


def _entry(
    article_id: str,
    title: str,
    abstract: str,
    concerns: list[str],
) -> dict:
    return {
        "id": article_id,
        "title": title,
        "abstract": abstract,
        "paper_text_sections": {
            "methods": abstract,
            "results": abstract,
        },
        "concerns": [{"concern_text": concern} for concern in concerns],
    }


class TestBM25ConcernRetriever:
    def test_retrieves_concerns_from_similar_article(self):
        corpus = [
            _entry(
                "train:1",
                "Single-cell RNA sequencing in glioblastoma",
                "Glioblastoma single-cell transcriptomics identifies tumor states and batch effects.",
                [
                    "The study lacks validation in an independent patient cohort.",
                    "Multiple-testing correction is not described for the differential expression analysis.",
                ],
            ),
            _entry(
                "train:2",
                "Zebrafish fin regeneration imaging study",
                "Live imaging tracks fin regeneration over time in zebrafish.",
                ["The manuscript does not report blinding during image scoring."],
            ),
        ]
        query = _entry(
            "val:1",
            "Transcriptomic states in glioblastoma single cells",
            "Single-cell RNA-seq of glioblastoma tumors reports transcriptional states and differential expression.",
            [],
        )

        retriever = BM25ConcernRetriever(corpus, top_k_docs=1, max_concerns=2)
        concerns = retriever.review_article(query)

        assert concerns
        assert "independent patient cohort" in concerns[0] or "Multiple-testing correction" in concerns[0]

    def test_excludes_same_article_id_from_retrieval(self):
        corpus = [
            _entry(
                "shared:1",
                "CRISPR screen in T cells",
                "Genome-wide CRISPR screen in activated T cells.",
                ["This concern should be excluded because it belongs to the query article."],
            ),
            _entry(
                "train:2",
                "CRISPR perturbation screen in activated T cells",
                "Activated T-cell CRISPR perturbation screen with flow cytometry readout.",
                ["Independent validation of top hits is missing."],
            ),
        ]
        query = _entry(
            "shared:1",
            "CRISPR screen in T cells",
            "Genome-wide CRISPR screen in activated T cells.",
            [],
        )

        retriever = BM25ConcernRetriever(corpus, top_k_docs=2, max_concerns=2)
        concerns = retriever.review_article(query)

        assert concerns == ["Independent validation of top hits is missing."]

    def test_respects_max_concerns(self):
        corpus = [
            _entry(
                "train:1",
                "Immune profiling in sepsis",
                "Sepsis immune profiling with cytokine measurements.",
                [f"Concern {idx}" for idx in range(1, 6)],
            )
        ]
        query = _entry(
            "val:1",
            "Immune profiling of septic patients",
            "Cytokine profiling in septic patients.",
            [],
        )

        retriever = BM25ConcernRetriever(corpus, top_k_docs=1, max_concerns=3)
        concerns = retriever.review_article(query)

        assert len(concerns) == 3


class TestBM25CLI:
    def test_cli_writes_benchmark_compatible_output(self, tmp_path: Path):
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        train_entry = _entry(
            "train:1",
            "Glioblastoma single-cell study",
            "Single-cell glioblastoma profiling with differential expression.",
            ["External validation cohort is missing."],
        )
        val_entry = _entry(
            "val:1",
            "Single-cell glioblastoma atlas",
            "Differential expression across glioblastoma cell states.",
            ["ground truth placeholder"],
        )

        (splits_dir / "train.jsonl").write_text(json.dumps(train_entry) + "\n", encoding="utf-8")
        (splits_dir / "val.jsonl").write_text(json.dumps(val_entry) + "\n", encoding="utf-8")

        output_path = tmp_path / "bm25_val.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--splits-dir",
                str(splits_dir),
                "--split",
                "val",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(rows) == 1
        assert rows[0]["article_id"] == "val:1"
        assert rows[0]["concerns"] == ["External validation cohort is missing."]

    def test_cli_dry_run(self, tmp_path: Path):
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        train_entry = _entry("train:1", "Paper", "Abstract", ["Concern"])
        val_entry = _entry("val:1", "Paper", "Abstract", ["placeholder"])
        (splits_dir / "train.jsonl").write_text(json.dumps(train_entry) + "\n", encoding="utf-8")
        (splits_dir / "val.jsonl").write_text(json.dumps(val_entry) + "\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--splits-dir", str(splits_dir), "--split", "val", "--dry-run"],
        )

        assert result.exit_code == 0
        assert "Dry run" in result.output

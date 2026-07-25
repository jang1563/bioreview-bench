from __future__ import annotations

import py_compile
from pathlib import Path

from click.testing import CliRunner

from bioreview_bench.scripts import collect_elife as collect_cli
from bioreview_bench.scripts.run_benchmark import main as run_benchmark_main


def test_run_benchmark_requires_tool_output() -> None:
    result = CliRunner().invoke(run_benchmark_main, [])
    assert result.exit_code == 2
    assert "--tool-output" in result.output or "--tool-name" in result.output


def test_standalone_run_benchmark_script_compiles() -> None:
    py_compile.compile("scripts/run_benchmark.py", doraise=True)


def test_run_benchmark_help_documents_matcher_configuration() -> None:
    result = CliRunner().invoke(run_benchmark_main, ["--help"])

    assert result.exit_code == 0
    assert "--embedding-model" in result.output
    assert "--embedding-revision" in result.output
    assert "--allow-fallback" in result.output
    assert "--no-embedding" in result.output
    assert "model-" in result.output
    assert "calibrat" in result.output.lower()


def test_run_benchmark_passes_embedding_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from bioreview_bench.evaluate import runner as eval_runner

    captured: dict[str, object] = {}

    def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return object(), []

    monkeypatch.setattr(eval_runner, "run_evaluation", fake_run_evaluation)
    tool_output = tmp_path / "tool.jsonl"
    tool_output.write_text("", encoding="utf-8")

    result = CliRunner().invoke(
        run_benchmark_main,
        [
            "--tool-output",
            str(tool_output),
            "--tool-name",
            "test-tool",
            "--embedding-model",
            "org/custom-review-encoder",
            "--embedding-revision",
            "abc123",
            "--allow-fallback",
        ],
    )

    assert result.exit_code == 0
    assert captured["use_embedding"] is True
    assert captured["embedding_model"] == "org/custom-review-encoder"
    assert captured["embedding_revision"] == "abc123"
    assert captured["allow_fallback"] is True


def test_run_benchmark_disables_partial_hf_publisher(tmp_path: Path) -> None:
    tool_output = tmp_path / "tool.jsonl"
    tool_output.write_text("", encoding="utf-8")

    result = CliRunner().invoke(
        run_benchmark_main,
        [
            "--tool-output",
            str(tool_output),
            "--tool-name",
            "test-tool",
            "--push-hf",
        ],
    )

    assert result.exit_code == 1
    assert "rights-audited dataset release" in result.output


def test_run_benchmark_retains_explicit_jaccard_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from bioreview_bench.evaluate import runner as eval_runner

    captured: dict[str, object] = {}

    def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return object(), []

    monkeypatch.setattr(eval_runner, "run_evaluation", fake_run_evaluation)
    tool_output = tmp_path / "tool.jsonl"
    tool_output.write_text("", encoding="utf-8")

    result = CliRunner().invoke(
        run_benchmark_main,
        [
            "--tool-output",
            str(tool_output),
            "--tool-name",
            "test-tool",
            "--no-embedding",
        ],
    )

    assert result.exit_code == 0
    assert captured["use_embedding"] is False
    assert captured["allow_fallback"] is False


def test_run_benchmark_surfaces_fail_closed_embedding_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from bioreview_bench.evaluate import runner as eval_runner

    def fail_closed(**kwargs):
        raise RuntimeError("embedding unavailable; fallback disabled")

    monkeypatch.setattr(eval_runner, "run_evaluation", fail_closed)
    tool_output = tmp_path / "tool.jsonl"
    tool_output.write_text("", encoding="utf-8")

    result = CliRunner().invoke(
        run_benchmark_main,
        [
            "--tool-output",
            str(tool_output),
            "--tool-name",
            "test-tool",
        ],
    )

    assert result.exit_code == 1
    assert "embedding unavailable; fallback disabled" in result.output


def test_collect_cli_passes_start_date_and_dry_run(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def fake_run(
        subjects: list[str],
        start_date: str,
        end_date: str | None,
        order: str,
        max_articles: int,
        output: Path,
        manifest_path: Path,
        model: str,
        dry_run: bool,
        no_extract: bool = False,
        append: bool = False,
        known_ids: set | None = None,
    ) -> None:
        captured["subjects"] = subjects
        captured["start_date"] = start_date
        captured["max_articles"] = max_articles
        captured["output"] = output
        captured["manifest_path"] = manifest_path
        captured["model"] = model
        captured["dry_run"] = dry_run
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(collect_cli, "_run", fake_run)

    out_path = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"
    result = CliRunner().invoke(
        collect_cli.main,
        [
            "--max-articles",
            "1",
            "--start-date",
            "2020-01-01",
            "--dry-run",
            "--output",
            str(out_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["start_date"] == "2020-01-01"
    assert captured["dry_run"] is True
    assert captured["max_articles"] == 1


def test_collect_cli_uses_elife_v1_1_default_output(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run(
        subjects: list[str],
        start_date: str,
        end_date: str | None,
        order: str,
        max_articles: int,
        output: Path,
        manifest_path: Path,
        model: str,
        dry_run: bool,
        no_extract: bool = False,
        append: bool = False,
        known_ids: set | None = None,
    ) -> None:
        captured["output"] = output
        captured["manifest_path"] = manifest_path

    monkeypatch.setattr(collect_cli, "_run", fake_run)

    result = CliRunner().invoke(
        collect_cli.main,
        ["--max-articles", "1", "--dry-run"],
    )

    assert result.exit_code == 0
    assert Path(captured["output"]).name == "elife_v1.1.jsonl"

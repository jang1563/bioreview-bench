from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from bioreview_bench.scripts import collect_elife as collect_cli
from bioreview_bench.scripts.run_benchmark import main as run_benchmark_main


def test_run_benchmark_requires_tool_output() -> None:
    result = CliRunner().invoke(run_benchmark_main, [])
    assert result.exit_code == 2
    assert "--tool-output" in result.output or "--tool-name" in result.output


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

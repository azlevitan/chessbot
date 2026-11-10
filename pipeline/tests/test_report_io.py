"""Report rebuilding, CLI and publication failure behavior without an engine."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from report import build_ladder, write_report
from run_layout import REPO_ROOT
from test_report import HEADER, generation, put, rows


def test_empty_run_produces_empty_ladder_and_header_only(tmp_path):
    assert build_ladder(tmp_path) == []
    write_report(tmp_path)
    assert json.loads((tmp_path / "ladder.json").read_text()) == []
    assert (tmp_path / "report.tsv").read_text().splitlines() == ["\t".join(HEADER)]


def test_rebuild_is_deterministic_and_preserves_all_input_bytes_and_mtimes(tmp_path):
    generation(tmp_path, 0)
    directory = generation(tmp_path, 1)
    put(directory / "train.json", {"final_loss": 0.75, "seconds": 40})
    inputs = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in (tmp_path / "gens").rglob("*") if path.is_file()}
    write_report(tmp_path)
    outputs = {name: ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_mtime_ns)
               for name in ("report.tsv", "ladder.json")}
    for name in outputs:
        (tmp_path / (name + ".tmp")).write_text("interrupted publication\n")
    write_report(tmp_path)
    assert outputs == {name: ((tmp_path / name).read_bytes(),
                              (tmp_path / name).stat().st_mtime_ns) for name in outputs}
    assert inputs == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in inputs}
    assert not list(tmp_path.rglob("*.tmp"))


def test_cli_writes_reports_and_prints_exact_saved_tsv_without_site_packages(tmp_path):
    generation(tmp_path, 0)
    generation(tmp_path, 1)
    result = subprocess.run(
        [sys.executable, "-S", str(REPO_ROOT / "pipeline/report.py"), "--run", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == (tmp_path / "report.tsv").read_text()
    assert len(json.loads((tmp_path / "ladder.json").read_text())) == 2
    assert len(rows(tmp_path)) == 2


@pytest.mark.parametrize("contents", ["invalid-json", "[]", "null", "{}",
                                      '{"promoted":"true","reason":"sprt_accept"}'])
def test_malformed_published_decision_does_not_replace_existing_reports(tmp_path, contents):
    generation(tmp_path, 0)
    directory = generation(tmp_path, 1)
    write_report(tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in ("ladder.json", "report.tsv")}
    (directory / "decision.json").write_text(contents)
    with pytest.raises(ValueError):
        write_report(tmp_path)
    assert before == {name: (tmp_path / name).read_bytes() for name in before}
    assert not list(tmp_path.rglob("*.tmp"))


def test_unreadable_optional_metadata_becomes_na_without_losing_generation(tmp_path):
    generation(tmp_path, 0)
    directory = generation(tmp_path, 1)
    for relative in ("train.json", "anchors/vs_random.json", "anchors/vs_material.json"):
        path = put(directory / relative, {})
        path.write_text("invalid-json")
    write_report(tmp_path)
    row = rows(tmp_path)[1]
    assert float(row["ladder_elo"]) == 50
    for key in ("train_loss", "val_top1", "train_secs", "vs_random", "vs_material"):
        assert row[key] == "NA"


def test_publication_failure_keeps_previous_tsv_complete_and_retry_repairs_pair(
        tmp_path, monkeypatch):
    generation(tmp_path, 0)
    write_report(tmp_path)
    before = (tmp_path / "report.tsv").read_bytes()
    generation(tmp_path, 1)
    original_replace = Path.replace

    def fail_tsv_publication(path, target):
        if Path(target).name == "report.tsv":
            raise OSError("injected publication failure")
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_tsv_publication)
        with pytest.raises(OSError, match="injected publication failure"):
            write_report(tmp_path)
    assert (tmp_path / "report.tsv").read_bytes() == before
    assert isinstance(json.loads((tmp_path / "ladder.json").read_text()), list)
    assert not list(tmp_path.rglob("*.tmp"))
    write_report(tmp_path)
    assert list(rows(tmp_path)) == [0, 1]
    assert [row["gen"] for row in json.loads((tmp_path / "ladder.json").read_text())] == [0, 1]

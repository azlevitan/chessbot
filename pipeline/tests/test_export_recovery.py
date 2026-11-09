"""An export failure must preserve trained weights and retry only export."""

import os
import subprocess

import pytest

import bootstrap
from run_layout import REPO_ROOT, read_json, read_run_json, write_json_atomic


def test_failed_parity_retries_export_without_retraining(engine_bin, tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    write_json_atomic(config, {"seed": 0})
    run = tmp_path / "run"
    pgn = tmp_path / "input.pgn"
    pgn.write_bytes((REPO_ROOT / "training/tests/fixtures/sample.pgn").read_bytes())
    args = ["--pgn", str(pgn),
            "--run", str(run), "--config", str(config), "--val-games", "1",
            "--steps", "2", "--batch-size", "16", "--device", "cpu"]
    original = bootstrap.training_process

    def fail_export(script, *arguments):
        if script == "export_onnx.py":
            (run / "gens/gen_0000/model.onnx.tmp").write_bytes(b"failed export")
            raise subprocess.CalledProcessError(1, "parity failure")
        return original(script, *arguments)

    monkeypatch.setattr(bootstrap, "training_process", fail_export)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.main(args)
    gen = run / "gens/gen_0000"
    protected = [gen / "model.pt", gen / "train.json",
                 *run.glob("supervised*/*.bin"), *run.glob("supervised*/*.json")]
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in protected}
    assert read_json(gen / "train.json")["steps"] == 2
    assert not (gen / "model.onnx").exists()
    assert not (run / "run.json").exists()

    def export_only(script, *arguments):
        assert script == "export_onnx.py", "completed training must be reused"
        options = dict(zip(map(str, arguments[::2]), map(str, arguments[1::2])))
        assert options["--parity-source"] == str(run / "supervised_val")
        return original(script, *arguments)

    monkeypatch.setattr(bootstrap, "training_process", export_only)
    monkeypatch.setattr(bootstrap, "convert_stage", lambda *a, **k: pytest.fail("conversion ran"))
    timestamp = pgn.stat().st_mtime_ns
    os.utime(pgn, ns=(timestamp, timestamp + 1_000_000_000))
    monkeypatch.setattr(bootstrap, "ENGINE_BIN", tmp_path / "missing-engine")
    bootstrap.main([*args, "--export-only"])
    assert {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in protected} == before
    assert bootstrap.valid_onnx(gen / "model.onnx")
    assert read_json(gen / "decision.json") == {"promoted": True, "reason": "bootstrap"}
    assert read_run_json(run)["stage"] == "selfplay"
    assert not list(run.rglob("*.tmp"))
    # A later failed forced re-export must retain the last published model/state.
    published = [gen / "model.onnx", gen / "decision.json", run / "run.json"]
    previous = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in published}
    monkeypatch.setattr(bootstrap, "training_process", fail_export)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.main([*args, "--export-only"])
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in published} == previous


@pytest.mark.parametrize("missing", ["model.pt", "train.json"])
def test_export_only_never_retrains_or_deletes_incomplete_checkpoint(
    engine_bin, tmp_path, monkeypatch, missing,
):
    # Build published data using Rust, but deliberately leave training incomplete.
    fixture = REPO_ROOT / "training/tests/fixtures/sample.pgn"
    run = tmp_path / "run"
    for directory in ("supervised", "supervised_val"):
        bootstrap.convert_stage(fixture, run / directory, "--max-games", "1")
    config = tmp_path / "config.json"
    write_json_atomic(config, {})
    gen = run / "gens/gen_0000"
    gen.mkdir(parents=True)
    checkpoint = gen / "model.pt"
    summary = gen / "train.json"
    checkpoint.write_bytes(b"retained weights")
    write_json_atomic(summary, {"steps": 2})
    (gen / missing).unlink()
    before = {path: path.read_bytes() for path in gen.iterdir()}
    monkeypatch.setattr(bootstrap, "training_process", lambda *a: pytest.fail("training ran"))
    with pytest.raises(ValueError, match="export-only requires complete model.pt"):
        bootstrap.main(["--pgn", str(fixture), "--run", str(run), "--config", str(config),
                        "--export-only"])
    assert {path: path.read_bytes() for path in gen.iterdir()} == before

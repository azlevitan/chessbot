"""Real loop interruption retains completed chunks and durable telemetry."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from config import load_config
from run_layout import REPO_ROOT, read_run_json
from telemetry import read_attempts, stages_summary
from test_loop import bare_run, command, invoke


def fake_engine(path, ready, resume):
    """Publish first chunk; block second until the interrupted run is restarted."""
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "options = dict(zip(sys.argv[2::2], sys.argv[3::2]))\n"
        "out = Path(options['--out'])\n"
        f"if out.stem == 'chunk_000001' and not Path({str(resume)!r}).exists():\n"
        f"    Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        "    time.sleep(30)\n"
        "games = int(options['--games'])\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_bytes(bytes(games * 2 * 492))\n"
        "metadata = {'format':'chessbot-chunk-v1', 'kind':'selfplay',\n"
        " 'samples':games*2, 'games':games, 'model':options['--model'],\n"
        " 'seed':int(options['--seed']), 'sims':int(options['--sims']),\n"
        " 'max_plies':int(options['--max-plies']), 'temp_plies':int(options['--temp-plies']),\n"
        " 'results':{'white':0,'black':0,'draw':games},\n"
        " 'terminations':{'checkmate':0,'resign':0,'rule_draw':0,'repetition':0,\n"
        "                 'insufficient_material':0,'ply_cap':games},\n"
        " 'resign':{'threshold':float(options['--resign-threshold']),\n"
        "           'consecutive':3,'playthrough_games':0,'false_positives':0}}\n"
        "out.with_suffix('.json').write_text(json.dumps(metadata))\n"
    )
    path.chmod(0o755)


def snapshot(paths):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def test_sigint_and_resume_preserve_chunks_manifest_and_attempt_metrics(tmp_path):
    run, ready, resume = (tmp_path / name for name in ("run", "ready", "resume"))
    cfg = load_config(REPO_ROOT / "pipeline/configs/ci_tiny.json")
    before = bare_run(run, config=cfg)
    initial = run / "gens/gen_0000"
    initial.mkdir(parents=True)
    (initial / "model.onnx").write_bytes(b"fake engine does not read this model")
    engine = tmp_path / "fake-engine"
    fake_engine(engine, ready, resume)
    env = os.environ.copy()
    env.pop("CHESSBOT_CRASH_AT", None)
    process = subprocess.Popen(command(run, engine, "--until-gen", 1), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 8
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        assert ready.exists()
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 130, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()

    assert read_run_json(run) == before
    assert not (run / "loop.lock").exists()
    directory = run / "gens/gen_0001"
    chunks = directory / "selfplay"
    kept = snapshot([chunks / "chunk_000000.bin", chunks / "chunk_000000.json"])
    manifest = snapshot([directory / "generation.json"])
    records_before = snapshot((directory / "telemetry").glob("*.json"))
    first = stages_summary(directory)["selfplay"]
    assert first["rss_complete"] and first["seconds"] > 0
    assert first["attempts"] == 1
    assert any(record.get("status") == "interrupted" for record in read_attempts(directory))
    with pytest.raises(ProcessLookupError):
        os.kill(int(ready.read_text()), 0)

    (directory / "telemetry/stage_abandoned.json.tmp").write_text("partial publication")
    resume.touch()
    resumed = invoke(run, engine, "--until-gen", 1, crash="1:selfplay", timeout=10)
    assert resumed.returncode == 17, resumed.stdout + resumed.stderr
    assert read_run_json(run) == before
    assert snapshot(kept) == kept
    assert snapshot(manifest) == manifest
    assert snapshot(records_before) == records_before
    assert len(list(chunks.glob("chunk_*.bin"))) == 2
    assert (directory / "selfplay_summary.json").is_file()
    assert json.loads((directory / "generation.json").read_text())["config"] == cfg

    records = read_attempts(directory)
    spans = [record for record in records if record["kind"] == "stage"]
    leaves = [record for record in records if record["kind"] == "process"]
    summary = stages_summary(directory)["selfplay"]
    assert summary["attempts"] == 2 and summary["rss_complete"]
    assert summary["seconds"] == sum(record["seconds"] for record in spans)
    assert summary["seconds"] != sum(record["seconds"] for record in records)
    assert summary["seconds"] > first["seconds"]
    assert summary["peak_rss_bytes"] >= max(record["peak_rss_bytes"] for record in leaves)
    assert len([record for record in leaves if record["unit"] == "chunk_000000"]) == 1
    assert not list(run.rglob("*.tmp"))

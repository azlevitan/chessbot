"""Real parity inputs stay Rust-encoded, bounded, and cover dynamic batches."""

from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

import export_onnx as exporter
from data.chunk_dataset import ChunkDataset, list_complete_chunks


@pytest.fixture
def chunks(engine_bin, tmp_path):
    fixture = Path(__file__).parent / "fixtures/sample.pgn"
    pgn = tmp_path / "input.pgn"
    pgn.write_bytes(fixture.read_bytes() * 2)
    directory = tmp_path / "chunks"
    subprocess.run([str(engine_bin), "convert-pgn", "--input", str(pgn),
                    "--out-dir", str(directory), "--chunk-samples", "25"], check=True)
    return directory


def test_real_positions_span_chunks_and_keep_strict_checks(chunks, monkeypatch):
    dataset = ChunkDataset(list_complete_chunks(chunks))
    indices = np.linspace(0, len(dataset) - 1, 64, dtype=np.int64)
    expected = torch.stack([dataset[int(i)]["planes"] for i in indices])
    seen = []
    session = object()

    def check(model, path, planes, *, legal_mask, rtol, session):
        assert rtol == 0 and session is not None
        assert legal_mask.shape == (len(planes), 4168)
        assert bool(legal_mask.any(dim=1).all())
        seen.append(planes.clone())
        return 3e-5

    monkeypatch.setattr(exporter, "verify_parity", check)
    assert exporter.verify_real_positions(None, Path("unused.onnx"), chunks, session) == 3e-5
    offset = 0
    for size in (1, 7, 32):
        count = (64 + size - 1) // size
        calls = seen[offset:offset + count]
        assert all(len(batch) == size for batch in calls)
        assert torch.equal(torch.cat(calls)[:64], expected)
        offset += count
    assert offset == len(seen)


def test_tiny_source_still_exercises_batches_1_7_32(chunks, monkeypatch):
    positions = exporter.load_parity_positions(chunks)
    positions = {key: value[:1] for key, value in positions.items()}
    sizes = []

    def check(model, path, planes, **options):
        sizes.append(len(planes))
        assert torch.equal(planes, positions["planes"].repeat(len(planes), 1, 1, 1))
        assert options["rtol"] == 0
        return 0.0

    monkeypatch.setattr(exporter, "verify_parity", check)
    exporter.verify_real_positions(None, Path("unused"), chunks, None, positions=positions)
    assert sizes == [1, 7, 32]


def test_empty_or_incomplete_source_cannot_pass(tmp_path):
    with pytest.raises(ValueError, match="no complete nonempty chunks"):
        exporter.load_parity_positions(tmp_path)
    (tmp_path / "partial.bin").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="no complete nonempty chunks"):
        exporter.load_parity_positions(tmp_path)


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_export_cannot_overwrite_checkpoint(tmp_path, alias):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"precious weights")
    out = checkpoint if alias == "same" else tmp_path / "output.onnx"
    if alias == "symlink":
        out.symlink_to(checkpoint)
    elif alias == "hardlink":
        out.hardlink_to(checkpoint)
    with pytest.raises(ValueError, match="aliases protected input"):
        exporter.validate_paths(checkpoint, out, None)
    assert checkpoint.read_bytes() == b"precious weights"


def test_export_cannot_overwrite_validation_source(chunks, tmp_path):
    source = chunks / "chunk_000000.bin"
    previous = source.read_bytes()
    with pytest.raises(ValueError):
        exporter.validate_paths(tmp_path / "model.pt", source, chunks)
    alias = tmp_path / "outside.onnx"
    alias.hardlink_to(source)
    with pytest.raises(ValueError):
        exporter.validate_paths(tmp_path / "model.pt", alias, chunks)
    assert source.read_bytes() == previous


def test_invalid_source_is_checked_before_output_write(tmp_path, monkeypatch):
    out = tmp_path / "existing.onnx"
    out.write_bytes(b"existing model")
    source = tmp_path / "empty"
    source.mkdir()
    monkeypatch.setattr(exporter, "export_onnx", lambda *args: pytest.fail("export ran"))
    assert exporter.main(["--checkpoint", str(tmp_path / "missing.pt"), "--out", str(out),
                          "--parity-source", str(source)]) == 1
    assert out.read_bytes() == b"existing model"

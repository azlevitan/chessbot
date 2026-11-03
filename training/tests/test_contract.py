"""GATE 2: Rust-written TrainingSample bytes decode to exactly what SPEC.md defines.

Expected values come from the test-only python-chess oracle (`oracle.py`) plus a
few hand-computed literal indices that guard against a shared mistake.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest

from data.replay_buffer import (
    FIELD_OFFSETS,
    NUM_MOVES,
    POLICY_SLOTS,
    SAMPLE_BYTES,
    SAMPLE_DTYPE,
    UNUSED_MOVE,
    ReplayBuffer,
    make_loader,
    open_samples,
    unpack_planes,
    unpack_policy,
)
from oracle import (
    TEST_FENS,
    canonical_ep,
    castling_rights,
    expected_index,
    expected_indices,
    expected_planes,
)

LABELS = [label for label, _ in TEST_FENS]


@pytest.fixture(scope="module")
def buffer(vectors_path: Path) -> ReplayBuffer:
    return ReplayBuffer(vectors_path)


def board_for(label: str) -> chess.Board:
    return chess.Board(dict(TEST_FENS)[label])


def test_layout(vectors_path: Path) -> None:
    assert SAMPLE_DTYPE.itemsize == SAMPLE_BYTES == 492
    assert {k: SAMPLE_DTYPE.fields[k][1] for k in FIELD_OFFSETS} == FIELD_OFFSETS
    assert vectors_path.stat().st_size == len(TEST_FENS) * SAMPLE_BYTES == 2460
    assert len(open_samples(vectors_path)) == len(TEST_FENS)


@pytest.mark.parametrize("i,label", list(enumerate(LABELS)))
def test_planes_match_oracle(buffer: ReplayBuffer, i: int, label: str) -> None:
    board = board_for(label)
    expected = expected_planes(board)
    raw = unpack_planes(buffer.record(i))
    assert raw.shape == (18, 8, 8) and raw.dtype == np.float32
    for p in range(18):
        assert np.array_equal(raw[p], expected[p]), f"{label}: plane {p} differs"
    assert np.array_equal(buffer[i]["planes"].numpy(), expected)


@pytest.mark.parametrize("i,label", list(enumerate(LABELS)))
def test_flags_match_oracle(buffer: ReplayBuffer, i: int, label: str) -> None:
    board = board_for(label)
    rec = buffer.record(i)
    flags = int(rec["castling_and_ep"])
    rights = castling_rights(board)
    assert [bool(flags & (1 << b)) for b in range(4)] == rights, label
    ep = canonical_ep(board)
    assert bool(flags & 0x10) == (ep is not None), label
    assert int(rec["ep_square"]) == (ep if ep is not None else 0), label
    assert int(rec["halfmove_clock"]) == board.halfmove_clock, label
    assert flags >> 5 == 0, f"{label}: reserved flag bits set"


def test_flag_spot_checks(buffer: ReplayBuffer) -> None:
    rec = {label: buffer.record(i) for i, label in enumerate(LABELS)}
    assert int(rec["startpos"]["castling_and_ep"]) == 0b01111
    assert int(rec["kiwipete"]["castling_and_ep"]) == 0b01111
    assert int(rec["en_passant"]["castling_and_ep"]) == 0b11111
    assert int(rec["en_passant"]["ep_square"]) == 45  # f6
    assert int(rec["underpromotion"]["castling_and_ep"]) == 0
    assert int(rec["black_check"]["halfmove_clock"]) == 1


@pytest.mark.parametrize("i,label", list(enumerate(LABELS)))
def test_move_indices_match_oracle(buffer: ReplayBuffer, i: int, label: str) -> None:
    board = board_for(label)
    rec = buffer.record(i)
    n_legal = board.legal_moves.count()
    n = int(rec["num_legal_moves"])
    assert n == n_legal <= POLICY_SLOTS, label
    active = rec["policy"][:n]
    idx = active["move_idx"].astype(int)
    assert set(idx.tolist()) == expected_indices(board), label
    assert np.all(np.diff(idx) > 0), f"{label}: entries not strictly ascending"
    assert np.allclose(active["prob"], 1.0 / n_legal), label
    padding = rec["policy"][n:]
    assert np.all(padding["move_idx"] == UNUSED_MOVE), label
    assert np.all(padding["prob"] == 0.0), label


@pytest.mark.parametrize("i,label", list(enumerate(LABELS)))
def test_dense_targets(buffer: ReplayBuffer, i: int, label: str) -> None:
    board = board_for(label)
    item = buffer[i]
    policy, mask = item["policy"].numpy(), item["legal_mask"].numpy()
    assert policy.shape == mask.shape == (NUM_MOVES,)
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert set(np.flatnonzero(mask).tolist()) == expected_indices(board)
    assert np.all(policy[~mask] == 0.0)
    assert int(item["wdl"]) == 1
    assert float(item["q"]) == 0.0


@pytest.mark.parametrize(
    "label,uci,index",
    [
        ("startpos", "e2e4", 796),
        ("underpromotion", "b7a8n", 4105),
        ("underpromotion", "b7b8r", 4110),
        ("underpromotion", "b7c8q", 3194),
        ("black_check", "g7g6", 14 * 64 + 22),  # canonical g2g3
        ("kiwipete", "e1g1", 262),
        ("kiwipete", "e1c1", 258),
    ],
)
def test_hand_computed_indices(buffer: ReplayBuffer, label: str, uci: str, index: int) -> None:
    board = board_for(label)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    assert expected_index(board, move) == index
    rec = buffer.record(LABELS.index(label))
    stored = rec["policy"][: int(rec["num_legal_moves"])]["move_idx"]
    assert index in stored.tolist()


def test_loader_batches_with_guardrails(buffer: ReplayBuffer) -> None:
    loader = make_loader(buffer, batch_size=4, shuffle=False)
    assert loader.num_workers == 0
    assert loader.pin_memory is False
    batch = next(iter(loader))
    assert batch["planes"].shape == (4, 18, 8, 8)
    assert batch["policy"].shape == (4, NUM_MOVES)
    assert batch["legal_mask"].shape == (4, NUM_MOVES)
    assert batch["wdl"].shape == (4,)


def test_reader_rejects_truncated_file(vectors_path: Path, tmp_path: Path) -> None:
    bad = tmp_path / "truncated.bin"
    bad.write_bytes(vectors_path.read_bytes()[: SAMPLE_BYTES - 1])
    with pytest.raises(ValueError):
        open_samples(bad)
    (tmp_path / "empty.bin").write_bytes(b"")
    with pytest.raises(ValueError):
        ReplayBuffer(tmp_path / "empty.bin")


def test_policy_decode_rejects_corrupt_count(vectors_path: Path) -> None:
    rec = np.frombuffer(vectors_path.read_bytes()[:SAMPLE_BYTES], dtype=SAMPLE_DTYPE).copy()[0]
    rec["num_legal_moves"] = POLICY_SLOTS + 1
    with pytest.raises(ValueError):
        unpack_policy(rec)

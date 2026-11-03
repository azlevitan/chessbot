"""GATE 5 (Python half): `engine convert-pgn` output matches an independent oracle.

The fixture PGN is replayed with python-chess; every Rust-written sample must
match the test-only oracle (`oracle.py`) for planes, legal mask, played-move
index and WDL label. No encoding happens outside `training/tests/` (Law 1).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest

from data.replay_buffer import POLICY_SLOTS, SAMPLE_BYTES, ReplayBuffer, unpack_planes, unpack_policy
from oracle import expected_index, expected_indices, expected_planes

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"
WDL = {"win": 2, "draw": 1, "loss": 0}


def fixture_positions() -> list[tuple[str, chess.Board, chess.Move, int]]:
    """(game label, board before the move, played move, expected WDL) per mainline ply."""
    out = []
    with FIXTURE.open() as f:
        while (game := chess.pgn.read_game(f)) is not None:
            result = game.headers["Result"]
            board = game.board()
            for move in game.mainline_moves():
                if result == "1/2-1/2":
                    wdl = WDL["draw"]
                else:
                    white_won = result == "1-0"
                    wdl = WDL["win"] if white_won == (board.turn == chess.WHITE) else WDL["loss"]
                out.append((game.headers["Event"], board.copy(), move, wdl))
                board.push(move)
    return out


POSITIONS = fixture_positions()


@pytest.fixture(scope="module")
def converted(engine_bin: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pgn") / "sample.bin"
    proc = subprocess.run(
        [str(engine_bin), "convert-pgn", "--input", str(FIXTURE), "--out", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "converted 3, skipped 0" in proc.stdout, proc.stdout + proc.stderr
    return out


@pytest.fixture(scope="module")
def buffer(converted: Path) -> ReplayBuffer:
    return ReplayBuffer(converted)


def test_fixture_covers_special_moves() -> None:
    moves = [(b, m) for _, b, m, _ in POSITIONS]
    assert any(b.is_en_passant(m) for b, m in moves)
    assert any(b.is_kingside_castling(m) for b, m in moves)
    assert any(b.is_queenside_castling(m) for b, m in moves)
    assert any(m.promotion == chess.QUEEN for _, m in moves)
    assert any(m.promotion == chess.KNIGHT for _, m in moves)
    assert any(b.turn == chess.BLACK and w == WDL["win"] for _, b, _, w in POSITIONS)


def test_sample_count(converted: Path, buffer: ReplayBuffer) -> None:
    assert len(buffer) == len(POSITIONS) == 60
    assert converted.stat().st_size == len(POSITIONS) * SAMPLE_BYTES


@pytest.mark.parametrize("i", range(len(POSITIONS)))
def test_sample_matches_oracle(buffer: ReplayBuffer, i: int) -> None:
    label, board, move, wdl = POSITIONS[i]
    where = f"{label} ply {i}: {board.fen()} {move.uci()}"
    rec = buffer.record(i)

    assert np.array_equal(unpack_planes(rec), expected_planes(board)), where
    assert int(rec["outcome_wdl"]) == wdl, where
    assert float(rec["q_value"]) == float(wdl - 1), where

    dense, mask = unpack_policy(rec)
    legal = expected_indices(board)
    active = set(np.flatnonzero(mask).tolist())
    played = expected_index(board, move)
    if len(legal) <= POLICY_SLOTS:
        assert active == legal, where
    else:
        assert len(active) == POLICY_SLOTS and active <= legal and played in active, where
    assert np.flatnonzero(dense).tolist() == [played], where
    assert dense[played] == 1.0, where


def test_items_load_as_tensors(buffer: ReplayBuffer) -> None:
    item = buffer[0]
    assert tuple(item["planes"].shape) == (18, 8, 8)

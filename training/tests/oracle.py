"""TEST-ONLY independent oracle for the SPEC.md encoding, built on python-chess.

This deliberately re-implements SPEC §1/§2 so the contract test can catch bugs
in either language. It must NEVER be imported outside `training/tests/`
(CLAUDE.md Law 1: production encoding lives only in Rust).
"""

from __future__ import annotations

import chess
import numpy as np

# Same order as `TEST_FENS` in engine/src/cli/dump_vectors.rs.
TEST_FENS: list[tuple[str, str]] = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("en_passant", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    ("underpromotion", "n1n5/1P6/8/8/8/8/8/K6k w - - 0 1"),
    ("black_check", "rnbqkbnr/ppppp1pp/5p2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2"),
]

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
UNDERPROMO_IDX = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def canon(square: int, turn: chess.Color) -> int:
    """Square from the side to move's perspective (vertical mirror for Black)."""
    return square if turn == chess.WHITE else chess.square_mirror(square)


def castling_rights(board: chess.Board) -> list[bool]:
    """[us K, us Q, them K, them Q] castling *rights*."""
    us, them = board.turn, not board.turn
    return [
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(them),
        board.has_queenside_castling_rights(them),
    ]


def canonical_ep(board: chess.Board) -> int | None:
    return None if board.ep_square is None else canon(board.ep_square, board.turn)


def expected_planes(board: chess.Board) -> np.ndarray:
    """SPEC §2 planes, float32[18, 8, 8]."""
    planes = np.zeros((18, 64), dtype=np.float32)
    us, them = board.turn, not board.turn
    for i, piece in enumerate(PIECE_ORDER):
        for sq in board.pieces(piece, us):
            planes[i, canon(sq, us)] = 1.0
        for sq in board.pieces(piece, them):
            planes[6 + i, canon(sq, us)] = 1.0
    for i, has_right in enumerate(castling_rights(board)):
        if has_right:
            planes[12 + i] = 1.0
    ep = canonical_ep(board)
    if ep is not None:
        planes[16, ep] = 1.0
    planes[17] = min(board.halfmove_clock, 100) / 100.0
    return planes.reshape(18, 8, 8)


def expected_index(board: chess.Board, move: chess.Move) -> int:
    """SPEC §1 move index. python-chess UCI castling is already e1g1 / e1c1."""
    f = canon(move.from_square, board.turn)
    t = canon(move.to_square, board.turn)
    if move.promotion in (None, chess.QUEEN):
        return f * 64 + t
    assert 48 <= f < 56, "underpromotion must start on canonical rank 7"
    from_file = f % 8
    delta = (t % 8) - from_file + 1
    return 4096 + from_file * 9 + delta * 3 + UNDERPROMO_IDX[move.promotion]


def expected_indices(board: chess.Board) -> set[int]:
    return {expected_index(board, m) for m in board.legal_moves}

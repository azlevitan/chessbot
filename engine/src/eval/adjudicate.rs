//! Match termination, with material adjudication only at the ply cap (SPEC §9).

use cozy_chess::{Board, Color, GameStatus, Piece};

use crate::cli::bin_writer::GameResult;

/// Check the current position before choosing a move. `history` includes the
/// initial position and every resulting position, including `board` itself.
/// Opening plies count toward `plies`. Natural endings take precedence over
/// the cap; matches never resign.
pub fn adjudicate(
    board: &Board,
    history: &[u64],
    plies: u32,
    max_plies: u32,
) -> Option<(GameResult, &'static str)> {
    match board.status() {
        GameStatus::Won => {
            return Some((
                GameResult::win_for(!board.side_to_move()),
                "checkmate",
            ));
        }
        GameStatus::Drawn => return Some((GameResult::Draw, "rule_draw")),
        GameStatus::Ongoing => {}
    }
    if history.iter().filter(|&&hash| hash == board.hash()).count() >= 3 {
        return Some((GameResult::Draw, "repetition"));
    }
    let pieces = board.occupied().len();
    let minors = board.pieces(Piece::Knight).len() + board.pieces(Piece::Bishop).len();
    if pieces == 2 || (pieces == 3 && minors == 1) {
        return Some((GameResult::Draw, "insufficient_material"));
    }
    if plies < max_plies {
        return None;
    }
    let difference: i32 = [
        (Piece::Pawn, 1),
        (Piece::Knight, 3),
        (Piece::Bishop, 3),
        (Piece::Rook, 5),
        (Piece::Queen, 9),
    ]
    .into_iter()
    .map(|(piece, weight)| {
        let white = (board.pieces(piece) & board.colors(Color::White)).len() as i32;
        let black = (board.pieces(piece) & board.colors(Color::Black)).len() as i32;
        weight * (white - black)
    })
    .sum();
    let result = match difference {
        5.. => GameResult::WhiteWins,
        ..=-5 => GameResult::BlackWins,
        _ => GameResult::Draw,
    };
    Some((result, "ply_cap"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::from_fen;
    use crate::uci::notation::move_from_uci;

    #[test]
    fn cap_material_boundaries_are_independent_of_side_to_move() {
        for (position, expected) in [
            ("4k3/8/8/8/8/8/8/R3K3", GameResult::WhiteWins),
            ("r3k3/8/8/8/8/8/8/4K3", GameResult::BlackWins),
            ("4k3/7p/8/8/8/8/8/R3K3", GameResult::Draw),
            ("r3k3/8/8/8/8/8/7P/4K3", GameResult::Draw),
            ("4k3/8/8/8/8/8/8/Q3K3", GameResult::WhiteWins),
            ("q3k3/8/8/8/8/8/8/4K3", GameResult::BlackWins),
            ("r3k3/8/8/8/8/8/8/Q3K3", GameResult::Draw),
            ("q3k3/8/8/8/8/8/8/R3K3", GameResult::Draw),
        ] {
            for side in ["w", "b"] {
                let board = from_fen(&format!("{position} {side} - - 0 1")).unwrap();
                let history = [board.hash()];
                assert_eq!(adjudicate(&board, &history, 19, 20), None);
                assert_eq!(
                    adjudicate(&board, &history, 20, 20),
                    Some((expected, "ply_cap")),
                    "{position} {side}"
                );
            }
        }
    }

    #[test]
    fn all_non_king_piece_weights_contribute_to_material() {
        for (fen, expected) in [
            ("4k3/7p/8/8/8/8/8/2BNK3 w - - 0 1", GameResult::WhiteWins),
            ("2bnk3/8/8/8/8/8/7P/4K3 b - - 0 1", GameResult::BlackWins),
            ("4k3/7p/8/8/8/8/8/2B1K3 w - - 0 1", GameResult::Draw),
        ] {
            let board = from_fen(fen).unwrap();
            assert_eq!(
                adjudicate(&board, &[board.hash()], 20, 20),
                Some((expected, "ply_cap"))
            );
        }
    }

    #[test]
    fn mate_and_rule_draws_precede_the_cap_and_repetition() {
        for (fen, expected, reason) in [
            ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", GameResult::WhiteWins, "checkmate"),
            ("7K/6q1/6k1/8/8/8/8/8 w - - 0 1", GameResult::BlackWins, "checkmate"),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", GameResult::Draw, "rule_draw"),
            ("4k3/8/8/8/8/8/8/Q3K3 w - - 100 1", GameResult::Draw, "rule_draw"),
        ] {
            let board = from_fen(fen).unwrap();
            assert_eq!(
                adjudicate(&board, &[board.hash(); 3], 20, 20),
                Some((expected, reason))
            );
        }
    }

    #[test]
    fn actual_game_repetition_requires_the_third_occurrence() {
        let mut board = Board::default();
        let mut history = vec![board.hash()];
        for cycle in 0..2 {
            for text in ["g1f3", "g8f6", "f3g1", "f6g8"] {
                board.play(move_from_uci(&board, text).unwrap());
                history.push(board.hash());
            }
            let expected = (cycle == 1).then_some((GameResult::Draw, "repetition"));
            assert_eq!(adjudicate(&board, &history, 8, 20), expected);
        }
        assert_eq!(
            adjudicate(&board, &history, 20, 20),
            Some((GameResult::Draw, "repetition"))
        );
    }

    #[test]
    fn only_the_specified_insufficient_material_cases_end_early() {
        for position in [
            "4k3/8/8/8/8/8/8/4K3",
            "4k3/8/8/8/8/8/8/2B1K3",
            "4k3/8/8/8/8/8/8/2N1K3",
            "2b1k3/8/8/8/8/8/8/4K3",
            "2n1k3/8/8/8/8/8/8/4K3",
        ] {
            let board = from_fen(&format!("{position} w - - 0 1")).unwrap();
            assert_eq!(
                adjudicate(&board, &[board.hash()], 20, 20),
                Some((GameResult::Draw, "insufficient_material"))
            );
        }
        for position in [
            "4k3/8/8/8/8/8/8/2NNK3",
            "2b1k3/8/8/8/8/8/8/2B1K3",
            "4k3/8/8/8/8/8/7P/4K3",
        ] {
            let board = from_fen(&format!("{position} w - - 0 1")).unwrap();
            assert_eq!(adjudicate(&board, &[board.hash()], 19, 20), None);
        }
    }
}

//! Embedded, legality-checked opening sequences for paired evaluation games.

use std::collections::HashSet;

use anyhow::{ensure, Context, Result};
use cozy_chess::{Board, Move};

use crate::uci::notation::move_from_uci;

const BOOK: &str = include_str!("../../data/openings.txt");

/// Fifty distinct openings, each containing 2–8 plies from the start position.
/// A malformed embedded book is a source-asset error and fails with context.
pub fn openings() -> Vec<Vec<Move>> {
    parse_book(BOOK).expect("invalid embedded engine/data/openings.txt")
}

fn parse_book(text: &str) -> Result<Vec<Vec<Move>>> {
    let lines: Vec<_> = text.lines().collect();
    ensure!(
        lines.len() == 50,
        "opening book must contain exactly 50 lines"
    );
    let mut seen = HashSet::new();
    lines
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            let number = index + 1;
            let tokens: Vec<_> = line.split_whitespace().collect();
            ensure!(
                (2..=8).contains(&tokens.len()),
                "opening line {number} must contain 2–8 plies"
            );
            ensure!(
                seen.insert(tokens.clone()),
                "duplicate opening at line {number}"
            );
            let mut board = Board::default();
            tokens
                .into_iter()
                .enumerate()
                .map(|(ply, token)| {
                    let mv = move_from_uci(&board, token).with_context(|| {
                        format!("opening line {number}, ply {}: {token}", ply + 1)
                    })?;
                    board.play_unchecked(mv);
                    Ok(mv)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_contains_legal_castling_in_standard_uci_form() {
        let book = openings();
        let mut has_castling = false;
        for opening in book {
            let mut board = Board::default();
            for mv in opening {
                assert!(board.is_legal(mv));
                has_castling |= board.color_on(mv.to) == Some(board.side_to_move());
                board.play_unchecked(mv);
            }
        }
        assert!(has_castling);
    }

    #[test]
    fn invalid_assets_report_count_duplicates_lengths_and_illegal_moves() {
        assert!(parse_book("e2e4 e7e5\n").is_err());
        let duplicate = "e2e4 e7e5\n".repeat(50);
        assert!(parse_book(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
        for replacement in ["e2e4", "e2e5 e7e5"] {
            let mut lines: Vec<_> = BOOK.lines().collect();
            lines[0] = replacement;
            assert!(parse_book(&lines.join("\n")).is_err());
        }
    }
}

//! UCI command loop: uci, isready, ucinewgame, position, go, quit.

use std::io::{BufRead, Write};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use cozy_chess::Board;

use super::notation::{move_from_uci, move_to_uci};
use crate::board::from_fen;
use crate::mcts::arena::MAX_NODES;
use crate::mcts::{search, SearchLimits, SearchResult};
use crate::nn::Evaluator;

pub struct UciEngine {
    board: Board,
    evaluator: Box<dyn Evaluator>,
}

impl UciEngine {
    pub fn new(evaluator: Box<dyn Evaluator>) -> Self {
        Self { board: Board::default(), evaluator }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Handle one command line. Returns `Ok(false)` on `quit`.
    pub fn handle(&mut self, line: &str, out: &mut impl Write) -> Result<bool> {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first().copied() {
            None => {}
            Some("uci") => {
                writeln!(out, "id name chessbot")?;
                writeln!(out, "id author azlevitan")?;
                writeln!(out, "uciok")?;
            }
            Some("isready") => writeln!(out, "readyok")?,
            Some("ucinewgame") => self.board = Board::default(),
            Some("position") => self.board = parse_position(&tokens[1..])?,
            Some("go") => self.go(&tokens[1..], out)?,
            Some("quit") => return Ok(false),
            Some(other) => writeln!(out, "info string unknown command: {other}")?,
        }
        out.flush()?;
        Ok(true)
    }

    fn go(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let limits = parse_go(args)?;
        let result = search(&self.board, self.evaluator.as_mut(), limits, MAX_NODES)?;
        writeln!(out, "{}", info_line(&self.board, &result))?;
        let best = result.best_move.map_or_else(|| "0000".to_string(), |m| move_to_uci(&self.board, m));
        writeln!(out, "bestmove {best}")?;
        Ok(())
    }
}

/// `startpos | fen <6 fields>` followed by optional `moves <uci>...`.
pub fn parse_position(args: &[&str]) -> Result<Board> {
    let moves_at = args.iter().position(|&t| t == "moves").unwrap_or(args.len());
    let mut board = match args.first().copied() {
        Some("startpos") => Board::default(),
        Some("fen") => from_fen(&args[1..moves_at].join(" "))?,
        _ => bail!("position needs `startpos` or `fen`"),
    };
    for &text in args.iter().skip(moves_at + 1) {
        let mv = move_from_uci(&board, text)?;
        board.play_unchecked(mv);
    }
    Ok(board)
}

/// Supports `nodes N` and `movetime MS`; other `go` arguments are ignored.
pub fn parse_go(args: &[&str]) -> Result<SearchLimits> {
    let mut limits = SearchLimits::default();
    let mut it = args.iter();
    while let Some(&key) = it.next() {
        match key {
            "nodes" => limits.nodes = Some(next_num(&mut it, key)?),
            "movetime" => limits.movetime = Some(Duration::from_millis(next_num(&mut it, key)?)),
            _ => {}
        }
    }
    Ok(limits)
}

fn next_num<T: std::str::FromStr>(it: &mut std::slice::Iter<&str>, key: &str) -> Result<T> {
    it.next()
        .and_then(|v| v.parse().ok())
        .with_context(|| format!("`go {key}` needs a non-negative integer"))
}

fn info_line(board: &Board, r: &SearchResult) -> String {
    let mates = r.best_move.is_some_and(|m| r.root_children.iter().any(|c| c.mv == m && c.mates));
    let score = if mates {
        "mate 1".to_string()
    } else {
        // Leela-style Q -> centipawn mapping, clamped away from +/-1.
        let q = r.root_q.clamp(-0.999, 0.999);
        format!("cp {}", (111.714_64 * (1.562_068_8 * q).tan()).round() as i32)
    };
    let mut pv_board = board.clone();
    let pv: Vec<String> = r
        .pv
        .iter()
        .map(|&m| {
            let s = move_to_uci(&pv_board, m);
            pv_board.play_unchecked(m);
            s
        })
        .collect();
    format!("info nodes {} score {score} pv {}", r.simulations, pv.join(" "))
}

/// Run the loop until `quit` or EOF. Errors are reported and the loop continues.
pub fn run(evaluator: Box<dyn Evaluator>, input: impl BufRead, mut out: impl Write) -> Result<()> {
    let mut engine = UciEngine::new(evaluator);
    for line in input.lines() {
        match engine.handle(&line?, &mut out) {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => {
                writeln!(out, "info string error: {e:#}")?;
                out.flush()?;
            }
        }
    }
    Ok(())
}

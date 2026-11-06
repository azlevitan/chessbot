//! Sequential, colour-paired evaluation games and deterministic match reports.

use anyhow::{ensure, Context, Result};
use cozy_chess::{Board, Color, Move};
use serde::Serialize;

use super::adjudicate::adjudicate;
use super::players::Player;
use crate::cli::bin_writer::GameResult;
use crate::nn::infer_opts::InferOptions;
use crate::uci::notation::move_to_uci;

pub struct MatchConfig {
    pub a: String,
    pub b: String,
    pub games: u32,
    pub sims: u32,
    pub seed: u64,
    pub opening_offset: usize,
    pub max_plies: u32,
    pub batch: usize,
    pub infer: InferOptions,
}

impl MatchConfig {
    /// Validate before constructing players or opening any output files.
    pub fn validate(&self, book: &[Vec<Move>]) -> Result<()> {
        ensure!(self.games > 0 && self.games % 2 == 0, "--games must be positive and even");
        ensure!(self.sims >= 2, "--sims must be at least 2");
        ensure!(self.batch > 0, "--batch must be positive");
        ensure!(self.max_plies > 0, "--max-plies must be positive");
        ensure!(!self.a.is_empty() && !self.b.is_empty(), "player specs must not be empty");
        self.infer.validate()?;
        self.seed
            .checked_add(u64::from(self.games - 1))
            .context("--seed + game index overflows u64")?;
        ensure!(book.len() == 50, "opening book must contain exactly 50 lines");
        for pair in 0..(self.games / 2).min(50) {
            let index = opening_index(self.opening_offset, pair);
            ensure!(
                (2..=8).contains(&book[index].len()),
                "opening {index} must contain 2–8 plies"
            );
            ensure!(
                book[index].len() <= self.max_plies as usize,
                "--max-plies is shorter than opening {index}"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct MatchGame {
    pub game: u32,
    pub opening: usize,
    pub white: &'static str,
    pub result: &'static str,
    pub termination: &'static str,
    pub plies: u32,
    /// Retained only for the optional moves log; never included in match JSON.
    #[serde(skip)]
    pub moves: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct MatchReport {
    pub format: &'static str,
    pub a: String,
    pub b: String,
    pub games: u32,
    pub a_wins: u32,
    pub b_wins: u32,
    pub draws: u32,
    pub a_score: f64,
    pub sims: u32,
    pub seed: u64,
    pub opening_offset: usize,
    pub max_plies: u32,
    pub results: Vec<MatchGame>,
}

fn opening_index(offset: usize, pair: u32) -> usize {
    (offset % 50 + pair as usize % 50) % 50
}

fn play_game(cfg: &MatchConfig, book: &[Vec<Move>], game: u32) -> Result<MatchGame> {
    let seed = cfg.seed + u64::from(game); // Checked by MatchConfig::validate.
    // Each invocation drops both players before the next game loads its models.
    let mut players = [
        Player::from_spec(&cfg.a, cfg.sims, cfg.batch, &cfg.infer, seed)
            .context("constructing player A")?,
        Player::from_spec(&cfg.b, cfg.sims, cfg.batch, &cfg.infer, seed)
            .context("constructing player B")?,
    ];
    let a_white = game % 2 == 0;
    let opening = opening_index(cfg.opening_offset, game / 2);
    let mut board = Board::default();
    let mut history = vec![board.hash()];
    let mut moves = Vec::new();
    loop {
        let plies = moves.len() as u32;
        if let Some((result, termination)) = adjudicate(&board, &history, plies, cfg.max_plies) {
            return Ok(MatchGame {
                game,
                opening,
                white: if a_white { "a" } else { "b" },
                result: match result {
                    GameResult::WhiteWins => "1-0",
                    GameResult::BlackWins => "0-1",
                    GameResult::Draw => "1/2-1/2",
                },
                termination,
                plies,
                moves,
            });
        }
        let mv = match book[opening].get(moves.len()) {
            Some(&mv) => mv,
            None => {
                let a_turn = (board.side_to_move() == Color::White) == a_white;
                players[usize::from(!a_turn)].choose(&board, &history)?
            }
        };
        ensure!(board.is_legal(mv), "game {game}, ply {plies}: illegal move {mv}");
        // Castling must be rendered against the pre-move board in UCI notation.
        moves.push(move_to_uci(&board, mv));
        for player in &mut players {
            player.observe(mv)?;
        }
        board.play_unchecked(mv);
        history.push(board.hash());
    }
}

/// Complete pairs sequentially, preserving at most two trees and evaluators.
pub fn run_match(cfg: &MatchConfig, book: &[Vec<Move>]) -> Result<MatchReport> {
    cfg.validate(book)?;
    let mut report = MatchReport {
        format: "chessbot-match-v1",
        a: cfg.a.clone(),
        b: cfg.b.clone(),
        games: cfg.games,
        a_wins: 0,
        b_wins: 0,
        draws: 0,
        a_score: 0.0,
        sims: cfg.sims,
        seed: cfg.seed,
        opening_offset: cfg.opening_offset,
        max_plies: cfg.max_plies,
        results: Vec::new(),
    };
    for game in 0..cfg.games {
        let result = play_game(cfg, book, game).with_context(|| format!("match game {game}"))?;
        match (result.result, result.white) {
            ("1/2-1/2", _) => report.draws += 1,
            ("1-0", "a") | ("0-1", "b") => report.a_wins += 1,
            _ => report.b_wins += 1,
        }
        report.results.push(result);
    }
    report.a_score = (f64::from(report.a_wins) + 0.5 * f64::from(report.draws))
        / f64::from(cfg.games);
    Ok(report)
}

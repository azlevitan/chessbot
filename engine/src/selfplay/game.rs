//! One self-play game: MCTS with root Dirichlet noise at every move,
//! visit-proportional move sampling for the opening plies, greedy afterwards.

use anyhow::{ensure, Result};
use cozy_chess::{Board, GameStatus, Move, Piece};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;

use crate::cli::bin_writer::GameResult;
use crate::mcts::arena::MAX_NODES;
use crate::mcts::{search_with_noise, RootNoise, SearchLimits};
use crate::nn::Evaluator;

#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    /// MCTS simulations per move (>= 2: the first only expands the root).
    pub sims: u32,
    /// Game is adjudicated a draw after this many plies.
    pub max_plies: u32,
    /// Plies played by sampling from visit counts (temperature 1).
    pub temp_plies: u32,
    pub noise: RootNoise,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self { sims: 200, max_plies: 512, temp_plies: 30, noise: RootNoise::default() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    Checkmate,
    /// Stalemate or the 50-move rule (cozy-chess `GameStatus::Drawn`).
    RuleDraw,
    Repetition,
    InsufficientMaterial,
    PlyCap,
}

/// A searched position: the board, its visit-share policy over every legal
/// move (zeros included) and the root value for the side to move.
#[derive(Clone, Debug)]
pub struct PlyRecord {
    pub board: Board,
    pub policy: Vec<(Move, f32)>,
    pub root_q: f32,
    pub played: Move,
}

#[derive(Clone, Debug)]
pub struct GameRecord {
    pub plies: Vec<PlyRecord>,
    pub result: GameResult,
    pub termination: Termination,
}

/// Bare kings, or king + one minor piece vs king.
fn insufficient_material(board: &Board) -> bool {
    let occupied = board.occupied().len();
    let minors = board.pieces(Piece::Knight).len() + board.pieces(Piece::Bishop).len();
    occupied == 2 || (occupied == 3 && minors == 1)
}

/// Game-over check before searching `board`; `history` holds every position hash so far.
fn game_over(board: &Board, history: &[u64], ply: u32, cfg: &SelfPlayConfig) -> Option<(GameResult, Termination)> {
    match board.status() {
        GameStatus::Won => return Some((GameResult::win_for(!board.side_to_move()), Termination::Checkmate)),
        GameStatus::Drawn => return Some((GameResult::Draw, Termination::RuleDraw)),
        GameStatus::Ongoing => {}
    }
    let hash = board.hash();
    if history.iter().filter(|&&h| h == hash).count() >= 3 {
        return Some((GameResult::Draw, Termination::Repetition));
    }
    if insufficient_material(board) {
        return Some((GameResult::Draw, Termination::InsufficientMaterial));
    }
    (ply >= cfg.max_plies).then_some((GameResult::Draw, Termination::PlyCap))
}

/// Play one game from the standard start position.
pub fn play_game(eval: &mut dyn Evaluator, cfg: &SelfPlayConfig, rng: &mut StdRng) -> Result<GameRecord> {
    ensure!(cfg.sims >= 2, "selfplay needs at least 2 simulations per move");
    let limits = SearchLimits {
        nodes: Some(cfg.sims),
        movetime: None,
        ..SearchLimits::default()
    };
    let mut board = Board::default();
    let mut history = vec![board.hash()];
    let mut plies = Vec::new();
    loop {
        if let Some((result, termination)) = game_over(&board, &history, plies.len() as u32, cfg) {
            return Ok(GameRecord { plies, result, termination });
        }
        let r = search_with_noise(&board, eval, limits, MAX_NODES, Some((&cfg.noise, &mut *rng)))?;
        let visits: Vec<u32> = r.root_children.iter().map(|c| c.visits).collect();
        let total: u32 = visits.iter().sum();
        ensure!(total > 0, "search produced no root visits");
        let policy: Vec<(Move, f32)> = r
            .root_children
            .iter()
            .map(|c| (c.mv, c.visits as f32 / total as f32))
            .collect();
        let played = if (plies.len() as u32) < cfg.temp_plies {
            let dist = WeightedIndex::new(&visits).expect("total visits > 0");
            r.root_children[dist.sample(rng)].mv
        } else {
            r.best_move.expect("ongoing position has a best move")
        };
        plies.push(PlyRecord { board: board.clone(), policy, root_q: r.root_q, played });
        board.play_unchecked(played);
        history.push(board.hash());
    }
}

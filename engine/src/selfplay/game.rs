//! One self-play game: MCTS with root Dirichlet noise at every move,
//! visit-proportional move sampling for the opening plies, greedy afterwards.

use anyhow::{ensure, Result};
use cozy_chess::{Board, Color, GameStatus, Move, Piece};
use rand::rngs::StdRng;

use crate::cli::bin_writer::GameResult;
use crate::mcts::RootNoise;
use crate::nn::Evaluator;

#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    /// New MCTS simulations per move (>= 2: a fresh root needs expansion).
    pub sims: u32,
    /// Game is adjudicated a draw after this many plies.
    pub max_plies: u32,
    /// Plies played by sampling from visit counts (temperature 1).
    pub temp_plies: u32,
    pub noise: RootNoise,
    /// Retain the played child's subtree between moves.
    pub reuse: bool,
    /// Resign below this Q; values <= -1 disable resignation.
    pub resign_threshold: f32,
    /// Required low-Q searches on a side's own consecutive turns.
    pub resign_consecutive: u32,
    /// Fraction of games that continue to audit would-be resignations.
    pub resign_playthrough: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            sims: 200,
            max_plies: 512,
            temp_plies: 30,
            noise: RootNoise::default(),
            reuse: true,
            resign_threshold: -0.95,
            resign_consecutive: 3,
            resign_playthrough: 0.1,
        }
    }
}

impl SelfPlayConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.sims >= 2, "selfplay needs at least 2 simulations per move");
        ensure!(self.resign_threshold.is_finite(), "--resign-threshold must be finite");
        ensure!(self.resign_consecutive > 0, "--resign-consecutive must be positive");
        ensure!(
            (0.0..=1.0).contains(&self.resign_playthrough),
            "--resign-playthrough must be between 0 and 1"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    Checkmate,
    Resign,
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
    /// Absent only for the final searched position when the side resigns.
    pub played: Option<Move>,
}

#[derive(Clone, Debug)]
pub struct GameRecord {
    pub plies: Vec<PlyRecord>,
    pub result: GameResult,
    pub termination: Termination,
    pub playthrough: bool,
    /// First side that reached the resignation streak in a playthrough game.
    pub would_resign: Option<Color>,
}

/// Bare kings, or king + one minor piece vs king.
fn insufficient_material(board: &Board) -> bool {
    let occupied = board.occupied().len();
    let minors = board.pieces(Piece::Knight).len() + board.pieces(Piece::Bishop).len();
    occupied == 2 || (occupied == 3 && minors == 1)
}

/// Game-over check before searching `board`; `history` holds every position hash so far.
pub(super) fn game_over(
    board: &Board,
    history: &[u64],
    ply: u32,
    cfg: &SelfPlayConfig,
) -> Option<(GameResult, Termination)> {
    match board.status() {
        GameStatus::Won => {
            return Some((
                GameResult::win_for(!board.side_to_move()),
                Termination::Checkmate,
            ))
        }
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
pub fn play_game(
    eval: &mut dyn Evaluator,
    cfg: &SelfPlayConfig,
    rng: &mut StdRng,
) -> Result<GameRecord> {
    let mut driver = super::driver::GameDriver::new(cfg, rng.clone())?;
    *rng = driver.rng.clone();
    while !driver.is_finished() {
        let step = super::batch_runner::run_step(eval, std::slice::from_mut(&mut driver), 1);
        *rng = driver.rng.clone();
        step?;
    }
    Ok(driver.into_record())
}

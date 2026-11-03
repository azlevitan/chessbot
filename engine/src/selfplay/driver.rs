//! One resumable self-play game; inference is supplied by the shared runner.

use anyhow::{ensure, Result};
use cozy_chess::{Board, Move};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;

use super::game::game_over;
use super::{GameRecord, PlyRecord, SelfPlayConfig, Termination};
use crate::cli::bin_writer::GameResult;
use crate::encoder::move_mapper::move_to_index;
use crate::mcts::arena::MAX_NODES;
use crate::mcts::batch::PendingBatch;
use crate::mcts::{ChildStats, SearchTree};

pub struct GameDriver {
    cfg: SelfPlayConfig,
    board: Board,
    history: Vec<u64>,
    tree: SearchTree,
    pub(super) rng: StdRng,
    plies: Vec<PlyRecord>,
    move_start: u32,
    finished: Option<(GameResult, Termination)>,
}

impl GameDriver {
    pub fn new(cfg: &SelfPlayConfig, rng: StdRng) -> Result<Self> {
        ensure!(
            cfg.sims >= 2,
            "selfplay needs at least 2 simulations per move"
        );
        let board = Board::default();
        let history = vec![board.hash()];
        Ok(Self {
            tree: SearchTree::new(board.clone(), &history, MAX_NODES),
            finished: game_over(&board, &history, 0, cfg),
            cfg: *cfg,
            board,
            history,
            rng,
            plies: Vec::new(),
            move_start: 0,
        })
    }

    pub fn is_finished(&self) -> bool {
        self.finished.is_some()
    }

    /// The guard reserves paths until results arrive, rolling back on failure.
    pub(crate) fn step_request(&mut self, leaf_batch: usize) -> Option<PendingBatch<'_>> {
        if self.is_finished() {
            return None;
        }
        let remaining = self.cfg.sims - (self.tree.simulations() - self.move_start);
        // Another game's inference can fail after our terminal-only request
        // completes the move budget. Resume at step_apply without gathering again.
        if remaining == 0 {
            return None;
        }
        self.tree.apply_root_noise(&self.cfg.noise, &mut self.rng);
        Some(PendingBatch::new(
            &mut self.tree,
            leaf_batch.min(remaining as usize),
        ))
    }

    /// Called after all pending evaluations have been applied and guards dropped.
    pub(crate) fn step_apply(&mut self) -> Result<()> {
        if self.is_finished() || self.tree.simulations() - self.move_start < self.cfg.sims {
            return Ok(());
        }
        let result = self.tree.result();
        let total: u32 = result.root_children.iter().map(|c| c.visits).sum();
        ensure!(total > 0, "search produced no root visits");
        let policy = result
            .root_children
            .iter()
            .map(|c| (c.mv, c.visits as f32 / total as f32))
            .collect();
        let played = choose_move(
            &self.board,
            &result.root_children,
            self.plies.len() as u32,
            self.cfg.temp_plies,
            &mut self.rng,
        );
        self.plies.push(PlyRecord {
            board: self.board.clone(),
            policy,
            root_q: result.root_q,
            played,
        });
        self.board.play_unchecked(played);
        self.history.push(self.board.hash());
        self.finished = game_over(
            &self.board,
            &self.history,
            self.plies.len() as u32,
            &self.cfg,
        );
        if !self.is_finished() {
            if self.cfg.reuse {
                self.tree.advance(played, self.board.clone(), &self.history);
            } else {
                self.tree = SearchTree::new(self.board.clone(), &self.history, MAX_NODES);
            }
            self.move_start = self.tree.simulations();
        }
        Ok(())
    }

    pub fn into_record(self) -> GameRecord {
        let (result, termination) = self.finished.expect("finish the game before emitting it");
        GameRecord {
            plies: self.plies,
            result,
            termination,
        }
    }
}

fn choose_move(
    board: &Board,
    children: &[ChildStats],
    ply: u32,
    temp_plies: u32,
    rng: &mut StdRng,
) -> Move {
    if ply < temp_plies {
        let dist =
            WeightedIndex::new(children.iter().map(|c| c.visits)).expect("positive root visits");
        children[dist.sample(rng)].mv
    } else {
        children
            .iter()
            .max_by(|a, b| {
                a.visits
                    .cmp(&b.visits)
                    .then_with(|| a.prior.total_cmp(&b.prior))
                    .then_with(|| move_to_index(board, b.mv).cmp(&move_to_index(board, a.mv)))
            })
            .expect("ongoing position has children")
            .mv
    }
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;

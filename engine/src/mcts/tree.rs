//! Resumable MCTS. Evaluation happens outside the tree.

use std::collections::HashSet;

use cozy_chess::{Board, GameStatus, Move};
use rand::RngCore;

use super::arena::{Arena, NodeState};
use super::noise::RootNoise;
use super::result::{build_result, SearchResult};
use super::search::C_PUCT;

/// A selected leaf's value is always from its own side to move's perspective.
#[derive(Debug)]
pub enum Leaf {
    Terminal(f32),
    NeedsEval { node: usize, board: Board },
}

/// Owns one arena and one active trajectory. The batch driver reserves paths
/// separately while their evaluations are pending.
pub struct SearchTree {
    pub(super) root: Board,
    pub(super) arena: Arena,
    pub(super) history_hashes: HashSet<u64>,
    pub(super) path: Vec<usize>,
    pub(super) path_hashes: Vec<u64>,
    pub(super) simulations: u32,
    pub(super) arena_full: bool,
    pub(super) root_noise_applied: bool,
}

impl SearchTree {
    /// History may include the root hash; repetition is only checked below it.
    pub fn new(root: Board, history_hashes: &[u64], capacity: usize) -> Self {
        Self {
            root,
            arena: Arena::new(capacity),
            history_hashes: history_hashes.iter().copied().collect(),
            path: Vec::with_capacity(64),
            path_hashes: Vec::with_capacity(64),
            simulations: 0,
            arena_full: false,
            root_noise_applied: false,
        }
    }

    /// The outstanding trajectory, including root and leaf.
    pub fn selected_path(&self) -> &[usize] {
        &self.path
    }

    /// Select without evaluating or counting a simulation. Finish the outstanding
    /// selection with a backup before selecting again.
    pub fn select_leaf(&mut self) -> Leaf {
        assert!(self.path.is_empty(), "back up the outstanding leaf first");
        let mut board = self.root.clone();
        let mut node = 0;
        self.path.push(node);
        loop {
            let hash = board.hash();
            if node != 0
                && (self.history_hashes.contains(&hash) || self.path_hashes.contains(&hash))
            {
                self.arena[node].state = NodeState::Terminal(0.0);
            }
            self.path_hashes.push(hash);
            if let NodeState::Terminal(v) = self.arena[node].state {
                return Leaf::Terminal(v);
            }
            if self.arena[node].state == NodeState::Unexpanded {
                let terminal = match board.status() {
                    GameStatus::Won => Some(-1.0),
                    GameStatus::Drawn => Some(0.0),
                    GameStatus::Ongoing => None,
                };
                if let Some(v) = terminal {
                    self.arena[node].state = NodeState::Terminal(v);
                    return Leaf::Terminal(v);
                }
                return Leaf::NeedsEval { node, board };
            }
            node = self.select_child(node);
            board.play_unchecked(self.arena[node].mv.expect("non-root node has a move"));
            self.path.push(node);
        }
    }

    fn select_child(&self, parent: usize) -> usize {
        let sqrt_n = (self.arena[parent].visits as f32).sqrt();
        let fpu_q = self.fpu_q(parent);
        let mut best = (f32::NEG_INFINITY, parent);
        for c in self.arena[parent].children() {
            let child = &self.arena[c];
            let q = if child.visits == 0 { fpu_q } else { child.q() };
            let score = q + C_PUCT * child.prior * sqrt_n / (1.0 + child.visits as f32);
            if score > best.0 {
                best = (score, c);
            }
        }
        best.1
    }

    fn fpu_q(&self, parent: usize) -> f32 {
        // The edge into s is scored for s's opponent. Negate before reducing.
        if parent == 0 {
            0.0
        } else {
            (-self.arena[parent].q() - 0.25).clamp(-1.0, 1.0)
        }
    }

    pub fn expand_and_backup(&mut self, node: usize, priors: &[(Move, f32)], value: f32) {
        assert_eq!(
            self.path.last(),
            Some(&node),
            "evaluation must match the selected leaf"
        );
        assert_eq!(self.arena[node].state, NodeState::Unexpanded);
        if !self.arena.expand(node, priors) {
            self.arena_full = true;
        }
        self.backup(value);
    }

    pub fn backup_terminal(&mut self, path: &[usize], value: f32) {
        assert!(!path.is_empty(), "select a terminal leaf first");
        assert_eq!(path, self.path, "backup must match the selected trajectory");
        assert_eq!(
            self.arena[*path.last().unwrap()].state,
            NodeState::Terminal(value)
        );
        self.backup(value);
    }

    fn backup(&mut self, value: f32) {
        // Each edge stores the value for the player who chose it. The root uses
        // the same convention and result() negates its stored mean.
        let mut v = -value;
        for &idx in self.path.iter().rev() {
            self.arena[idx].visits += 1;
            self.arena[idx].value_sum += v;
            v = -v;
        }
        self.simulations += 1;
        self.path.clear();
        self.path_hashes.clear();
    }

    /// Mix noise once per root, after expansion or immediately after reuse.
    pub fn apply_root_noise(&mut self, noise: &RootNoise, rng: &mut dyn RngCore) {
        if self.root_noise_applied || self.arena[0].state != NodeState::Expanded {
            return;
        }
        let children = self.arena[0].children();
        let mut priors: Vec<_> = children.clone().map(|c| self.arena[c].prior).collect();
        noise.apply(&mut priors, rng);
        for (c, p) in children.zip(priors) {
            self.arena[c].prior = p;
        }
        self.root_noise_applied = true;
    }

    /// A snapshot; it does not mutate or finish an outstanding selection.
    pub fn result(&self) -> SearchResult {
        build_result(&self.arena, self.simulations, self.arena_full)
    }
}

#[cfg(test)]
#[path = "tree_tests.rs"]
mod tests;

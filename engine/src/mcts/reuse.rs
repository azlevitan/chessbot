//! Retain the played child's subtree, compacting siblings in breadth-first order.

use std::collections::VecDeque;

use cozy_chess::{Board, GameStatus, Move};

use super::arena::{Arena, NodeState};
use super::tree::SearchTree;

impl SearchTree {
    /// Promote a matching child, or reset to `new_board` and return false.
    /// Finish the current search (including all pending evaluations) first.
    pub fn advance(&mut self, mv: Move, new_board: Board, history_hashes: &[u64]) -> bool {
        assert!(
            self.path.is_empty(),
            "finish the outstanding search before advancing"
        );
        let capacity = self.arena.capacity();
        let child = self.arena[0]
            .children()
            .find(|&i| self.arena[i].mv == Some(mv));
        let matches = child.is_some() && {
            let mut expected = self.root.clone();
            expected.play_unchecked(mv);
            expected == new_board
        };
        if !matches {
            *self = Self::new(new_board, history_hashes, capacity);
            return false;
        }

        let child = child.unwrap();
        let mut arena = Arena::new(capacity);
        arena[0] = self.arena[child];
        let mut pending = VecDeque::from([(child, 0)]);
        while let Some((old, new)) = pending.pop_front() {
            let first_child = arena.len();
            for index in self.arena[old].children() {
                let copied = arena.push_copy(self.arena[index]);
                pending.push_back((index, copied));
            }
            arena[new].first_child = if arena[new].num_children == 0 {
                0
            } else {
                first_child as u32
            };
        }
        // Root value_sum already has the correct sign: the old incoming edge
        // and a root both store the value for the opposite side to move.
        arena[0].mv = None;
        arena[0].prior = 1.0;
        // A repeated non-root position is a search draw, but a game continues
        // until its actual termination rule fires. Re-evaluate that new root.
        if arena[0].state == NodeState::Terminal(0.0) && new_board.status() == GameStatus::Ongoing {
            arena[0].state = if arena[0].num_children > 0 {
                NodeState::Expanded
            } else {
                NodeState::Unexpanded
            };
        }
        self.simulations = arena[0].visits;
        self.arena = arena;
        self.root = new_board;
        self.history_hashes = history_hashes.iter().copied().collect();
        self.path_hashes.clear();
        self.arena_full = false;
        self.root_noise_applied = false;
        true
    }
}

#[cfg(test)]
#[path = "reuse_tests.rs"]
mod tests;

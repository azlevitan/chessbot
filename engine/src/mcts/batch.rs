//! One tree's pending leaves, virtual losses, and collision/error cleanup.

use std::collections::{HashSet, VecDeque};

use anyhow::Result;
use cozy_chess::Board;

use super::tree::{Leaf, SearchTree};
use crate::nn::{evaluate_boards, BoardEvaluation, Evaluator};

/// Complete at most `budget` simulations. Terminals count toward the budget
/// without occupying evaluator slots; collisions finish the current gather.
pub(super) fn search_batch(
    tree: &mut SearchTree,
    eval: &mut dyn Evaluator,
    budget: usize,
) -> Result<u32> {
    assert!(budget > 0);
    let before = tree.simulations;
    let batch = PendingBatch::new(tree, budget);
    let outputs = evaluate_boards(eval, batch.boards())?;
    batch.complete(&mut outputs.into_iter());
    Ok(tree.simulations - before)
}

/// Dropping this guard also rolls back on inference/validation failure. Completed
/// terminal backups remain real simulations; pending selections never count.
pub(crate) struct PendingBatch<'a> {
    tree: &'a mut SearchTree,
    paths: VecDeque<Vec<usize>>,
    boards: Vec<Board>,
    nodes: HashSet<usize>,
    virtual_loss: bool,
}

impl<'a> PendingBatch<'a> {
    pub(crate) fn new(tree: &'a mut SearchTree, budget: usize) -> Self {
        assert!(budget > 0);
        let mut batch = Self {
            tree,
            paths: VecDeque::new(),
            boards: Vec::new(),
            nodes: HashSet::new(),
            // No temporary arithmetic at batch 1: preserve sequential results exactly.
            virtual_loss: budget > 1,
        };
        batch.gather(budget);
        batch
    }

    pub(crate) fn boards(&self) -> &[Board] {
        &self.boards
    }

    /// Consume this tree's prefix of an already validated combined output batch.
    pub(crate) fn complete(mut self, outputs: &mut impl Iterator<Item = BoardEvaluation>) {
        while let Some(path) = self.paths.pop_front() {
            let (priors, value) = outputs.next().expect("one output per pending path");
            self.undo(&path);
            let node = *path.last().unwrap();
            self.tree.path = path;
            self.tree.expand_and_backup(node, &priors, value);
        }
    }

    fn gather(&mut self, budget: usize) {
        let mut terminals = 0;
        while terminals + self.paths.len() < budget {
            match self.tree.select_leaf() {
                Leaf::Terminal(value) => {
                    let path = self.tree.selected_path().to_vec();
                    self.tree.backup_terminal(&path, value);
                    terminals += 1;
                }
                Leaf::NeedsEval { node, board } => {
                    let path = std::mem::take(&mut self.tree.path);
                    self.tree.path_hashes.clear();
                    if self.virtual_loss {
                        for &idx in path.iter().skip(1) {
                            self.tree.arena[idx].visits += 1;
                            self.tree.arena[idx].value_sum -= 1.0;
                        }
                    }
                    if !self.nodes.insert(node) {
                        self.undo(&path);
                        break;
                    }
                    self.paths.push_back(path);
                    self.boards.push(board);
                    // Root initialization is always a singleton and must finish
                    // before the driver adds noise or selects any child.
                    if node == 0 {
                        break;
                    }
                }
            }
        }
    }

    fn undo(&mut self, path: &[usize]) {
        if self.virtual_loss {
            for &idx in path.iter().skip(1) {
                self.tree.arena[idx].visits -= 1;
                self.tree.arena[idx].value_sum += 1.0;
            }
        }
    }
}

impl Drop for PendingBatch<'_> {
    fn drop(&mut self) {
        while let Some(path) = self.paths.pop_front() {
            self.undo(&path);
        }
        self.tree.path.clear();
        self.tree.path_hashes.clear();
    }
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;

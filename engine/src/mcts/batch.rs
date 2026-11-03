//! One tree's pending leaves, virtual losses, and collision/error cleanup.

use std::collections::{HashSet, VecDeque};

use anyhow::Result;
use cozy_chess::Board;

use super::tree::{Leaf, SearchTree};
use crate::nn::{evaluate_boards, Evaluator};

/// Complete at most `budget` simulations. Terminals count toward the budget
/// without occupying evaluator slots; collisions finish the current gather.
pub(super) fn search_batch(
    tree: &mut SearchTree,
    eval: &mut dyn Evaluator,
    budget: usize,
) -> Result<u32> {
    assert!(budget > 0);
    let before = tree.simulations;
    let mut batch = PendingBatch {
        tree,
        paths: VecDeque::new(),
        boards: Vec::new(),
        nodes: HashSet::new(),
        // No temporary arithmetic at batch 1: preserve sequential results exactly.
        virtual_loss: budget > 1,
    };
    batch.gather(budget);
    let outputs = evaluate_boards(eval, &batch.boards)?;
    for (priors, value) in outputs {
        let path = batch.paths.pop_front().expect("one path per output");
        batch.undo(&path);
        let node = *path.last().unwrap();
        batch.tree.path = path;
        batch.tree.expand_and_backup(node, &priors, value);
    }
    Ok(batch.tree.simulations - before)
}

/// Dropping this guard also rolls back on inference/validation failure. Completed
/// terminal backups remain real simulations; pending selections never count.
struct PendingBatch<'a> {
    tree: &'a mut SearchTree,
    paths: VecDeque<Vec<usize>>,
    boards: Vec<Board>,
    nodes: HashSet<usize>,
    virtual_loss: bool,
}

impl PendingBatch<'_> {
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

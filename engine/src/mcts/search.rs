//! Search limits and evaluator driver for the resumable tree.

use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Result};
use cozy_chess::Board;
use rand::RngCore;

use super::arena::MAX_NODES;
use super::batch::search_batch;
use super::noise::RootNoise;
use super::tree::SearchTree;
use crate::board::legal_moves;
use crate::nn::Evaluator;

pub const C_PUCT: f32 = 1.5;
/// Simulations used when `go` gives neither `nodes` nor `movetime`.
pub const DEFAULT_NODES: u32 = 800;

#[derive(Clone, Copy, Debug)]
pub struct SearchLimits {
    /// Maximum number of simulations.
    pub nodes: Option<u32>,
    /// Wall-clock budget, checked between batches.
    pub movetime: Option<Duration>,
    /// Maximum leaves per evaluator call. One preserves sequential search.
    pub batch: usize,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            nodes: None,
            movetime: None,
            batch: 1,
        }
    }
}

pub use super::result::{ChildStats, SearchResult};

/// Run MCTS from `root` with at most `capacity` (<= 65,536) arena nodes.
pub fn search(
    root: &Board,
    eval: &mut dyn Evaluator,
    limits: SearchLimits,
    capacity: usize,
) -> Result<SearchResult> {
    search_with_noise(root, eval, limits, capacity, None)
}

/// `search`, optionally with root Dirichlet noise for self-play exploration.
pub fn search_with_noise(
    root: &Board,
    eval: &mut dyn Evaluator,
    limits: SearchLimits,
    capacity: usize,
    mut noise: Option<(&RootNoise, &mut dyn RngCore)>,
) -> Result<SearchResult> {
    ensure!(limits.batch > 0, "search batch must be positive");
    let capacity = capacity.clamp(1, MAX_NODES);
    let mut tree = SearchTree::new(root.clone(), &[], capacity);
    let root_moves = legal_moves(root).len();
    if root_moves == 0 {
        return Ok(tree.result());
    }
    if 1 + root_moves > capacity {
        bail!("arena capacity {capacity} cannot hold the root's {root_moves} children");
    }
    let max_sims = match limits {
        SearchLimits {
            nodes: None,
            movetime: None,
            ..
        } => DEFAULT_NODES,
        SearchLimits { nodes, .. } => nodes.unwrap_or(u32::MAX),
    };
    let start = Instant::now();
    let mut sims = 0;
    while sims < max_sims
        && !limits
            .movetime
            .is_some_and(|t| sims > 0 && start.elapsed() >= t)
    {
        let budget = limits.batch.min((max_sims - sims) as usize);
        sims += search_batch(&mut tree, eval, budget)?;
        if sims == 1 {
            if let Some((n, rng)) = noise.take() {
                tree.apply_root_noise(n, rng);
            }
        }
    }
    Ok(tree.result())
}

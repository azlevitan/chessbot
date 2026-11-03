//! PUCT search over the fixed arena (SPEC.md §5).
//!
//! Values are always backed up with alternating sign: a node's `value_sum` is
//! from the perspective of the side that made the move into it.

use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use cozy_chess::{Board, GameStatus, Move};
use rand::RngCore;

use super::arena::{Arena, NodeState, MAX_NODES};
use super::noise::RootNoise;
use crate::board::legal_moves;
use crate::nn::{evaluate_board, Evaluator};

pub const C_PUCT: f32 = 1.5;
/// Simulations used when `go` gives neither `nodes` nor `movetime`.
pub const DEFAULT_NODES: u32 = 800;

#[derive(Clone, Copy, Debug, Default)]
pub struct SearchLimits {
    /// Maximum number of simulations.
    pub nodes: Option<u32>,
    /// Wall-clock budget, checked between simulations.
    pub movetime: Option<Duration>,
}

#[derive(Clone, Debug)]
pub struct ChildStats {
    pub mv: Move,
    pub prior: f32,
    pub visits: u32,
    /// Mean value from the root side to move's perspective.
    pub q: f32,
    /// The child position is checkmate (i.e. `mv` mates).
    pub mates: bool,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    /// `None` when the root has no legal moves.
    pub best_move: Option<Move>,
    /// Root value from the side to move's perspective.
    pub root_q: f32,
    pub pv: Vec<Move>,
    pub simulations: u32,
    pub nodes_used: usize,
    pub arena_full: bool,
    pub root_children: Vec<ChildStats>,
}

/// Index of the child of `parent` maximizing `Q + c_puct * P * sqrt(N) / (1 + n)`.
fn select_child(arena: &Arena, parent: usize) -> usize {
    let sqrt_n = (arena[parent].visits as f32).sqrt();
    let mut best = (f32::NEG_INFINITY, parent);
    for c in arena[parent].children() {
        let child = &arena[c];
        let score = child.q() + C_PUCT * child.prior * sqrt_n / (1.0 + child.visits as f32);
        if score > best.0 {
            best = (score, c);
        }
    }
    best.1
}

/// Value of `node` for its side to move, expanding it if possible.
fn evaluate_leaf(
    arena: &mut Arena,
    node: usize,
    board: &Board,
    eval: &mut dyn Evaluator,
    arena_full: &mut bool,
) -> Result<f32> {
    if let NodeState::Terminal(v) = arena[node].state {
        return Ok(v);
    }
    let terminal = match board.status() {
        GameStatus::Won => Some(-1.0), // side to move is checkmated
        GameStatus::Drawn => Some(0.0),
        GameStatus::Ongoing => None,
    };
    if let Some(v) = terminal {
        arena[node].state = NodeState::Terminal(v);
        return Ok(v);
    }
    let (priors, value) = evaluate_board(eval, board)?;
    if !arena.expand(node, &priors) {
        *arena_full = true;
    }
    Ok(value)
}

/// Run MCTS from `root` with at most `capacity` (<= 65,536) arena nodes.
pub fn search(
    root: &Board,
    eval: &mut dyn Evaluator,
    limits: SearchLimits,
    capacity: usize,
) -> Result<SearchResult> {
    search_with_noise(root, eval, limits, capacity, None)
}

/// Mix root Dirichlet noise into the root children's priors (right after the
/// first simulation expands the root).
fn noise_root(arena: &mut Arena, noise: &RootNoise, rng: &mut dyn RngCore) {
    let children = arena[0].children();
    let mut priors: Vec<f32> = children.clone().map(|c| arena[c].prior).collect();
    noise.apply(&mut priors, rng);
    for (c, p) in children.zip(priors) {
        arena[c].prior = p;
    }
}

/// `search`, optionally with root Dirichlet noise for self-play exploration.
pub fn search_with_noise(
    root: &Board,
    eval: &mut dyn Evaluator,
    limits: SearchLimits,
    capacity: usize,
    mut noise: Option<(&RootNoise, &mut dyn RngCore)>,
) -> Result<SearchResult> {
    let mut arena = Arena::new(capacity.min(MAX_NODES));
    let mut arena_full = false;
    let root_moves = legal_moves(root).len();
    if root_moves == 0 {
        return Ok(empty_result(&arena));
    }
    if 1 + root_moves > arena.capacity() {
        bail!("arena capacity {} cannot hold the root's {root_moves} children", arena.capacity());
    }
    let max_sims = match limits {
        SearchLimits { nodes: None, movetime: None } => DEFAULT_NODES,
        SearchLimits { nodes, .. } => nodes.unwrap_or(u32::MAX),
    };
    let start = Instant::now();
    let mut sims = 0u32;
    let mut path = Vec::with_capacity(64);

    while sims < max_sims && !limits.movetime.is_some_and(|t| sims > 0 && start.elapsed() >= t) {
        let mut board = root.clone();
        path.clear();
        path.push(0usize);
        let mut node = 0usize;
        while arena[node].state == NodeState::Expanded {
            node = select_child(&arena, node);
            board.play_unchecked(arena[node].mv.expect("non-root node has a move"));
            path.push(node);
        }
        let leaf_value = evaluate_leaf(&mut arena, node, &board, eval, &mut arena_full)?;
        // Leaf value is for the leaf's side to move; the leaf node stores it from
        // its parent's perspective, so start negated and flip at every level.
        let mut v = -leaf_value;
        for &idx in path.iter().rev() {
            arena[idx].visits += 1;
            arena[idx].value_sum += v;
            v = -v;
        }
        sims += 1;
        if sims == 1 {
            if let Some((n, rng)) = noise.take() {
                noise_root(&mut arena, n, rng);
            }
        }
    }
    Ok(build_result(&arena, sims, arena_full))
}

fn empty_result(arena: &Arena) -> SearchResult {
    SearchResult {
        best_move: None,
        root_q: 0.0,
        pv: Vec::new(),
        simulations: 0,
        nodes_used: arena.len(),
        arena_full: false,
        root_children: Vec::new(),
    }
}

/// Most-visited child of `node` (ties -> higher Q), if any child was visited.
fn best_child(arena: &Arena, node: usize) -> Option<usize> {
    arena[node]
        .children()
        .filter(|&c| arena[c].visits > 0)
        .max_by(|&a, &b| {
            (arena[a].visits, arena[a].q())
                .partial_cmp(&(arena[b].visits, arena[b].q()))
                .expect("Q values are finite")
        })
}

fn build_result(arena: &Arena, simulations: u32, arena_full: bool) -> SearchResult {
    let root_children = arena[0]
        .children()
        .map(|c| ChildStats {
            mv: arena[c].mv.expect("child has a move"),
            prior: arena[c].prior,
            visits: arena[c].visits,
            q: arena[c].q(),
            mates: arena[c].state == NodeState::Terminal(-1.0),
        })
        .collect();
    let mut pv = Vec::new();
    let mut node = 0;
    while let Some(c) = best_child(arena, node) {
        pv.push(arena[c].mv.expect("child has a move"));
        node = c;
    }
    let root = &arena[0];
    SearchResult {
        best_move: pv.first().copied(),
        root_q: if root.visits == 0 { 0.0 } else { -root.value_sum / root.visits as f32 },
        pv,
        simulations,
        nodes_used: arena.len(),
        arena_full,
        root_children,
    }
}

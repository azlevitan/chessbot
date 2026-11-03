//! Public search snapshots and deterministic visit-based move selection.

use cozy_chess::Move;

use super::arena::{Arena, NodeState};

#[derive(Clone, Debug)]
pub struct ChildStats {
    pub mv: Move,
    pub prior: f32,
    pub visits: u32,
    /// Mean value from the root side to move's perspective.
    pub q: f32,
    /// Terminal value from the child's side to move's perspective, if known.
    pub terminal: Option<f32>,
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

pub(super) fn build_result(arena: &Arena, simulations: u32, arena_full: bool) -> SearchResult {
    let root_children = arena[0]
        .children()
        .map(|c| ChildStats {
            mv: arena[c].mv.expect("child has a move"),
            prior: arena[c].prior,
            visits: arena[c].visits,
            q: arena[c].q(),
            terminal: match arena[c].state {
                NodeState::Terminal(v) => Some(v),
                _ => None,
            },
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
        root_q: if root.visits == 0 {
            0.0
        } else {
            -root.value_sum / root.visits as f32
        },
        pv,
        simulations,
        nodes_used: arena.len(),
        arena_full,
        root_children,
    }
}

/// Most-visited child (ties -> higher Q); preserve existing move ordering.
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

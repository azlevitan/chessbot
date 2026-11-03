//! Focused selection tests using edge values in their stored perspective.

use super::*;
use crate::board::legal_moves;

#[test]
fn fpu_negates_the_parent_edge_reduces_and_clamps_but_keeps_root_zero() {
    let mut tree = SearchTree::new(Board::default(), &[], 8);
    let mv = legal_moves(&tree.root)[0];
    assert!(tree.arena.expand(0, &[(mv, 1.0)]));
    for (edge_q, expected) in [
        (-1.0, 0.75),
        (-0.8, 0.55),
        (0.0, -0.25),
        (0.8, -1.0),
        (1.0, -1.0),
    ] {
        for node in [0, 1] {
            tree.arena[node].visits = 1;
            tree.arena[node].value_sum = edge_q;
        }
        assert_eq!(tree.fpu_q(0), 0.0);
        assert!((tree.fpu_q(1) - expected).abs() < 1e-6);
    }
}

#[test]
fn puct_uses_reduced_fpu_only_for_unvisited_children() {
    let mut tree = SearchTree::new(Board::default(), &[], 8);
    let moves = legal_moves(&tree.root);
    assert!(tree.arena.expand(0, &[(moves[0], 1.0)]));
    let parent = 1;
    assert!(tree
        .arena
        .expand(parent, &[(moves[0], 0.5), (moves[1], 0.5)]));
    let first = tree.arena[parent].first_child as usize;
    tree.arena[parent].visits = 2;
    tree.arena[parent].value_sum = 1.2; // s is losing: FPU = -0.85.
    tree.arena[first].visits = 1;
    tree.arena[first].value_sum = -0.2;
    // Visited: -0.2 + 0.53; unvisited: -0.85 + 1.06.
    assert_eq!(tree.select_child(parent), first);
    tree.arena[parent].value_sum = -1.6; // s is winning: FPU = +0.55.
    assert_eq!(tree.select_child(parent), first + 1);
}

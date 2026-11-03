//! Internal compaction invariants without exposing arena storage publicly.

use super::*;
use crate::board::legal_moves;
use crate::mcts::{Leaf, SearchLimits};
use crate::nn::ort_infer::UniformEvaluator;

#[test]
fn breadth_first_copy_keeps_only_descendants_and_remaps_every_child_range() {
    let board = Board::default();
    let moves = legal_moves(&board);
    let mut tree = SearchTree::new(board.clone(), &[], 32);
    assert!(tree.arena.expand(0, &[(moves[0], 0.4), (moves[1], 0.6)]));
    assert!(tree.arena.expand(2, &[(moves[2], 0.3), (moves[3], 0.7)]));
    assert!(tree.arena.expand(1, &[(moves[4], 1.0)])); // discarded sibling
    assert!(tree.arena.expand(4, &[(moves[5], 1.0)]));
    assert!(tree.arena.expand(3, &[(moves[6], 0.2), (moves[7], 0.8)]));
    let old_indices = [2, 3, 4, 7, 8, 6];
    for (visits, &index) in old_indices.iter().enumerate() {
        tree.arena[index].visits = visits as u32 + 10;
        tree.arena[index].value_sum = visits as f32 - 2.0;
    }
    tree.arena[8].state = NodeState::Terminal(-1.0);
    let old: Vec<_> = old_indices.iter().map(|&i| tree.arena[i]).collect();
    tree.arena_full = true;
    tree.root_noise_applied = true;
    let mut next = board.clone();
    next.play_unchecked(moves[1]);
    let history = [board.hash(), next.hash()];
    assert!(tree.advance(moves[1], next.clone(), &history));
    assert_eq!(tree.arena.len(), old_indices.len());
    assert_eq!(tree.arena.capacity(), 32);
    assert_eq!(tree.arena[0].mv, None);
    assert_eq!(tree.arena[0].prior, 1.0);
    assert_eq!(tree.root, next);
    assert_eq!(tree.history_hashes, history.into_iter().collect());
    assert!(!tree.arena_full && !tree.root_noise_applied);
    assert!(tree.path.is_empty() && tree.path_hashes.is_empty());
    assert_eq!(tree.simulations, old[0].visits);
    let mut reached = vec![0];
    for (new, before) in old.iter().enumerate() {
        let after = tree.arena[new];
        assert_eq!(after.visits, before.visits);
        assert_eq!(after.value_sum, before.value_sum);
        assert_eq!(after.state, before.state);
        assert_eq!(after.num_children, before.num_children);
        if new != 0 {
            assert_eq!(after.mv, before.mv);
            assert_eq!(after.prior, before.prior);
        }
        for child in after.children() {
            assert!(child > new && child < tree.arena.len());
            assert_eq!(
                old_indices[child],
                before.first_child as usize + child - after.first_child as usize
            );
            reached.push(child);
        }
    }
    assert_eq!(reached, (0..tree.arena.len()).collect::<Vec<_>>());
}

#[test]
fn compaction_reclaims_space_and_clears_the_old_arena_full_flag() {
    let board = Board::default();
    let mut tree = SearchTree::new(board.clone(), &[], 25);
    let limits = SearchLimits {
        nodes: Some(40),
        ..SearchLimits::default()
    };
    let result = tree.search(&mut UniformEvaluator, limits, None).unwrap();
    assert!(result.arena_full);
    let mv = result.best_move.unwrap();
    let mut next = board.clone();
    next.play_unchecked(mv);
    assert!(tree.advance(mv, next, &[board.hash()]));
    assert_eq!(tree.arena.len(), 1);
    assert!(!tree.result().arena_full);
    tree.search(
        &mut UniformEvaluator,
        SearchLimits {
            nodes: Some(1),
            ..limits
        },
        None,
    )
    .unwrap();
    assert!(tree.arena.len() > 1 && tree.arena.len() <= 25);
}

#[test]
fn a_repetition_draw_promoted_to_an_ongoing_root_can_expand_again() {
    let board = Board::default();
    let mv = legal_moves(&board)[0];
    let mut next = board.clone();
    next.play_unchecked(mv);
    let history = [next.hash(), board.hash()];
    let mut tree = SearchTree::new(board.clone(), &history, 64);
    let Leaf::NeedsEval { node, .. } = tree.select_leaf() else {
        panic!("fresh root")
    };
    tree.expand_and_backup(node, &[(mv, 1.0)], 0.0);
    assert!(matches!(tree.select_leaf(), Leaf::Terminal(0.0)));
    let path = tree.selected_path().to_vec();
    tree.backup_terminal(&path, 0.0);
    assert!(tree.advance(mv, next.clone(), &[board.hash(), next.hash()]));
    assert_eq!(tree.arena[0].visits, 1);
    assert!(matches!(
        tree.select_leaf(),
        Leaf::NeedsEval { node: 0, .. }
    ));
    tree.expand_and_backup(0, &[(legal_moves(&next)[0], 1.0)], 0.4);
    assert_eq!(tree.arena[0].visits, 2);
    assert_eq!(tree.arena[0].state, NodeState::Expanded);
}

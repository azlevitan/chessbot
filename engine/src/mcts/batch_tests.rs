//! Inspect all edges after completed batches, collisions, and failed inference.

use super::*;
use crate::board::{from_fen, legal_moves};
use crate::mcts::arena::{NodeState, MAX_NODES};
use crate::nn::ort_infer::UniformEvaluator;
use crate::nn::{evaluate_board, NetOutput};

fn assert_clean(tree: &SearchTree) {
    assert!(tree.path.is_empty());
    assert!(tree.path_hashes.is_empty());
    assert_eq!(tree.arena[0].visits, tree.simulations);
    assert!(tree.arena.len() <= MAX_NODES);
    for i in 0..tree.arena.len() {
        let node = &tree.arena[i];
        assert!(node.value_sum.is_finite());
        assert!(
            node.value_sum.abs() <= node.visits as f32 + 1e-4,
            "node {i}"
        );
        if node.state == NodeState::Expanded {
            assert!(node.children().all(|c| c < tree.arena.len()));
            let children: u32 = node.children().map(|c| tree.arena[c].visits).sum();
            assert_eq!(node.visits, 1 + children, "node {i}");
        }
    }
}

#[test]
fn every_expanded_node_conserves_visits_at_all_batch_sizes() {
    for batch in [1, 4, 8, 16] {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1",
            "k7/2P5/1K6/8/8/8/8/8 w - - 0 1",
        ] {
            let mut tree = SearchTree::new(from_fen(fen).unwrap(), &[], MAX_NODES);
            while tree.simulations < 400 {
                let budget = batch.min((400 - tree.simulations) as usize);
                let done = search_batch(&mut tree, &mut UniformEvaluator, budget).unwrap();
                assert!(done > 0 && done as usize <= budget);
                assert_clean(&tree);
            }
            assert_eq!(tree.simulations, 400);
            assert_eq!(
                tree.result()
                    .root_children
                    .iter()
                    .map(|c| c.visits)
                    .sum::<u32>(),
                399
            );
        }
    }
}

#[test]
fn forced_collision_undoes_only_its_own_reservation() {
    let board = Board::default();
    let mut tree = SearchTree::new(board.clone(), &[], MAX_NODES);
    let Leaf::NeedsEval { node, .. } = tree.select_leaf() else {
        panic!()
    };
    // One child forces the second selection to collide with the pending first.
    tree.expand_and_backup(node, &[(legal_moves(&board)[0], 1.0)], 0.0);
    {
        let mut pending = PendingBatch {
            tree: &mut tree,
            paths: VecDeque::new(),
            boards: Vec::new(),
            nodes: HashSet::new(),
            virtual_loss: true,
        };
        pending.gather(8);
        assert_eq!(pending.paths.len(), 1);
        assert_eq!(pending.tree.arena[0].visits, 1); // root excluded
        assert_eq!(pending.tree.arena[1].visits, 1);
        assert_eq!(pending.tree.arena[1].value_sum, -1.0); // choosing side loses
        assert_eq!(pending.tree.simulations, 1);
    }
    assert_eq!(tree.arena[1].visits, 0);
    assert_eq!(tree.arena[1].value_sum, 0.0);
    assert_clean(&tree);
    assert_eq!(
        search_batch(&mut tree, &mut UniformEvaluator, 8).unwrap(),
        1
    );
    assert_clean(&tree);
}

struct BrokenEvaluator(u8);
impl Evaluator for BrokenEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        let mut out = UniformEvaluator.evaluate(planes, batch)?;
        match self.0 {
            0 => anyhow::bail!("injected inference failure"),
            1 => {
                out.pop();
            }
            2 => {
                out.last_mut().unwrap().policy_logits.pop();
            }
            _ => out.last_mut().unwrap().wdl_logits[0] = f32::NAN,
        }
        Ok(out)
    }
}

#[test]
fn inference_and_output_errors_release_every_pending_path_and_allow_resume() {
    for broken in 0..4 {
        for batch in [1, 8] {
            let mut tree = SearchTree::new(Board::default(), &[], MAX_NODES);
            search_batch(&mut tree, &mut UniformEvaluator, 1).unwrap();
            let before = format!("{:?}", tree.result());
            assert!(search_batch(&mut tree, &mut BrokenEvaluator(broken), batch).is_err());
            assert_eq!(format!("{:?}", tree.result()), before);
            assert_clean(&tree);
            assert!(search_batch(&mut tree, &mut UniformEvaluator, batch).unwrap() > 0);
            assert_clean(&tree);
        }
    }
}

#[test]
fn terminal_only_batches_use_no_evaluator_and_count_exactly_the_budget() {
    let board = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    let mut tree = SearchTree::new(board, &[], MAX_NODES);
    assert_eq!(
        search_batch(&mut tree, &mut BrokenEvaluator(0), 7).unwrap(),
        7
    );
    assert_clean(&tree);
}

#[test]
fn arena_exhaustion_keeps_searching_without_leaking_reservations() {
    let mut tree = SearchTree::new(Board::default(), &[], 25);
    while tree.simulations < 400 {
        let budget = 16.min((400 - tree.simulations) as usize);
        search_batch(&mut tree, &mut UniformEvaluator, budget).unwrap();
        assert_clean(&tree);
    }
    assert!(tree.result().arena_full);
    assert!(tree.arena.len() <= 25);
}

#[test]
fn shared_edges_have_one_reservation_per_pending_trajectory() {
    let board = Board::default();
    let mut tree = SearchTree::new(board.clone(), &[], MAX_NODES);
    let Leaf::NeedsEval { node, .. } = tree.select_leaf() else {
        panic!()
    };
    tree.expand_and_backup(node, &[(legal_moves(&board)[0], 1.0)], 0.0);
    let Leaf::NeedsEval { node, board } = tree.select_leaf() else {
        panic!()
    };
    let (priors, v) = evaluate_board(&mut UniformEvaluator, &board).unwrap();
    tree.expand_and_backup(node, &priors, v);
    {
        let mut pending = PendingBatch {
            tree: &mut tree,
            paths: VecDeque::new(),
            boards: Vec::new(),
            nodes: HashSet::new(),
            virtual_loss: true,
        };
        pending.gather(8);
        assert_eq!(pending.paths.len(), 8);
        assert_eq!(pending.tree.arena[1].visits, 9);
        assert_eq!(pending.tree.arena[1].value_sum, -8.0);
    }
    assert_eq!(tree.arena[1].visits, 1);
    assert_eq!(tree.arena[1].value_sum, 0.0);
    assert_clean(&tree);
}

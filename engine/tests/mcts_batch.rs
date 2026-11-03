//! Phase 7.2: batch-1 parity, batched mate search, throughput and inference order.

use std::time::Duration;

use anyhow::Result;
use cozy_chess::Board;
use engine::board::{from_fen, legal_moves};
use engine::encoder::move_mapper::NUM_MOVES;
use engine::encoder::tensor_encoder::{encode_planes, TENSOR_LEN};
use engine::mcts::arena::MAX_NODES;
use engine::mcts::{search, Leaf, SearchLimits, SearchTree};
use engine::nn::ort_infer::UniformEvaluator;
use engine::nn::{evaluate_board, evaluate_boards, Evaluator, NetOutput};
use engine::uci::notation::move_to_uci;

const MATES: [(&str, &[&str]); 3] = [
    (
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        &["f3f7"],
    ),
    ("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", &["a8a1"]),
    ("k7/2P5/1K6/8/8/8/8/8 w - - 0 1", &["c7c8q", "c7c8r"]),
];

fn limits(nodes: u32, batch: usize) -> SearchLimits {
    SearchLimits {
        nodes: Some(nodes),
        batch,
        ..SearchLimits::default()
    }
}

#[derive(Default)]
struct CountingEvaluator {
    batches: Vec<usize>,
}
impl Evaluator for CountingEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        self.batches.push(batch);
        UniformEvaluator.evaluate(planes, batch)
    }
}

#[test]
fn batch_sizes_four_eight_sixteen_solve_every_gate_four_mate() {
    for batch in [4, 8, 16] {
        for (fen, mates) in MATES {
            let board = from_fen(fen).unwrap();
            let result =
                search(&board, &mut UniformEvaluator, limits(400, batch), MAX_NODES).unwrap();
            let best = move_to_uci(&board, result.best_move.unwrap());
            assert!(
                mates.contains(&best.as_str()),
                "batch {batch}: {fen}: {best}"
            );
            assert_eq!(result.simulations, 400);
            assert_eq!(
                result.root_children.iter().map(|c| c.visits).sum::<u32>(),
                399
            );
            assert!(result.root_children.iter().any(|c| c.mates));
            assert!(result.nodes_used <= MAX_NODES);
        }
    }
}

#[test]
fn startpos_averages_at_least_six_leaves_per_call_at_batch_eight() {
    let mut eval = CountingEvaluator::default();
    let result = search(&Board::default(), &mut eval, limits(400, 8), MAX_NODES).unwrap();
    let evaluated: usize = eval.batches.iter().sum();
    let average = evaluated as f64 / eval.batches.len() as f64;
    assert!(
        average >= 6.0,
        "average {average}, batches {:?}",
        eval.batches
    );
    assert!(eval.batches.iter().all(|&b| b <= 8));
    assert_eq!(eval.batches[0], 1);
    assert_eq!(result.simulations, 400);
}

#[test]
fn partial_batches_and_small_budgets_never_overshoot() {
    assert_eq!(SearchLimits::default().batch, 1);
    for batch in [1, 4, 8, 16] {
        for nodes in [0, 1, 2, 7, 18, 399] {
            let mut eval = CountingEvaluator::default();
            let result = search(
                &Board::default(),
                &mut eval,
                limits(nodes, batch),
                MAX_NODES,
            )
            .unwrap();
            assert_eq!(result.simulations, nodes);
            assert_eq!(eval.batches.iter().sum::<usize>(), nodes as usize);
            assert!(eval.batches.iter().all(|&b| b > 0 && b <= batch));
        }
    }
    assert!(search(
        &Board::default(),
        &mut UniformEvaluator,
        limits(10, 0),
        MAX_NODES
    )
    .is_err());
}

#[test]
fn zero_time_finishes_root_initialization_and_terminal_roots_do_not_evaluate() {
    let timed = SearchLimits {
        nodes: Some(400),
        movetime: Some(Duration::ZERO),
        batch: 8,
    };
    let result = search(&Board::default(), &mut UniformEvaluator, timed, MAX_NODES).unwrap();
    assert_eq!(result.simulations, 1);
    let board = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    let mut eval = CountingEvaluator::default();
    let result = search(&board, &mut eval, limits(400, 8), MAX_NODES).unwrap();
    assert_eq!(result.simulations, 0);
    assert!(result.best_move.is_none());
    assert!(eval.batches.is_empty());
}

/// Distinct input-derived values and policy logits expose ordering/masking bugs.
struct InputEvaluator;
impl Evaluator for InputEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        assert_eq!(planes.len(), batch * TENSOR_LEN);
        Ok(planes
            .chunks_exact(TENSOR_LEN)
            .map(|p| NetOutput {
                policy_logits: (0..NUM_MOVES).map(|i| (i % 7) as f32).collect(),
                wdl_logits: [0.0, p[17 * 64], p[4 * 64..5 * 64].iter().sum()],
            })
            .collect())
    }
}

#[test]
fn evaluate_boards_preserves_encoding_order_legal_mask_and_scalar_results() {
    let boards = [
        Board::default(),
        from_fen(MATES[1].0).unwrap(),
        from_fen(MATES[0].0).unwrap(),
    ];
    struct CheckingEvaluator<'a>(&'a [Board], usize);
    impl Evaluator for CheckingEvaluator<'_> {
        fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
            self.1 += 1;
            assert_eq!(batch, self.0.len());
            for (board, row) in self.0.iter().zip(planes.chunks_exact(TENSOR_LEN)) {
                assert_eq!(row, encode_planes(board));
            }
            InputEvaluator.evaluate(planes, batch)
        }
    }
    let mut eval = CheckingEvaluator(&boards, 0);
    let results = evaluate_boards(&mut eval, &boards).unwrap();
    assert_eq!(eval.1, 1);
    for (board, result) in boards.iter().zip(results) {
        assert_eq!(result, evaluate_board(&mut InputEvaluator, board).unwrap());
        assert_eq!(result.0.len(), legal_moves(board).len());
        assert!((result.0.iter().map(|(_, p)| p).sum::<f32>() - 1.0).abs() < 1e-6);
    }
    assert!(evaluate_boards(&mut eval, &[]).unwrap().is_empty());
    assert_eq!(eval.1, 1);
}

#[test]
fn batch_one_matches_sequential_tree_with_nonuniform_nonzero_evaluations() {
    let board = Board::default();
    let mut tree = SearchTree::new(board.clone(), &[], MAX_NODES);
    for _ in 0..300 {
        match tree.select_leaf() {
            Leaf::Terminal(v) => {
                let path = tree.selected_path().to_vec();
                tree.backup_terminal(&path, v);
            }
            Leaf::NeedsEval { node, board } => {
                let (priors, value) = evaluate_board(&mut InputEvaluator, &board).unwrap();
                tree.expand_and_backup(node, &priors, value);
            }
        }
    }
    let result = search(&board, &mut InputEvaluator, limits(300, 1), MAX_NODES).unwrap();
    assert_eq!(format!("{:?}", tree.result()), format!("{result:?}"));
}

//! MCTS: arena cap, perspective negation, priors, and mate-in-1 with the dummy net.

use anyhow::Result;
use cozy_chess::Board;
use engine::board::{from_fen, legal_moves};
use engine::encoder::move_mapper::NUM_MOVES;
use engine::encoder::tensor_encoder::TENSOR_LEN;
use engine::mcts::arena::MAX_NODES;
use engine::mcts::{search, SearchLimits};
use engine::nn::ort_infer::UniformEvaluator;
use engine::nn::{legal_priors, wdl_value, Evaluator, NetOutput};
use engine::uci::notation::move_to_uci;

/// Same 3 positions as pipeline/test_uci_mate.py (sets verified with python-chess).
const MATES: [(&str, &[&str]); 3] = [
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", &["f3f7"]),
    ("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", &["a8a1"]),
    ("k7/2P5/1K6/8/8/8/8/8 w - - 0 1", &["c7c8q", "c7c8r"]),
];

fn nodes(n: u32) -> SearchLimits {
    SearchLimits { nodes: Some(n), movetime: None }
}

/// Always reports the same WDL for the side to move: p = [0.2, 0.2, 0.6] -> v = +0.4.
struct ConstEvaluator;
impl Evaluator for ConstEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        assert_eq!(planes.len(), batch * TENSOR_LEN);
        let out = NetOutput { policy_logits: vec![0.0; NUM_MOVES], wdl_logits: [0.0, 0.0, 3f32.ln()] };
        Ok(vec![out; batch])
    }
}

#[test]
fn finds_mate_in_one_with_uniform_net() {
    for (fen, mates) in MATES {
        let board = from_fen(fen).unwrap();
        let r = search(&board, &mut UniformEvaluator, nodes(400), MAX_NODES).unwrap();
        let best = move_to_uci(&board, r.best_move.unwrap());
        assert!(mates.contains(&best.as_str()), "{fen}: got {best}, want one of {mates:?}");
        assert_eq!(r.simulations, 400);
        assert!(r.root_children.iter().any(|c| c.mates), "{fen}: mate child not marked");
    }
}

#[test]
fn values_are_negated_between_plies() {
    let board = Board::default();
    assert!((wdl_value(&[0.0, 0.0, 3f32.ln()]) - 0.4).abs() < 1e-6);
    // 1 root expansion + 20 sims: every root child visited exactly once (FPU = 0 > -0.4).
    let r = search(&board, &mut ConstEvaluator, nodes(21), MAX_NODES).unwrap();
    assert_eq!(r.root_children.len(), 20);
    for c in &r.root_children {
        assert_eq!(c.visits, 1, "{}", c.mv);
        // Child position is +0.4 for the opponent, so -0.4 for the root side.
        assert!((c.q + 0.4).abs() < 1e-6, "{}: q = {}", c.mv, c.q);
    }
    // Root: one +0.4 (its own eval) and twenty -0.4 backups, from its own view.
    let expected = (0.4 - 20.0 * 0.4) / 21.0;
    assert!((r.root_q - expected).abs() < 1e-5, "root_q {}", r.root_q);
}

#[test]
fn arena_never_exceeds_capacity() {
    let board = Board::default();
    let r = search(&board, &mut UniformEvaluator, nodes(200), 25).unwrap();
    assert!(r.nodes_used <= 25);
    assert!(r.arena_full, "children of root children cannot fit in 25 nodes");
    assert!(r.best_move.is_some());
    // Too small to even expand the root: a clean error, not a panic.
    assert!(search(&board, &mut UniformEvaluator, nodes(10), 5).is_err());
    // Requested capacity is clamped to the 65,536 hard cap.
    let r = search(&board, &mut UniformEvaluator, nodes(3000), 10 * MAX_NODES).unwrap();
    assert!(r.nodes_used <= MAX_NODES);
}

#[test]
fn full_default_search_stays_within_cap() {
    let board = from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let r = search(&board, &mut UniformEvaluator, nodes(5000), MAX_NODES).unwrap();
    assert!(r.nodes_used <= MAX_NODES);
    assert!(r.arena_full, "5000 sims of a 48-move position should fill the arena");
    assert!(r.best_move.is_some());
}

#[test]
fn no_legal_moves_gives_no_best_move() {
    let stalemate = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    let r = search(&stalemate, &mut UniformEvaluator, nodes(50), MAX_NODES).unwrap();
    assert!(r.best_move.is_none());
    assert_eq!(r.simulations, 0);
}

#[test]
fn priors_cover_exactly_the_legal_moves() {
    let board = from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let mut logits = vec![0.0f32; NUM_MOVES];
    for (i, l) in logits.iter_mut().enumerate() {
        *l = (i % 7) as f32; // illegal logits are large too; they must be ignored
    }
    let priors = legal_priors(&board, &logits);
    assert_eq!(priors.len(), legal_moves(&board).len());
    let sum: f32 = priors.iter().map(|p| p.1).sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(priors.iter().all(|p| p.1 > 0.0));
}

//! Phase 7.3: retained search statistics, clean noise, and deterministic self-play.

use std::process::Command;

use cozy_chess::Board;
use engine::board::legal_moves;
use engine::encoder::sample::{TrainingSample, SAMPLE_BYTES};
use engine::mcts::arena::MAX_NODES;
use engine::mcts::{RootNoise, SearchLimits, SearchTree};
use engine::nn::ort_infer::UniformEvaluator;
use engine::nn::{Evaluator, NetOutput};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

fn limits(nodes: u32, batch: usize) -> SearchLimits {
    SearchLimits {
        nodes: Some(nodes),
        batch,
        ..SearchLimits::default()
    }
}

struct NonzeroValue;
impl Evaluator for NonzeroValue {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> anyhow::Result<Vec<NetOutput>> {
        let mut outputs = UniformEvaluator.evaluate(planes, batch)?;
        for output in &mut outputs {
            output.wdl_logits = [0.0, 0.0, 3f32.ln()];
        }
        Ok(outputs)
    }
}

#[test]
fn advance_preserves_child_visits_and_value_then_adds_exactly_new_playouts() {
    for batch in [1, 4, 8, 16] {
        let board = Board::default();
        let mut tree = SearchTree::new(board.clone(), &[board.hash()], MAX_NODES * 2);
        let before = tree
            .search(&mut NonzeroValue, limits(400, batch), None)
            .unwrap();
        let best = before.best_move.unwrap();
        let child = before.root_children.iter().find(|c| c.mv == best).unwrap();
        assert!(child.visits > 1 && child.q.abs() > 0.01);
        let mut next = board.clone();
        next.play_unchecked(best);
        assert!(tree.advance(best, next.clone(), &[board.hash(), next.hash()]));
        let reused = tree.result();
        assert_eq!(reused.simulations, child.visits);
        assert!((reused.root_q + child.q).abs() < 1e-6);
        assert!(reused.nodes_used < before.nodes_used);
        let after = tree
            .search(&mut NonzeroValue, limits(16, batch), None)
            .unwrap();
        assert_eq!(after.simulations, child.visits + 16);
        assert!(after.simulations >= 16 && after.nodes_used <= MAX_NODES);
        assert_eq!(
            after.root_children.iter().map(|c| c.visits).sum::<u32>(),
            after.simulations - 1
        );
        assert!((-1.0..=1.0).contains(&after.root_q));
    }
}

#[test]
fn missing_move_or_mismatched_board_resets_and_uses_the_supplied_board() {
    let board = Board::default();
    let mv = legal_moves(&board)[0];
    let mut next = board.clone();
    next.play_unchecked(mv);
    for populated in [false, true] {
        let mut tree = SearchTree::new(board.clone(), &[], MAX_NODES);
        if populated {
            tree.search(&mut UniformEvaluator, limits(50, 1), None)
                .unwrap();
        }
        // Empty tree: missing move. Populated tree: mismatched destination board.
        let supplied = if populated {
            board.clone()
        } else {
            next.clone()
        };
        assert!(!tree.advance(mv, supplied.clone(), &[supplied.hash()]));
        assert_eq!(tree.result().simulations, 0);
        assert_eq!(tree.result().nodes_used, 1);
        let result = tree
            .search(&mut UniformEvaluator, limits(16, 1), None)
            .unwrap();
        assert_eq!(result.simulations, 16);
        assert_eq!(
            result
                .root_children
                .iter()
                .map(|c| c.mv)
                .collect::<Vec<_>>(),
            legal_moves(&supplied)
        );
    }
}

#[test]
fn reused_children_have_clean_priors_and_receive_noise_once_before_search() {
    let board = Board::default();
    let mut tree = SearchTree::new(board.clone(), &[], MAX_NODES);
    let noise = RootNoise::default();
    let mut rng = StdRng::seed_from_u64(9);
    let first = tree
        .search(
            &mut UniformEvaluator,
            limits(400, 8),
            Some((&noise, &mut rng)),
        )
        .unwrap();
    let mv = first.best_move.unwrap();
    let mut next = board.clone();
    next.play_unchecked(mv);
    assert!(tree.advance(mv, next, &[board.hash()]));
    let clean = tree.result();
    let uniform = 1.0 / clean.root_children.len() as f32;
    assert!(!clean.root_children.is_empty());
    assert!(clean
        .root_children
        .iter()
        .all(|c| (c.prior - uniform).abs() < 1e-6));
    let mut expected_rng = rng.clone();
    let mut expected: Vec<_> = clean.root_children.iter().map(|c| c.prior).collect();
    noise.apply(&mut expected, &mut expected_rng);
    // No playouts: any noise must be applied immediately to the reused root.
    let noisy = tree
        .search(
            &mut UniformEvaluator,
            limits(0, 8),
            Some((&noise, &mut rng)),
        )
        .unwrap();
    assert_eq!(
        noisy
            .root_children
            .iter()
            .map(|c| c.prior)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(noisy.simulations, clean.simulations);
    tree.search(
        &mut UniformEvaluator,
        limits(8, 8),
        Some((&noise, &mut rng)),
    )
    .unwrap();
    assert_eq!(
        tree.result()
            .root_children
            .iter()
            .map(|c| c.prior)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        rng.next_u64(),
        expected_rng.next_u64(),
        "noise must not consume RNG twice"
    );
}

#[test]
fn seed_nine_selfplay_is_byte_identical_and_policies_are_normalized_with_both_modes() {
    let dir = std::env::temp_dir().join(format!("chessbot-reuse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut modes = Vec::new();
    for no_reuse in [false, true] {
        let mut previous = None;
        for run in 0..2 {
            let out = dir.join(format!("reuse-{no_reuse}-{run}.bin"));
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_engine"));
            cmd.args([
                "selfplay",
                "--model",
                "uniform",
                "--seed",
                "9",
                "--sims",
                "16",
                "--games",
                "1",
                "--max-plies",
                "30",
                "--ep",
                "cpu",
                "--out",
            ])
            .arg(&out);
            if no_reuse {
                cmd.arg("--no-reuse");
            }
            let output = cmd.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let bytes = std::fs::read(&out).unwrap();
            assert!(!bytes.is_empty() && bytes.len() % SAMPLE_BYTES == 0);
            for row in bytes.chunks_exact(SAMPLE_BYTES) {
                let sample = TrainingSample::from_bytes(row);
                let policy = sample.policy;
                let n = sample.num_legal_moves as usize;
                assert!(n > 0 && n <= 64);
                assert!((policy[..n].iter().map(|p| p.prob).sum::<f32>() - 1.0).abs() < 1e-5);
            }
            let sidecar: serde_json::Value =
                serde_json::from_slice(&std::fs::read(out.with_extension("json")).unwrap())
                    .unwrap();
            assert_eq!(
                sidecar["samples"].as_u64().unwrap() as usize * SAMPLE_BYTES,
                bytes.len()
            );
            assert_eq!(sidecar["games"], 1);
            if let Some((prior_bytes, prior_sidecar)) = previous {
                assert_eq!(bytes, prior_bytes);
                assert_eq!(sidecar, prior_sidecar);
            }
            previous = Some((bytes, sidecar));
        }
        modes.push(previous.unwrap().0);
    }
    assert_ne!(
        modes[0], modes[1],
        "--no-reuse must discard retained visits"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_defaults_to_reuse_and_no_reuse_is_a_boolean_switch() {
    let base = vec!["--out".to_owned(), "unused.bin".to_owned()];
    assert!(engine::cli::selfplay::parse_args(&base).unwrap().cfg.reuse);
    let mut disabled = base;
    disabled.push("--no-reuse".to_owned());
    assert!(
        !engine::cli::selfplay::parse_args(&disabled)
            .unwrap()
            .cfg
            .reuse
    );
}

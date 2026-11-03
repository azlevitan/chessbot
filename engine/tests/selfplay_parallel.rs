//! Phase 7.4: scheduling-independent bytes, shared inference, and temperature.

use std::path::{Path, PathBuf};
use std::process::Command;

use engine::encoder::move_mapper::move_to_index;
use engine::encoder::sample::SAMPLE_BYTES;
use engine::nn::ort_infer::UniformEvaluator;
use engine::nn::{Evaluator, NetOutput};
use engine::selfplay::batch_runner::{run_games, BatchOptions};
use engine::selfplay::{game_samples, play_game, SelfPlayConfig};
use rand::{rngs::StdRng, SeedableRng};

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("chessbot-parallel-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn chunk(
    dir: &Path,
    parallel: usize,
    leaf_batch: usize,
    no_reuse: bool,
) -> (Vec<u8>, serde_json::Value) {
    let out = dir.join(format!("p{parallel}-k{leaf_batch}-r{no_reuse}.bin"));
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_engine"));
    cmd.args([
        "selfplay",
        "--model",
        "uniform",
        "--games",
        "4",
        "--sims",
        "16",
        "--max-plies",
        "40",
        "--temp-plies",
        "10",
        "--seed",
        "11",
        "--ep",
        "cpu",
        "--parallel",
        &parallel.to_string(),
        "--leaf-batch",
        &leaf_batch.to_string(),
        "--out",
    ])
    .arg(&out);
    if no_reuse {
        cmd.arg("--no-reuse");
    }
    let result = cmd.output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let sidecar: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.with_extension("json")).unwrap()).unwrap();
    assert_eq!(
        sidecar["samples"].as_u64().unwrap() as usize * SAMPLE_BYTES,
        bytes.len()
    );
    assert_eq!(sidecar["games"], 4);
    assert!(std::fs::read_dir(dir).unwrap().all(|p| !p
        .unwrap()
        .path()
        .to_string_lossy()
        .ends_with(".tmp")));
    (bytes, sidecar)
}

#[test]
fn parallel_one_and_four_produce_byte_identical_chunks_and_sidecars() {
    let dir = TestDir::new("bytes");
    for no_reuse in [false, true] {
        for leaf_batch in [1, 4] {
            let sequential = chunk(&dir.0, 1, leaf_batch, no_reuse);
            for parallel in [3, 4] {
                let concurrent = chunk(&dir.0, parallel, leaf_batch, no_reuse);
                assert_eq!(
                    sequential, concurrent,
                    "P={parallel}, K={leaf_batch}, no-reuse={no_reuse}"
                );
            }
        }
    }
}

#[test]
fn zero_temperature_plays_an_argmax_of_every_saved_policy() {
    let cfg = SelfPlayConfig {
        sims: 16,
        max_plies: 40,
        temp_plies: 0,
        ..SelfPlayConfig::default()
    };
    let mut count = 0;
    run_games(
        &mut UniformEvaluator,
        4,
        11,
        &cfg,
        BatchOptions {
            parallel: 4,
            leaf_batch: 4,
        },
        |_, game| {
            for (ply, sample) in game.plies.iter().zip(game_samples(&game)) {
                let index = move_to_index(&ply.board, ply.played) as u16;
                let policy = sample.policy;
                let active = &policy[..sample.num_legal_moves as usize];
                let played = active
                    .iter()
                    .find(|entry| entry.move_idx == index)
                    .unwrap()
                    .prob;
                assert_eq!(
                    played,
                    active.iter().map(|entry| entry.prob).fold(0.0, f32::max)
                );
                assert!((active.iter().map(|entry| entry.prob).sum::<f32>() - 1.0).abs() < 1e-5);
                count += 1;
            }
            Ok(())
        },
    )
    .unwrap();
    assert!(count > 0);
    let sampled = play_game(
        &mut UniformEvaluator,
        &SelfPlayConfig {
            temp_plies: 30,
            ..cfg
        },
        &mut StdRng::seed_from_u64(11),
    )
    .unwrap();
    let greedy = play_game(&mut UniformEvaluator, &cfg, &mut StdRng::seed_from_u64(11)).unwrap();
    assert_eq!(sampled.plies[0].policy, greedy.plies[0].policy);
    assert!(greedy.plies[0].policy.iter().filter(|p| p.1 > 0.0).count() > 1);
}

#[derive(Default)]
struct Counting {
    batches: Vec<usize>,
}
impl Evaluator for Counting {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> anyhow::Result<Vec<NetOutput>> {
        self.batches.push(batch);
        UniformEvaluator.evaluate(planes, batch)
    }
}

#[test]
fn each_step_combines_games_and_honors_the_remaining_simulation_budget() {
    let mut eval = Counting::default();
    let cfg = SelfPlayConfig {
        sims: 10,
        max_plies: 1,
        ..SelfPlayConfig::default()
    };
    let mut ids = Vec::new();
    run_games(
        &mut eval,
        4,
        11,
        &cfg,
        BatchOptions {
            parallel: 4,
            leaf_batch: 8,
        },
        |id, game| {
            ids.push(id);
            assert_eq!(game.plies.len(), 1);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(eval.batches, [4, 32, 4]); // root, eight leaves/game, one tail leaf/game
    assert_eq!(ids, [0, 1, 2, 3]);
}

#[test]
fn empty_games_and_zero_ply_games_make_no_inference_calls() {
    let mut eval = Counting::default();
    let cfg = SelfPlayConfig {
        max_plies: 0,
        ..SelfPlayConfig::default()
    };
    let mut emitted = Vec::new();
    for games in [0, 5] {
        run_games(
            &mut eval,
            games,
            0,
            &cfg,
            BatchOptions {
                parallel: 2,
                leaf_batch: 8,
            },
            |id, game| {
                assert!(game.plies.is_empty());
                emitted.push(id);
                Ok(())
            },
        )
        .unwrap();
    }
    assert_eq!(emitted, [0, 1, 2, 3, 4]);
    assert!(eval.batches.is_empty());
}

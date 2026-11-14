//! Live observations preserve search bytes and precede the first completed game.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use engine::nn::ort_infer::UniformEvaluator;
use engine::nn::{Evaluator, NetOutput};
use engine::selfplay::batch_runner::{run_games, run_games_with_progress, BatchOptions};
use engine::selfplay::progress::{ProgressPhase, ProgressSnapshot};
use engine::selfplay::{game_samples, SelfPlayConfig};
use rand::{rngs::StdRng, Rng, SeedableRng};

struct CheckedEvaluator(Rc<RefCell<Option<ProgressSnapshot>>>);
impl Evaluator for CheckedEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        let snapshot = self.0.borrow().unwrap();
        assert_eq!(snapshot.phase, ProgressPhase::Evaluating);
        assert_eq!(snapshot.batch_positions, batch);
        assert!(snapshot.eval_calls > 0);
        UniformEvaluator.evaluate(planes, batch)
    }
}

#[test]
fn progress_observes_moves_before_a_game_finishes_without_changing_output_bytes() {
    for parallel in [1, 4] {
        let cfg = SelfPlayConfig { sims: 8, max_plies: 6, resign_threshold: -1.0,
                                   ..SelfPlayConfig::default() };
        let options = BatchOptions { parallel, leaf_batch: 4 };
        let mut baseline = Vec::new();
        run_games(&mut UniformEvaluator, 4, 11, &cfg, options, |_, game| {
            for sample in game_samples(&game) { baseline.extend_from_slice(sample.as_bytes()); }
            Ok(())
        }).unwrap();
        let latest = Rc::new(RefCell::new(None));
        let mut snapshots = Vec::new();
        let mut observed = Vec::new();
        run_games_with_progress(&mut CheckedEvaluator(Rc::clone(&latest)), 4, 11,
            &cfg, options, |_, game| {
                for sample in game_samples(&game) { observed.extend_from_slice(sample.as_bytes()); }
                Ok(())
            }, |snapshot| {
                *latest.borrow_mut() = Some(snapshot);
                snapshots.push(snapshot);
            }).unwrap();
        assert_eq!(baseline, observed);
        assert!(snapshots.iter().any(|s| s.plies > 0 && s.completed == 0 && s.emitted == 0));
        assert!(snapshots.windows(2).all(|pair| pair[1].plies >= pair[0].plies));
        let last = snapshots.last().unwrap();
        assert_eq!(last.phase, ProgressPhase::Complete);
        assert_eq!((last.completed, last.emitted, last.active, last.waiting), (4, 4, 0, 0));
        assert_eq!(last.first_game, None);
        assert_eq!(last.plies, 24);
        assert!(last.eval_positions >= last.eval_calls);
    }
}

#[test]
fn finished_later_game_is_visible_while_output_waits_for_the_earlier_game() {
    let seed = (0..100).find(|&seed| {
        StdRng::seed_from_u64(seed).gen::<f32>() < 0.5
            && StdRng::seed_from_u64(seed + 1).gen::<f32>() >= 0.5
    }).unwrap();
    let cfg = SelfPlayConfig { sims: 8, max_plies: 3, resign_threshold: 0.1,
                               resign_consecutive: 1, resign_playthrough: 0.5,
                               ..SelfPlayConfig::default() };
    let mut order = Vec::new();
    let mut snapshots = Vec::new();
    run_games_with_progress(&mut UniformEvaluator, 2, seed, &cfg,
        BatchOptions { parallel: 2, leaf_batch: 4 }, |index, _| {
            order.push(index);
            Ok(())
        }, |snapshot| snapshots.push(snapshot)).unwrap();
    assert_eq!(order, vec![0, 1]);
    assert!(snapshots.iter().any(|s| s.completed == 1 && s.waiting == 1
                                    && s.emitted == 0 && s.active == 1));
    let last = snapshots.last().unwrap();
    assert_eq!((last.completed, last.emitted, last.waiting), (2, 2, 0));
    assert_eq!(last.plies, 3);
}

#[test]
fn zero_games_and_failed_inference_have_explicit_terminal_observations() {
    let mut snapshots = Vec::new();
    run_games_with_progress(&mut UniformEvaluator, 0, 0, &SelfPlayConfig::default(),
        BatchOptions::default(), |_, _| panic!("no game is emitted"),
        |snapshot| snapshots.push(snapshot)).unwrap();
    assert_eq!(snapshots.last().unwrap().phase, ProgressPhase::Complete);
    assert_eq!(snapshots.last().unwrap().games, 0);

    struct Broken;
    impl Evaluator for Broken {
        fn evaluate(&mut self, _: &[f32], _: usize) -> Result<Vec<NetOutput>> {
            anyhow::bail!("injected inference error")
        }
    }
    snapshots.clear();
    let result = run_games_with_progress(&mut Broken, 1, 0, &SelfPlayConfig::default(),
        BatchOptions::default(), |_, _| panic!("failed game is not emitted"),
        |snapshot| snapshots.push(snapshot));
    assert!(result.is_err());
    assert_eq!(snapshots.last().unwrap().phase, ProgressPhase::Failed);
    assert_eq!(snapshots.last().unwrap().eval_calls, 1);
    assert_eq!(snapshots.last().unwrap().eval_positions, 0);
}

#[test]
fn cli_reports_model_loading_and_final_publication_or_failure() {
    use std::{fs, process::Command, time::{SystemTime, UNIX_EPOCH}};
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let directory = std::env::temp_dir().join(format!("chessbot-progress-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let output = directory.join("empty.bin");
    let result = Command::new(env!("CARGO_BIN_EXE_engine"))
        .args(["selfplay", "--games", "0", "--out"]).arg(&output).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("phase=loading_model"));
    assert!(stderr.contains("phase=complete"));
    assert!(output.is_file() && output.with_extension("json").is_file());

    let result = Command::new(env!("CARGO_BIN_EXE_engine"))
        .args(["selfplay", "--model"]).arg(directory.join("missing.onnx"))
        .arg("--out").arg(directory.join("failed.bin")).output().unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("phase=failed"));
    assert!(!directory.join("failed.bin").exists());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn failed_game_output_does_not_claim_successful_emission_or_completion() {
    let cfg = SelfPlayConfig { sims: 8, max_plies: 1, ..SelfPlayConfig::default() };
    let mut snapshots = Vec::new();
    let result = run_games_with_progress(&mut UniformEvaluator, 1, 0, &cfg,
        BatchOptions::default(), |_, _| anyhow::bail!("injected writer error"),
        |snapshot| snapshots.push(snapshot));
    assert!(result.is_err());
    let last = snapshots.last().unwrap();
    assert_eq!(last.phase, ProgressPhase::Failed);
    assert_eq!((last.completed, last.emitted, last.plies), (1, 0, 1));
    assert!(!snapshots.iter().any(|s| s.phase == ProgressPhase::Complete));
}

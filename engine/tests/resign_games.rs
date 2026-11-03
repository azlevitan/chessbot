//! Resignation in the real driver, final-result labels, and seeded scheduling.

use cozy_chess::{Board, Color};
use engine::cli::bin_writer::{outcome_for, GameResult};
use engine::encoder::sample::SAMPLE_BYTES;
use engine::nn::ort_infer::UniformEvaluator;
use engine::nn::{Evaluator, NetOutput};
use engine::selfplay::batch_runner::{run_games, BatchOptions};
use engine::selfplay::stats::SelfPlayStats;
use engine::selfplay::{game_samples, play_game, SelfPlayConfig, Termination};
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};

/// Two simulations: a losing root and a winning child give that root Q = -1.
struct ScriptedLoss { evaluated: usize, losing_root: usize }
impl Evaluator for ScriptedLoss {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> anyhow::Result<Vec<NetOutput>> {
        let mut outputs = UniformEvaluator.evaluate(planes, batch)?;
        for out in &mut outputs {
            if self.evaluated == self.losing_root {
                out.wdl_logits = [20.0, -20.0, -20.0];
            } else if self.evaluated > self.losing_root {
                out.wdl_logits = [-20.0, -20.0, 20.0];
            }
            self.evaluated += 1;
        }
        Ok(outputs)
    }
}

#[test]
fn either_colour_resigns_before_playing_and_the_final_search_is_saved_and_labelled() {
    for (side, losing_root) in [(Color::White, 0), (Color::Black, 2)] {
        let cfg = SelfPlayConfig {
            sims: 2, max_plies: 10, reuse: false,
            resign_threshold: -0.9, resign_consecutive: 1, resign_playthrough: 0.0,
            ..SelfPlayConfig::default()
        };
        let game = play_game(
            &mut ScriptedLoss { evaluated: 0, losing_root },
            &cfg, &mut StdRng::seed_from_u64(17),
        ).unwrap();
        assert_eq!(game.termination, Termination::Resign);
        assert_eq!(game.result, GameResult::win_for(!side));
        assert_eq!(game.plies.len(), losing_root / 2 + 1);
        let last = game.plies.last().unwrap();
        assert_eq!(last.board.side_to_move(), side);
        assert_eq!(last.played, None);
        assert!(last.root_q < -0.99);
        assert!(!game.playthrough);
        assert_eq!(game.would_resign, None);
        let mut board = Board::default();
        for ply in &game.plies {
            assert_eq!(ply.board, board);
            if let Some(mv) = ply.played { board.play_unchecked(mv); }
        }
        let samples = game_samples(&game);
        assert_eq!(samples.len(), game.plies.len());
        for (sample, ply) in samples.iter().zip(&game.plies) {
            assert_eq!({ sample.outcome_wdl }, outcome_for(ply.board.side_to_move(), game.result).0);
            assert_eq!({ sample.q_value }, ply.root_q);
            let policy = sample.policy;
            let active = &policy[..sample.num_legal_moves as usize];
            assert!((active.iter().map(|p| p.prob).sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert_eq!({ samples.last().unwrap().outcome_wdl }, 0);
        let mut stats = SelfPlayStats::default();
        stats.record(&game);
        assert_eq!(stats.sidecar("scripted", 17, &cfg)["terminations"]["resign"], 1);
    }
}

fn forced_cfg() -> SelfPlayConfig {
    SelfPlayConfig {
        sims: 8, max_plies: 12, resign_threshold: 0.5, resign_consecutive: 2,
        ..SelfPlayConfig::default()
    }
}

#[test]
fn playthrough_continues_after_the_same_trigger_and_uses_the_real_draw_labels() {
    for probability in [0.0, 1.0] {
        let cfg = SelfPlayConfig { resign_playthrough: probability, ..forced_cfg() };
        let game = play_game(&mut UniformEvaluator, &cfg, &mut StdRng::seed_from_u64(11)).unwrap();
        let mut stats = SelfPlayStats::default();
        stats.record(&game);
        let meta = stats.sidecar("uniform", 11, &cfg);
        if probability == 0.0 {
            assert_eq!(game.termination, Termination::Resign);
            assert_eq!(game.plies.len(), 3);
            assert_eq!(game.result, GameResult::BlackWins);
            assert_eq!(meta["resign"]["playthrough_games"], 0);
            assert_eq!(meta["resign"]["false_positives"], 0);
        } else {
            assert_eq!(game.would_resign, Some(Color::White));
            assert_eq!(game.termination, Termination::PlyCap);
            assert_eq!(game.plies.len(), cfg.max_plies as usize);
            assert!(game.plies.iter().all(|p| p.played.is_some()));
            assert_eq!(game.result, GameResult::Draw);
            assert!(game_samples(&game).iter().all(|s| s.outcome_wdl == 1));
            assert_eq!(meta["resign"]["playthrough_games"], 1);
            assert_eq!(meta["resign"]["false_positives"], 1);
        }
    }
}

#[test]
fn every_game_consumes_the_audit_draw_even_without_search_or_resignation() {
    for threshold in [-1.0, -0.95] {
        for probability in [0.0, 0.1, 1.0] {
            let cfg = SelfPlayConfig {
                max_plies: 0, resign_threshold: threshold, resign_playthrough: probability,
                ..SelfPlayConfig::default()
            };
            let mut rng = StdRng::seed_from_u64(23);
            let mut expected = rng.clone();
            let draw = expected.gen::<f32>();
            let game = play_game(&mut UniformEvaluator, &cfg, &mut rng).unwrap();
            assert_eq!(game.playthrough, draw < probability);
            assert_eq!(rng.next_u64(), expected.next_u64());
            assert!(game.plies.is_empty());
            assert_eq!(game.would_resign, None);
        }
    }
}

#[test]
fn mixed_audit_and_resigned_games_keep_identical_samples_and_counters_across_parallelism() {
    for reuse in [false, true] {
        for leaf_batch in [1, 4] {
            let cfg = SelfPlayConfig { reuse, resign_playthrough: 0.5, ..forced_cfg() };
            let mut reference = None;
            for parallel in [1, 3, 4] {
                let mut stats = SelfPlayStats::default();
                let mut records = Vec::new();
                run_games(&mut UniformEvaluator, 9, 11, &cfg, BatchOptions { parallel, leaf_batch }, |id, game| {
                    assert_eq!(id as usize, records.len());
                    let mut rng = StdRng::seed_from_u64(11 + u64::from(id));
                    assert_eq!(game.playthrough, rng.gen::<f32>() < cfg.resign_playthrough);
                    let bytes: Vec<_> = game_samples(&game).iter()
                        .flat_map(|s| s.as_bytes().to_vec()).collect();
                    assert_eq!(bytes.len(), game.plies.len() * SAMPLE_BYTES);
                    stats.record(&game);
                    records.push((bytes, game.termination, game.playthrough, game.would_resign));
                    Ok(())
                }).unwrap();
                let meta = stats.sidecar("uniform", 11, &cfg);
                let audited = meta["resign"]["playthrough_games"].as_u64().unwrap();
                assert!(audited > 0 && audited < 9);
                assert_eq!(meta["terminations"]["resign"].as_u64().unwrap(), 9 - audited);
                assert_eq!(meta["resign"]["false_positives"].as_u64().unwrap(), audited);
                let actual = (records, meta);
                if let Some(expected) = &reference {
                    assert_eq!(&actual, expected, "P={parallel}, K={leaf_batch}, reuse={reuse}");
                } else {
                    reference = Some(actual);
                }
            }
        }
    }
}

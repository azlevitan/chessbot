//! Board-dependent evaluations expose incorrect routing between concurrent games.

use engine::encoder::tensor_encoder::TENSOR_LEN;
use engine::nn::ort_infer::UniformEvaluator;
use engine::nn::{Evaluator, NetOutput};
use engine::selfplay::batch_runner::{run_games, BatchOptions};
use engine::selfplay::{game_samples, SelfPlayConfig};

struct PositionEvaluator;
impl Evaluator for PositionEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> anyhow::Result<Vec<NetOutput>> {
        let mut outputs = UniformEvaluator.evaluate(planes, batch)?;
        for (row, output) in planes.chunks_exact(TENSOR_LEN).zip(&mut outputs) {
            let signature = row[..768]
                .iter()
                .enumerate()
                .map(|(i, &value)| (i + 1) * value as usize)
                .sum::<usize>();
            for (index, logit) in output.policy_logits.iter_mut().enumerate() {
                *logit = ((index * 13 + signature) % 37) as f32 / 7.0;
            }
            output.wdl_logits = [
                (signature % 7) as f32 * 0.1,
                -0.2,
                (signature % 11) as f32 * 0.2,
            ];
        }
        Ok(outputs)
    }
}

#[test]
fn nonuniform_values_and_policies_reach_the_correct_game_including_partial_windows() {
    let cfg = SelfPlayConfig {
        sims: 16,
        max_plies: 60,
        temp_plies: 10,
        ..SelfPlayConfig::default()
    };
    let mut reference = None;
    for parallel in [1, 3, 4] {
        let mut games = Vec::new();
        run_games(
            &mut PositionEvaluator,
            7,
            u64::MAX - 2,
            &cfg,
            BatchOptions {
                parallel,
                leaf_batch: 4,
            },
            |id, game| {
                assert_eq!(id as usize, games.len());
                let bytes: Vec<u8> = game_samples(&game)
                    .iter()
                    .flat_map(|sample| sample.as_bytes().iter().copied())
                    .collect();
                assert!(game.plies.iter().any(|p| p.root_q.abs() > 0.01));
                games.push((
                    bytes,
                    game.plies.iter().map(|p| p.played).collect::<Vec<_>>(),
                ));
                Ok(())
            },
        )
        .unwrap();
        if let Some(expected) = &reference {
            assert_eq!(&games, expected, "parallel {parallel}");
        } else {
            reference = Some(games);
        }
    }
}

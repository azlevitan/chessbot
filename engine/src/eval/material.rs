//! Material-only baseline over the canonical input planes (SPEC.md §9).

use anyhow::{ensure, Result};

use crate::encoder::move_mapper::NUM_MOVES;
use crate::encoder::tensor_encoder::TENSOR_LEN;
use crate::nn::{Evaluator, NetOutput};

/// Uniform policy and a material value from the already-encoded side to move.
#[derive(Default)]
pub struct MaterialEvaluator;

impl Evaluator for MaterialEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        ensure!(
            batch.checked_mul(TENSOR_LEN) == Some(planes.len()),
            "planes length mismatch"
        );
        Ok(planes
            .chunks_exact(TENSOR_LEN)
            .map(|row| {
                let difference: f32 = [1.0, 3.0, 3.0, 5.0, 9.0, 0.0]
                    .iter()
                    .enumerate()
                    .map(|(piece, weight)| {
                        let us: f32 = row[piece * 64..(piece + 1) * 64].iter().sum();
                        let them: f32 = row[(piece + 6) * 64..(piece + 7) * 64].iter().sum();
                        weight * (us - them)
                    })
                    .sum();
                let value = (0.25 * difference).tanh();
                NetOutput {
                    policy_logits: vec![0.0; NUM_MOVES],
                    wdl_logits: [
                        ((1.0 - value) / 2.0).max(1e-6).ln(),
                        1e-6f32.ln(),
                        ((1.0 + value) / 2.0).max(1e-6).ln(),
                    ],
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::wdl_value;

    #[test]
    fn batch_uses_all_piece_weights_and_ignores_auxiliary_planes() {
        let mut planes = vec![0.0; 3 * TENSOR_LEN];
        for (piece, count) in [2, 1, 1, 1, 1, 1].into_iter().enumerate() {
            planes[piece * 64..piece * 64 + count].fill(1.0);
            let start = TENSOR_LEN + (piece + 6) * 64;
            planes[start..start + count].fill(1.0);
        }
        planes[2 * TENSOR_LEN..].fill(1.0);
        let outputs = MaterialEvaluator.evaluate(&planes, 3).unwrap();
        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            assert_eq!(output.policy_logits, vec![0.0; NUM_MOVES]);
            assert!(output.wdl_logits.iter().all(|x| x.is_finite()));
            assert_eq!(output.wdl_logits[1], 1e-6f32.ln());
        }
        let expected = (0.25f32 * 22.0).tanh();
        assert!((wdl_value(&outputs[0].wdl_logits) - expected).abs() < 3e-6);
        assert!((wdl_value(&outputs[1].wdl_logits) + expected).abs() < 3e-6);
        assert_eq!(wdl_value(&outputs[2].wdl_logits), 0.0);
    }

    #[test]
    fn tensor_length_must_match_the_batch() {
        for (length, batch) in [(TENSOR_LEN - 1, 1), (TENSOR_LEN, 2), (1, 0)] {
            assert!(MaterialEvaluator
                .evaluate(&vec![0.0; length], batch)
                .is_err());
        }
        assert!(MaterialEvaluator.evaluate(&[], 0).unwrap().is_empty());
    }
}

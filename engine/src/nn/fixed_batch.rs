//! Keep every physical evaluator call the same size, preserving logical rows.

use anyhow::{ensure, Context, Result};

use super::{Evaluator, NetOutput};
use crate::encoder::move_mapper::NUM_MOVES;
use crate::encoder::tensor_encoder::TENSOR_LEN;

/// One owned evaluator/session plus a bounded, reusable zero-padding buffer.
pub struct FixedBatchEvaluator {
    inner: Box<dyn Evaluator>,
    width: usize,
    planes: Vec<f32>,
}

impl FixedBatchEvaluator {
    pub fn new(inner: Box<dyn Evaluator>, width: usize) -> Result<Self> {
        ensure!((1..=64).contains(&width), "fixed inference batch must be in 1..64");
        Ok(Self { inner, width, planes: vec![0.0; width * TENSOR_LEN] })
    }
}

impl Evaluator for FixedBatchEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        let expected = batch.checked_mul(TENSOR_LEN).context("input batch size overflow")?;
        ensure!(batch > 0 && planes.len() == expected, "planes length mismatch");
        let mut result = Vec::with_capacity(batch);
        for chunk in planes.chunks(self.planes.len()) {
            let actual = chunk.len() / TENSOR_LEN;
            // Clear every row before copying: a smaller tail must never retain
            // input from an earlier full call, including after an inner error.
            self.planes.fill(0.0);
            self.planes[..chunk.len()].copy_from_slice(chunk);
            let outputs = self.inner.evaluate(&self.planes, self.width)?;
            ensure!(outputs.len() == self.width, "fixed evaluator output row count mismatch");
            for output in &outputs {
                ensure!(output.policy_logits.len() == NUM_MOVES, "fixed evaluator policy width mismatch");
                ensure!(output.policy_logits.iter().chain(output.wdl_logits.iter()).all(|x| x.is_finite()),
                        "fixed evaluator returned non-finite logits");
            }
            result.extend(outputs.into_iter().take(actual));
        }
        Ok(result)
    }
}

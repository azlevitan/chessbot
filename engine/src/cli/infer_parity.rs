//! Compare real exact-shape and fixed-batch sessions on mixed Rust chess inputs.

use std::io::{self, Write};
use std::path::Path;

use anyhow::{ensure, Result};
use serde_json::{json, Value};

use super::bench::BENCH_FENS;
use crate::board::from_fen;
use crate::encoder::move_mapper::NUM_MOVES;
use crate::encoder::tensor_encoder::encode_planes;
use crate::nn::fixed_batch::FixedBatchEvaluator;
use crate::nn::infer_opts::{Ep, InferOptions};
use crate::nn::ort_infer::OrtEvaluator;
use crate::nn::{Evaluator, NetOutput};

const FORMAT: &str = "chessbot-infer-parity-v1";
const BATCHES: [usize; 7] = [1, 7, 20, 31, 32, 33, 65];
const RAW_ATOL: f64 = 1e-4;
const RAW_RTOL: f64 = 1e-6;
const PROB_ATOL: f64 = 1e-5;

#[derive(Default)]
struct Comparison {
    max_raw_abs: f64,
    max_policy_prob_abs: f64,
    max_wdl_prob_abs: f64,
    raw_close: bool,
    probs_close: bool,
}

fn emit(out: &mut impl Write, value: Value) -> Result<()> {
    serde_json::to_writer(&mut *out, &value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

fn validate(outputs: &[NetOutput], batch: usize) -> Result<()> {
    ensure!(outputs.len() == batch, "output row count does not match requested batch");
    for output in outputs {
        ensure!(output.policy_logits.len() == NUM_MOVES, "policy output width mismatch");
        ensure!(output.policy_logits.iter().chain(output.wdl_logits.iter()).all(|x| x.is_finite()),
                "nonfinite policy or WDL logits");
    }
    Ok(())
}

fn probabilities(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let values: Vec<_> = logits.iter().map(|value| (f64::from(*value) - maximum).exp()).collect();
    let total: f64 = values.iter().sum();
    values.into_iter().map(|value| value / total).collect()
}

fn probability_difference(reference: &[f32], candidate: &[f32]) -> f64 {
    probabilities(reference).into_iter().zip(probabilities(candidate))
        .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
}

fn compare(reference: &[NetOutput], candidate: &[NetOutput], batch: usize) -> Result<Comparison> {
    validate(reference, batch)?;
    validate(candidate, batch)?;
    let mut result = Comparison { raw_close: true, probs_close: true, ..Comparison::default() };
    for (reference, candidate) in reference.iter().zip(candidate) {
        for (a, b) in reference.policy_logits.iter().chain(reference.wdl_logits.iter())
            .zip(candidate.policy_logits.iter().chain(candidate.wdl_logits.iter())) {
            let difference = (f64::from(*a) - f64::from(*b)).abs();
            result.max_raw_abs = result.max_raw_abs.max(difference);
            result.raw_close &= difference <= RAW_ATOL + RAW_RTOL * f64::from(*a).abs();
        }
        result.max_policy_prob_abs = result.max_policy_prob_abs.max(
            probability_difference(&reference.policy_logits, &candidate.policy_logits));
        result.max_wdl_prob_abs = result.max_wdl_prob_abs.max(
            probability_difference(&reference.wdl_logits, &candidate.wdl_logits));
    }
    result.probs_close = result.max_policy_prob_abs <= PROB_ATOL && result.max_wdl_prob_abs <= PROB_ATOL;
    Ok(result)
}

pub(super) fn run(model: &Path, opts: &InferOptions, memory_pattern: bool, width: usize) -> Result<()> {
    let mut reference = OrtEvaluator::new_with_memory_pattern(model, opts, memory_pattern)?;
    let raw_candidate = OrtEvaluator::new_with_memory_pattern(model, opts, memory_pattern)?;
    let mut candidate = FixedBatchEvaluator::new(Box::new(raw_candidate), width)?;
    let corpus = BENCH_FENS.iter().map(|(_, fen)| from_fen(fen).map(|board| encode_planes(&board)))
        .collect::<Result<Vec<_>>>()?;
    let requested_ep = match opts.ep { Ep::Cpu => "cpu", Ep::CoreMl => "coreml", Ep::Auto => "auto" };
    let selected_ep = if opts.ep == Ep::Auto {
        if cfg!(target_os = "macos") { "coreml" } else { "cpu" }
    } else { requested_ep };
    let mut out = io::stdout().lock();
    emit(&mut out, json!({
        "format": FORMAT, "kind": "session", "model": model,
        "requested_ep": requested_ep, "selected_ep": selected_ep,
        "provider_note": "Provider selection permits CPU fallback; two temporary parity sessions.",
        "threads": opts.threads, "memory_pattern": memory_pattern, "fixed_batch": width,
        "batches": BATCHES, "corpus": "engine::cli::bench::BENCH_FENS",
        "raw_atol": RAW_ATOL, "raw_rtol": RAW_RTOL, "prob_atol": PROB_ATOL,
        "policy_probability_mask": "none", "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH, "ort_build": ort::info(),
    }))?;
    let mut passed = true;
    for batch in BATCHES {
        let planes: Vec<_> = (0..batch).flat_map(|row| corpus[row % corpus.len()].iter().copied()).collect();
        eprintln!("infer-parity: requested {batch}, fixed width {width}");
        let comparison = (|| {
            let expected = reference.evaluate(&planes, batch)?;
            let actual = candidate.evaluate(&planes, batch)?;
            compare(&expected, &actual, batch)
        })();
        let value = match comparison {
            Ok(result) => {
                let pass = result.raw_close && result.probs_close;
                passed &= pass;
                json!({"format": FORMAT, "kind": "parity", "requested_batch": batch,
                    "fixed_batch": width, "physical_calls": batch.div_ceil(width),
                    "max_raw_abs": result.max_raw_abs, "raw_close": result.raw_close,
                    "max_policy_prob_abs": result.max_policy_prob_abs,
                    "max_wdl_prob_abs": result.max_wdl_prob_abs,
                    "probabilities_close": result.probs_close, "pass": pass})
            }
            Err(error) => {
                passed = false;
                json!({"format": FORMAT, "kind": "parity", "requested_batch": batch,
                    "fixed_batch": width, "pass": false, "error": error.to_string()})
            }
        };
        emit(&mut out, value)?;
    }
    emit(&mut out, json!({"format": FORMAT, "kind": "summary", "batches_checked": BATCHES.len(),
                          "fixed_batch": width, "pass": passed}))?;
    ensure!(passed, "fixed-batch inference parity failed; inspect parity JSONL records");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> NetOutput {
        NetOutput { policy_logits: vec![0.0; NUM_MOVES], wdl_logits: [0.0, 0.0, 0.0] }
    }

    #[test]
    fn raw_relative_tolerance_uses_each_reference_logit() {
        let mut reference = row();
        reference.policy_logits[0] = 1000.0;
        let mut candidate = reference.clone();
        candidate.policy_logits[0] += 0.0005;
        assert!(compare(&[reference.clone()], &[candidate], 1).unwrap().raw_close);
        let mut candidate = reference.clone();
        candidate.policy_logits[1] = 0.0002;
        let result = compare(&[reference], &[candidate], 1).unwrap();
        assert!(!result.raw_close);
    }

    #[test]
    fn probability_tolerance_is_independent_of_raw_logit_tolerance() {
        let reference = row();
        let mut candidate = row();
        candidate.wdl_logits[0] = 0.00009;
        let result = compare(&[reference], &[candidate], 1).unwrap();
        assert!(result.raw_close);
        assert!(!result.probs_close);
    }

    #[test]
    fn rejects_shapes_nonfinite_outputs_and_row_reordering() {
        assert!(compare(&[], &[row()], 1).is_err());
        let mut wrong_width = row();
        wrong_width.policy_logits.pop();
        assert!(compare(&[row()], &[wrong_width], 1).is_err());
        for value in [f32::NAN, f32::INFINITY] {
            let mut nonfinite = row();
            nonfinite.wdl_logits[0] = value;
            assert!(compare(&[nonfinite], &[row()], 1).is_err());
        }
        let mut other = row();
        other.wdl_logits = [-0.5, 0.0, 0.5];
        assert!(!compare(&[row(), other.clone()], &[other, row()], 2).unwrap().raw_close);
    }
}

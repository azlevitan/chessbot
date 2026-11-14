//! Stable physical batches preserve logical inputs, output order and failures.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use engine::encoder::move_mapper::NUM_MOVES;
use engine::encoder::tensor_encoder::TENSOR_LEN;
use engine::nn::fixed_batch::FixedBatchEvaluator;
use engine::nn::infer_opts::{Ep, InferOptions};
use engine::nn::ort_infer::load_evaluator;
use engine::nn::{Evaluator, NetOutput};

type Calls = Rc<RefCell<Vec<(usize, Vec<f32>)>>>;

fn oracle(row: &[f32]) -> NetOutput {
    let value = row[0] + row[TENSOR_LEN - 1] * 0.25;
    NetOutput {
        policy_logits: (0..NUM_MOVES).map(|index| value + (index % 13) as f32 / 16.0).collect(),
        wdl_logits: [-value, value * 0.5, value * 0.25],
    }
}

struct Recording {
    calls: Calls,
    broken: Option<u8>,
}

impl Evaluator for Recording {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        let mut calls = self.calls.borrow_mut();
        calls.push((batch, planes.to_vec()));
        let mut result: Vec<_> = planes.chunks_exact(TENSOR_LEN).map(oracle).collect();
        if calls.len() == 2 {
            if let Some(broken) = self.broken.take() {
                match broken {
                    0 => { result.pop(); }
                    1 => { result[0].policy_logits.pop(); }
                    2 => result[0].policy_logits[0] = f32::NAN,
                    3 => result[0].wdl_logits[2] = f32::INFINITY,
                    _ => anyhow::bail!("injected physical-call failure"),
                }
            }
        }
        Ok(result)
    }
}

fn input(batch: usize) -> Vec<f32> {
    let mut result = vec![0.0; batch * TENSOR_LEN];
    for (index, row) in result.chunks_exact_mut(TENSOR_LEN).enumerate() {
        row[0] = index as f32 + 1.0;
        row[TENSOR_LEN - 1] = index as f32 * 0.5;
    }
    result
}

fn recorder(width: usize, broken: Option<u8>) -> (FixedBatchEvaluator, Calls) {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let inner = Recording { calls: Rc::clone(&calls), broken };
    (FixedBatchEvaluator::new(Box::new(inner), width).unwrap(), calls)
}

fn assert_outputs(outputs: &[NetOutput], planes: &[f32]) {
    assert_eq!(outputs.len() * TENSOR_LEN, planes.len());
    for (output, row) in outputs.iter().zip(planes.chunks_exact(TENSOR_LEN)) {
        let expected = oracle(row);
        assert_eq!(output.policy_logits, expected.policy_logits);
        assert_eq!(output.wdl_logits, expected.wdl_logits);
    }
}

#[test]
fn every_physical_call_has_fixed_width_and_padding_never_leaks_between_requests() {
    let (mut eval, calls) = recorder(32, None);
    for requested in [1, 20, 31, 32, 33, 65] {
        calls.borrow_mut().clear();
        let planes = input(requested);
        assert_outputs(&eval.evaluate(&planes, requested).unwrap(), &planes);
        let observed = calls.borrow();
        assert_eq!(observed.len(), requested.div_ceil(32));
        for ((physical, actual), logical) in observed.iter().zip(planes.chunks(32 * TENSOR_LEN)) {
            assert_eq!(*physical, 32);
            assert_eq!(actual.len(), 32 * TENSOR_LEN);
            assert_eq!(&actual[..logical.len()], logical);
            assert!(actual[logical.len()..].iter().all(|value| *value == 0.0));
        }
    }
}

#[test]
fn invalid_width_or_logical_input_is_rejected_before_calling_the_inner_evaluator() {
    for width in [0, 65] {
        let inner = Recording { calls: Rc::new(RefCell::new(Vec::new())), broken: None };
        assert!(FixedBatchEvaluator::new(Box::new(inner), width).is_err());
    }
    let (mut eval, calls) = recorder(32, None);
    assert!(eval.evaluate(&[], 0).is_err());
    assert!(eval.evaluate(&vec![0.0; TENSOR_LEN - 1], 1).is_err());
    assert!(eval.evaluate(&vec![0.0; TENSOR_LEN], 2).is_err());
    assert!(calls.borrow().is_empty());
}

#[test]
fn malformed_nonfinite_or_failed_second_chunks_return_errors_and_allow_a_clean_retry() {
    for broken in 0..5 {
        let (mut eval, calls) = recorder(32, Some(broken));
        let planes = input(33);
        assert!(eval.evaluate(&planes, 33).is_err(), "accepted broken mode {broken}");
        assert_eq!(calls.borrow().len(), 2);
        assert_outputs(&eval.evaluate(&planes, 33).unwrap(), &planes);
        assert_eq!(calls.borrow().len(), 4);
    }
}

#[test]
fn real_ort_fixture_returns_only_requested_rows_for_partial_and_split_batches() {
    let model = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"),
                                           "/tests/fixtures/match_uniform.onnx"));
    let mut eval = load_evaluator(Some(model), &InferOptions {
        ep: Ep::Cpu, threads: 1, infer_batch: Some(32),
    }).unwrap();
    for batch in [1, 20, 31, 32, 33, 65] {
        let outputs = eval.evaluate(&input(batch), batch).unwrap();
        assert_eq!(outputs.len(), batch);
        for output in outputs {
            assert_eq!(output.policy_logits, vec![0.0; NUM_MOVES]);
            assert_eq!(output.wdl_logits, [0.0; 3]);
        }
    }
}

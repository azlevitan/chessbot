//! Exact-shape ORT latency diagnostic; search and training outputs are untouched.

use std::collections::HashSet;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use cozy_chess::Board;
use serde_json::{json, Value};

use crate::encoder::move_mapper::NUM_MOVES;
use crate::encoder::tensor_encoder::{encode_planes, TENSOR_LEN};
use crate::nn::infer_opts::{Ep, InferOptions};
use crate::nn::ort_infer::OrtEvaluator;
use crate::nn::{Evaluator, NetOutput};

const FORMAT: &str = "chessbot-infer-bench-v1";
const USAGE: &str = "usage: engine infer-bench --model <path.onnx>
    [--ep cpu|coreml|auto] [--threads 1] [--batches 32,31,32,20]
    [--cycles 5] [--repeat 2] [--memory-pattern on|off] [--pad-to N]
Measure exact batch shapes through one ORT session; stream JSONL measurements.
Each requested batch repeats before changing size. Batches/padding: 1..64;
at most 10,000 calls. Padding is diagnostic only and uses zero-filled extra rows.";

struct Args {
    model: PathBuf,
    infer: InferOptions,
    batches: Vec<usize>,
    cycles: usize,
    repeat: usize,
    memory_pattern: bool,
    pad_to: Option<usize>,
}

fn number(flag: &str, text: &str) -> Result<usize> {
    text.parse().with_context(|| format!("bad {flag} value {text:?}"))
}

fn parse(args: &[String]) -> Result<Args> {
    let mut model = None;
    let mut a = Args {
        model: PathBuf::new(), infer: InferOptions { ep: Ep::Cpu, threads: 1 },
        batches: vec![32, 31, 32, 20], cycles: 5, repeat: 2,
        memory_pattern: true, pad_to: None,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        ensure!(matches!(flag.as_str(), "--model" | "--ep" | "--threads" | "--batches"
                         | "--cycles" | "--repeat" | "--memory-pattern" | "--pad-to"),
                "unknown argument for infer-bench: {flag}");
        let value = it.next().with_context(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--ep" | "--threads" => { a.infer.parse_flag(flag, value)?; }
            "--batches" => {
                a.batches = value.split(',').map(|v| number(flag, v.trim())).collect::<Result<_>>()?;
            }
            "--cycles" => a.cycles = number(flag, value)?,
            "--repeat" => a.repeat = number(flag, value)?,
            "--pad-to" => a.pad_to = Some(number(flag, value)?),
            "--memory-pattern" => a.memory_pattern = match value.as_str() {
                "on" => true, "off" => false,
                _ => bail!("--memory-pattern must be on or off"),
            },
            _ => unreachable!(),
        }
    }
    a.model = model.context("infer-bench requires --model <path.onnx>")?;
    ensure!(!a.model.as_os_str().is_empty(), "--model must not be empty");
    ensure!(!a.batches.is_empty() && a.batches.iter().all(|b| (1..=64).contains(b)),
            "--batches must contain sizes in 1..64");
    ensure!(a.cycles > 0 && a.repeat > 0, "--cycles and --repeat must be positive");
    let calls = a.batches.len().checked_mul(a.cycles).and_then(|n| n.checked_mul(a.repeat));
    ensure!(calls.is_some_and(|n| n <= 10_000), "infer-bench is limited to 10,000 calls");
    if let Some(pad) = a.pad_to {
        ensure!((1..=64).contains(&pad) && a.batches.iter().all(|b| *b <= pad),
                "--pad-to must be in 1..64 and cover every requested batch");
    }
    a.infer.validate()?;
    Ok(a)
}

fn emit(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

fn check_outputs(outputs: &[NetOutput], batch: usize) -> Result<()> {
    ensure!(outputs.len() == batch, "evaluator returned the wrong row count");
    for output in outputs {
        ensure!(output.policy_logits.len() == NUM_MOVES, "policy output shape mismatch");
        ensure!(output.policy_logits.iter().chain(output.wdl_logits.iter()).all(|x| x.is_finite()),
                "evaluator returned nonfinite logits");
    }
    Ok(())
}

fn ep_name(ep: Ep) -> &'static str {
    match ep { Ep::Cpu => "cpu", Ep::CoreMl => "coreml", Ep::Auto => "auto" }
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let mut a = parse(args)?;
    a.model = a.model.canonicalize().with_context(|| format!("opening model {}", a.model.display()))?;
    eprintln!("infer-bench: loading {} (selected provider {}; CPU fallback may apply)",
              a.model.display(), ep_name(a.infer.ep));
    let load = Instant::now();
    let mut eval = OrtEvaluator::new_with_memory_pattern(&a.model, &a.infer, a.memory_pattern)?;
    let load_seconds = load.elapsed().as_secs_f64();
    let encoded = encode_planes(&Board::default());
    let max_batch = a.pad_to.unwrap_or_else(|| *a.batches.iter().max().unwrap());
    let mut planes = vec![0.0; max_batch * TENSOR_LEN];
    let selected = match a.infer.ep {
        Ep::Auto if cfg!(target_os = "macos") => "coreml",
        Ep::Auto => "cpu",
        ep => ep_name(ep),
    };
    let mut out = io::stdout().lock();
    emit(&mut out, &json!({
        "format": FORMAT, "kind": "session", "model": a.model,
        "selected_ep": selected, "requested_ep": ep_name(a.infer.ep), "threads": a.infer.threads,
        "memory_pattern": a.memory_pattern, "pad_to": a.pad_to, "batches": a.batches,
        "cycles": a.cycles, "repeat": a.repeat, "load_seconds": load_seconds,
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH, "ort_build": ort::info(),
    }))?;
    let mut seen = HashSet::new();
    let (mut index, mut previous, mut inference_seconds) = (0usize, None, 0.0);
    let (mut requested_positions, mut physical_positions) = (0usize, 0usize);
    for cycle in 1..=a.cycles {
        for requested in &a.batches {
            let physical = a.pad_to.unwrap_or(*requested);
            let input = &mut planes[..physical * TENSOR_LEN];
            input.fill(0.0);
            for row in input.chunks_exact_mut(TENSOR_LEN).take(*requested) {
                row.copy_from_slice(&encoded);
            }
            for repeat_index in 1..=a.repeat {
                index += 1;
                eprintln!("infer-bench: call {index}, cycle {cycle}, requested {requested}, physical {physical}");
                let started = Instant::now();
                let outputs = eval.evaluate(input, physical)?;
                let seconds = started.elapsed().as_secs_f64();
                check_outputs(&outputs, physical)?;
                drop(outputs);
                let first_for_shape = seen.insert(physical);
                let shape_changed = previous.is_some_and(|batch| batch != physical);
                previous = Some(physical);
                inference_seconds += seconds;
                requested_positions += requested;
                physical_positions += physical;
                emit(&mut out, &json!({
                    "format": FORMAT, "kind": "call", "index": index, "cycle": cycle,
                    "requested_batch": requested, "physical_batch": physical,
                    "shape_changed": shape_changed, "first_for_shape": first_for_shape,
                    "seconds": seconds, "repeat_index": repeat_index,
                }))?;
            }
        }
    }
    emit(&mut out, &json!({
        "format": FORMAT, "kind": "summary", "calls": index,
        "requested_positions": requested_positions, "physical_positions": physical_positions,
        "inference_seconds": inference_seconds,
    }))
}

#[cfg(test)]
mod tests {
    use super::{check_outputs, NetOutput, NUM_MOVES};

    fn finite_output() -> NetOutput {
        NetOutput { policy_logits: vec![0.0; NUM_MOVES], wdl_logits: [-0.5, 0.0, 0.5] }
    }

    #[test]
    fn accepts_finite_outputs_and_rejects_incorrect_row_counts() {
        assert!(check_outputs(&[finite_output(), finite_output()], 2).is_ok());
        assert!(check_outputs(&[], 1).is_err());
        assert!(check_outputs(&[finite_output()], 2).is_err());
    }

    #[test]
    fn rejects_an_incorrect_policy_width() {
        let mut output = finite_output();
        output.policy_logits.pop();
        assert!(check_outputs(&[output], 1).is_err());
    }

    #[test]
    fn rejects_nonfinite_values_in_either_head() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut policy = finite_output();
            policy.policy_logits[0] = value;
            assert!(check_outputs(&[policy], 1).is_err());
            let mut wdl = finite_output();
            wdl.wdl_logits[1] = value;
            assert!(check_outputs(&[wdl], 1).is_err());
        }
    }
}

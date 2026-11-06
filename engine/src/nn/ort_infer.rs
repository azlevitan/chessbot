//! ONNX Runtime inference (SPEC.md §4) and the uniform dummy fallback.

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{anyhow, ensure, Context, Result};
use ort::environment::Environment;
#[cfg(target_os = "macos")]
use ort::ep::CoreML;
use ort::session::Session;
use ort::value::Tensor;

use super::{Evaluator, NetOutput};
use super::infer_opts::InferOptions;
#[cfg(target_os = "macos")]
use super::infer_opts::Ep;
use crate::encoder::move_mapper::NUM_MOVES;
use crate::encoder::tensor_encoder::TENSOR_LEN;

pub const INPUT_NAME: &str = "input_planes";
pub const POLICY_NAME: &str = "policy_logits";
pub const WDL_NAME: &str = "wdl_logits";

// ort rc.15 permits only one environment construction for the entire process,
// including after all sessions have been dropped. Keep its shared handle alive
// across both match players and all games; session options remain independent.
fn shared_environment() -> Result<Environment> {
    static ENV: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        ort::init()
            .with_name("chessbot")
            .build()
            .map_err(|error| error.to_string())
    })
    .as_ref()
    .cloned()
    .map_err(|error| anyhow!("initializing ONNX Runtime: {error}"))
}

/// Dummy network: uniform policy over legal moves and a dead-even value (v = 0).
#[derive(Default)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        ensure!(planes.len() == batch * TENSOR_LEN, "planes length mismatch");
        let out = NetOutput { policy_logits: vec![0.0; NUM_MOVES], wdl_logits: [0.0; 3] };
        Ok(vec![out; batch])
    }
}

/// An exported `ChessNet` running in ONNX Runtime (CoreML EP when available).
pub struct OrtEvaluator {
    session: Session,
    _env: Environment,
}

impl OrtEvaluator {
    pub fn new(model_path: &Path, opts: &InferOptions) -> Result<Self> {
        opts.validate()?;
        let env = shared_environment()?;
        let mut builder = Session::builder(&env)
            .map_err(|e| anyhow!("creating session builder: {e}"))?
            .with_intra_threads(opts.threads)
            .map_err(|e| anyhow!("setting intra-op threads: {e}"))?;
        // CoreML registration failure is non-fatal: ORT falls back to the CPU EP.
        #[cfg(target_os = "macos")]
        if opts.ep != Ep::Cpu {
            builder = builder
                .with_execution_providers([CoreML::default().build()])
                .map_err(|e| anyhow!("registering CoreML EP: {e}"))?;
        }
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("loading {}: {e}", model_path.display()))?;
        let inputs: Vec<&str> = session.inputs().iter().map(|o| o.name()).collect();
        let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
        ensure!(inputs == [INPUT_NAME], "model inputs {inputs:?}, want [{INPUT_NAME:?}]");
        ensure!(
            outputs.contains(&POLICY_NAME) && outputs.contains(&WDL_NAME),
            "model outputs {outputs:?}, want {POLICY_NAME:?} and {WDL_NAME:?}"
        );
        Ok(Self { session, _env: env })
    }
}

impl Evaluator for OrtEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        ensure!(batch > 0 && planes.len() == batch * TENSOR_LEN, "planes length mismatch");
        let input = Tensor::from_array(([batch, 18, 8, 8], planes.to_vec()))
            .map_err(|e| anyhow!("building input tensor: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![INPUT_NAME => input])
            .map_err(|e| anyhow!("running model: {e}"))?;
        let (_, policy) = outputs[POLICY_NAME]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("reading {POLICY_NAME}: {e}"))?;
        let (_, wdl) = outputs[WDL_NAME]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("reading {WDL_NAME}: {e}"))?;
        ensure!(policy.len() == batch * NUM_MOVES, "policy output has {} values", policy.len());
        ensure!(wdl.len() == batch * 3, "wdl output has {} values", wdl.len());
        Ok((0..batch)
            .map(|i| NetOutput {
                policy_logits: policy[i * NUM_MOVES..(i + 1) * NUM_MOVES].to_vec(),
                wdl_logits: [wdl[i * 3], wdl[i * 3 + 1], wdl[i * 3 + 2]],
            })
            .collect())
    }
}

/// `Some(path)` -> ONNX model, `None` -> uniform dummy evaluator.
pub fn load_evaluator(model_path: Option<&Path>, opts: &InferOptions) -> Result<Box<dyn Evaluator>> {
    opts.validate()?;
    match model_path {
        Some(path) => Ok(Box::new(
            OrtEvaluator::new(path, opts).with_context(|| format!("model {}", path.display()))?,
        )),
        None => Ok(Box::new(UniformEvaluator)),
    }
}

#[cfg(test)]
mod tests {
    use super::shared_environment;

    #[test]
    fn concurrent_loaders_and_later_games_can_share_the_process_environment() {
        let loaders: Vec<_> = (0..4)
            .map(|_| std::thread::spawn(shared_environment))
            .collect();
        let environments: Vec<_> = loaders
            .into_iter()
            .map(|loader| loader.join().unwrap().unwrap())
            .collect();
        drop(environments);
        assert!(shared_environment().is_ok());
    }
}

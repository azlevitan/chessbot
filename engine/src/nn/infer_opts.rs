//! Portable execution-provider, thread and physical batch options (SPEC.md §8).

use anyhow::{bail, ensure, Context, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ep {
    #[default]
    Auto,
    Cpu,
    CoreMl,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferOptions {
    pub ep: Ep,
    pub threads: usize,
    /// Optional fixed physical ONNX batch; padding never adds logical positions.
    pub infer_batch: Option<usize>,
}

impl Default for InferOptions {
    fn default() -> Self {
        Self { ep: Ep::Auto, threads: 1, infer_batch: None }
    }
}

impl InferOptions {
    /// Parse a shared CLI option; return false for flags owned by the caller.
    pub fn parse_flag(&mut self, flag: &str, value: &str) -> Result<bool> {
        match flag {
            "--ep" => {
                self.ep = match value {
                    "auto" => Ep::Auto,
                    "cpu" => Ep::Cpu,
                    "coreml" => Ep::CoreMl,
                    _ => bail!("invalid --ep {value:?}: expected auto, cpu, or coreml"),
                };
            }
            "--threads" => {
                let threads = value.parse::<usize>().context("--threads requires a positive integer")?;
                ensure!(threads > 0, "--threads must be at least 1");
                self.threads = threads;
            }
            "--infer-batch" => {
                let width = value.parse::<usize>().context("--infer-batch requires an integer in 1..64")?;
                ensure!((1..=64).contains(&width), "--infer-batch must be in 1..64");
                self.infer_batch = Some(width);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Validate direct API callers as well as CLI input, even for uniform nets.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.threads > 0, "--threads must be at least 1");
        ensure!(self.infer_batch.map_or(true, |width| (1..=64).contains(&width)),
                "--infer-batch must be in 1..64");
        ensure!(
            self.ep != Ep::CoreMl || cfg!(target_os = "macos"),
            "--ep coreml is only supported on macOS"
        );
        Ok(())
    }
}

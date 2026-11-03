//! Portable execution-provider and ONNX Runtime thread options (SPEC.md §8).

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
}

impl Default for InferOptions {
    fn default() -> Self {
        Self { ep: Ep::Auto, threads: 1 }
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
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Validate direct API callers as well as CLI input, even for uniform nets.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.threads > 0, "--threads must be at least 1");
        ensure!(
            self.ep != Ep::CoreMl || cfg!(target_os = "macos"),
            "--ep coreml is only supported on macOS"
        );
        Ok(())
    }
}

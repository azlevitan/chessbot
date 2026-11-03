//! Lock-step inference and bounded, game-index-ordered output.

use std::collections::VecDeque;

use anyhow::{ensure, Result};
use rand::{rngs::StdRng, SeedableRng};

use super::driver::GameDriver;
use super::{GameRecord, SelfPlayConfig};
use crate::nn::{evaluate_boards, Evaluator};

#[derive(Clone, Copy, Debug)]
pub struct BatchOptions {
    pub parallel: usize,
    pub leaf_batch: usize,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            parallel: 16,
            leaf_batch: 1,
        }
    }
}

impl BatchOptions {
    pub fn validate(&self, cfg: &SelfPlayConfig) -> Result<()> {
        ensure!(self.parallel > 0, "--parallel must be positive");
        ensure!(self.leaf_batch > 0, "--leaf-batch must be positive");
        ensure!(
            cfg.sims >= 2,
            "selfplay needs at least 2 simulations per move"
        );
        Ok(())
    }
}

/// One call over all pending leaves, with every output validated before backup.
/// If evaluation fails, dropping the guards undoes every tree's virtual losses.
pub(crate) fn run_step(
    eval: &mut dyn Evaluator,
    drivers: &mut [GameDriver],
    leaf_batch: usize,
) -> Result<()> {
    let pending: Vec<_> = drivers
        .iter_mut()
        .filter_map(|d| d.step_request(leaf_batch))
        .collect();
    let boards: Vec<_> = pending
        .iter()
        .flat_map(|p| p.boards().iter().cloned())
        .collect();
    let mut outputs = evaluate_boards(eval, &boards)?.into_iter();
    for request in pending {
        request.complete(&mut outputs);
    }
    for driver in drivers {
        driver.step_apply()?;
    }
    Ok(())
}

/// At most P games (active or finished) occupy the reorder window. A slow early
/// game cannot cause completed later games to accumulate without bound.
pub fn run_games(
    eval: &mut dyn Evaluator,
    games: u32,
    seed: u64,
    cfg: &SelfPlayConfig,
    opts: BatchOptions,
    mut emit: impl FnMut(u32, GameRecord) -> Result<()>,
) -> Result<()> {
    opts.validate(cfg)?;
    let mut drivers = VecDeque::new();
    let (mut next, mut emitted) = (0, 0);
    while emitted < games {
        while next < games && drivers.len() < opts.parallel {
            let rng = StdRng::seed_from_u64(seed.wrapping_add(next as u64));
            drivers.push_back(GameDriver::new(cfg, rng)?);
            next += 1;
        }
        run_step(eval, drivers.make_contiguous(), opts.leaf_batch)?;
        while drivers.front().is_some_and(GameDriver::is_finished) {
            let game = drivers.pop_front().unwrap().into_record();
            emit(emitted, game)?;
            emitted += 1;
        }
    }
    Ok(())
}

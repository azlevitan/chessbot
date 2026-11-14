//! Lock-step inference and bounded, game-index-ordered output.

use std::collections::VecDeque;

use anyhow::{ensure, Result};
use rand::{rngs::StdRng, SeedableRng};

use super::driver::GameDriver;
use super::progress::{ProgressPhase, ProgressSnapshot};
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
        cfg.validate()
    }
}

/// One call over all pending leaves, with every output validated before backup.
/// If evaluation fails, dropping the guards undoes every tree's virtual losses.
pub(crate) fn run_step(
    eval: &mut dyn Evaluator,
    drivers: &mut [GameDriver],
    leaf_batch: usize,
) -> Result<()> {
    run_step_observed(eval, drivers, leaf_batch, |_, _| {})
}

fn run_step_observed(
    eval: &mut dyn Evaluator,
    drivers: &mut [GameDriver],
    leaf_batch: usize,
    mut evaluating: impl FnMut(usize, bool),
) -> Result<()> {
    let pending: Vec<_> = drivers
        .iter_mut()
        .filter_map(|d| d.step_request(leaf_batch))
        .collect();
    let boards: Vec<_> = pending
        .iter()
        .flat_map(|p| p.boards().iter().cloned())
        .collect();
    if !boards.is_empty() { evaluating(boards.len(), true); }
    let mut outputs = evaluate_boards(eval, &boards)?.into_iter();
    if !boards.is_empty() { evaluating(boards.len(), false); }
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
    emit: impl FnMut(u32, GameRecord) -> Result<()>,
) -> Result<()> {
    run_games_with_progress(eval, games, seed, cfg, opts, emit, |_| {})
}

/// Observations do not consume RNG or change the order of search or output.
pub fn run_games_with_progress(
    eval: &mut dyn Evaluator,
    games: u32,
    seed: u64,
    cfg: &SelfPlayConfig,
    opts: BatchOptions,
    mut emit: impl FnMut(u32, GameRecord) -> Result<()>,
    mut observe: impl FnMut(ProgressSnapshot),
) -> Result<()> {
    opts.validate(cfg)?;
    let mut drivers = VecDeque::new();
    let (mut next, mut emitted) = (0, 0);
    let mut emitted_plies = 0;
    let mut progress = ProgressSnapshot { games, phase: ProgressPhase::Searching,
                                         ..ProgressSnapshot::default() };
    observe(progress);
    while emitted < games {
        while next < games && drivers.len() < opts.parallel {
            let rng = StdRng::seed_from_u64(seed.wrapping_add(next as u64));
            drivers.push_back(GameDriver::new(cfg, rng)?);
            next += 1;
        }
        refresh(&mut progress, &drivers, next, emitted, emitted_plies);
        observe(progress);
        let step = run_step_observed(eval, drivers.make_contiguous(), opts.leaf_batch,
            |positions, entering| {
                progress.phase = if entering { ProgressPhase::Evaluating } else { ProgressPhase::Searching };
                progress.batch_positions = if entering { positions } else { 0 };
                if entering { progress.eval_calls += 1; } else { progress.eval_positions += positions as u64; }
                observe(progress);
            });
        if let Err(error) = step {
            progress.phase = ProgressPhase::Failed;
            observe(progress);
            return Err(error);
        }
        refresh(&mut progress, &drivers, next, emitted, emitted_plies);
        observe(progress);
        while drivers.front().is_some_and(GameDriver::is_finished) {
            progress.phase = ProgressPhase::Writing;
            observe(progress);
            let game = drivers.pop_front().unwrap().into_record();
            let plies = game.plies.iter().filter(|ply| ply.played.is_some()).count() as u64;
            if let Err(error) = emit(emitted, game) {
                progress.phase = ProgressPhase::Failed;
                observe(progress);
                return Err(error);
            }
            emitted_plies += plies;
            emitted += 1;
            progress.phase = ProgressPhase::Searching;
            refresh(&mut progress, &drivers, next, emitted, emitted_plies);
            observe(progress);
        }
    }
    progress.phase = ProgressPhase::Complete;
    observe(progress);
    Ok(())
}

fn refresh(progress: &mut ProgressSnapshot, drivers: &VecDeque<GameDriver>,
           started: u32, emitted: u32, emitted_plies: u64) {
    let waiting = drivers.iter().filter(|driver| driver.is_finished()).count();
    progress.started = started;
    progress.emitted = emitted;
    progress.completed = emitted + waiting as u32;
    progress.active = drivers.len() - waiting;
    progress.waiting = waiting;
    progress.plies = emitted_plies + drivers.iter().map(|driver| driver.played_plies() as u64).sum::<u64>();
    progress.first_game = drivers.front().map(|_| emitted);
    progress.first_ply = drivers.front().map_or(0, GameDriver::played_plies);
}

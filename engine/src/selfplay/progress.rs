//! Small shared snapshots keep CLI progress visible during blocked inference.

use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProgressPhase {
    #[default]
    LoadingModel,
    Searching,
    Evaluating,
    Writing,
    Complete,
    Failed,
}

impl ProgressPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LoadingModel => "loading_model",
            Self::Searching => "searching",
            Self::Evaluating => "evaluating",
            Self::Writing => "writing",
            Self::Complete => "complete",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgressSnapshot {
    pub phase: ProgressPhase,
    pub games: u32,
    pub started: u32,
    pub completed: u32,
    pub emitted: u32,
    pub active: usize,
    pub waiting: usize,
    /// Actual board moves, excluding a final resignation training sample.
    pub plies: u64,
    /// Zero-based oldest game still in the bounded reorder window.
    pub first_game: Option<u32>,
    pub first_ply: usize,
    pub eval_calls: u64,
    /// Positions whose combined evaluation and validation returned successfully.
    pub eval_positions: u64,
    pub batch_positions: usize,
}

pub fn format_progress(
    snapshot: ProgressSnapshot,
    elapsed: Duration,
    in_flight: Duration,
    idle: Duration,
) -> String {
    let first = snapshot.first_game.map_or_else(
        || "none".to_owned(),
        |index| format!("{}:ply{}", index + 1, snapshot.first_ply),
    );
    format!(
        "selfplay progress: elapsed={:.1}s phase={} completed={}/{} emitted={} \
         active={} waiting={} moves={} first_game={} eval_calls={} eval_positions={} \
         batch_positions={} in_flight={:.1}s since_progress={:.1}s",
        elapsed.as_secs_f64(), snapshot.phase.as_str(), snapshot.completed, snapshot.games,
        snapshot.emitted, snapshot.active, snapshot.waiting, snapshot.plies, first,
        snapshot.eval_calls, snapshot.eval_positions, snapshot.batch_positions,
        in_flight.as_secs_f64(), idle.as_secs_f64(),
    )
}

struct State {
    snapshot: ProgressSnapshot,
    last_progress: Instant,
    evaluating_since: Option<Instant>,
    stopped: bool,
}

/// One small watchdog thread; no worker pool, tree copy, or tensor is retained.
/// Create before loading a model to distinguish startup from stalled inference.
pub struct ProgressReporter {
    shared: Arc<(Mutex<State>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

impl ProgressReporter {
    pub fn stderr(games: u32, interval: Duration) -> Self {
        Self::with_sink(games, interval, |line| eprintln!("{line}"))
    }

    fn with_sink(games: u32, interval: Duration, sink: impl Fn(String) + Send + 'static) -> Self {
        let started = Instant::now();
        let snapshot = ProgressSnapshot { games, ..ProgressSnapshot::default() };
        sink(format_progress(snapshot, Duration::ZERO, Duration::ZERO, Duration::ZERO));
        let shared = Arc::new((Mutex::new(State {
            snapshot, last_progress: started, evaluating_since: None, stopped: false,
        }), Condvar::new()));
        let worker = Arc::clone(&shared);
        let thread = thread::spawn(move || loop {
            let (lock, wake) = &*worker;
            let state = lock.lock().unwrap();
            let state = if state.stopped { state } else {
                wake.wait_timeout(state, interval.max(Duration::from_millis(1))).unwrap().0
            };
            let now = Instant::now();
            let line = format_progress(state.snapshot, now.duration_since(started),
                state.evaluating_since.map_or(Duration::ZERO, |when| now.duration_since(when)),
                now.duration_since(state.last_progress));
            let stopped = state.stopped;
            drop(state);
            sink(line);
            if stopped { break; }
        });
        Self { shared, thread: Some(thread) }
    }

    pub fn update(&self, snapshot: ProgressSnapshot) {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap();
        let old = state.snapshot;
        let now = Instant::now();
        if (old.plies, old.completed, old.emitted, old.eval_positions)
            != (snapshot.plies, snapshot.completed, snapshot.emitted, snapshot.eval_positions) {
            state.last_progress = now;
        }
        if snapshot.phase != ProgressPhase::Evaluating {
            state.evaluating_since = None;
        } else if old.phase != ProgressPhase::Evaluating {
            state.evaluating_since = Some(now);
        }
        state.snapshot = snapshot;
        if matches!(snapshot.phase, ProgressPhase::Complete | ProgressPhase::Failed) {
            wake.notify_one();
        }
    }

    pub fn set_phase(&self, phase: ProgressPhase) {
        let mut snapshot = self.shared.0.lock().unwrap().snapshot;
        snapshot.phase = phase;
        self.update(snapshot);
    }
}

impl Drop for ProgressReporter {
    fn drop(&mut self) {
        let (lock, wake) = &*self.shared;
        lock.lock().unwrap().stopped = true;
        wake.notify_one();
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

#[cfg(test)]
#[path = "progress_tests.rs"]
mod tests;

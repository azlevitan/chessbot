//! Self-play sidecar counters; resignation remains disabled until Phase 7.5.

use serde::Serialize;
use serde_json::{json, Value};

use super::{GameRecord, SelfPlayConfig, Termination};
use crate::cli::bin_writer::GameResult;

#[derive(Default, Serialize)]
pub struct Results {
    white: u64,
    black: u64,
    draw: u64,
}

#[derive(Default, Serialize)]
pub struct Terminations {
    checkmate: u64,
    resign: u64,
    rule_draw: u64,
    repetition: u64,
    insufficient_material: u64,
    ply_cap: u64,
}

#[derive(Default)]
pub struct SelfPlayStats {
    games: u64,
    results: Results,
    terminations: Terminations,
}

impl SelfPlayStats {
    pub fn record(&mut self, game: &GameRecord) {
        self.games += 1;
        match game.result {
            GameResult::WhiteWins => self.results.white += 1,
            GameResult::BlackWins => self.results.black += 1,
            GameResult::Draw => self.results.draw += 1,
        }
        match game.termination {
            Termination::Checkmate => self.terminations.checkmate += 1,
            Termination::RuleDraw => self.terminations.rule_draw += 1,
            Termination::Repetition => self.terminations.repetition += 1,
            Termination::InsufficientMaterial => self.terminations.insufficient_material += 1,
            Termination::PlyCap => self.terminations.ply_cap += 1,
        }
    }

    pub fn sidecar(&self, model: &str, seed: u64, cfg: &SelfPlayConfig) -> Value {
        json!({
            "kind": "selfplay", "games": self.games, "model": model,
            "seed": seed, "sims": cfg.sims, "max_plies": cfg.max_plies,
            "temp_plies": cfg.temp_plies, "results": self.results,
            "terminations": self.terminations,
            "resign": { "threshold": -1.0, "consecutive": 0,
                        "playthrough_games": 0, "false_positives": 0 }
        })
    }
}

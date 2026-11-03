//! Streaming `.bin` writer for `TrainingSample` records plus game-result labelling.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use cozy_chess::Color;

use crate::encoder::sample::TrainingSample;

/// Final result of a game, from White's point of view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    WhiteWins,
    BlackWins,
    Draw,
}

impl GameResult {
    /// Parse a PGN result token (`1-0`, `0-1`, `1/2-1/2`); `*` and others -> `None`.
    pub fn from_pgn(token: &str) -> Option<Self> {
        match token {
            "1-0" => Some(Self::WhiteWins),
            "0-1" => Some(Self::BlackWins),
            "1/2-1/2" => Some(Self::Draw),
            _ => None,
        }
    }

    /// The side `winner` won (i.e. the opponent was mated or resigned).
    pub fn win_for(winner: Color) -> Self {
        if winner == Color::White { Self::WhiteWins } else { Self::BlackWins }
    }
}

/// `(outcome_wdl, value)` for the side to move `stm`: Win = (2, 1), Draw = (1, 0), Loss = (0, -1).
pub fn outcome_for(stm: Color, result: GameResult) -> (u8, f32) {
    match (result, stm) {
        (GameResult::Draw, _) => (1, 0.0),
        (GameResult::WhiteWins, Color::White) | (GameResult::BlackWins, Color::Black) => (2, 1.0),
        _ => (0, -1.0),
    }
}

/// Buffered, append-only writer of 492-byte samples.
pub struct SampleWriter {
    out: BufWriter<File>,
    count: u64,
}

impl SampleWriter {
    /// Create (truncate) `path`, making parent directories as needed.
    pub fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        Ok(Self { out: BufWriter::new(file), count: 0 })
    }

    pub fn write(&mut self, sample: &TrainingSample) -> Result<()> {
        self.out.write_all(sample.as_bytes())?;
        self.count += 1;
        Ok(())
    }

    /// Flush buffered samples to disk (call after each game).
    pub fn flush(&mut self) -> Result<()> {
        self.out.flush().context("flushing sample file")
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Flush and return the total number of samples written.
    pub fn finish(mut self) -> Result<u64> {
        self.flush()?;
        Ok(self.count)
    }

    /// Flush and synchronize records before an atomic chunk publication.
    pub fn finish_synced(mut self) -> Result<u64> {
        self.flush()?;
        self.out.get_ref().sync_all().context("synchronizing sample file")?;
        Ok(self.count)
    }
}

//! Streaming PGN output rotation; chunk boundaries may fall within a game.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;

use super::chunk_writer::ChunkWriter;
use crate::encoder::sample::TrainingSample;

pub(super) struct PgnOutput {
    destination: PathBuf,
    chunk_samples: Option<u64>,
    source: String,
    index: u64,
    writer: Option<ChunkWriter>,
    games: u64,
    first_game: u64,
}

impl PgnOutput {
    pub fn new(destination: PathBuf, chunk_samples: Option<u64>, source: String, first_game: u64) -> Self {
        Self { destination, chunk_samples, source, index: 0, writer: None, games: 0, first_game }
    }

    fn open(&mut self, game: u64) -> Result<()> {
        if self.writer.is_none() {
            let path = if self.chunk_samples.is_some() {
                self.destination.join(format!("chunk_{:06}.bin", self.index))
            } else {
                self.destination.clone()
            };
            self.writer = Some(ChunkWriter::create(&path)?);
            self.first_game = game;
        }
        Ok(())
    }

    fn publish(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            writer.finish(json!({ "kind": "pgn", "games": self.games,
                "source": self.source, "first_game": self.first_game }))?;
            self.index += 1;
            self.games = 0;
        }
        Ok(())
    }

    pub fn write_game(&mut self, samples: &[TrainingSample], game: u64) -> Result<()> {
        for sample in samples {
            if let (Some(limit), Some(writer)) = (self.chunk_samples, &self.writer) {
                if writer.count() == limit {
                    self.publish()?;
                }
            }
            self.open(game)?;
            self.writer.as_mut().unwrap().write(sample)?;
        }
        self.open(game)?;
        // Attribute a game once, to the chunk containing its final sample.
        self.games += 1;
        self.writer.as_mut().unwrap().flush()?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        // Preserve --out's empty-file behavior; directory mode has no empty tail.
        if self.chunk_samples.is_none() {
            self.open(self.first_game)?;
        } else {
            std::fs::create_dir_all(&self.destination)?;
        }
        self.publish()
    }
}

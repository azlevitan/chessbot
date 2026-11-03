//! Atomic binary chunks: the JSON sidecar is published last as the commit marker.

use std::fs::{self, File};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

use super::bin_writer::SampleWriter;
use crate::encoder::sample::TrainingSample;

pub struct ChunkWriter {
    writer: SampleWriter,
    bin: PathBuf,
    json: PathBuf,
    bin_tmp: PathBuf,
    json_tmp: PathBuf,
}

fn temporary(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    name.into()
}

fn remove_stale(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("removing stale {}", path.display())),
    }
}

impl ChunkWriter {
    pub fn create(bin_path: &Path) -> Result<Self> {
        let bin = bin_path.to_path_buf();
        let json = bin.with_extension("json");
        let bin_tmp = temporary(&bin);
        let json_tmp = temporary(&json);
        // Invalidate the old commit marker before touching its binary, even if
        // the replacement happens to contain exactly the same sample count.
        for path in [&json, &bin, &bin_tmp, &json_tmp] {
            remove_stale(path)?;
        }
        let writer = SampleWriter::create(&bin_tmp)?;
        Ok(Self { writer, bin, json, bin_tmp, json_tmp })
    }

    pub fn write(&mut self, sample: &TrainingSample) -> Result<()> {
        self.writer.write(sample)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()
    }

    pub fn count(&self) -> u64 {
        self.writer.count()
    }

    pub fn finish(self, mut sidecar: Value) -> Result<u64> {
        let object = sidecar.as_object_mut().context("chunk sidecar must be a JSON object")?;
        object.insert("format".into(), "chessbot-chunk-v1".into());
        object.insert("samples".into(), self.count().into());
        let metadata = serde_json::to_vec_pretty(&sidecar)?;
        let count = self.writer.finish_synced()?;
        fs::rename(&self.bin_tmp, &self.bin).context("publishing chunk binary")?;
        let mut file = File::create(&self.json_tmp).context("creating chunk sidecar")?;
        file.write_all(&metadata)?;
        file.write_all(b"\n")?;
        file.sync_all().context("synchronizing chunk sidecar")?;
        drop(file);
        fs::rename(&self.json_tmp, &self.json).context("publishing chunk sidecar")?;
        Ok(count)
    }
}

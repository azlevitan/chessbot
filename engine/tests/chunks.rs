//! Phase 6.2: atomic publication, sidecars, and streaming PGN chunk boundaries.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use cozy_chess::Board;
use engine::cli::chunk_writer::ChunkWriter;
use engine::encoder::sample::SAMPLE_BYTES;
use engine::encoder::tensor_encoder::encode_sample;
use serde_json::{json, Value};

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../training/tests/fixtures/sample.pgn");

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("chessbot-chunks-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

fn run(dir: &Path, args: &[&str], input: Option<&[u8]>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_engine"))
        .current_dir(dir).args(args)
        .stdin(if input.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    if let Some(bytes) = input { child.stdin.take().unwrap().write_all(bytes).unwrap(); }
    child.wait_with_output().unwrap()
}

fn convert(dir: &Path, flags: &[&str]) {
    let mut args = vec!["convert-pgn", "--input", FIXTURE];
    args.extend_from_slice(flags);
    let output = run(dir, &args, None);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}

fn sidecar(bin: &Path) -> Value {
    let value: Value = serde_json::from_slice(&fs::read(bin.with_extension("json")).unwrap()).unwrap();
    assert_eq!(value["format"], "chessbot-chunk-v1");
    assert_eq!(fs::metadata(bin).unwrap().len(), value["samples"].as_u64().unwrap() * SAMPLE_BYTES as u64);
    value
}

fn no_temps(dir: &Path) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        assert_ne!(path.extension().and_then(|x| x.to_str()), Some("tmp"));
        if path.is_dir() { no_temps(&path); }
    }
}

#[test]
fn publication_is_last_and_stale_files_are_invalidated() {
    let dir = TempDir::new();
    let bin = dir.0.join("chunk_000000.bin");
    for name in ["chunk_000000.bin", "chunk_000000.json", "chunk_000000.bin.tmp", "chunk_000000.json.tmp"] {
        fs::write(dir.0.join(name), b"stale").unwrap();
    }
    let sample = encode_sample(&Board::default(), &[], 1, 0.0);
    let mut writer = ChunkWriter::create(&bin).unwrap();
    assert!(!bin.exists());
    assert!(!bin.with_extension("json").exists());
    assert!(!dir.0.join("chunk_000000.json.tmp").exists());
    writer.write(&sample).unwrap();
    writer.flush().unwrap();
    assert_eq!(fs::metadata(dir.0.join("chunk_000000.bin.tmp")).unwrap().len(), 492);
    drop(writer); // Simulate interruption before publication.
    assert!(!bin.with_extension("json").exists());
    let mut writer = ChunkWriter::create(&bin).unwrap();
    writer.write(&sample).unwrap();
    assert_eq!(writer.finish(json!({"format": "wrong", "samples": 999, "kind": "pgn"})).unwrap(), 1);
    assert_eq!(sidecar(&bin)["samples"], 1);
    assert_eq!(fs::read(&bin).unwrap(), sample.as_bytes());
    no_temps(&dir.0);
}

#[test]
fn sidecar_failure_never_marks_binary_complete() {
    let dir = TempDir::new();
    let bin = dir.0.join("broken.bin");
    let mut writer = ChunkWriter::create(&bin).unwrap();
    writer.write(&encode_sample(&Board::default(), &[], 1, 0.0)).unwrap();
    fs::create_dir(dir.0.join("broken.json.tmp")).unwrap();
    assert!(writer.finish(json!({})).is_err());
    assert!(bin.exists());
    assert!(!bin.with_extension("json").exists());
}

#[test]
fn uniform_selfplay_writes_metadata_and_preserves_determinism() {
    let dir = TempDir::new();
    for name in ["a.bin", "b.bin"] {
        let output = run(&dir.0, &["selfplay", "--games", "2", "--sims", "8", "--max-plies", "20",
            "--seed", "3", "--out", name], None);
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let meta = sidecar(&dir.0.join(name));
        assert_eq!(meta["kind"], "selfplay");
        assert_eq!(meta["games"], 2);
        assert_eq!(meta["model"], "uniform");
        assert_eq!((meta["seed"].as_u64(), meta["sims"].as_u64(), meta["max_plies"].as_u64()),
                   (Some(3), Some(8), Some(20)));
        assert_eq!(meta["temp_plies"], 30);
        for key in ["results", "terminations"] {
            assert_eq!(meta[key].as_object().unwrap().values().map(|v| v.as_u64().unwrap()).sum::<u64>(), 2);
        }
        assert_eq!(meta["terminations"]["resign"], 0);
        assert_eq!(meta["resign"], json!({"threshold": -1.0, "consecutive": 0,
            "playthrough_games": 0, "false_positives": 0}));
    }
    assert_eq!(fs::read(dir.0.join("a.bin")).unwrap(), fs::read(dir.0.join("b.bin")).unwrap());
    assert_eq!(sidecar(&dir.0.join("a.bin")), sidecar(&dir.0.join("b.bin")));
    no_temps(&dir.0);
}

#[test]
fn pgn_chunks_split_exactly_and_preserve_single_output_bytes() {
    let dir = TempDir::new();
    convert(&dir.0, &["--out-dir", "chunks", "--chunk-samples", "25"]);
    convert(&dir.0, &["--out", "all.bin"]);
    let mut bytes = Vec::new();
    let mut games = 0;
    for (i, (samples, first, completed)) in [(25, 0, 0), (25, 0, 2), (10, 2, 1)].into_iter().enumerate() {
        let bin = dir.0.join(format!("chunks/chunk_{i:06}.bin"));
        let meta = sidecar(&bin);
        assert_eq!(meta["samples"], samples);
        assert_eq!(meta["first_game"], first);
        assert_eq!(meta["games"], completed);
        assert_eq!(meta["source"], FIXTURE);
        assert_eq!(meta["kind"], "pgn");
        games += meta["games"].as_u64().unwrap();
        bytes.extend(fs::read(bin).unwrap());
    }
    assert_eq!(games, 3);
    assert_eq!(bytes, fs::read(dir.0.join("all.bin")).unwrap());
    assert_eq!(sidecar(&dir.0.join("all.bin"))["games"], 3);
    assert_eq!(fs::read_dir(dir.0.join("chunks")).unwrap().count(), 6);
    no_temps(&dir.0);
}

#[test]
fn skip_limits_stdin_and_exact_boundary_have_correct_metadata() {
    let dir = TempDir::new();
    convert(&dir.0, &["--out", "skip.bin", "--skip-games", "1"]);
    let skip = sidecar(&dir.0.join("skip.bin"));
    assert_eq!(skip["games"], 2);
    assert_eq!(skip["samples"], 30);
    assert_eq!(skip["first_game"], 1);
    convert(&dir.0, &["--out", "limited.bin", "--skip-games", "1", "--max-samples", "1"]);
    let limited = sidecar(&dir.0.join("limited.bin"));
    assert_eq!(limited["samples"], 17); // Finish the whole second game.
    assert_eq!(limited["games"], 1);
    assert_eq!(limited["first_game"], 1);
    convert(&dir.0, &["--out", "one.bin", "--skip-games", "1", "--max-games", "1"]);
    assert_eq!(sidecar(&dir.0.join("one.bin"))["samples"], 17);
    let input = fs::read(FIXTURE).unwrap();
    let output = run(&dir.0, &["convert-pgn", "--input", "-", "--out", "stdin.bin", "--skip-games", "1"], Some(&input));
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(sidecar(&dir.0.join("stdin.bin"))["source"], "-");
    assert_eq!(fs::read(dir.0.join("stdin.bin")).unwrap(), fs::read(dir.0.join("skip.bin")).unwrap());
    convert(&dir.0, &["--out-dir", "exact", "--chunk-samples", "30"]);
    assert_eq!(fs::read_dir(dir.0.join("exact")).unwrap().count(), 4);
    no_temps(&dir.0);
}

#[test]
fn rejects_conflicting_outputs_and_invalid_chunk_sizes() {
    let dir = TempDir::new();
    for flags in [vec![], vec!["--out", "a.bin", "--out-dir", "a"],
        vec!["--out-dir", "a", "--chunk-samples", "0"],
        vec!["--out-dir", "a", "--chunk-samples", "bad"],
        vec!["--out", "a.bin", "--chunk-samples", "25"]] {
        let mut args = vec!["convert-pgn", "--input", FIXTURE];
        args.extend(flags);
        assert!(!run(&dir.0, &args, None).status.success());
    }
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 0);
}

#[test]
fn empty_single_output_has_a_valid_sidecar() {
    let dir = TempDir::new();
    convert(&dir.0, &["--out", "empty.bin", "--max-samples", "0"]);
    let meta = sidecar(&dir.0.join("empty.bin"));
    assert_eq!(meta["samples"], 0);
    assert_eq!(meta["games"], 0);
    convert(&dir.0, &["--out-dir", "empty", "--skip-games", "99"]);
    assert_eq!(fs::read_dir(dir.0.join("empty")).unwrap().count(), 0);
    no_temps(&dir.0);
}

#[test]
fn invalid_games_emit_no_partial_samples_and_keep_absolute_indices() {
    let dir = TempDir::new();
    let mut input = b"[Result \"1-0\"]\n\n1. e4 e5 2. Ke3 1-0\n\n".to_vec();
    input.extend(fs::read(FIXTURE).unwrap());
    let output = run(&dir.0, &["convert-pgn", "--input", "-", "--out-dir", "chunks",
        "--max-games", "2"], Some(&input));
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let meta = sidecar(&dir.0.join("chunks/chunk_000000.bin"));
    assert_eq!(meta["first_game"], 1);
    assert_eq!(meta["samples"], 30);
    assert_eq!(meta["games"], 1);
    assert!(String::from_utf8_lossy(&output.stdout).contains("converted 1, skipped 1"));
    no_temps(&dir.0);
}

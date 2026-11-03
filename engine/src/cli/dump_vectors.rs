//! `engine dump-test-vectors --out <path>`: encode known FENs for the
//! cross-language contract test (TASKS.md Phase 1 / Phase 2).

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::board::{from_fen, legal_moves};
use crate::encoder::tensor_encoder::encode_sample;

/// `(label, FEN)` pairs, written in this order.
pub const TEST_FENS: [(&str, &str); 5] = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("en_passant", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    ("underpromotion", "n1n5/1P6/8/8/8/8/8/K6k w - - 0 1"),
    ("black_check", "rnbqkbnr/ppppp1pp/5p2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2"),
];

/// Encode every test FEN: uniform policy over all legal moves, draw, q = 0.
pub fn build_test_vectors() -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for (label, fen) in TEST_FENS {
        let board = from_fen(fen).with_context(|| format!("test vector {label}"))?;
        let moves = legal_moves(&board);
        let p = 1.0 / moves.len() as f32;
        let policy: Vec<_> = moves.into_iter().map(|mv| (mv, p)).collect();
        bytes.extend_from_slice(encode_sample(&board, &policy, 1, 0.0).as_bytes());
    }
    Ok(bytes)
}

/// Write the test vectors to `out`, creating parent directories.
pub fn write_test_vectors(out: &Path) -> Result<usize> {
    let bytes = build_test_vectors()?;
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(out, &bytes).with_context(|| format!("writing {}", out.display()))?;
    Ok(bytes.len())
}

/// Parse `--out <path>` (default `data_store/test_vectors.bin`) and run.
pub fn run(args: &[String]) -> Result<()> {
    let mut out = String::from("data_store/test_vectors.bin");
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => out = it.next().context("--out requires a path")?.clone(),
            other => bail!("unknown argument for dump-test-vectors: {other}"),
        }
    }
    let n = write_test_vectors(Path::new(&out))?;
    println!("wrote {} samples ({n} bytes) to {out}", TEST_FENS.len());
    Ok(())
}

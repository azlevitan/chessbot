//! `engine convert-pgn --input <pgn> --out <bin> [--max-games N]`: supervised
//! bootstrapping data. Each position before a mainline move becomes one sample
//! with prob 1.0 on the played move, 0.0 on every other legal move, and the
//! final game result (from the side to move) as both WDL label and `q_value`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context, Result};

use super::bin_writer::{outcome_for, SampleWriter};
use crate::board::legal_moves;
use crate::encoder::sample::TrainingSample;
use crate::encoder::tensor_encoder::encode_sample;
use crate::pgn::{san_to_move, PgnReader, RawGame};

/// Warnings printed for skipped games before going quiet.
const MAX_WARNINGS: u64 = 20;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConvertStats {
    pub games_read: u64,
    pub converted: u64,
    pub skipped: u64,
    pub samples: u64,
}

/// Encode every mainline position of one game. Errors (no result, non-standard
/// variant, bad FEN, illegal/ambiguous SAN) mean the whole game is skipped.
pub fn convert_game(game: &RawGame) -> Result<Vec<TrainingSample>> {
    if !game.is_standard() {
        bail!("non-standard variant {:?}", game.tag("Variant").unwrap_or(""));
    }
    let Some(result) = game.result() else {
        bail!("game has no decisive or drawn result");
    };
    let mut board = game.start_board()?;
    let mut samples = Vec::with_capacity(game.sans.len());
    for (ply, san) in game.sans.iter().enumerate() {
        let played = san_to_move(&board, san).with_context(|| format!("ply {}", ply + 1))?;
        let policy: Vec<_> = legal_moves(&board)
            .into_iter()
            .map(|mv| (mv, if mv == played { 1.0 } else { 0.0 }))
            .collect();
        let (wdl, q) = outcome_for(board.side_to_move(), result);
        samples.push(encode_sample(&board, &policy, wdl, q));
        board.play_unchecked(played);
    }
    Ok(samples)
}

/// Stream games from `input` into `writer`, stopping after `max_games` games read.
pub fn convert(input: impl BufRead, writer: &mut SampleWriter, max_games: Option<u64>) -> Result<ConvertStats> {
    let mut reader = PgnReader::new(input);
    let mut stats = ConvertStats::default();
    while max_games.map_or(true, |m| stats.games_read < m) {
        let Some(game) = reader.next_game()? else { break };
        stats.games_read += 1;
        match convert_game(&game) {
            Ok(samples) => {
                for s in &samples {
                    writer.write(s)?;
                }
                writer.flush()?;
                stats.converted += 1;
                stats.samples += samples.len() as u64;
            }
            Err(e) => {
                stats.skipped += 1;
                if stats.skipped <= MAX_WARNINGS {
                    eprintln!("convert-pgn: skipping game {}: {e:#}", stats.games_read);
                } else if stats.skipped == MAX_WARNINGS + 1 {
                    eprintln!("convert-pgn: further skip warnings suppressed");
                }
            }
        }
    }
    Ok(stats)
}

/// Convert the PGN file at `input` into the sample file `out`.
pub fn convert_file(input: &Path, out: &Path, max_games: Option<u64>) -> Result<ConvertStats> {
    let file = File::open(input).with_context(|| format!("opening {}", input.display()))?;
    let mut writer = SampleWriter::create(out)?;
    let stats = convert(BufReader::new(file), &mut writer, max_games)?;
    writer.finish()?;
    Ok(stats)
}

pub fn run(args: &[String]) -> Result<()> {
    let (mut input, mut out, mut max_games) = (None, None, None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--input" => input = Some(it.next().context("--input requires a path")?.clone()),
            "--out" => out = Some(it.next().context("--out requires a path")?.clone()),
            "--max-games" => {
                let n = it.next().context("--max-games requires a number")?;
                max_games = Some(n.parse().with_context(|| format!("bad --max-games {n:?}"))?);
            }
            other => bail!("unknown argument for convert-pgn: {other}"),
        }
    }
    let input = input.context("convert-pgn requires --input <pgn>")?;
    let out = out.context("convert-pgn requires --out <bin>")?;
    let s = convert_file(Path::new(&input), Path::new(&out), max_games)?;
    println!(
        "games read {}, converted {}, skipped {}; wrote {} samples to {out}",
        s.games_read, s.converted, s.skipped, s.samples
    );
    Ok(())
}

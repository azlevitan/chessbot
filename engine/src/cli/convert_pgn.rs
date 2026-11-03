//! `engine convert-pgn --input <pgn> --out <bin> [--max-games N]`: supervised
//! bootstrapping data. Each position before a mainline move becomes one sample
//! with prob 1.0 on the played move, 0.0 on every other legal move, and the
//! final game result (from the side to move) as both WDL label and `q_value`.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};

use super::bin_writer::{outcome_for, SampleWriter};
use super::pgn_output::PgnOutput;
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

#[derive(Default)]
struct ConvertOptions {
    max_games: Option<u64>,
    max_samples: Option<u64>,
    skip_games: u64,
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
    convert_stream(input, &ConvertOptions { max_games, ..Default::default() }, |samples, _| {
        for sample in samples {
            writer.write(sample)?;
        }
        writer.flush()
    })
}

fn convert_stream(
    input: impl BufRead, opts: &ConvertOptions,
    mut write_game: impl FnMut(&[TrainingSample], u64) -> Result<()>,
) -> Result<ConvertStats> {
    let mut reader = PgnReader::new(input);
    let mut stats = ConvertStats::default();
    for _ in 0..opts.skip_games {
        if reader.next_game()?.is_none() {
            return Ok(stats);
        }
        stats.games_read += 1;
    }
    while opts.max_games.map_or(true, |m| stats.games_read - opts.skip_games < m)
        && opts.max_samples.map_or(true, |m| stats.samples < m)
    {
        let Some(game) = reader.next_game()? else { break };
        let game_index = stats.games_read;
        stats.games_read += 1;
        match convert_game(&game) {
            Ok(samples) => {
                write_game(&samples, game_index)?;
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
    let mut writer = PgnOutput::new(out.into(), None, input.display().to_string(), 0);
    let opts = ConvertOptions { max_games, ..Default::default() };
    let stats = convert_stream(BufReader::new(file), &opts, |s, i| writer.write_game(s, i))?;
    writer.finish()?;
    Ok(stats)
}

pub fn run(args: &[String]) -> Result<()> {
    let (mut input, mut out, mut out_dir) = (None, None, None);
    let mut opts = ConvertOptions::default();
    let mut chunk_samples = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let value = it.next().with_context(|| format!("{arg} requires a value"))?;
        match arg.as_str() {
            "--input" => input = Some(value.clone()),
            "--out" => out = Some(PathBuf::from(value)),
            "--out-dir" => out_dir = Some(PathBuf::from(value)),
            "--max-games" | "--max-samples" | "--skip-games" | "--chunk-samples" => {
                let n = value.parse::<u64>().with_context(|| format!("bad {arg} {value:?}"))?;
                match arg.as_str() {
                    "--max-games" => opts.max_games = Some(n),
                    "--max-samples" => opts.max_samples = Some(n),
                    "--skip-games" => opts.skip_games = n,
                    _ => {
                        ensure!(n > 0, "--chunk-samples must be positive");
                        chunk_samples = Some(n);
                    }
                }
            }
            other => bail!("unknown argument for convert-pgn: {other}"),
        }
    }
    let input = input.context("convert-pgn requires --input <pgn>")?;
    ensure!(out.is_some() != out_dir.is_some(), "provide exactly one of --out or --out-dir");
    ensure!(chunk_samples.is_none() || out_dir.is_some(), "--chunk-samples requires --out-dir");
    let limit = out_dir.as_ref().map(|_| chunk_samples.unwrap_or(262144));
    let destination = out.or(out_dir).unwrap();
    let reader: Box<dyn BufRead> = if input == "-" {
        Box::new(io::stdin().lock())
    } else {
        Box::new(BufReader::new(File::open(&input).with_context(|| format!("opening {input}"))?))
    };
    let mut writer = PgnOutput::new(destination.clone(), limit, input, opts.skip_games);
    let s = convert_stream(reader, &opts, |samples, i| writer.write_game(samples, i))?;
    writer.finish()?;
    println!(
        "games read {}, converted {}, skipped {}; wrote {} samples to {}",
        s.games_read, s.converted, s.skipped, s.samples, destination.display()
    );
    Ok(())
}

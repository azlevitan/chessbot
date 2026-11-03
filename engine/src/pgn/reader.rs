//! Streaming PGN reader: one game at a time from any `BufRead`, constant memory.
//!
//! Skips `{}` comments, `;` line comments, `%` escape lines, nested `()`
//! variations, `$n` NAGs, move numbers and annotation glyphs.

use std::io::BufRead;

use anyhow::Result;
use cozy_chess::Board;

use crate::board::from_fen;
use crate::cli::bin_writer::GameResult;

/// One game's tags and mainline SAN tokens (not yet validated).
#[derive(Debug, Default, Clone)]
pub struct RawGame {
    pub tags: Vec<(String, String)>,
    pub sans: Vec<String>,
    /// Game-termination token from the movetext (`1-0`, `0-1`, `1/2-1/2`, `*`).
    pub termination: Option<String>,
}

impl RawGame {
    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Result from the movetext termination token, else the `Result` tag.
    pub fn result(&self) -> Option<GameResult> {
        self.termination.as_deref().or(self.tag("Result")).and_then(GameResult::from_pgn)
    }

    /// `true` unless a `Variant` tag names something other than standard chess.
    pub fn is_standard(&self) -> bool {
        self.tag("Variant").map_or(true, |v| v.eq_ignore_ascii_case("standard"))
    }

    /// Starting position: the `FEN` tag if present, else the standard start.
    pub fn start_board(&self) -> Result<Board> {
        match self.tag("FEN") {
            Some(fen) => from_fen(fen),
            None => Ok(Board::default()),
        }
    }

    fn is_empty(&self) -> bool {
        self.tags.is_empty() && self.sans.is_empty() && self.termination.is_none()
    }
}

pub struct PgnReader<R: BufRead> {
    input: R,
    line: Vec<u8>,
    pending_tag: Option<String>,
    in_comment: bool,
    var_depth: u32,
}

/// Parse `[Key "Value"]` (with `\"` and `\\` escapes in the value).
fn parse_tag(line: &str) -> Option<(String, String)> {
    let inner = line.trim().strip_prefix('[')?.strip_suffix(']')?;
    let (key, rest) = inner.split_once(char::is_whitespace)?;
    let value = rest.trim().strip_prefix('"')?.strip_suffix('"')?;
    Some((key.to_string(), value.replace("\\\"", "\"").replace("\\\\", "\\")))
}

/// Strip a leading move number (`12.`, `12...`, `12.e4` -> `e4`).
fn strip_move_number(tok: &str) -> &str {
    let digits = tok.len() - tok.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits > 0 && tok[digits..].starts_with('.') {
        tok[digits..].trim_start_matches('.')
    } else {
        tok
    }
}

impl<R: BufRead> PgnReader<R> {
    pub fn new(input: R) -> Self {
        Self { input, line: Vec::new(), pending_tag: None, in_comment: false, var_depth: 0 }
    }

    /// Handle one movetext token; returns `true` if it terminated the game.
    fn push_token(&self, game: &mut RawGame, tok: &str) -> bool {
        if matches!(tok, "1-0" | "0-1" | "1/2-1/2" | "*") {
            game.termination = Some(tok.to_string());
            return true;
        }
        let tok = strip_move_number(tok);
        let glyph_only = tok.chars().all(|c| c == '!' || c == '?');
        if !tok.is_empty() && !tok.starts_with('$') && !glyph_only {
            game.sans.push(tok.to_string());
        }
        false
    }

    /// Tokenize one movetext line into `game`; returns `true` on game termination.
    fn scan_movetext(&mut self, game: &mut RawGame, text: &str) -> bool {
        let mut tok = String::new();
        let mut done = false;
        for c in text.chars().chain(std::iter::once(' ')) {
            if self.in_comment {
                self.in_comment = c != '}';
                continue;
            }
            let delimiter = matches!(c, '{' | ';' | '(' | ')') || c.is_whitespace();
            if delimiter && !tok.is_empty() {
                if self.var_depth == 0 && self.push_token(game, &tok) {
                    done = true;
                }
                tok.clear();
            }
            match c {
                '{' => self.in_comment = true,
                ';' => break,
                '(' => self.var_depth += 1,
                ')' => self.var_depth = self.var_depth.saturating_sub(1),
                _ if delimiter => {}
                _ => tok.push(c),
            }
            if done {
                break;
            }
        }
        done
    }

    /// Next game, or `None` at end of input.
    pub fn next_game(&mut self) -> Result<Option<RawGame>> {
        let mut game = RawGame::default();
        if let Some(tag) = self.pending_tag.take().and_then(|l| parse_tag(&l)) {
            game.tags.push(tag);
        }
        loop {
            self.line.clear();
            if self.input.read_until(b'\n', &mut self.line)? == 0 {
                self.in_comment = false;
                self.var_depth = 0;
                return Ok((!game.is_empty()).then_some(game));
            }
            let text = String::from_utf8_lossy(&self.line).into_owned();
            let trimmed = text.trim();
            let free = !self.in_comment && self.var_depth == 0;
            if free && trimmed.starts_with('%') {
                continue;
            }
            if free && trimmed.starts_with('[') {
                if !game.sans.is_empty() {
                    // New game's tags without a termination token on the last one.
                    self.pending_tag = Some(text);
                    return Ok(Some(game));
                }
                game.tags.extend(parse_tag(trimmed));
                continue;
            }
            if self.scan_movetext(&mut game, &text) {
                self.in_comment = false;
                self.var_depth = 0;
                return Ok(Some(game));
            }
        }
    }
}

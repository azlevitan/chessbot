//! Standard Algebraic Notation (SAN) -> legal `cozy_chess::Move`.
//!
//! Resolution is done by matching against the legal move list, so the result
//! is always legal; castling is matched via `standard_to` (king destination).

use anyhow::{anyhow, bail, Result};
use cozy_chess::{Board, File, Move, Piece, Rank, Square};

use crate::board::legal_moves;
use crate::encoder::move_mapper::standard_to;

/// Parsed SAN pieces before matching against the board.
#[derive(Debug, Default)]
struct SanParts {
    piece: Option<Piece>, // None = pawn
    from_file: Option<File>,
    from_rank: Option<Rank>,
    to: Option<Square>,
    promotion: Option<Piece>,
    castle: Option<File>, // Some(G) = O-O, Some(C) = O-O-O
}

fn piece_from_char(c: char) -> Option<Piece> {
    match c.to_ascii_uppercase() {
        'N' => Some(Piece::Knight),
        'B' => Some(Piece::Bishop),
        'R' => Some(Piece::Rook),
        'Q' => Some(Piece::Queen),
        'K' => Some(Piece::King),
        _ => None,
    }
}

fn parse_parts(san: &str) -> Result<SanParts> {
    let body = san.trim_end_matches(['+', '#', '!', '?']);
    match body {
        "O-O" | "0-0" => return Ok(SanParts { castle: Some(File::G), ..Default::default() }),
        "O-O-O" | "0-0-0" => return Ok(SanParts { castle: Some(File::C), ..Default::default() }),
        _ => {}
    }
    let mut parts = SanParts::default();
    let mut chars: Vec<char> = body.chars().filter(|&c| c != 'x' && c != ':' && c != '-').collect();
    // Moving piece (uppercase letter only; a lowercase 'b' is the b-file).
    if let Some(&c) = chars.first() {
        if c.is_ascii_uppercase() {
            parts.piece = Some(piece_from_char(c).ok_or_else(|| anyhow!("bad piece in {san:?}"))?);
            chars.remove(0);
        }
    }
    // Promotion: `=Q`, or a trailing piece letter on a pawn move (`e8Q`).
    if let Some(eq) = chars.iter().position(|&c| c == '=') {
        let promo = chars.get(eq + 1).and_then(|&c| piece_from_char(c));
        parts.promotion = Some(promo.ok_or_else(|| anyhow!("bad promotion in {san:?}"))?);
        chars.truncate(eq);
    } else if parts.piece.is_none() {
        if let Some(&c) = chars.last().filter(|c| c.is_ascii_uppercase()) {
            parts.promotion = Some(piece_from_char(c).ok_or_else(|| anyhow!("bad promotion in {san:?}"))?);
            chars.pop();
        }
    }
    if chars.len() < 2 {
        bail!("SAN {san:?} has no destination square");
    }
    let dest: String = chars[chars.len() - 2..].iter().collect();
    parts.to = Some(dest.parse().map_err(|_| anyhow!("bad destination in {san:?}"))?);
    for &c in &chars[..chars.len() - 2] {
        if let Ok(f) = File::try_from(c) {
            parts.from_file = Some(f);
        } else if let Ok(r) = Rank::try_from(c) {
            parts.from_rank = Some(r);
        } else {
            bail!("bad disambiguation {c:?} in {san:?}");
        }
    }
    Ok(parts)
}

/// True if `mv` is a castling move in cozy-chess's king-captures-rook form.
fn is_castle(board: &Board, mv: Move) -> bool {
    board.piece_on(mv.from) == Some(Piece::King) && board.color_on(mv.to) == Some(board.side_to_move())
}

fn matches(board: &Board, mv: Move, p: &SanParts) -> bool {
    if let Some(file) = p.castle {
        return is_castle(board, mv) && standard_to(board, mv).file() == file;
    }
    let piece = p.piece.unwrap_or(Piece::Pawn);
    board.piece_on(mv.from) == Some(piece)
        && !is_castle(board, mv)
        && Some(mv.to) == p.to
        && mv.promotion == p.promotion
        && p.from_file.map_or(true, |f| mv.from.file() == f)
        && p.from_rank.map_or(true, |r| mv.from.rank() == r)
}

/// Resolve a SAN token (e.g. `Nbd7`, `exd6`, `e8=N+`, `O-O-O`) on `board`.
pub fn san_to_move(board: &Board, san: &str) -> Result<Move> {
    let parts = parse_parts(san)?;
    let mut found = legal_moves(board).into_iter().filter(|&mv| matches(board, mv, &parts));
    let mv = found.next().ok_or_else(|| anyhow!("SAN {san:?} matches no legal move"))?;
    if found.next().is_some() {
        bail!("SAN {san:?} is ambiguous");
    }
    Ok(mv)
}

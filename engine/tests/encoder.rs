//! Tensor / TrainingSample encoder contract tests (SPEC.md §2, §3).

use cozy_chess::{Board, Move};
use engine::board::{from_fen, legal_moves};
use engine::cli::dump_vectors::{build_test_vectors, write_test_vectors, TEST_FENS};
use engine::encoder::sample::{TrainingSample, POLICY_SLOTS, SAMPLE_BYTES, UNUSED_MOVE};
use engine::encoder::tensor_encoder::{
    encode_planes, encode_sample, sample_to_planes, TENSOR_LEN,
};

fn plane(planes: &[f32], p: usize) -> &[f32] {
    &planes[p * 64..(p + 1) * 64]
}

fn plane_sum(planes: &[f32], p: usize) -> f32 {
    plane(planes, p).iter().sum()
}

/// Colour-swapped vertical mirror of a FEN (same position from the other side).
fn mirror_fen(fen: &str) -> String {
    let f: Vec<&str> = fen.split(' ').collect();
    let swap = |s: &str| -> String {
        s.chars()
            .map(|c| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() })
            .collect()
    };
    let rows: Vec<&str> = f[0].split('/').rev().collect();
    let stm = if f[1] == "w" { "b" } else { "w" };
    let mut castling: Vec<char> = swap(f[2]).chars().collect();
    castling.sort_by_key(|c| "KQkq-".find(*c));
    let ep = match f[3] {
        "-" => "-".to_string(),
        sq => format!("{}{}", &sq[..1], if &sq[1..] == "3" { "6" } else { "3" }),
    };
    let castling: String = castling.into_iter().collect();
    format!("{} {stm} {castling} {ep} {} {}", swap(&rows.join("/")), f[4], f[5])
}

#[test]
fn struct_size_is_492() {
    assert_eq!(std::mem::size_of::<TrainingSample>(), SAMPLE_BYTES);
    assert_eq!(SAMPLE_BYTES, 492);
    assert_eq!(TENSOR_LEN, 1152);
}

#[test]
fn startpos_planes() {
    let planes = encode_planes(&Board::default());
    let expected = [8.0, 2.0, 2.0, 2.0, 1.0, 1.0];
    for (i, &n) in expected.iter().enumerate() {
        assert_eq!(plane_sum(&planes, i), n, "us plane {i}");
        assert_eq!(plane_sum(&planes, 6 + i), n, "them plane {i}");
    }
    assert!(plane(&planes, 0)[8..16].iter().all(|&v| v == 1.0)); // us pawns on rank 2
    assert!(plane(&planes, 6)[48..56].iter().all(|&v| v == 1.0)); // them pawns on rank 7
    assert_eq!(plane(&planes, 5)[4], 1.0); // us king e1
    assert_eq!(plane(&planes, 11)[60], 1.0); // them king e8
    for p in 12..16 {
        assert_eq!(plane_sum(&planes, p), 64.0, "castling plane {p}");
    }
    assert_eq!(plane_sum(&planes, 16), 0.0);
    assert_eq!(plane_sum(&planes, 17), 0.0);
}

#[test]
fn black_to_move_equals_white_mirror() {
    let extra = [
        "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
        "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 7 30",
    ];
    for fen in TEST_FENS.iter().map(|t| t.1).chain(extra) {
        let a = from_fen(fen).unwrap();
        let b = from_fen(&mirror_fen(fen)).unwrap();
        assert_ne!(a.side_to_move(), b.side_to_move());
        assert_eq!(encode_planes(&a), encode_planes(&b), "mirror mismatch for {fen}");
    }
}

#[test]
fn en_passant_square_is_canonical() {
    let white = from_fen(TEST_FENS[2].1).unwrap(); // f6 target
    let s = encode_sample(&white, &[], 1, 0.0);
    assert_eq!((s.castling_and_ep & 0x10, s.ep_square), (0x10, 45));
    assert_eq!(plane(&encode_planes(&white), 16)[45], 1.0);

    let black = from_fen("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3").unwrap();
    let s = encode_sample(&black, &[], 1, 0.0);
    assert_eq!((s.castling_and_ep & 0x10, s.ep_square), (0x10, 43)); // d3 ^ 56 = d6
    assert_eq!(plane_sum(&encode_planes(&black), 16), 1.0);

    let none = encode_sample(&Board::default(), &[], 1, 0.0);
    assert_eq!((none.castling_and_ep & 0x10, none.ep_square), (0, 0));
}

#[test]
fn castling_bits_and_halfmove() {
    let kiwi = encode_sample(&from_fen(TEST_FENS[1].1).unwrap(), &[], 1, 0.0);
    assert_eq!(kiwi.castling_and_ep, 0b1111);
    // Black to move: us (black) has q, them (white) has K.
    let b = from_fen("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 7 30").unwrap();
    let s = encode_sample(&b, &[], 1, 0.0);
    assert_eq!(s.castling_and_ep, 0b0110);
    assert_eq!(s.halfmove_clock, 7);
    let planes = encode_planes(&b);
    assert_eq!(
        [12, 13, 14, 15].map(|p| plane_sum(&planes, p)),
        [0.0, 64.0, 64.0, 0.0]
    );
    assert!(plane(&planes, 17).iter().all(|&v| (v - 0.07).abs() < 1e-6));

    // cozy-chess rejects clocks > 100, so 100 is the boundary (the clamp is defensive).
    assert!(from_fen("8/8/8/8/8/8/8/K6k w - - 150 200").is_err());
    let max = from_fen("8/8/8/8/8/8/8/K6k w - - 100 200").unwrap();
    assert_eq!(encode_sample(&max, &[], 1, 0.0).halfmove_clock, 100);
    assert!(plane(&encode_planes(&max), 17).iter().all(|&v| v == 1.0));
}

#[test]
fn bytes_roundtrip_and_planes_match() {
    for (_, fen) in TEST_FENS {
        let board = from_fen(fen).unwrap();
        let policy: Vec<(Move, f32)> = legal_moves(&board).into_iter().map(|m| (m, 0.5)).collect();
        let s = encode_sample(&board, &policy, 2, -0.25);
        let back = TrainingSample::from_bytes(s.as_bytes());
        assert_eq!(back.as_bytes(), s.as_bytes());
        assert_eq!(sample_to_planes(&back), encode_planes(&board), "{fen}");
        assert_eq!({ back.outcome_wdl }, 2);
        assert_eq!({ back.q_value }, -0.25);
    }
}

#[test]
fn policy_slots_sorted_and_padded() {
    let board = from_fen(TEST_FENS[1].1).unwrap(); // Kiwipete: 48 legal moves
    let moves = legal_moves(&board);
    assert_eq!(moves.len(), 48);
    let policy: Vec<_> = moves.iter().map(|&m| (m, 1.0 / 48.0)).collect();
    let s = encode_sample(&board, &policy, 1, 0.0);
    assert_eq!({ s.num_legal_moves }, 48);
    let entries = s.policy_entries();
    assert!(entries.windows(2).all(|w| w[0].0 < w[1].0), "entries not sorted ascending");
    let total: f32 = entries.iter().map(|e| e.1).sum();
    assert!((total - 1.0).abs() < 1e-5);
    let slots = s.policy;
    assert!(slots[48..].iter().all(|e| { e.move_idx } == UNUSED_MOVE && { e.prob } == 0.0));
}

#[test]
fn policy_truncates_to_top_64_and_renormalizes() {
    // 218 legal moves (the known maximum).
    let board = from_fen("R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1").unwrap();
    let moves = legal_moves(&board);
    assert_eq!(moves.len(), 218);
    let policy: Vec<_> = moves.iter().enumerate().map(|(i, &m)| (m, (i + 1) as f32)).collect();
    let s = encode_sample(&board, &policy, 1, 0.0);
    assert_eq!({ s.num_legal_moves } as usize, POLICY_SLOTS);
    let entries = s.policy_entries();
    let total: f32 = entries.iter().map(|e| e.1).sum();
    assert!((total - 1.0).abs() < 1e-5);
    // Smallest kept raw prob is 155 (moves 155..=218 kept); normalized by their sum.
    let raw_sum: f32 = (155..=218).map(|v| v as f32).sum();
    let min_kept = entries.iter().map(|e| e.1).fold(f32::MAX, f32::min);
    assert!((min_kept - 155.0 / raw_sum).abs() < 1e-6);
}

#[test]
fn dump_test_vectors_layout() {
    let bytes = build_test_vectors().unwrap();
    assert_eq!(bytes.len(), TEST_FENS.len() * SAMPLE_BYTES);
    assert_eq!(bytes.len(), 2460);
    let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("tv/test_vectors.bin");
    assert_eq!(write_test_vectors(&out).unwrap(), 2460);
    assert_eq!(std::fs::read(&out).unwrap(), bytes);
    for (i, (_, fen)) in TEST_FENS.iter().enumerate() {
        let s = TrainingSample::from_bytes(&bytes[i * SAMPLE_BYTES..(i + 1) * SAMPLE_BYTES]);
        let n = legal_moves(&from_fen(fen).unwrap()).len();
        assert_eq!({ s.num_legal_moves } as usize, n.min(POLICY_SLOTS), "{fen}");
    }
}

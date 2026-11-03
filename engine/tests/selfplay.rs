//! Root Dirichlet noise and self-play game invariants (uniform dummy network).

use cozy_chess::Board;
use engine::board::legal_moves;
use engine::cli::bin_writer::{outcome_for, GameResult};
use engine::cli::selfplay::generate;
use engine::encoder::sample::{TrainingSample, SAMPLE_BYTES};
use engine::mcts::arena::MAX_NODES;
use engine::mcts::{search, search_with_noise, RootNoise, SearchLimits};
use engine::nn::ort_infer::UniformEvaluator;
use engine::selfplay::{game_samples, play_game, SelfPlayConfig, Termination};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn small_cfg() -> SelfPlayConfig {
    SelfPlayConfig { sims: 24, max_plies: 60, ..SelfPlayConfig::default() }
}

#[test]
fn dirichlet_noise_keeps_a_distribution() {
    let noise = RootNoise::default();
    let mut rng = StdRng::seed_from_u64(7);
    let eta = noise.sample_dirichlet(20, &mut rng);
    assert!((eta.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(eta.iter().all(|&x| x >= 0.0));
    let mut priors = vec![0.05f32; 20];
    noise.apply(&mut priors, &mut rng);
    assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(priors.iter().all(|&p| p >= 0.75 * 0.05 - 1e-7), "eps = 0.25 keeps 75% of the prior");
    assert!(priors.iter().any(|&p| (p - 0.05).abs() > 1e-4), "noise changed nothing");
}

#[test]
fn noisy_search_perturbs_root_priors_only_when_asked() {
    let board = Board::default();
    let limits = SearchLimits {
        nodes: Some(50),
        movetime: None,
        ..SearchLimits::default()
    };
    let plain = search(&board, &mut UniformEvaluator, limits, MAX_NODES).unwrap();
    assert!(plain.root_children.iter().all(|c| (c.prior - 0.05).abs() < 1e-6));
    let mut rng = StdRng::seed_from_u64(1);
    let noise = RootNoise::default();
    let noisy =
        search_with_noise(&board, &mut UniformEvaluator, limits, MAX_NODES, Some((&noise, &mut rng))).unwrap();
    let sum: f32 = noisy.root_children.iter().map(|c| c.prior).sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(noisy.root_children.iter().any(|c| (c.prior - 0.05).abs() > 1e-3));
    assert_eq!(noisy.simulations, 50);
}

#[test]
fn game_records_are_valid_and_seeded() {
    let cfg = small_cfg();
    let game = play_game(&mut UniformEvaluator, &cfg, &mut StdRng::seed_from_u64(42)).unwrap();
    assert!(!game.plies.is_empty() && game.plies.len() <= cfg.max_plies as usize);
    if game.termination == Termination::PlyCap {
        assert_eq!(game.result, GameResult::Draw);
        assert_eq!(game.plies.len(), cfg.max_plies as usize);
    }
    let mut board = Board::default();
    for p in &game.plies {
        assert_eq!(p.board, board, "plies form one continuous game");
        assert_eq!(p.policy.len(), legal_moves(&board).len(), "policy covers every legal move");
        let sum: f32 = p.policy.iter().map(|e| e.1).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!((-1.0..=1.0).contains(&p.root_q));
        if let Some(played) = p.played {
            assert!(p.policy.iter().any(|&(mv, prob)| mv == played && prob > 0.0), "played an unvisited move");
            board.play_unchecked(played);
        } else {
            assert_eq!(game.termination, Termination::Resign);
            assert_eq!(p.board, game.plies.last().unwrap().board);
        }
    }
    let again = play_game(&mut UniformEvaluator, &cfg, &mut StdRng::seed_from_u64(42)).unwrap();
    let moves = |g: &engine::selfplay::GameRecord| g.plies.iter().map(|p| p.played).collect::<Vec<_>>();
    assert_eq!(moves(&game), moves(&again), "same seed -> same game");
    let other = play_game(&mut UniformEvaluator, &cfg, &mut StdRng::seed_from_u64(43)).unwrap();
    assert_ne!(moves(&game), moves(&other), "different seed -> different game");
}

#[test]
fn samples_carry_outcome_and_root_q() {
    let game = play_game(&mut UniformEvaluator, &small_cfg(), &mut StdRng::seed_from_u64(5)).unwrap();
    let samples = game_samples(&game);
    assert_eq!(samples.len(), game.plies.len());
    for (s, p) in samples.iter().zip(&game.plies) {
        let (wdl, q) = (s.outcome_wdl, s.q_value);
        assert_eq!(wdl, outcome_for(p.board.side_to_move(), game.result).0);
        assert_eq!(q, p.root_q);
        assert_eq!({ s.num_legal_moves } as usize, p.policy.len().min(64));
    }
}

#[test]
fn checkmate_outcome_labels_winner() {
    use cozy_chess::Color::{Black, White};
    assert_eq!(outcome_for(White, GameResult::WhiteWins), (2, 1.0));
    assert_eq!(outcome_for(Black, GameResult::WhiteWins), (0, -1.0));
    assert_eq!(outcome_for(Black, GameResult::Draw), (1, 0.0));
    assert_eq!(GameResult::win_for(Black), GameResult::BlackWins);
}

#[test]
fn generate_writes_whole_samples() {
    let dir = std::env::temp_dir().join(format!("chessbot-sp-{}", std::process::id()));
    let out = dir.join("sp.bin");
    let n = generate(&mut UniformEvaluator, 2, 9, &small_cfg(), &out).unwrap();
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(bytes.len() as u64, n * SAMPLE_BYTES as u64);
    let s = TrainingSample::from_bytes(&bytes[..SAMPLE_BYTES]);
    assert_eq!({ s.num_legal_moves }, 20, "first sample is the start position");
    std::fs::remove_dir_all(dir).unwrap();
}

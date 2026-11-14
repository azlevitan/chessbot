//! Phase 8.1: evaluation baselines, player lifecycle, and the embedded book.

use std::collections::HashSet;

use cozy_chess::{Board, GameStatus};
use engine::board::from_fen;
use engine::eval::material::MaterialEvaluator;
use engine::eval::openings::openings;
use engine::eval::players::Player;
use engine::mcts::arena::MAX_NODES;
use engine::mcts::SearchResult;
use engine::nn::evaluate_board;
use engine::nn::infer_opts::{Ep, InferOptions};
use engine::uci::notation::{move_from_uci, move_to_uci};

fn opts() -> InferOptions {
    InferOptions {
        ep: Ep::Cpu,
        threads: 1,
        ..InferOptions::default()
    }
}

fn player(spec: &str, sims: u32, batch: usize) -> Player {
    Player::from_spec(spec, sims, batch, &opts(), 17).unwrap()
}

fn snapshot(player: &Player) -> SearchResult {
    match player {
        Player::Mcts { tree, .. } => tree.result(),
        Player::Random { .. } => panic!("expected MCTS player"),
    }
}

#[test]
fn material_is_even_at_start_and_scores_a_queen_for_the_side_to_move() {
    let (_, start) = evaluate_board(&mut MaterialEvaluator, &Board::default()).unwrap();
    assert_eq!(start, 0.0);
    for (side, sign) in [("w", 1.0), ("b", -1.0)] {
        let board = from_fen(&format!("4k3/8/8/8/8/8/8/3QK3 {side} - - 0 1")).unwrap();
        let (priors, value) = evaluate_board(&mut MaterialEvaluator, &board).unwrap();
        assert!(sign * value > 0.8);
        assert!(priors
            .iter()
            .all(|(_, p)| (*p - 1.0 / priors.len() as f32).abs() < 1e-6));
    }
}

#[test]
fn material_mcts_captures_the_hanging_queen_at_64_simulations() {
    let board = from_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1").unwrap();
    for batch in [1, 8] {
        let mut player = player("material", 64, batch);
        let mv = player.choose(&board, &[board.hash()]).unwrap();
        assert_eq!(move_to_uci(&board, mv), "e4d5");
        assert_eq!(snapshot(&player).simulations, 64);
    }
}

#[test]
fn the_book_has_50_distinct_legal_openings_of_two_to_eight_plies() {
    let book = openings();
    assert_eq!(book.len(), 50);
    assert_eq!(include_str!("../data/openings.txt").lines().count(), 50);
    let mut seen = HashSet::new();
    for line in book {
        assert!((2..=8).contains(&line.len()));
        let mut board = Board::default();
        let mut uci = Vec::new();
        for mv in line {
            assert!(board.is_legal(mv));
            uci.push(move_to_uci(&board, mv));
            board.play_unchecked(mv);
        }
        assert!(seen.insert(uci), "duplicate opening sequence");
        assert_eq!(board.status(), GameStatus::Ongoing);
    }
}

#[test]
fn random_players_with_the_same_seed_produce_the_same_legal_sequence() {
    let mut first = player("random", 64, 8);
    let mut second = player("random", 64, 8);
    let mut board = Board::default();
    let mut history = vec![board.hash()];
    let mut plies = 0;
    while plies < 80 && board.status() == GameStatus::Ongoing {
        let mv = first.choose(&board, &history).unwrap();
        assert_eq!(second.choose(&board, &history).unwrap(), mv);
        assert!(board.is_legal(mv));
        first.observe(mv).unwrap();
        second.observe(mv).unwrap();
        board.play_unchecked(mv);
        history.push(board.hash());
        plies += 1;
    }
    assert!(plies >= 4);
}

#[test]
fn both_players_observe_openings_and_each_played_move_without_losing_reuse() {
    let mut players = [player("material", 128, 8), player("uniform", 128, 8)];
    let mut board = Board::default();
    let mut history = vec![board.hash()];
    for mv in openings()[0].iter().copied() {
        for player in &mut players {
            player.observe(mv).unwrap();
        }
        board.play_unchecked(mv);
        history.push(board.hash());
    }
    for _ in 0..6 {
        let turn = board.side_to_move() as usize;
        let before = snapshot(&players[turn]).simulations;
        let mv = players[turn].choose(&board, &history).unwrap();
        let result = snapshot(&players[turn]);
        assert_eq!(result.simulations, before + 128);
        assert!(result.nodes_used <= MAX_NODES);
        let child_visits = result
            .root_children
            .iter()
            .find(|c| c.mv == mv)
            .unwrap()
            .visits;
        assert!(child_visits > 0);
        board.play_unchecked(mv);
        history.push(board.hash());
        for player in &mut players {
            player.observe(mv).unwrap();
            if let Player::Mcts {
                board: stored,
                history: hashes,
                ..
            } = player
            {
                assert_eq!(*stored, board);
                assert_eq!(*hashes, history);
            }
        }
        assert_eq!(snapshot(&players[turn]).simulations, child_visits);
    }
}

#[test]
fn changing_the_position_rebuilds_the_tree_and_same_position_adds_new_simulations() {
    let mut player = player("material", 64, 8);
    let board = Board::default();
    player.choose(&board, &[board.hash()]).unwrap();
    player.choose(&board, &[]).unwrap();
    assert_eq!(snapshot(&player).simulations, 128);
    let supplied = from_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let mv = player.choose(&supplied, &[supplied.hash()]).unwrap();
    assert_eq!(move_to_uci(&supplied, mv), "e4d5");
    assert_eq!(snapshot(&player).simulations, 64);
}

#[test]
fn changed_history_rebuilds_the_same_position_and_marks_repetitions_as_draws() {
    let mut board = from_fen("7k/8/8/8/8/8/8/K5Q1 w - - 0 1").unwrap();
    let mut history = vec![board.hash()];
    for text in ["g1g2", "h8h7", "g2g1", "h7h8"] {
        let mv = move_from_uci(&board, text).unwrap();
        board.play_unchecked(mv);
        history.push(board.hash());
    }
    // A queen-up material value is near 1: allow enough exploration for the
    // repeating move to be selected, so this tests history rather than FPU.
    let mut player = player("material", 800, 1);
    player.choose(&board, &[board.hash()]).unwrap();
    let mv = player.choose(&board, &history).unwrap();
    let result = snapshot(&player);
    assert_eq!(result.simulations, 800);
    let repeat = move_from_uci(&board, "g1g2").unwrap();
    let child = result
        .root_children
        .iter()
        .find(|c| c.mv == repeat)
        .unwrap();
    assert_eq!(child.terminal, Some(0.0));
    assert_ne!(mv, repeat);
}

#[test]
fn terminal_positions_return_errors_for_each_baseline() {
    for fen in [
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 100 1",
    ] {
        let board = from_fen(fen).unwrap();
        assert_ne!(board.status(), GameStatus::Ongoing);
        for spec in ["uniform", "material", "random"] {
            assert!(player(spec, 64, 8).choose(&board, &[board.hash()]).is_err());
        }
    }
}

#[test]
fn factory_rejects_invalid_search_limits_options_and_unloadable_models() {
    for spec in ["uniform", "material"] {
        for sims in [0, 1] {
            assert!(Player::from_spec(spec, sims, 8, &opts(), 0).is_err());
        }
        assert!(Player::from_spec(spec, 64, 0, &opts(), 0).is_err());
    }
    let bad = InferOptions {
        threads: 0,
        ..opts()
    };
    for spec in ["uniform", "material", "random"] {
        assert!(Player::from_spec(spec, 64, 8, &bad, 0).is_err());
    }
    let missing = concat!(env!("CARGO_MANIFEST_DIR"), "/data/does-not-exist.onnx");
    assert!(Player::from_spec(missing, 64, 8, &opts(), 0).is_err());
    assert!(Player::from_spec("", 64, 8, &opts(), 0).is_err());
}

#[test]
fn an_illegal_observation_does_not_change_the_board_history_or_search() {
    let mut player = player("material", 64, 8);
    let board = Board::default();
    player.choose(&board, &[]).unwrap();
    let before = snapshot(&player);
    assert!(player.observe("e2e5".parse().unwrap()).is_err());
    if let Player::Mcts {
        board: stored,
        history,
        ..
    } = &player
    {
        assert_eq!(*stored, board);
        assert_eq!(*history, vec![board.hash()]);
    }
    assert_eq!(snapshot(&player).simulations, before.simulations);
    assert_eq!(snapshot(&player).nodes_used, before.nodes_used);
    player.choose(&board, &[]).unwrap();
    assert_eq!(snapshot(&player).simulations, before.simulations + 64);
}

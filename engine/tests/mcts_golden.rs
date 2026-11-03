//! Seeded visit snapshots: the sequential baseline for later batched search.

use engine::board::from_fen;
use engine::mcts::arena::MAX_NODES;
use engine::mcts::{search_with_noise, Leaf, RootNoise, SearchLimits, SearchTree};
use engine::nn::evaluate_board;
use engine::nn::ort_infer::UniformEvaluator;
use engine::uci::notation::move_to_uci;
use rand::{rngs::StdRng, SeedableRng};

const POSITIONS: [(&str, &str); 3] = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    (
        "scholar",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    ),
];

fn snapshot(fen: &str) -> String {
    let board = from_fen(fen).unwrap();
    let result = search_with_noise(
        &board,
        &mut UniformEvaluator,
        SearchLimits {
            nodes: Some(300),
            movetime: None,
        },
        MAX_NODES,
        Some((&RootNoise::default(), &mut StdRng::seed_from_u64(1))),
    )
    .unwrap();
    assert_eq!(result.simulations, 300);
    assert_eq!(
        result.root_children.iter().map(|c| c.visits).sum::<u32>(),
        299
    );
    assert!(result.nodes_used <= MAX_NODES);
    let mut children: Vec<_> = result
        .root_children
        .iter()
        .map(|c| (move_to_uci(&board, c.mv), c.visits))
        .collect();
    children.sort();
    children
        .iter()
        .map(|(mv, n)| format!("{mv}:{n}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn seeded_root_visits_are_deterministic() {
    let golden = include_str!("fixtures/mcts_7_1.txt");
    for (name, fen) in POSITIONS {
        let first = snapshot(fen);
        assert_eq!(first, snapshot(fen), "{name}");
        let expected = golden
            .lines()
            .find(|line| line.starts_with(&format!("{name} ")))
            .unwrap();
        assert_eq!(format!("{name} {first}"), expected, "{name}");
    }
}

#[test]
fn pausing_between_selections_and_evaluation_matches_the_sequential_driver() {
    for (name, fen) in POSITIONS {
        let board = from_fen(fen).unwrap();
        let mut tree = SearchTree::new(board.clone(), &[], MAX_NODES);
        let mut rng = StdRng::seed_from_u64(1);
        let mut completed = 0;
        for stop in [1, 41, 97, 300] {
            while completed < stop {
                let leaf = tree.select_leaf();
                // Taking a snapshot while evaluation is pending must not count
                // the selection as a completed simulation.
                assert_eq!(tree.result().simulations, completed);
                assert_eq!(tree.selected_path().first(), Some(&0));
                match leaf {
                    Leaf::Terminal(v) => {
                        let path = tree.selected_path().to_vec();
                        tree.backup_terminal(&path, v);
                    }
                    Leaf::NeedsEval { node, board } => {
                        let (priors, value) =
                            evaluate_board(&mut UniformEvaluator, &board).unwrap();
                        tree.expand_and_backup(node, &priors, value);
                    }
                }
                completed += 1;
                if completed == 1 {
                    tree.apply_root_noise(&RootNoise::default(), &mut rng);
                }
                assert!(tree.selected_path().is_empty());
            }
            assert_eq!(tree.result().simulations, stop);
        }
        let expected = search_with_noise(
            &board,
            &mut UniformEvaluator,
            SearchLimits {
                nodes: Some(300),
                movetime: None,
            },
            MAX_NODES,
            Some((&RootNoise::default(), &mut StdRng::seed_from_u64(1))),
        )
        .unwrap();
        assert_eq!(
            format!("{:?}", tree.result()),
            format!("{expected:?}"),
            "{name}"
        );
    }
}

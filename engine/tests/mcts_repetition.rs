//! History and trajectory repetition become draws before neural evaluation.

use engine::board::from_fen;
use engine::encoder::move_mapper::NUM_MOVES;
use engine::encoder::tensor_encoder::TENSOR_LEN;
use engine::mcts::arena::MAX_NODES;
use engine::mcts::{Leaf, SearchTree};
use engine::nn::{evaluate_board, Evaluator, NetOutput};
use engine::uci::notation::{move_from_uci, move_to_uci};

const FEN: &str = "7k/8/8/8/8/8/8/K5Q1 w - - 0 1";
const CYCLE: [&str; 4] = ["g1g2", "h8h7", "g2g1", "h7h8"];

/// Test-only value signal: the side with the queen is ahead. Uniform policy
/// alone cannot distinguish a known draw from an unevaluated winning position.
struct QueenValue;
impl Evaluator for QueenValue {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> anyhow::Result<Vec<NetOutput>> {
        assert_eq!(planes.len(), batch * TENSOR_LEN);
        Ok(planes
            .chunks_exact(TENSOR_LEN)
            .map(|p| {
                let us_queen = p[4 * 64..5 * 64].contains(&1.0);
                NetOutput {
                    policy_logits: vec![0.0; NUM_MOVES],
                    wdl_logits: if us_queen {
                        [0.0, 0.0, 9f32.ln()]
                    } else {
                        [9f32.ln(), 0.0, 0.0]
                    },
                }
            })
            .collect())
    }
}

#[test]
fn game_history_marks_the_repeating_root_child_and_prefers_a_winning_alternative() {
    let mut board = from_fen(FEN).unwrap();
    let mut history = vec![board.hash()];
    for text in CYCLE {
        let mv = move_from_uci(&board, text).unwrap();
        board.play_unchecked(mv);
        history.push(board.hash());
    }
    assert_eq!(board.hash(), history[0]);
    let mut tree = SearchTree::new(board.clone(), &history, MAX_NODES);
    let mut evaluated = 0;
    for _ in 0..200 {
        match tree.select_leaf() {
            Leaf::Terminal(v) => {
                let path = tree.selected_path().to_vec();
                tree.backup_terminal(&path, v);
            }
            Leaf::NeedsEval { node, board } => {
                let (priors, value) = evaluate_board(&mut QueenValue, &board).unwrap();
                tree.expand_and_backup(node, &priors, value);
                evaluated += 1;
            }
        }
    }
    let result = tree.result();
    let repeated = result
        .root_children
        .iter()
        .find(|c| move_to_uci(&board, c.mv) == "g1g2")
        .unwrap();
    assert_eq!(result.simulations, 200);
    assert_eq!(repeated.terminal, Some(0.0));
    assert!(repeated.visits > 0);
    assert_eq!(repeated.q, 0.0);
    assert!(!repeated.mates);
    assert_ne!(move_to_uci(&board, result.best_move.unwrap()), "g1g2");
    assert!(result.root_q > 0.0);
    assert!(evaluated > 0 && evaluated < 200);
}

#[test]
fn a_trajectory_cycle_is_terminal_without_game_history_and_stays_cached() {
    let mut tree = SearchTree::new(from_fen(FEN).unwrap(), &[], MAX_NODES);
    // Restrict priors to one legal move at each step so selection must follow
    // the cycle. The fifth selection reaches the root board below the root.
    for (sim, text) in CYCLE.iter().enumerate() {
        let Leaf::NeedsEval { node, board } = tree.select_leaf() else {
            panic!("no position has repeated yet");
        };
        let mv = move_from_uci(&board, text).unwrap();
        tree.expand_and_backup(node, &[(mv, 1.0)], 0.0);
        assert_eq!(tree.result().simulations, sim as u32 + 1);
    }
    for sim in 5..=6 {
        assert!(matches!(tree.select_leaf(), Leaf::Terminal(0.0)));
        let path = tree.selected_path().to_vec();
        assert_eq!(path.len(), 5);
        tree.backup_terminal(&path, 0.0);
        assert_eq!(tree.result().simulations, sim);
        assert_eq!(tree.result().nodes_used, 5);
    }
}

//! Match player construction, greedy search, and observation-driven tree reuse.

use std::path::Path;

use anyhow::{anyhow, ensure, Result};
use cozy_chess::{Board, GameStatus, Move};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use super::material::MaterialEvaluator;
use crate::board::legal_moves;
use crate::encoder::move_mapper::move_to_index;
use crate::mcts::{arena::MAX_NODES, ChildStats, SearchLimits, SearchTree};
use crate::nn::{infer_opts::InferOptions, ort_infer::load_evaluator, Evaluator};

/// One independent player. MCTS keeps its current board and full game history
/// so `observe` can advance the tree without asking the caller for a board.
pub enum Player {
    Mcts {
        eval: Box<dyn Evaluator>,
        tree: SearchTree,
        batch: usize,
        sims: u32,
        board: Board,
        history: Vec<u64>,
    },
    Random {
        rng: StdRng,
    },
}

impl Player {
    /// `spec` is `uniform`, `material`, `random`, or an ONNX model path.
    /// `seed` is the game's seed; random players need no search budget.
    pub fn from_spec(
        spec: &str,
        sims: u32,
        batch: usize,
        opts: &InferOptions,
        seed: u64,
    ) -> Result<Self> {
        opts.validate()?;
        if spec == "random" {
            return Ok(Self::Random {
                rng: StdRng::seed_from_u64(seed),
            });
        }
        ensure!(sims >= 2, "MCTS needs at least 2 simulations per move");
        ensure!(batch > 0, "search batch must be positive");
        ensure!(!spec.is_empty(), "player spec must not be empty");
        let eval: Box<dyn Evaluator> = match spec {
            "uniform" => load_evaluator(None, opts)?,
            "material" => Box::new(MaterialEvaluator),
            path => load_evaluator(Some(Path::new(path)), opts)?,
        };
        let board = Board::default();
        let history = vec![board.hash()];
        Ok(Self::Mcts {
            eval,
            tree: SearchTree::new(board.clone(), &history, MAX_NODES),
            batch,
            sims,
            board,
            history,
        })
    }

    /// Search without playing a move. `history` holds chronological game hashes
    /// and may include the current position. A different position/history resets
    /// the tree. Report every committed move to both players via `observe`.
    pub fn choose(&mut self, board: &Board, history: &[u64]) -> Result<Move> {
        ensure!(
            board.status() == GameStatus::Ongoing,
            "cannot choose in a finished position"
        );
        match self {
            Self::Random { rng } => legal_moves(board)
                .choose(rng)
                .copied()
                .ok_or_else(|| anyhow!("position has no legal moves")),
            Self::Mcts {
                eval,
                tree,
                batch,
                sims,
                board: current,
                history: saved,
            } => {
                let mut hashes = history.to_vec();
                if hashes.last() != Some(&board.hash()) {
                    hashes.push(board.hash());
                }
                if current != board || *saved != hashes {
                    *current = board.clone();
                    *saved = hashes;
                    *tree = SearchTree::new(board.clone(), saved, MAX_NODES);
                }
                let result = tree.search(
                    eval.as_mut(),
                    SearchLimits {
                        nodes: Some(*sims),
                        movetime: None,
                        batch: *batch,
                    },
                    None,
                )?;
                greedy_move(board, &result.root_children)
                    .ok_or_else(|| anyhow!("search produced no legal moves"))
            }
        }
    }

    /// Observe each actual move exactly once, including this player's own move
    /// and opening/opponent moves. Random players have no tree to update.
    /// An illegal observation leaves the MCTS player's state unchanged.
    pub fn observe(&mut self, mv: Move) -> Result<()> {
        if let Self::Mcts {
            tree,
            board,
            history,
            ..
        } = self
        {
            let mut next = board.clone();
            next.try_play(mv)
                .map_err(|_| anyhow!("cannot observe illegal move {mv}"))?;
            history.push(next.hash());
            tree.advance(mv, next.clone(), history);
            *board = next;
        }
        Ok(())
    }
}

// Match the self-play T=0 contract rather than SearchResult.best_move's Q tie.
fn greedy_move(board: &Board, children: &[ChildStats]) -> Option<Move> {
    children
        .iter()
        .max_by(|a, b| {
            a.visits
                .cmp(&b.visits)
                .then_with(|| a.prior.total_cmp(&b.prior))
                .then_with(|| move_to_index(board, b.mv).cmp(&move_to_index(board, a.mv)))
        })
        .map(|c| c.mv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uci::notation::move_from_uci;

    fn child(board: &Board, text: &str, visits: u32, prior: f32, q: f32) -> ChildStats {
        ChildStats {
            mv: move_from_uci(board, text).unwrap(),
            prior,
            visits,
            q,
            terminal: None,
            mates: false,
        }
    }

    #[test]
    fn greedy_orders_by_visits_then_prior_then_canonical_index_not_q() {
        let board = Board::default();
        let a = child(&board, "e2e4", 2, 0.1, -1.0);
        let b = child(&board, "d2d4", 1, 0.9, 1.0);
        assert_eq!(greedy_move(&board, &[a.clone(), b.clone()]), Some(a.mv));
        let b = ChildStats { visits: 2, ..b };
        assert_eq!(greedy_move(&board, &[a.clone(), b.clone()]), Some(b.mv));
        let a = ChildStats {
            prior: 0.9,
            q: 1.0,
            ..a
        };
        let b = ChildStats { q: -1.0, ..b };
        assert_eq!(greedy_move(&board, &[b.clone(), a.clone()]), Some(b.mv));
        assert_eq!(greedy_move(&board, &[a, b.clone()]), Some(b.mv));
    }

    #[test]
    fn greedy_uses_black_canonical_indices() {
        let mut board = Board::default();
        board.play(move_from_uci(&board, "e2e4").unwrap());
        let children = [
            child(&board, "e7e5", 1, 0.5, 1.0),
            child(&board, "b8a6", 1, 0.5, -1.0),
        ];
        assert_eq!(greedy_move(&board, &children), Some(children[1].mv));
    }
}

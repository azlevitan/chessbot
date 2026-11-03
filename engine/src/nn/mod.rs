//! Neural-network evaluation: the `Evaluator` trait plus prior / value helpers.

pub mod infer_opts;
pub mod ort_infer;

use anyhow::{ensure, Result};
use cozy_chess::{Board, Move};

use crate::board::legal_moves;
use crate::encoder::move_mapper::{move_to_index, NUM_MOVES};
use crate::encoder::tensor_encoder::{encode_planes_into, TENSOR_LEN};

/// Raw network output for one position.
#[derive(Clone, Debug)]
pub struct NetOutput {
    /// `NUM_MOVES` policy logits (unmasked).
    pub policy_logits: Vec<f32>,
    /// `[Loss, Draw, Win]` logits from the side to move's perspective.
    pub wdl_logits: [f32; 3],
}

/// Anything that can score a batch of `18x8x8` planes (flat, plane-major).
pub trait Evaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>>;
}

/// Legal move priors and a scalar value, both for the board's side to move.
pub type BoardEvaluation = (Vec<(Move, f32)>, f32);

/// Scalar value `p_win - p_loss` in `[-1, 1]` after softmax (SPEC.md §5).
pub fn wdl_value(wdl_logits: &[f32; 3]) -> f32 {
    let max = wdl_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = wdl_logits.map(|l| (l - max).exp());
    let sum: f32 = exp.iter().sum();
    (exp[2] - exp[0]) / sum
}

/// Softmax over the logits of legal moves only (equivalent to masking every
/// illegal logit to -1e9). Returns `(move, prior)` for every legal move.
pub fn legal_priors(board: &Board, policy_logits: &[f32]) -> Vec<(Move, f32)> {
    debug_assert_eq!(policy_logits.len(), NUM_MOVES);
    let moves = legal_moves(board);
    let logits: Vec<f32> = moves
        .iter()
        .map(|&m| policy_logits[move_to_index(board, m)])
        .collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    moves
        .into_iter()
        .zip(exp)
        .map(|(m, e)| (m, e / sum))
        .collect()
}

/// Encode `board`, run the evaluator, and return `(legal priors, value)`.
pub fn evaluate_board(eval: &mut dyn Evaluator, board: &Board) -> Result<BoardEvaluation> {
    Ok(evaluate_boards(eval, std::slice::from_ref(board))?
        .pop()
        .unwrap())
}

/// Encode boards in order and run exactly one evaluator call (none if empty).
/// Every output is validated before any search expansion can use the batch.
pub fn evaluate_boards(eval: &mut dyn Evaluator, boards: &[Board]) -> Result<Vec<BoardEvaluation>> {
    if boards.is_empty() {
        return Ok(Vec::new());
    }
    let mut planes = vec![0.0f32; boards.len() * TENSOR_LEN];
    for (board, row) in boards.iter().zip(planes.chunks_exact_mut(TENSOR_LEN)) {
        encode_planes_into(board, row);
    }
    let outputs = eval.evaluate(&planes, boards.len())?;
    ensure!(
        outputs.len() == boards.len(),
        "evaluator returned {} outputs for batch {}",
        outputs.len(),
        boards.len()
    );
    for out in &outputs {
        ensure!(
            out.policy_logits.len() == NUM_MOVES,
            "policy output length mismatch"
        );
        ensure!(
            out.policy_logits
                .iter()
                .chain(&out.wdl_logits)
                .all(|v| v.is_finite()),
            "evaluator returned non-finite logits"
        );
    }
    Ok(boards
        .iter()
        .zip(outputs)
        .map(|(board, out)| {
            (
                legal_priors(board, &out.policy_logits),
                wdl_value(&out.wdl_logits),
            )
        })
        .collect())
}

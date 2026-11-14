//! Self-play data generation: play games with noisy MCTS and label each
//! searched position with the final result from its side to move.

pub mod batch_runner;
pub mod driver;
pub mod game;
pub mod progress;
pub mod resign;
pub mod stats;

pub use game::{play_game, GameRecord, PlyRecord, SelfPlayConfig, Termination};

use crate::cli::bin_writer::outcome_for;
use crate::encoder::sample::TrainingSample;
use crate::encoder::tensor_encoder::encode_sample;

/// One `TrainingSample` per searched ply: visit-share policy, game outcome
/// as the WDL label, and the MCTS root value as `q_value`.
pub fn game_samples(game: &GameRecord) -> Vec<TrainingSample> {
    game.plies
        .iter()
        .map(|p| {
            let (wdl, _) = outcome_for(p.board.side_to_move(), game.result);
            encode_sample(&p.board, &p.policy, wdl, p.root_q)
        })
        .collect()
}

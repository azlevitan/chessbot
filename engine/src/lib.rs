//! Chess engine core: board wrapper, canonical encoders, NN inference, MCTS,
//! UCI and CLI tools. See SPEC.md for every on-disk / on-tensor contract.

pub mod board;
pub mod cli;
pub mod encoder;
pub mod mcts;
pub mod nn;
pub mod pgn;
pub mod selfplay;
pub mod uci;

//! Fixed-capacity arena MCTS with PUCT selection (SPEC.md §5).

pub mod arena;
mod batch;
pub mod noise;
mod result;
pub mod search;
pub mod tree;

pub use noise::RootNoise;
pub use search::{search, search_with_noise, ChildStats, SearchLimits, SearchResult, C_PUCT};
pub use tree::{Leaf, SearchTree};

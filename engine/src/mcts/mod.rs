//! Fixed-capacity arena MCTS with PUCT selection (SPEC.md §5).

pub mod arena;
pub(crate) mod batch;
pub mod noise;
mod result;
mod reuse;
pub mod search;
pub mod tree;

pub use noise::RootNoise;
pub use search::{search, search_with_noise, ChildStats, SearchLimits, SearchResult, C_PUCT};
pub use tree::{Leaf, SearchTree};

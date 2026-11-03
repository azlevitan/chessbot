//! Fixed-capacity arena MCTS with PUCT selection (SPEC.md §5).

pub mod arena;
pub mod noise;
pub mod search;

pub use noise::RootNoise;
pub use search::{search, search_with_noise, ChildStats, SearchLimits, SearchResult, C_PUCT};

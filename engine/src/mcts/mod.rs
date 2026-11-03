//! Fixed-capacity arena MCTS with PUCT selection (SPEC.md §5).

pub mod arena;
pub mod search;

pub use search::{search, ChildStats, SearchLimits, SearchResult, C_PUCT};

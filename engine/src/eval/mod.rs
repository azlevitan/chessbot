//! Deterministic evaluation players and the embedded opening book (SPEC §9).

pub mod adjudicate;
pub mod match_runner;
pub mod material;
pub mod openings;
pub mod players;

pub use material::MaterialEvaluator;
pub use openings::openings;
pub use players::Player;

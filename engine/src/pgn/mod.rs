//! PGN parsing for supervised bootstrapping: streaming reader + SAN resolver.

pub mod reader;
pub mod san;

pub use reader::{PgnReader, RawGame};
pub use san::san_to_move;

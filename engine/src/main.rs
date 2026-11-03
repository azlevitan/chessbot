//! `engine` binary entry point: dispatches subcommands.

use anyhow::{bail, Result};

const USAGE: &str = "usage: engine <command> [args]\n\ncommands:\n  dump-test-vectors [--out <path>]";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("dump-test-vectors") => engine::cli::dump_vectors::run(&args[1..]),
        Some("-h" | "--help") | None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => bail!("unknown command: {other}\n\n{USAGE}"),
    }
}

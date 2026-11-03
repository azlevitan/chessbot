//! Self-play scheduling options are validated before opening output files.

use std::process::Command;

#[test]
fn cli_defaults_and_invalid_limits_are_checked_before_output_is_touched() {
    let dir = std::env::temp_dir().join(format!("chessbot-sp-options-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("keep.bin");
    let base = vec!["--out".to_string(), out.display().to_string()];
    let defaults = engine::cli::selfplay::parse_args(&base).unwrap();
    assert_eq!(
        (
            defaults.batch.parallel,
            defaults.batch.leaf_batch,
            defaults.cfg.temp_plies
        ),
        (16, 1, 30)
    );
    std::fs::write(&out, b"keep binary").unwrap();
    std::fs::write(out.with_extension("json"), b"keep sidecar").unwrap();
    for (flag, value) in [
        ("--parallel", "0"),
        ("--leaf-batch", "0"),
        ("--parallel", "-1"),
        ("--leaf-batch", "bad"),
        ("--sims", "1"),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_engine"))
            .arg("selfplay")
            .args(&base)
            .args([flag, value])
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert_eq!(std::fs::read(&out).unwrap(), b"keep binary");
        assert_eq!(
            std::fs::read(out.with_extension("json")).unwrap(),
            b"keep sidecar"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

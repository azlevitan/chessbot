//! Phase 7.5: per-colour streaks and the playthrough false-positive audit.

use cozy_chess::Color::{Black, White};
use engine::cli::bin_writer::GameResult;
use engine::selfplay::resign::ResignTracker;
use engine::selfplay::stats::SelfPlayStats;
use engine::selfplay::{GameRecord, SelfPlayConfig, Termination};

#[test]
fn recovery_resets_only_that_sides_streak_and_resigns_on_its_sixth_move() {
    let mut tracker = ResignTracker::new(-0.9, 3, false);
    let values = [-0.95, -0.95, -0.5, -0.95, -0.95, -0.95];
    for (i, q) in values.into_iter().enumerate() {
        assert_eq!(tracker.update(White, q), (i == 5).then_some(White));
        assert_eq!(tracker.update(Black, 0.0), None);
    }
    assert_eq!(tracker.would_resign(), None);
}

#[test]
fn sides_have_independent_streaks_and_equality_does_not_resign() {
    let mut tracker = ResignTracker::new(-0.9, 3, false);
    assert_eq!(tracker.update(Black, -0.95), None);
    assert_eq!(tracker.update(White, -0.95), None);
    assert_eq!(tracker.update(Black, -0.95), None);
    assert_eq!(tracker.update(White, -0.9), None);
    assert_eq!(tracker.update(Black, -0.95), Some(Black));
    assert_eq!(tracker.update(White, -0.95), None);
    assert_eq!(tracker.update(White, -0.95), None);
    assert_eq!(tracker.update(White, -0.95), Some(White));
}

#[test]
fn playthrough_never_resigns_and_keeps_the_first_trigger_through_recovery() {
    let mut tracker = ResignTracker::new(-0.9, 3, true);
    for (i, q) in [-0.95, -0.95, -0.5, -0.95, -0.95, -0.95].into_iter().enumerate() {
        assert_eq!(tracker.update(White, q), None);
        assert_eq!(tracker.would_resign(), (i == 5).then_some(White));
    }
    assert!(tracker.is_playthrough());
    assert_eq!(tracker.update(White, 0.0), None);
    for _ in 0..4 {
        assert_eq!(tracker.update(Black, -1.0), None);
    }
    assert_eq!(tracker.would_resign(), Some(White));
}

#[test]
fn thresholds_at_or_below_minus_one_disable_both_resignation_and_audit_triggers() {
    for threshold in [-1.0, -1.5] {
        for playthrough in [false, true] {
            let mut tracker = ResignTracker::new(threshold, 1, playthrough);
            for side in [White, Black, White, Black] {
                assert_eq!(tracker.update(side, -1.0), None);
            }
            assert_eq!(tracker.would_resign(), None);
        }
    }
}

#[test]
fn audit_counts_a_false_positive_only_when_the_first_would_be_resigner_did_not_lose() {
    let cfg = SelfPlayConfig::default();
    for side in [White, Black] {
        let mut stats = SelfPlayStats::default();
        for result in [GameResult::WhiteWins, GameResult::BlackWins, GameResult::Draw] {
            let mut tracker = ResignTracker::new(-0.9, 1, true);
            tracker.update(side, -0.95);
            let mut one = SelfPlayStats::default();
            let game = GameRecord {
                plies: Vec::new(), result, termination: Termination::PlyCap,
                playthrough: tracker.is_playthrough(), would_resign: tracker.would_resign(),
            };
            one.record(&game);
            let audit = one.sidecar("uniform", 0, &cfg);
            assert_eq!(audit["resign"]["playthrough_games"], 1);
            assert_eq!(audit["resign"]["false_positives"], u64::from(result != GameResult::win_for(!side)));
            stats.record(&game);
        }
        // An audit game with no trigger contributes to the denominator only.
        stats.record(&GameRecord {
            plies: Vec::new(), result: GameResult::Draw, termination: Termination::PlyCap,
            playthrough: true, would_resign: None,
        });
        stats.record(&GameRecord {
            plies: Vec::new(), result: GameResult::Draw, termination: Termination::PlyCap,
            playthrough: false, would_resign: None,
        });
        let meta = stats.sidecar("uniform", 0, &cfg);
        assert_eq!(meta["games"], 5);
        assert_eq!(meta["resign"]["playthrough_games"], 4);
        assert_eq!(meta["resign"]["false_positives"], 2);
    }
}

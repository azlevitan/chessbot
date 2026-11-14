use super::*;
use std::sync::mpsc;

#[test]
fn formatter_distinguishes_completed_games_from_ordered_output_and_busy_inference() {
    let snapshot = ProgressSnapshot {
        phase: ProgressPhase::Evaluating,
        games: 50,
        started: 32,
        completed: 3,
        emitted: 0,
        active: 29,
        waiting: 3,
        plies: 123,
        first_game: Some(0),
        first_ply: 4,
        eval_calls: 8,
        eval_positions: 192,
        batch_positions: 29,
    };
    let line = format_progress(snapshot, Duration::from_secs(10),
                               Duration::from_secs(6), Duration::from_secs(7));
    for text in ["phase=evaluating", "completed=3/50", "emitted=0", "waiting=3",
                 "moves=123", "first_game=1:ply4", "eval_positions=192",
                 "batch_positions=29", "in_flight=6.0s", "since_progress=7.0s"] {
        assert!(line.contains(text), "missing {text} in {line}");
    }
}

#[test]
fn watchdog_reports_while_caller_is_blocked_and_stops_without_waiting_for_interval() {
    let (send, receive) = mpsc::channel();
    let reporter = ProgressReporter::with_sink(2, Duration::from_millis(10), move |line| {
        let _ = send.send(line);
    });
    assert!(receive.recv_timeout(Duration::from_secs(1)).unwrap().contains("phase=loading_model"));
    reporter.update(ProgressSnapshot {
        phase: ProgressPhase::Evaluating,
        games: 2,
        active: 2,
        batch_positions: 2,
        ..ProgressSnapshot::default()
    });
    let deadline = Instant::now() + Duration::from_secs(1);
    let heartbeat = loop {
        let line = receive.recv_timeout(deadline.saturating_duration_since(Instant::now())).unwrap();
        if line.contains("phase=evaluating") { break line; }
    };
    assert!(heartbeat.contains("completed=0/2"));
    assert!(heartbeat.contains("in_flight="));
    drop(reporter);
    assert!(receive.recv_timeout(Duration::from_secs(1)).is_ok());
    while receive.try_recv().is_ok() {}
    assert!(matches!(receive.try_recv(), Err(mpsc::TryRecvError::Disconnected)));
}

#[test]
fn repeated_busy_snapshots_do_not_reset_time_since_last_real_progress() {
    let reporter = ProgressReporter::with_sink(1, Duration::from_secs(60), |_| {});
    let busy = ProgressSnapshot { phase: ProgressPhase::Evaluating, games: 1,
                                  ..ProgressSnapshot::default() };
    reporter.update(busy);
    let before = reporter.shared.0.lock().unwrap().last_progress;
    reporter.update(busy);
    assert_eq!(reporter.shared.0.lock().unwrap().last_progress, before);
    reporter.update(ProgressSnapshot { eval_positions: 1, ..busy });
    assert!(reporter.shared.0.lock().unwrap().last_progress >= before);
}

//! Temperature decisions, simulation budgets, and combined-batch recovery.

use super::*;
use crate::board::legal_moves;
use crate::nn::ort_infer::UniformEvaluator;
use crate::nn::{Evaluator, NetOutput};
use crate::selfplay::batch_runner::run_step;
use rand::{RngCore, SeedableRng};

fn child(mv: Move, visits: u32, prior: f32, q: f32) -> ChildStats {
    ChildStats {
        mv,
        visits,
        prior,
        q,
        terminal: None,
        mates: false,
    }
}

#[test]
fn greedy_ties_use_prior_then_canonical_index_for_both_colors() {
    let mut black = Board::default();
    black.play_unchecked("e2e4".parse().unwrap());
    for board in [Board::default(), black] {
        let mut moves = legal_moves(&board);
        moves.sort_by_key(|&mv| move_to_index(&board, mv));
        let mut children = vec![
            child(moves[0], 4, 0.1, 1.0),
            child(moves[2], 4, 0.4, 0.9),
            child(moves[1], 4, 0.4, -0.9),
            child(moves[3], 0, 1.0, 1.0),
        ];
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(choose_move(&board, &children, 10, 10, &mut rng), moves[1]);
        children[0].visits = 5;
        assert_eq!(choose_move(&board, &children, 10, 10, &mut rng), moves[0]);
    }
}

#[test]
fn temperature_boundary_samples_visits_then_stops_consuming_rng() {
    let board = Board::default();
    let moves = legal_moves(&board);
    let children = [child(moves[0], 1, 0.5, 0.0), child(moves[1], 3, 0.5, 0.0)];
    let mut counts = [0, 0];
    let mut rng = StdRng::seed_from_u64(17);
    for _ in 0..4000 {
        let mv = choose_move(&board, &children, 9, 10, &mut rng);
        counts[usize::from(mv == moves[1])] += 1;
    }
    assert!((counts[0] as f32 / 4000.0 - 0.25).abs() < 0.03);
    let mut untouched = rng.clone();
    for ply in [10, 11, 100] {
        assert_eq!(choose_move(&board, &children, ply, 10, &mut rng), moves[1]);
    }
    assert_eq!(rng.next_u64(), untouched.next_u64());
}

#[test]
fn each_move_adds_exactly_the_budget_including_retained_visits() {
    for reuse in [false, true] {
        let cfg = SelfPlayConfig {
            sims: 18,
            max_plies: 8,
            reuse,
            ..SelfPlayConfig::default()
        };
        let mut driver = GameDriver::new(&cfg, StdRng::seed_from_u64(9)).unwrap();
        while !driver.is_finished() {
            let before = driver.tree.simulations();
            let baseline = driver.move_start;
            let plies = driver.plies.len();
            let pending = driver.step_request(8).unwrap();
            let outputs =
                crate::nn::evaluate_boards(&mut UniformEvaluator, pending.boards()).unwrap();
            pending.complete(&mut outputs.into_iter());
            let after = driver.tree.simulations();
            assert!(after > before && after - before <= 8);
            assert!(after - baseline <= cfg.sims);
            driver.step_apply().unwrap();
            if driver.plies.len() > plies {
                assert_eq!(after - baseline, cfg.sims);
            }
            let snapshot = driver.tree.result();
            assert!(snapshot.nodes_used <= MAX_NODES);
            if !snapshot.root_children.is_empty() {
                assert_eq!(
                    snapshot.simulations,
                    1 + snapshot.root_children.iter().map(|c| c.visits).sum::<u32>()
                );
            }
        }
    }
}

struct Broken(u8);
impl Evaluator for Broken {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        let mut outputs = UniformEvaluator.evaluate(planes, batch)?;
        match self.0 {
            0 => anyhow::bail!("injected inference error"),
            1 => {
                outputs.pop();
            }
            2 => {
                outputs.last_mut().unwrap().policy_logits.pop();
            }
            _ => outputs.last_mut().unwrap().wdl_logits[0] = f32::NAN,
        }
        Ok(outputs)
    }
}

#[test]
fn combined_inference_errors_release_all_trees_and_allow_resume() {
    for broken in 0..4 {
        let cfg = SelfPlayConfig {
            sims: 20,
            max_plies: 2,
            ..SelfPlayConfig::default()
        };
        let mut drivers: Vec<_> = (0..3)
            .map(|seed| GameDriver::new(&cfg, StdRng::seed_from_u64(seed)).unwrap())
            .collect();
        run_step(&mut UniformEvaluator, &mut drivers, 8).unwrap();
        // Apply noise before taking snapshots, so the failure only rolls back reservations.
        for driver in &mut drivers {
            driver.tree.apply_root_noise(&cfg.noise, &mut driver.rng);
        }
        let before: Vec<_> = drivers
            .iter()
            .map(|d| format!("{:?}", d.tree.result()))
            .collect();
        assert!(run_step(&mut Broken(broken), &mut drivers, 8).is_err());
        for (driver, expected) in drivers.iter().zip(before) {
            assert_eq!(format!("{:?}", driver.tree.result()), expected);
        }
        run_step(&mut UniformEvaluator, &mut drivers, 8).unwrap();
        assert!(drivers.iter().all(|d| d.tree.simulations() == 9));
    }
}

#[test]
fn terminal_only_steps_complete_simulations_without_inference() {
    let cfg = SelfPlayConfig {
        sims: 5,
        max_plies: 1,
        ..SelfPlayConfig::default()
    };
    let mut driver = GameDriver::new(&cfg, StdRng::seed_from_u64(2)).unwrap();
    // Every child repeats a historical board, so all child searches are terminal.
    for mv in legal_moves(&driver.board) {
        let mut child = driver.board.clone();
        child.play_unchecked(mv);
        driver.history.push(child.hash());
    }
    driver.tree = SearchTree::new(driver.board.clone(), &driver.history, MAX_NODES);
    run_step(&mut UniformEvaluator, std::slice::from_mut(&mut driver), 4).unwrap();
    run_step(&mut Broken(0), std::slice::from_mut(&mut driver), 4).unwrap();
    assert!(driver.is_finished());
    assert_eq!(driver.tree.simulations(), 5);
}

#[test]
fn terminal_completed_game_resumes_when_another_games_inference_fails() {
    let cfg = SelfPlayConfig {
        sims: 5,
        max_plies: 1,
        ..SelfPlayConfig::default()
    };
    let mut drivers: Vec<_> = (0..2)
        .map(|seed| GameDriver::new(&cfg, StdRng::seed_from_u64(seed)).unwrap())
        .collect();
    let terminal = &mut drivers[0];
    for mv in legal_moves(&terminal.board) {
        let mut board = terminal.board.clone();
        board.play_unchecked(mv);
        terminal.history.push(board.hash());
    }
    terminal.tree = SearchTree::new(terminal.board.clone(), &terminal.history, MAX_NODES);
    run_step(&mut UniformEvaluator, &mut drivers, 4).unwrap();
    assert!(run_step(&mut Broken(0), &mut drivers, 4).is_err());
    assert_eq!(drivers[0].tree.simulations(), cfg.sims);
    assert_eq!(drivers[1].tree.simulations(), 1);
    run_step(&mut UniformEvaluator, &mut drivers, 4).unwrap();
    assert!(drivers.iter().all(GameDriver::is_finished));
}

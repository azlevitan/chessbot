//! Per-side resignation streaks and the first would-be resigner in audit games.

use cozy_chess::Color;

#[derive(Clone, Debug)]
pub struct ResignTracker {
    threshold: f32,
    consecutive: u32,
    playthrough: bool,
    streaks: [u32; 2],
    would_resign: Option<Color>,
}

impl ResignTracker {
    pub fn new(threshold: f32, consecutive: u32, playthrough: bool) -> Self {
        assert!(threshold.is_finite(), "resignation threshold must be finite");
        assert!(consecutive > 0, "resignation consecutive count must be positive");
        Self {
            threshold,
            consecutive,
            playthrough,
            streaks: [0; 2],
            would_resign: None,
        }
    }

    /// Call once per completed move search, with Q from that side's perspective.
    pub fn update(&mut self, stm: Color, root_q: f32) -> Option<Color> {
        if self.threshold <= -1.0 {
            return None;
        }
        let streak = &mut self.streaks[stm as usize];
        *streak = if root_q < self.threshold {
            streak.saturating_add(1)
        } else {
            0
        };
        if *streak < self.consecutive {
            return None;
        }
        if self.playthrough {
            self.would_resign.get_or_insert(stm);
            None
        } else {
            Some(stm)
        }
    }

    pub fn is_playthrough(&self) -> bool {
        self.playthrough
    }

    pub fn would_resign(&self) -> Option<Color> {
        self.would_resign
    }
}

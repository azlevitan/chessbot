//! Root Dirichlet noise for self-play exploration (AlphaZero: alpha = 0.3, eps = 0.25).

use rand::Rng;
use rand_distr::{Distribution, Gamma};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootNoise {
    pub alpha: f32,
    pub eps: f32,
}

impl Default for RootNoise {
    fn default() -> Self {
        Self { alpha: 0.3, eps: 0.25 }
    }
}

impl RootNoise {
    /// Sample `eta ~ Dir(alpha)` of length `n` via normalized `Gamma(alpha, 1)` draws.
    pub fn sample_dirichlet<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Vec<f32> {
        let gamma = Gamma::new(self.alpha, 1.0).expect("alpha must be > 0");
        let mut eta: Vec<f32> = (0..n).map(|_| gamma.sample(rng)).collect();
        let sum: f32 = eta.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            eta.iter_mut().for_each(|x| *x /= sum);
        } else {
            // Every draw underflowed to 0 (tiny alpha): fall back to uniform.
            eta.fill(1.0 / n as f32);
        }
        eta
    }

    /// Mix noise into `priors` in place: `p <- (1 - eps) * p + eps * eta`.
    pub fn apply<R: Rng + ?Sized>(&self, priors: &mut [f32], rng: &mut R) {
        if priors.is_empty() {
            return;
        }
        let eta = self.sample_dirichlet(priors.len(), rng);
        for (p, e) in priors.iter_mut().zip(eta) {
            *p = (1.0 - self.eps) * *p + self.eps * e;
        }
    }
}

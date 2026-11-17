use std::f32::consts::TAU;

/// Deterministic 64-bit SplitMix/XorShift-style PRNG with zero external dependencies.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid zero lockup state.
        let initial = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        let mut rng = Self { state: initial };
        rng.next_u64();
        rng
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24 bits of mantissa precision in [0.0, 1.0).
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1_u32 << 24) as f32)
    }

    #[inline]
    pub fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    #[inline]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % (bound as u64)) as usize
    }

    #[inline]
    pub fn next_gaussian(&mut self, sigma: f32) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let mag = sigma * (-2.0 * u1.ln()).sqrt();
        mag * (TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let mut c = Rng::new(43);

        let seq_a: Vec<u64> = (0..32).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..32).map(|_| b.next_u64()).collect();
        let seq_c: Vec<u64> = (0..32).map(|_| c.next_u64()).collect();

        assert_eq!(seq_a, seq_b);
        assert_ne!(seq_a, seq_c);
    }

    #[test]
    fn gaussian_is_finite_and_centered() {
        let mut rng = Rng::new(1337);
        let mut sum = 0.0_f32;
        let n = 4000;
        for _ in 0..n {
            let g = rng.next_gaussian(1.0);
            assert!(g.is_finite());
            sum += g;
        }
        let mean = sum / (n as f32);
        assert!(mean.abs() < 0.1, "mean={mean}");
    }
}

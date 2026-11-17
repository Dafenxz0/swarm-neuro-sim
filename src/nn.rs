use crate::rng::Rng;

pub const NUM_RAYS: usize = 7;
pub const INPUTS: usize = NUM_RAYS + 1; // 7 raycast proximities + normalized speed
pub const HIDDEN: usize = 10;
pub const OUTPUTS: usize = 2; // [steering, throttle]

pub const WEIGHTS_IH_LEN: usize = INPUTS * HIDDEN;
pub const BIAS_H_LEN: usize = HIDDEN;
pub const WEIGHTS_HO_LEN: usize = HIDDEN * OUTPUTS;
pub const BIAS_O_LEN: usize = OUTPUTS;
pub const GENOME_LEN: usize = WEIGHTS_IH_LEN + BIAS_H_LEN + WEIGHTS_HO_LEN + BIAS_O_LEN;
pub const BRAIN_SNAPSHOT_LEN: usize = INPUTS + HIDDEN + OUTPUTS + GENOME_LEN;

/// Compact feedforward neural network (`8 -> 10 -> 2`) with `tanh` activations.
#[derive(Clone, Debug)]
pub struct NeuralNet {
    pub weights: [f32; GENOME_LEN],
    pub last_inputs: [f32; INPUTS],
    pub last_hidden: [f32; HIDDEN],
    pub last_outputs: [f32; OUTPUTS],
}

impl NeuralNet {
    pub fn random(rng: &mut Rng) -> Self {
        let mut weights = [0.0_f32; GENOME_LEN];
        let ih_scale = (2.0 / (INPUTS + HIDDEN) as f32).sqrt();
        let ho_scale = (2.0 / (HIDDEN + OUTPUTS) as f32).sqrt();

        for (i, w) in weights.iter_mut().enumerate() {
            if i < WEIGHTS_IH_LEN {
                *w = rng.next_gaussian(ih_scale);
            } else if i < WEIGHTS_IH_LEN + BIAS_H_LEN {
                *w = rng.next_signed() * 0.2;
            } else if i < WEIGHTS_IH_LEN + BIAS_H_LEN + WEIGHTS_HO_LEN {
                *w = rng.next_gaussian(ho_scale);
            } else {
                *w = rng.next_signed() * 0.1;
            }
        }
        // Slightly bias initial throttle neuron positive so gen-0 agents start driving.
        weights[GENOME_LEN - 1] = 0.45 + rng.next_signed() * 0.25;

        Self {
            weights,
            last_inputs: [0.0; INPUTS],
            last_hidden: [0.0; HIDDEN],
            last_outputs: [0.0; OUTPUTS],
        }
    }

    pub fn from_weights(weights: [f32; GENOME_LEN]) -> Self {
        Self {
            weights,
            last_inputs: [0.0; INPUTS],
            last_hidden: [0.0; HIDDEN],
            last_outputs: [0.0; OUTPUTS],
        }
    }

    pub fn forward(&mut self, inputs: &[f32; INPUTS]) -> [f32; OUTPUTS] {
        self.last_inputs = *inputs;

        let ih_offset = 0;
        let bh_offset = WEIGHTS_IH_LEN;
        let ho_offset = bh_offset + BIAS_H_LEN;
        let bo_offset = ho_offset + WEIGHTS_HO_LEN;

        for h in 0..HIDDEN {
            let mut sum = self.weights[bh_offset + h];
            let row = ih_offset + h * INPUTS;
            for (i, &inp) in inputs.iter().enumerate() {
                sum += inp * self.weights[row + i];
            }
            self.last_hidden[h] = sum.tanh();
        }

        for o in 0..OUTPUTS {
            let mut sum = self.weights[bo_offset + o];
            let row = ho_offset + o * HIDDEN;
            for (h, &hid) in self.last_hidden.iter().enumerate() {
                sum += hid * self.weights[row + h];
            }
            self.last_outputs[o] = sum.tanh();
        }

        self.last_outputs
    }

    pub fn crossover(parent_a: &Self, parent_b: &Self, rng: &mut Rng) -> Self {
        let mut child = [0.0_f32; GENOME_LEN];
        let cut = rng.next_usize(GENOME_LEN);
        for (i, w) in child.iter_mut().enumerate() {
            *w = if i < cut {
                if rng.next_f32() < 0.85 {
                    parent_a.weights[i]
                } else {
                    parent_b.weights[i]
                }
            } else if rng.next_f32() < 0.85 {
                parent_b.weights[i]
            } else {
                parent_a.weights[i]
            };
        }
        Self::from_weights(child)
    }

    pub fn mutate(&mut self, rng: &mut Rng, rate: f32, sigma: f32) {
        for w in &mut self.weights {
            let roll = rng.next_f32();
            if roll < rate {
                let delta = if rng.next_f32() < 0.08 {
                    rng.next_gaussian(sigma * 2.5)
                } else {
                    rng.next_gaussian(sigma)
                };
                *w = (*w + delta).clamp(-6.0, 6.0);
            }
        }
    }

    pub fn write_snapshot(&self, out: &mut [f32]) {
        debug_assert!(out.len() >= BRAIN_SNAPSHOT_LEN);
        let mut cursor = 0;
        out[cursor..cursor + INPUTS].copy_from_slice(&self.last_inputs);
        cursor += INPUTS;
        out[cursor..cursor + HIDDEN].copy_from_slice(&self.last_hidden);
        cursor += HIDDEN;
        out[cursor..cursor + OUTPUTS].copy_from_slice(&self.last_outputs);
        cursor += OUTPUTS;
        out[cursor..cursor + GENOME_LEN].copy_from_slice(&self.weights);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_pass_outputs_in_tanh_bounds_and_deterministic() {
        let mut rng = Rng::new(77);
        let mut net = NeuralNet::random(&mut rng);
        let inputs = [0.5, 0.8, 0.2, 1.0, 0.3, 0.7, 0.1, 0.4];

        let out1 = net.forward(&inputs);
        let out2 = net.forward(&inputs);

        assert_eq!(out1, out2);
        for val in out1 {
            assert!((-1.0..=1.0).contains(&val));
            assert!(val.is_finite());
        }
        for val in net.last_hidden {
            assert!((-1.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn zero_weights_produce_zero_outputs() {
        let mut net = NeuralNet::from_weights([0.0; GENOME_LEN]);
        let out = net.forward(&[1.0; INPUTS]);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn crossover_and_mutation_are_deterministic_for_seed() {
        let mut rng_init = Rng::new(10);
        let pa = NeuralNet::random(&mut rng_init);
        let pb = NeuralNet::random(&mut rng_init);

        let mut rng1 = Rng::new(999);
        let mut c1 = NeuralNet::crossover(&pa, &pb, &mut rng1);
        c1.mutate(&mut rng1, 0.2, 0.3);

        let mut rng2 = Rng::new(999);
        let mut c2 = NeuralNet::crossover(&pa, &pb, &mut rng2);
        c2.mutate(&mut rng2, 0.2, 0.3);

        assert_eq!(c1.weights, c2.weights);
        assert_ne!(c1.weights, pa.weights);
        assert_ne!(c1.weights, pb.weights);
    }
}

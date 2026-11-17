use crate::nn::{NeuralNet, BRAIN_SNAPSHOT_LEN, INPUTS, NUM_RAYS};
use crate::rng::Rng;
use crate::track::{ray_cast, segments_intersect, Segment, Track};

pub const RAY_MAX_DIST: f32 = 220.0;
pub const AGENT_RADIUS: f32 = 7.0;
pub const MAX_SPEED: f32 = 4.6;
pub const MIN_SPEED: f32 = 0.0;
pub const MAX_TURN_RATE: f32 = 0.115;
pub const MAX_GEN_TICKS: u32 = 1600;
pub const STALL_GRACE_TICKS: u32 = 110;

pub const RAY_ANGLES: [f32; NUM_RAYS] = [-1.25, -0.78, -0.36, 0.0, 0.36, 0.78, 1.25];
pub const AGENT_STRIDE: usize = 6; // [x, y, angle, alive(0/1), is_best(0/1), fitness]
pub const RAY_STRIDE: usize = 3; // [hit_x, hit_y, norm_dist]
pub const HISTORY_MAX: usize = 256;

#[derive(Clone, Debug)]
pub struct Agent {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub speed: f32,
    pub alive: bool,
    pub fitness: f32,
    pub distance_traveled: f32,
    pub checkpoints_passed: u32,
    pub next_checkpoint: usize,
    pub ticks_since_progress: u32,
    pub brain: NeuralNet,
    pub ray_hits: [(f32, f32, f32); NUM_RAYS],
}

impl Agent {
    fn new(spawn_x: f32, spawn_y: f32, spawn_angle: f32, brain: NeuralNet) -> Self {
        Self {
            x: spawn_x,
            y: spawn_y,
            angle: spawn_angle,
            speed: 0.0,
            alive: true,
            fitness: 0.0,
            distance_traveled: 0.0,
            checkpoints_passed: 0,
            next_checkpoint: 0,
            ticks_since_progress: 0,
            brain,
            ray_hits: [(spawn_x, spawn_y, 1.0); NUM_RAYS],
        }
    }

    fn reset(&mut self, spawn_x: f32, spawn_y: f32, spawn_angle: f32) {
        self.x = spawn_x;
        self.y = spawn_y;
        self.angle = spawn_angle;
        self.speed = 0.0;
        self.alive = true;
        self.fitness = 0.0;
        self.distance_traveled = 0.0;
        self.checkpoints_passed = 0;
        self.next_checkpoint = 0;
        self.ticks_since_progress = 0;
        self.ray_hits = [(spawn_x, spawn_y, 1.0); NUM_RAYS];
    }
}

pub struct Simulator {
    pub rng: Rng,
    pub track: Track,
    pub agents: Vec<Agent>,
    pub generation: u32,
    pub gen_ticks: u32,
    pub best_fitness: f32,
    pub all_time_best_fitness: f32,
    pub avg_fitness: f32,
    pub alive_count: u32,
    pub best_agent_idx: usize,
    pub mutation_rate: f32,
    pub mutation_sigma: f32,
    // Flat telemetry buffers read directly from JS over WASM linear memory.
    pub agent_render_buf: Vec<f32>,
    pub best_rays_buf: [f32; NUM_RAYS * RAY_STRIDE],
    pub best_brain_buf: [f32; BRAIN_SNAPSHOT_LEN],
    pub walls_buf: Vec<f32>,
    pub checkpoints_buf: Vec<f32>,
    pub history_buf: Vec<f32>, // [best, avg] pairs
}

impl Simulator {
    pub fn new(pop_size: usize, seed: u64, preset_id: u32) -> Self {
        let count = pop_size.clamp(20, 1000);
        let mut rng = Rng::new(seed);
        let track = Track::from_preset(preset_id);

        let mut agents = Vec::with_capacity(count);
        for _ in 0..count {
            let brain = NeuralNet::random(&mut rng);
            agents.push(Agent::new(
                track.spawn_x,
                track.spawn_y,
                track.spawn_angle,
                brain,
            ));
        }

        let mut sim = Self {
            rng,
            track,
            agents,
            generation: 1,
            gen_ticks: 0,
            best_fitness: 0.0,
            all_time_best_fitness: 0.0,
            avg_fitness: 0.0,
            alive_count: count as u32,
            best_agent_idx: 0,
            mutation_rate: 0.12,
            mutation_sigma: 0.25,
            agent_render_buf: vec![0.0; count * AGENT_STRIDE],
            best_rays_buf: [0.0; NUM_RAYS * RAY_STRIDE],
            best_brain_buf: [0.0; BRAIN_SNAPSHOT_LEN],
            walls_buf: Vec::new(),
            checkpoints_buf: Vec::new(),
            history_buf: Vec::with_capacity(HISTORY_MAX * 2),
        };

        sim.rebuild_track_buffers();
        sim.sense_all();
        sim.sync_telemetry();
        sim
    }

    pub fn set_preset(&mut self, preset_id: u32) {
        self.track = Track::from_preset(preset_id);
        self.rebuild_track_buffers();
        self.reset_population();
    }

    pub fn reset_population(&mut self) {
        for agent in &mut self.agents {
            agent.brain = NeuralNet::random(&mut self.rng);
            agent.reset(
                self.track.spawn_x,
                self.track.spawn_y,
                self.track.spawn_angle,
            );
        }
        self.generation = 1;
        self.gen_ticks = 0;
        self.best_fitness = 0.0;
        self.all_time_best_fitness = 0.0;
        self.avg_fitness = 0.0;
        self.alive_count = self.agents.len() as u32;
        self.best_agent_idx = 0;
        self.history_buf.clear();
        self.sense_all();
        self.sync_telemetry();
    }

    pub fn add_wall(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        self.track.add_wall(x1, y1, x2, y2);
        self.rebuild_track_buffers();
    }

    pub fn erase_walls_near(&mut self, x: f32, y: f32, radius: f32) {
        if self.track.erase_walls_near(x, y, radius) > 0 {
            self.rebuild_track_buffers();
        }
    }

    pub fn clear_custom_walls(&mut self) {
        self.track.clear_custom_walls();
        self.rebuild_track_buffers();
    }

    pub fn mutate_population(&mut self, sigma: f32) {
        let best_idx = self.best_agent_idx;
        for (idx, agent) in self.agents.iter_mut().enumerate() {
            if idx != best_idx {
                agent.brain.mutate(&mut self.rng, 0.35, sigma);
            }
        }
        self.sync_telemetry();
    }

    pub fn step_many(&mut self, steps: u32) {
        for _ in 0..steps {
            self.step_one_tick();
        }
        self.sync_telemetry();
    }

    pub fn step_one_tick(&mut self) {
        if self.alive_count == 0 || self.gen_ticks >= MAX_GEN_TICKS {
            self.evolve_next_generation();
            return;
        }

        self.gen_ticks += 1;
        let cp_count = self.track.checkpoints.len();
        let walls = &self.track.walls;
        let checkpoints = &self.track.checkpoints;
        let r_sq = AGENT_RADIUS * AGENT_RADIUS;

        let mut alive = 0_u32;
        let mut best_idx = 0_usize;
        let mut best_fit = -1.0_f32;
        let mut sum_fit = 0.0_f32;

        for (idx, agent) in self.agents.iter_mut().enumerate() {
            if agent.alive {
                // 1. Raycast sensors + NN forward pass.
                let mut inputs = [0.0_f32; INPUTS];
                for (r_idx, &offset) in RAY_ANGLES.iter().enumerate() {
                    let ray_ang = agent.angle + offset;
                    let dx = ray_ang.cos();
                    let dy = ray_ang.sin();
                    let dist = ray_cast(agent.x, agent.y, dx, dy, RAY_MAX_DIST, walls);
                    let norm = (dist / RAY_MAX_DIST).clamp(0.0, 1.0);
                    inputs[r_idx] = 1.0 - norm; // proximity signal: 1 = imminent wall, 0 = clear
                    agent.ray_hits[r_idx] = (agent.x + dx * dist, agent.y + dy * dist, norm);
                }
                inputs[NUM_RAYS] = (agent.speed / MAX_SPEED).clamp(0.0, 1.0);

                let out = agent.brain.forward(&inputs);
                let steer = out[0];
                let throttle = (out[1] + 1.0) * 0.5; // [0.0, 1.0]

                // 2. Kinematics.
                let target_speed = MIN_SPEED + throttle * MAX_SPEED;
                agent.speed = agent.speed * 0.8 + target_speed * 0.2;
                let turn_scale = 0.35 + 0.65 * (agent.speed / MAX_SPEED);
                agent.angle += steer * MAX_TURN_RATE * turn_scale;

                let prev_x = agent.x;
                let prev_y = agent.y;
                agent.x += agent.angle.cos() * agent.speed;
                agent.y += agent.angle.sin() * agent.speed;

                let dx = agent.x - prev_x;
                let dy = agent.y - prev_y;
                let step_dist = (dx * dx + dy * dy).sqrt();
                agent.distance_traveled += step_dist;
                agent.ticks_since_progress += 1;

                // 3. Wall collision.
                let mut crashed = false;
                for seg in walls {
                    if seg.dist_sq_to_point(agent.x, agent.y) <= r_sq {
                        crashed = true;
                        break;
                    }
                }

                if crashed {
                    agent.alive = false;
                } else if cp_count > 0 {
                    // 4. Checkpoint progression.
                    let move_seg = Segment::new(prev_x, prev_y, agent.x, agent.y);
                    let gate = checkpoints[agent.next_checkpoint];
                    if segments_intersect(move_seg, gate) {
                        agent.checkpoints_passed += 1;
                        agent.next_checkpoint = (agent.next_checkpoint + 1) % cp_count;
                        agent.ticks_since_progress = 0;
                    }
                }

                // 5. Stall culling so stationary gen-0 agents end quickly.
                if agent.alive
                    && ((self.gen_ticks > STALL_GRACE_TICKS && agent.distance_traveled < 22.0)
                        || agent.ticks_since_progress > 260)
                {
                    agent.alive = false;
                }

                // 6. Fitness score: checkpoints + proximity to next gate + distance bonus.
                let gate_bonus = if cp_count > 0 {
                    let (gx, gy) = checkpoints[agent.next_checkpoint].midpoint();
                    let dist_to_gate = ((gx - agent.x).powi(2) + (gy - agent.y).powi(2)).sqrt();
                    (120.0 - dist_to_gate * 0.2).max(0.0)
                } else {
                    0.0
                };
                agent.fitness = (agent.checkpoints_passed as f32) * 200.0
                    + gate_bonus
                    + agent.distance_traveled * 0.08;
            }

            if agent.alive {
                alive += 1;
            }
            sum_fit += agent.fitness;
            if agent.fitness > best_fit {
                best_fit = agent.fitness;
                best_idx = idx;
            }
        }

        // Prefer alive top-scoring agent for live sensor/NN inspection.
        let mut best_alive_idx = best_idx;
        let mut best_alive_fit = -1.0_f32;
        for (idx, agent) in self.agents.iter().enumerate() {
            if agent.alive && agent.fitness > best_alive_fit {
                best_alive_fit = agent.fitness;
                best_alive_idx = idx;
            }
        }

        self.alive_count = alive;
        self.best_fitness = best_fit.max(0.0);
        if self.best_fitness > self.all_time_best_fitness {
            self.all_time_best_fitness = self.best_fitness;
        }
        self.avg_fitness = sum_fit / (self.agents.len() as f32);
        self.best_agent_idx = if alive > 0 { best_alive_idx } else { best_idx };

        if self.alive_count == 0 || self.gen_ticks >= MAX_GEN_TICKS {
            self.evolve_next_generation();
        }
    }

    pub fn evolve_next_generation(&mut self) {
        let count = self.agents.len();
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&a, &b| {
            self.agents[b]
                .fitness
                .total_cmp(&self.agents[a].fitness)
                .then_with(|| a.cmp(&b))
        });

        let best_fit = self.agents[order[0]].fitness;
        let sum_fit: f32 = self.agents.iter().map(|a| a.fitness).sum();
        let avg_fit = sum_fit / (count as f32);

        if best_fit > self.all_time_best_fitness {
            self.all_time_best_fitness = best_fit;
        }
        self.best_fitness = best_fit;
        self.avg_fitness = avg_fit;

        if self.history_buf.len() >= HISTORY_MAX * 2 {
            self.history_buf.drain(0..2);
        }
        self.history_buf.push(best_fit);
        self.history_buf.push(avg_fit);

        let elite_count = (count / 16).clamp(4, 14);
        let pool_limit = (count / 2).max(elite_count + 2).min(count);
        let mut next_brains = Vec::with_capacity(count);

        for &idx in order.iter().take(elite_count) {
            next_brains.push(self.agents[idx].brain.clone());
        }

        // Keep top half as breeding pool with tournament selection.
        while next_brains.len() < count {
            let p1 = self.tournament_pick(&order[..pool_limit]);
            let p2 = self.tournament_pick(&order[..pool_limit]);
            let mut child = if self.rng.next_f32() < 0.75 {
                NeuralNet::crossover(
                    &self.agents[p1].brain,
                    &self.agents[p2].brain,
                    &mut self.rng,
                )
            } else {
                self.agents[p1].brain.clone()
            };
            child.mutate(&mut self.rng, self.mutation_rate, self.mutation_sigma);
            next_brains.push(child);
        }

        for (agent, brain) in self.agents.iter_mut().zip(next_brains.into_iter()) {
            agent.brain = brain;
            agent.reset(
                self.track.spawn_x,
                self.track.spawn_y,
                self.track.spawn_angle,
            );
        }

        self.generation += 1;
        self.gen_ticks = 0;
        self.alive_count = count as u32;
        self.best_agent_idx = 0;
        self.sense_all();
        self.sync_telemetry();
    }

    fn tournament_pick(&mut self, pool: &[usize]) -> usize {
        let mut best = pool[self.rng.next_usize(pool.len())];
        let mut best_fit = self.agents[best].fitness;
        for _ in 0..3 {
            let cand = pool[self.rng.next_usize(pool.len())];
            if self.agents[cand].fitness > best_fit {
                best = cand;
                best_fit = self.agents[cand].fitness;
            }
        }
        best
    }

    fn sense_all(&mut self) {
        let walls = &self.track.walls;
        for agent in &mut self.agents {
            let mut inputs = [0.0_f32; INPUTS];
            for (r_idx, &offset) in RAY_ANGLES.iter().enumerate() {
                let ray_ang = agent.angle + offset;
                let dx = ray_ang.cos();
                let dy = ray_ang.sin();
                let dist = ray_cast(agent.x, agent.y, dx, dy, RAY_MAX_DIST, walls);
                let norm = (dist / RAY_MAX_DIST).clamp(0.0, 1.0);
                inputs[r_idx] = 1.0 - norm;
                agent.ray_hits[r_idx] = (agent.x + dx * dist, agent.y + dy * dist, norm);
            }
            inputs[NUM_RAYS] = (agent.speed / MAX_SPEED).clamp(0.0, 1.0);
            agent.brain.forward(&inputs);
        }
    }

    fn rebuild_track_buffers(&mut self) {
        self.walls_buf.clear();
        self.walls_buf.reserve(self.track.walls.len() * 4);
        for seg in &self.track.walls {
            self.walls_buf.push(seg.x1);
            self.walls_buf.push(seg.y1);
            self.walls_buf.push(seg.x2);
            self.walls_buf.push(seg.y2);
        }

        self.checkpoints_buf.clear();
        self.checkpoints_buf
            .reserve(self.track.checkpoints.len() * 4);
        for cp in &self.track.checkpoints {
            self.checkpoints_buf.push(cp.x1);
            self.checkpoints_buf.push(cp.y1);
            self.checkpoints_buf.push(cp.x2);
            self.checkpoints_buf.push(cp.y2);
        }
    }

    fn sync_telemetry(&mut self) {
        let best_idx = self.best_agent_idx.min(self.agents.len() - 1);
        for (i, agent) in self.agents.iter().enumerate() {
            let base = i * AGENT_STRIDE;
            self.agent_render_buf[base] = agent.x;
            self.agent_render_buf[base + 1] = agent.y;
            self.agent_render_buf[base + 2] = agent.angle;
            self.agent_render_buf[base + 3] = if agent.alive { 1.0 } else { 0.0 };
            self.agent_render_buf[base + 4] = if i == best_idx { 1.0 } else { 0.0 };
            self.agent_render_buf[base + 5] = agent.fitness;
        }

        let best = &self.agents[best_idx];
        for (i, &(hx, hy, norm)) in best.ray_hits.iter().enumerate() {
            let base = i * RAY_STRIDE;
            self.best_rays_buf[base] = hx;
            self.best_rays_buf[base + 1] = hy;
            self.best_rays_buf[base + 2] = norm;
        }
        best.brain.write_snapshot(&mut self.best_brain_buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn population_defaults_to_200_plus_and_evolution_improves_fitness() {
        let mut sim = Simulator::new(220, 2026, 0);
        assert!(sim.agents.len() >= 200);

        // Run generation 1 to completion.
        while sim.generation == 1 {
            sim.step_one_tick();
        }
        assert_eq!(sim.history_buf.len(), 2);
        let gen1_best = sim.history_buf[0];

        // Evolve across 10 more generations and track peak generation best fitness.
        let mut peak_later = 0.0_f32;
        while sim.generation <= 10 {
            sim.step_one_tick();
            if sim.history_buf.len() >= 4 {
                let latest = sim.history_buf[sim.history_buf.len() - 2];
                if latest > peak_later {
                    peak_later = latest;
                }
            }
        }

        assert!(
            peak_later > gen1_best,
            "expected neuroevolution to improve fitness: gen1={gen1_best}, peak_later={peak_later}"
        );
    }

    #[test]
    fn elites_are_preserved_across_generations() {
        let mut sim = Simulator::new(60, 7, 0);
        sim.agents[12].fitness = 9999.0;
        let elite_weights = sim.agents[12].brain.weights;

        sim.evolve_next_generation();
        assert_eq!(sim.generation, 2);
        assert_eq!(sim.agents[0].brain.weights, elite_weights);
    }
}

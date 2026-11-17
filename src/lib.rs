pub mod nn;
pub mod rng;
pub mod sim;
pub mod track;

use nn::BRAIN_SNAPSHOT_LEN;
use sim::{Simulator, AGENT_STRIDE, RAY_STRIDE};
use track::{WORLD_HEIGHT, WORLD_WIDTH};

/// Allocates a new neuroevolution simulator and returns an opaque handle.
///
/// # Safety
/// The returned pointer must eventually be released via [`sim_destroy`].
#[no_mangle]
pub unsafe extern "C" fn sim_create(pop_size: u32, seed: u32, preset_id: u32) -> *mut Simulator {
    let sim = Simulator::new(pop_size as usize, u64::from(seed), preset_id);
    Box::into_raw(Box::new(sim))
}

/// Frees a simulator previously allocated with [`sim_create`].
///
/// # Safety
/// `ptr` must be null or a valid pointer returned by [`sim_create`] that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn sim_destroy(ptr: *mut Simulator) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Advances the simulation by `steps` physics ticks and updates telemetry buffers.
///
/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_step(ptr: *mut Simulator, steps: u32) {
    if let Some(sim) = ptr.as_mut() {
        sim.step_many(steps);
    }
}

/// Switches the active preset track and resets the population.
///
/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_set_preset(ptr: *mut Simulator, preset_id: u32) {
    if let Some(sim) = ptr.as_mut() {
        sim.set_preset(preset_id);
    }
}

/// Resets the population with fresh random weights on the current track.
///
/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_reset(ptr: *mut Simulator) {
    if let Some(sim) = ptr.as_mut() {
        sim.reset_population();
    }
}

/// Immediately forces evolution to the next generation.
///
/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_next_generation(ptr: *mut Simulator) {
    if let Some(sim) = ptr.as_mut() {
        sim.evolve_next_generation();
    }
}

/// Applies an immediate mutation burst to non-elite agents.
///
/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_mutate_burst(ptr: *mut Simulator, sigma: f32) {
    if let Some(sim) = ptr.as_mut() {
        sim.mutate_population(sigma);
    }
}

/// Sets mutation hyperparameters `(rate, sigma)`.
///
/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_set_mutation(ptr: *mut Simulator, rate: f32, sigma: f32) {
    if let Some(sim) = ptr.as_mut() {
        sim.mutation_rate = rate.clamp(0.01, 0.9);
        sim.mutation_sigma = sigma.clamp(0.02, 2.0);
    }
}

/// Adds a user-drawn wall segment to the active track.
///
/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_add_wall(ptr: *mut Simulator, x1: f32, y1: f32, x2: f32, y2: f32) {
    if let Some(sim) = ptr.as_mut() {
        sim.add_wall(x1, y1, x2, y2);
    }
}

/// Erases non-boundary walls within `radius` of `(x, y)`.
///
/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_erase_walls(ptr: *mut Simulator, x: f32, y: f32, radius: f32) {
    if let Some(sim) = ptr.as_mut() {
        sim.erase_walls_near(x, y, radius);
    }
}

#[no_mangle]
pub extern "C" fn sim_world_width() -> f32 {
    WORLD_WIDTH
}

#[no_mangle]
pub extern "C" fn sim_world_height() -> f32 {
    WORLD_HEIGHT
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_generation(ptr: *const Simulator) -> u32 {
    ptr.as_ref().map_or(0, |s| s.generation)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_gen_ticks(ptr: *const Simulator) -> u32 {
    ptr.as_ref().map_or(0, |s| s.gen_ticks)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_alive_count(ptr: *const Simulator) -> u32 {
    ptr.as_ref().map_or(0, |s| s.alive_count)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_agent_count(ptr: *const Simulator) -> u32 {
    ptr.as_ref().map_or(0, |s| s.agents.len() as u32)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_best_fitness(ptr: *const Simulator) -> f32 {
    ptr.as_ref().map_or(0.0, |s| s.best_fitness)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_all_time_best(ptr: *const Simulator) -> f32 {
    ptr.as_ref().map_or(0.0, |s| s.all_time_best_fitness)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_avg_fitness(ptr: *const Simulator) -> f32 {
    ptr.as_ref().map_or(0.0, |s| s.avg_fitness)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_best_agent_next_cp(ptr: *const Simulator) -> u32 {
    ptr.as_ref()
        .map_or(0, |s| s.agents[s.best_agent_idx].next_checkpoint as u32)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_best_agent_laps(ptr: *const Simulator) -> u32 {
    ptr.as_ref().map_or(0, |s| {
        let cps = s.track.checkpoints.len().max(1) as u32;
        s.agents[s.best_agent_idx].checkpoints_passed / cps
    })
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_agents_ptr(ptr: *const Simulator) -> *const f32 {
    ptr.as_ref()
        .map_or(std::ptr::null(), |s| s.agent_render_buf.as_ptr())
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_agents_len(ptr: *const Simulator) -> u32 {
    ptr.as_ref()
        .map_or(0, |s| (s.agents.len() * AGENT_STRIDE) as u32)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_best_rays_ptr(ptr: *const Simulator) -> *const f32 {
    ptr.as_ref()
        .map_or(std::ptr::null(), |s| s.best_rays_buf.as_ptr())
}

#[no_mangle]
pub extern "C" fn sim_best_rays_len() -> u32 {
    (nn::NUM_RAYS * RAY_STRIDE) as u32
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_best_brain_ptr(ptr: *const Simulator) -> *const f32 {
    ptr.as_ref()
        .map_or(std::ptr::null(), |s| s.best_brain_buf.as_ptr())
}

#[no_mangle]
pub extern "C" fn sim_best_brain_len() -> u32 {
    BRAIN_SNAPSHOT_LEN as u32
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_walls_ptr(ptr: *const Simulator) -> *const f32 {
    ptr.as_ref()
        .map_or(std::ptr::null(), |s| s.walls_buf.as_ptr())
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_walls_len(ptr: *const Simulator) -> u32 {
    ptr.as_ref().map_or(0, |s| s.walls_buf.len() as u32)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_checkpoints_ptr(ptr: *const Simulator) -> *const f32 {
    ptr.as_ref()
        .map_or(std::ptr::null(), |s| s.checkpoints_buf.as_ptr())
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_checkpoints_len(ptr: *const Simulator) -> u32 {
    ptr.as_ref().map_or(0, |s| s.checkpoints_buf.len() as u32)
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_history_ptr(ptr: *const Simulator) -> *const f32 {
    ptr.as_ref()
        .map_or(std::ptr::null(), |s| s.history_buf.as_ptr())
}

/// # Safety
/// `ptr` must be a valid non-null pointer from [`sim_create`].
#[no_mangle]
pub unsafe extern "C" fn sim_history_len(ptr: *const Simulator) -> u32 {
    ptr.as_ref().map_or(0, |s| s.history_buf.len() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_lifecycle_and_buffers_work() {
        unsafe {
            let sim = sim_create(220, 42, 0);
            assert!(!sim.is_null());
            assert_eq!(sim_agent_count(sim), 220);
            assert_eq!(sim_generation(sim), 1);
            assert_eq!(sim_best_brain_len(), BRAIN_SNAPSHOT_LEN as u32);
            assert_eq!(sim_best_rays_len(), (nn::NUM_RAYS * RAY_STRIDE) as u32);

            sim_step(sim, 30);
            assert!(sim_gen_ticks(sim) > 0);
            assert!(!sim_agents_ptr(sim).is_null());
            assert_eq!(sim_agents_len(sim), 220 * AGENT_STRIDE as u32);

            let walls_before = sim_walls_len(sim);
            sim_add_wall(sim, 400.0, 300.0, 480.0, 320.0);
            assert_eq!(sim_walls_len(sim), walls_before + 4);
            sim_erase_walls(sim, 440.0, 310.0, 20.0);
            assert_eq!(sim_walls_len(sim), walls_before);

            sim_next_generation(sim);
            assert_eq!(sim_generation(sim), 2);
            assert_eq!(sim_history_len(sim), 2);

            sim_destroy(sim);
        }
    }
}

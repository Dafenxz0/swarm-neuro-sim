# Swarm Neuro-Sim

[![CI & Deploy Pages](https://github.com/Dafenxz0/swarm-neuro-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/Dafenxz0/swarm-neuro-sim/actions/workflows/ci.yml)

**Live Demo:** https://dafenxz0.github.io/swarm-neuro-sim/

A real-time neuroevolution swarm simulator compiled from pure Rust to WebAssembly (`wasm32-unknown-unknown`) with zero runtime crate or npm dependencies, paired with a dark HTML5 Canvas telemetry lab UI.

## Features

- **220 Autonomous Agents in Real Time:** Each agent senses walls with 7 forward raycasts plus speed (`8 -> 10 -> 2` `tanh` MLP) and controls steering and throttle at 60+ FPS.
- **Pure-Rust Neuroevolution Engine:** Elitism (top ~6%), tournament selection, crossover, and Gaussian weight mutation (`src/nn.rs`, `src/sim.rs`).
- **Zero-Allocation WASM Linear Memory Bridge:** Rust updates flat `f32` telemetry buffers in place (`#[no_mangle] pub unsafe extern "C"`), which the browser reads directly via `Float32Array` views over `wasm.memory.buffer`.
- **Interactive Telemetry Lab (`web/`):**
  - Live neural network inspector showing active weights and neuron activations for the best agent
  - Live best & mean fitness evolution chart
  - Draw Walls / Erase Walls mouse tools (right-click drag erases anytime)
  - Simulation speed multiplier (`1x`, `5x`, `20x`)
  - 4 preset tracks (`Circuit Oval`, `Technical S-Curves`, `Spiral Gauntlet`, `Open Slalom`)

## Controls

| Input | Action |
| --- | --- |
| `Space` | Pause / Resume |
| `1` / `2` / `3` | Set speed `1x` / `5x` / `20x` |
| `D` / `E` | Switch to Draw Wall / Erase Wall tool |
| Left-drag (Draw mode) | Draw custom wall segments |
| Right-drag (or Erase mode) | Erase nearby custom/track walls |
| `R` | Reset population |

## Architecture

```text
src/
├── rng.rs    # Deterministic SplitMix64 PRNG + Box-Muller Gaussian sampler
├── nn.rs     # 8→10→2 feedforward MLP, crossover, Gaussian mutation, snapshot
├── track.rs  # 4 preset tracks, ray-segment intersection, collision & gates
├── sim.rs    # 220-agent physics loop, fitness scoring, elitism & evolution
└── lib.rs    # Zero-dependency WASM C-ABI exports & shared telemetry pointers
web/
├── index.html
├── style.css
└── app.js    # WebAssembly loader, Canvas 2D renderer, NN & fitness visualizers
```

## Local Build & Test

```bash
rustup target add wasm32-unknown-unknown
cargo test
cargo build --target wasm32-unknown-unknown --release
cp target/wasm32-unknown-unknown/release/swarm_neuro_sim.wasm web/swarm_neuro_sim.wasm
python3 -m http.server --directory web 8080
```

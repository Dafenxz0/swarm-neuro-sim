const POP_SIZE = 220;
const AGENT_STRIDE = 6;
const NUM_RAYS = 7;
const RAY_STRIDE = 3;
const INPUTS = 8;
const HIDDEN = 10;
const OUTPUTS = 2;
const WEIGHTS_IH_LEN = INPUTS * HIDDEN;
const BIAS_H_LEN = HIDDEN;
const WEIGHTS_HO_LEN = HIDDEN * OUTPUTS;
const MAX_GEN_TICKS = 1600;

const INPUT_LABELS = ["R-72°", "R-45°", "R-21°", "R0°", "R+21°", "R+45°", "R+72°", "SPD"];
const OUTPUT_LABELS = ["STEER", "THROT"];

async function loadWasm() {
  const url = "./swarm_neuro_sim.wasm";
  if (WebAssembly.instantiateStreaming) {
    try {
      const { instance } = await WebAssembly.instantiateStreaming(fetch(url), {});
      return instance.exports;
    } catch {
      // Fallback if server MIME type is not application/wasm.
    }
  }
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${resp.status}`);
  }
  const bytes = await resp.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

function showBanner(msg) {
  const banner = document.getElementById("status-banner");
  banner.textContent = msg;
  banner.classList.remove("hidden");
}

function main() {
  const arenaCanvas = document.getElementById("arena-canvas");
  const arenaCtx = arenaCanvas.getContext("2d");
  const nnCanvas = document.getElementById("nn-canvas");
  const nnCtx = nnCanvas.getContext("2d");
  const histCanvas = document.getElementById("history-canvas");
  const histCtx = histCanvas.getContext("2d");

  const elGen = document.getElementById("stat-gen");
  const elAlive = document.getElementById("stat-alive");
  const elBest = document.getElementById("stat-best");
  const elPeak = document.getElementById("stat-peak");
  const elAvg = document.getElementById("stat-avg");
  const elLaps = document.getElementById("stat-laps");
  const elTicks = document.getElementById("stat-ticks");
  const elTickFill = document.getElementById("tick-fill");
  const elNnOutputs = document.getElementById("nn-outputs");
  const elHistCount = document.getElementById("history-count");

  const btnPause = document.getElementById("btn-pause");
  const btnNextGen = document.getElementById("btn-next-gen");
  const btnMutate = document.getElementById("btn-mutate");
  const btnReset = document.getElementById("btn-reset");
  const selectPreset = document.getElementById("preset-select");
  const chkCheckpoints = document.getElementById("chk-checkpoints");
  const chkRays = document.getElementById("chk-rays");
  const sliderMutRate = document.getElementById("slider-mut-rate");
  const sliderMutSigma = document.getElementById("slider-mut-sigma");
  const valMutRate = document.getElementById("val-mut-rate");
  const valMutSigma = document.getElementById("val-mut-sigma");

  let wasm = null;
  let simPtr = 0;
  let worldW = 1000;
  let worldH = 700;
  let paused = false;
  let speed = 1;
  let activeTool = "observe";

  // Viewport transform from world space -> canvas CSS pixels
  const view = { scale: 1, offsetX: 0, offsetY: 0, width: 1, height: 1 };

  // Drawing / erasing state
  let pointerDown = false;
  let pointerButton = 0;
  let lastWorldPt = null;
  let curWorldPt = null;

  function resizeArena() {
    const rect = arenaCanvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    arenaCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
    arenaCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
    arenaCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const pad = 18;
    const availW = Math.max(100, rect.width - pad * 2);
    const availH = Math.max(100, rect.height - pad * 2);
    const scale = Math.min(availW / worldW, availH / worldH);

    view.width = rect.width;
    view.height = rect.height;
    view.scale = scale;
    view.offsetX = (rect.width - worldW * scale) * 0.5;
    view.offsetY = (rect.height - worldH * scale) * 0.5;
  }

  function clientToWorld(clientX, clientY) {
    const rect = arenaCanvas.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;
    return {
      x: (cx - view.offsetX) / view.scale,
      y: (cy - view.offsetY) / view.scale,
    };
  }

  function updateCursor() {
    if (activeTool === "observe") {
      arenaCanvas.style.cursor = "default";
    } else if (activeTool === "draw") {
      arenaCanvas.style.cursor = "crosshair";
    } else {
      arenaCanvas.style.cursor = "cell";
    }
  }

  document.querySelectorAll("[data-speed]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-speed]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      speed = Number(btn.dataset.speed) || 1;
    });
  });

  document.querySelectorAll("[data-tool]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-tool]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTool = btn.dataset.tool || "observe";
      updateCursor();
    });
  });

  btnPause.addEventListener("click", () => {
    paused = !paused;
    btnPause.textContent = paused ? "RESUME" : "PAUSE";
  });

  btnNextGen.addEventListener("click", () => {
    if (simPtr) wasm.sim_next_generation(simPtr);
  });

  btnMutate.addEventListener("click", () => {
    if (simPtr) wasm.sim_mutate_burst(simPtr, 0.45);
  });

  btnReset.addEventListener("click", () => {
    if (simPtr) wasm.sim_reset(simPtr);
  });

  selectPreset.addEventListener("change", () => {
    if (simPtr) wasm.sim_set_preset(simPtr, Number(selectPreset.value));
  });

  function syncMutationSliders() {
    const rate = Number(sliderMutRate.value) / 100;
    const sigma = Number(sliderMutSigma.value) / 100;
    valMutRate.textContent = `${sliderMutRate.value}%`;
    valMutSigma.textContent = sigma.toFixed(2);
    if (simPtr) wasm.sim_set_mutation(simPtr, rate, sigma);
  }
  sliderMutRate.addEventListener("input", syncMutationSliders);
  sliderMutSigma.addEventListener("input", syncMutationSliders);

  window.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;
    if (e.code === "Space") {
      e.preventDefault();
      btnPause.click();
    } else if (e.key === "1") {
      document.querySelector('[data-speed="1"]').click();
    } else if (e.key === "2") {
      document.querySelector('[data-speed="5"]').click();
    } else if (e.key === "3") {
      document.querySelector('[data-speed="20"]').click();
    } else if (e.key === "r" || e.key === "R") {
      btnReset.click();
    } else if (e.key === "d" || e.key === "D") {
      document.querySelector('[data-tool="draw"]').click();
    } else if (e.key === "e" || e.key === "E") {
      document.querySelector('[data-tool="erase"]').click();
    }
  });

  arenaCanvas.addEventListener("contextmenu", (e) => e.preventDefault());

  arenaCanvas.addEventListener("pointerdown", (e) => {
    if (!simPtr) return;
    pointerDown = true;
    pointerButton = e.button;
    const pt = clientToWorld(e.clientX, e.clientY);
    lastWorldPt = pt;
    curWorldPt = pt;
    arenaCanvas.setPointerCapture(e.pointerId);

    if (pointerButton === 2 || activeTool === "erase") {
      wasm.sim_erase_walls(simPtr, pt.x, pt.y, 18.0);
    }
  });

  arenaCanvas.addEventListener("pointermove", (e) => {
    const pt = clientToWorld(e.clientX, e.clientY);
    curWorldPt = pt;
    if (!pointerDown || !simPtr || !lastWorldPt) return;

    if (pointerButton === 2 || activeTool === "erase") {
      wasm.sim_erase_walls(simPtr, pt.x, pt.y, 18.0);
      lastWorldPt = pt;
    } else if (activeTool === "draw") {
      const dx = pt.x - lastWorldPt.x;
      const dy = pt.y - lastWorldPt.y;
      if (dx * dx + dy * dy >= 144) {
        wasm.sim_add_wall(simPtr, lastWorldPt.x, lastWorldPt.y, pt.x, pt.y);
        lastWorldPt = pt;
      }
    }
  });

  arenaCanvas.addEventListener("pointerup", (e) => {
    if (pointerDown && simPtr && lastWorldPt && curWorldPt) {
      if (pointerButton !== 2 && activeTool === "draw") {
        const dx = curWorldPt.x - lastWorldPt.x;
        const dy = curWorldPt.y - lastWorldPt.y;
        if (dx * dx + dy * dy >= 16) {
          wasm.sim_add_wall(simPtr, lastWorldPt.x, lastWorldPt.y, curWorldPt.x, curWorldPt.y);
        }
      }
    }
    pointerDown = false;
    lastWorldPt = null;
  });

  function drawArena(f32) {
    arenaCtx.clearRect(0, 0, view.width, view.height);

    arenaCtx.save();
    arenaCtx.translate(view.offsetX, view.offsetY);
    arenaCtx.scale(view.scale, view.scale);

    // Subtle telemetry grid
    arenaCtx.strokeStyle = "rgba(30, 39, 52, 0.45)";
    arenaCtx.lineWidth = 1 / view.scale;
    arenaCtx.beginPath();
    for (let x = 0; x <= worldW; x += 50) {
      arenaCtx.moveTo(x, 0);
      arenaCtx.lineTo(x, worldH);
    }
    for (let y = 0; y <= worldH; y += 50) {
      arenaCtx.moveTo(0, y);
      arenaCtx.lineTo(worldW, y);
    }
    arenaCtx.stroke();

    // Checkpoints
    if (chkCheckpoints.checked) {
      const cpPtr = wasm.sim_checkpoints_ptr(simPtr) >>> 2;
      const cpLen = wasm.sim_checkpoints_len(simPtr);
      const nextCp = wasm.sim_best_agent_next_cp(simPtr);
      arenaCtx.lineWidth = 2;
      arenaCtx.setLineDash([5, 5]);
      for (let i = 0; i < cpLen; i += 4) {
        const idx = i >> 2;
        arenaCtx.strokeStyle =
          idx === nextCp ? "rgba(0, 229, 163, 0.72)" : "rgba(56, 189, 248, 0.22)";
        arenaCtx.beginPath();
        arenaCtx.moveTo(f32[cpPtr + i], f32[cpPtr + i + 1]);
        arenaCtx.lineTo(f32[cpPtr + i + 2], f32[cpPtr + i + 3]);
        arenaCtx.stroke();
      }
      arenaCtx.setLineDash([]);
    }

    // Track & custom walls
    const wPtr = wasm.sim_walls_ptr(simPtr) >>> 2;
    const wLen = wasm.sim_walls_len(simPtr);
    arenaCtx.strokeStyle = "#cbd5e1";
    arenaCtx.lineWidth = 3;
    arenaCtx.lineCap = "round";
    arenaCtx.beginPath();
    for (let i = 0; i < wLen; i += 4) {
      arenaCtx.moveTo(f32[wPtr + i], f32[wPtr + i + 1]);
      arenaCtx.lineTo(f32[wPtr + i + 2], f32[wPtr + i + 3]);
    }
    arenaCtx.stroke();

    // Agents
    const aPtr = wasm.sim_agents_ptr(simPtr) >>> 2;
    const aLen = wasm.sim_agents_len(simPtr);
    let bestAgent = null;

    for (let i = 0; i < aLen; i += AGENT_STRIDE) {
      const x = f32[aPtr + i];
      const y = f32[aPtr + i + 1];
      const ang = f32[aPtr + i + 2];
      const alive = f32[aPtr + i + 3] > 0.5;
      const isBest = f32[aPtr + i + 4] > 0.5;

      if (isBest) {
        bestAgent = { x, y, ang, alive };
        continue;
      }

      if (!alive) {
        arenaCtx.fillStyle = "rgba(255, 77, 106, 0.22)";
        arenaCtx.fillRect(x - 2, y - 2, 4, 4);
        continue;
      }

      arenaCtx.save();
      arenaCtx.translate(x, y);
      arenaCtx.rotate(ang);
      arenaCtx.fillStyle = "rgba(56, 189, 248, 0.52)";
      arenaCtx.beginPath();
      arenaCtx.moveTo(8, 0);
      arenaCtx.lineTo(-5, -4.5);
      arenaCtx.lineTo(-3, 0);
      arenaCtx.lineTo(-5, 4.5);
      arenaCtx.closePath();
      arenaCtx.fill();
      arenaCtx.restore();
    }

    // Best agent rays + highlight
    if (bestAgent) {
      if (chkRays.checked) {
        const rPtr = wasm.sim_best_rays_ptr(simPtr) >>> 2;
        const rLen = wasm.sim_best_rays_len();
        for (let i = 0; i < rLen; i += RAY_STRIDE) {
          const hx = f32[rPtr + i];
          const hy = f32[rPtr + i + 1];
          const norm = f32[rPtr + i + 2];
          arenaCtx.strokeStyle =
            norm < 0.35 ? "rgba(255, 77, 106, 0.65)" : "rgba(0, 229, 163, 0.38)";
          arenaCtx.lineWidth = 1.4;
          arenaCtx.beginPath();
          arenaCtx.moveTo(bestAgent.x, bestAgent.y);
          arenaCtx.lineTo(hx, hy);
          arenaCtx.stroke();

          arenaCtx.fillStyle = norm < 0.35 ? "#ff4d6a" : "#00e5a3";
          arenaCtx.beginPath();
          arenaCtx.arc(hx, hy, 2.5, 0, Math.PI * 2);
          arenaCtx.fill();
        }
      }

      arenaCtx.save();
      arenaCtx.translate(bestAgent.x, bestAgent.y);
      arenaCtx.rotate(bestAgent.ang);
      arenaCtx.fillStyle = bestAgent.alive ? "#00e5a3" : "#f59e0b";
      arenaCtx.strokeStyle = "#080b0f";
      arenaCtx.lineWidth = 1.5;
      arenaCtx.beginPath();
      arenaCtx.moveTo(11, 0);
      arenaCtx.lineTo(-7, -6);
      arenaCtx.lineTo(-4, 0);
      arenaCtx.lineTo(-7, 6);
      arenaCtx.closePath();
      arenaCtx.fill();
      arenaCtx.stroke();
      arenaCtx.restore();
    }

    // Erase brush cursor
    if (curWorldPt && (activeTool === "erase" || (pointerDown && pointerButton === 2))) {
      arenaCtx.strokeStyle = "rgba(255, 77, 106, 0.8)";
      arenaCtx.lineWidth = 1.5;
      arenaCtx.beginPath();
      arenaCtx.arc(curWorldPt.x, curWorldPt.y, 18, 0, Math.PI * 2);
      arenaCtx.stroke();
    }

    arenaCtx.restore();
  }

  function drawNeuralNet(f32) {
    const w = nnCanvas.width;
    const h = nnCanvas.height;
    nnCtx.clearRect(0, 0, w, h);

    const bPtr = wasm.sim_best_brain_ptr(simPtr) >>> 2;
    const inOff = bPtr;
    const hidOff = inOff + INPUTS;
    const outOff = hidOff + HIDDEN;
    const wihOff = outOff + OUTPUTS;
    const whoOff = wihOff + WEIGHTS_IH_LEN + BIAS_H_LEN;

    const layerX = [56, 168, 266];
    const makeY = (n, pad) =>
      Array.from({ length: n }, (_, i) =>
        n === 1 ? h * 0.5 : pad + ((h - pad * 2) * i) / (n - 1)
      );

    const inY = makeY(INPUTS, 20);
    const hidY = makeY(HIDDEN, 16);
    const outY = makeY(OUTPUTS, 75);

    // Input -> Hidden weights
    for (let hi = 0; hi < HIDDEN; hi++) {
      for (let ii = 0; ii < INPUTS; ii++) {
        const wt = f32[wihOff + hi * INPUTS + ii];
        const mag = Math.min(1, Math.abs(wt) / 2.5);
        if (mag < 0.06) continue;
        nnCtx.strokeStyle =
          wt >= 0
            ? `rgba(0, 229, 163, ${0.12 + mag * 0.65})`
            : `rgba(255, 77, 106, ${0.12 + mag * 0.65})`;
        nnCtx.lineWidth = 0.6 + mag * 1.8;
        nnCtx.beginPath();
        nnCtx.moveTo(layerX[0], inY[ii]);
        nnCtx.lineTo(layerX[1], hidY[hi]);
        nnCtx.stroke();
      }
    }

    // Hidden -> Output weights
    for (let oi = 0; oi < OUTPUTS; oi++) {
      for (let hi = 0; hi < HIDDEN; hi++) {
        const wt = f32[whoOff + oi * HIDDEN + hi];
        const mag = Math.min(1, Math.abs(wt) / 2.5);
        if (mag < 0.06) continue;
        nnCtx.strokeStyle =
          wt >= 0
            ? `rgba(0, 229, 163, ${0.15 + mag * 0.72})`
            : `rgba(255, 77, 106, ${0.15 + mag * 0.72})`;
        nnCtx.lineWidth = 0.8 + mag * 2.1;
        nnCtx.beginPath();
        nnCtx.moveTo(layerX[1], hidY[hi]);
        nnCtx.lineTo(layerX[2], outY[oi]);
        nnCtx.stroke();
      }
    }

    const drawNode = (x, y, act, r, label, alignRight) => {
      const clamped = Math.max(-1, Math.min(1, act));
      const intensity = Math.abs(clamped);
      nnCtx.fillStyle =
        clamped >= 0
          ? `rgba(0, 229, 163, ${0.18 + intensity * 0.82})`
          : `rgba(255, 77, 106, ${0.18 + intensity * 0.82})`;
      nnCtx.strokeStyle = "#94a3b8";
      nnCtx.lineWidth = 1;
      nnCtx.beginPath();
      nnCtx.arc(x, y, r, 0, Math.PI * 2);
      nnCtx.fill();
      nnCtx.stroke();

      if (label) {
        nnCtx.fillStyle = "#94a3b8";
        nnCtx.font = '10px "JetBrains Mono", monospace';
        nnCtx.textBaseline = "middle";
        nnCtx.textAlign = alignRight ? "left" : "right";
        nnCtx.fillText(label, alignRight ? x + r + 6 : x - r - 6, y);
      }
    };

    for (let i = 0; i < INPUTS; i++) {
      drawNode(layerX[0], inY[i], f32[inOff + i], 6.5, INPUT_LABELS[i], false);
    }
    for (let i = 0; i < HIDDEN; i++) {
      drawNode(layerX[1], hidY[i], f32[hidOff + i], 6.5, null, false);
    }
    for (let i = 0; i < OUTPUTS; i++) {
      drawNode(layerX[2], outY[i], f32[outOff + i], 8.5, OUTPUT_LABELS[i], true);
    }

    const steer = f32[outOff];
    const throttle = (f32[outOff + 1] + 1) * 0.5;
    elNnOutputs.textContent = `STR ${steer >= 0 ? "+" : ""}${steer.toFixed(2)} · THR ${throttle.toFixed(2)}`;
  }

  function drawHistory(f32) {
    const w = histCanvas.width;
    const h = histCanvas.height;
    histCtx.clearRect(0, 0, w, h);

    const hPtr = wasm.sim_history_ptr(simPtr) >>> 2;
    const hLen = wasm.sim_history_len(simPtr);
    const gens = hLen >> 1;
    elHistCount.textContent = `${gens} GENS`;

    histCtx.strokeStyle = "rgba(30, 39, 52, 0.8)";
    histCtx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = (h * i) / 4;
      histCtx.beginPath();
      histCtx.moveTo(0, y);
      histCtx.lineTo(w, y);
      histCtx.stroke();
    }

    if (gens < 2) {
      histCtx.fillStyle = "#64748b";
      histCtx.font = '10px "JetBrains Mono", monospace';
      histCtx.fillText("Completing generation 1...", 12, h * 0.5 + 3);
      return;
    }

    let maxFit = 10;
    for (let i = 0; i < hLen; i += 2) {
      if (f32[hPtr + i] > maxFit) maxFit = f32[hPtr + i];
    }

    const padX = 8;
    const padY = 12;
    const spanX = w - padX * 2;
    const spanY = h - padY * 2;

    const plotSeries = (offset, color, width) => {
      histCtx.strokeStyle = color;
      histCtx.lineWidth = width;
      histCtx.beginPath();
      for (let g = 0; g < gens; g++) {
        const x = padX + (spanX * g) / (gens - 1);
        const val = f32[hPtr + g * 2 + offset];
        const y = h - padY - (val / maxFit) * spanY;
        if (g === 0) histCtx.moveTo(x, y);
        else histCtx.lineTo(x, y);
      }
      histCtx.stroke();
    };

    plotSeries(1, "rgba(56, 189, 248, 0.75)", 1.5);
    plotSeries(0, "#00e5a3", 2.0);
  }

  function frame() {
    if (!paused && simPtr) {
      wasm.sim_step(simPtr, speed);
    }

    if (simPtr) {
      const f32 = new Float32Array(wasm.memory.buffer);
      drawArena(f32);
      drawNeuralNet(f32);
      drawHistory(f32);

      const gen = wasm.sim_generation(simPtr);
      const ticks = wasm.sim_gen_ticks(simPtr);
      const alive = wasm.sim_alive_count(simPtr);
      const total = wasm.sim_agent_count(simPtr);
      const best = wasm.sim_best_fitness(simPtr);
      const peak = wasm.sim_all_time_best(simPtr);
      const avg = wasm.sim_avg_fitness(simPtr);
      const laps = wasm.sim_best_agent_laps(simPtr);

      elGen.textContent = String(gen);
      elAlive.textContent = `${alive}/${total}`;
      elBest.textContent = best.toFixed(1);
      elPeak.textContent = peak.toFixed(1);
      elAvg.textContent = avg.toFixed(1);
      elLaps.textContent = String(laps);
      elTicks.textContent = `${ticks}/${MAX_GEN_TICKS}`;
      elTickFill.style.width = `${Math.min(100, (ticks / MAX_GEN_TICKS) * 100)}%`;
    }

    requestAnimationFrame(frame);
  }

  window.addEventListener("resize", resizeArena);

  loadWasm()
    .then((exports) => {
      wasm = exports;
      worldW = wasm.sim_world_width();
      worldH = wasm.sim_world_height();
      const seed = (Date.now() & 0xffff_ffff) >>> 0;
      simPtr = wasm.sim_create(POP_SIZE, seed, 0);
      syncMutationSliders();
      resizeArena();
      updateCursor();
      requestAnimationFrame(frame);
    })
    .catch((err) => {
      showBanner(
        `WASM failed to load (${err.message}). Serve this folder over HTTP (e.g. GitHub Pages or python -m http.server).`
      );
    });
}

main();

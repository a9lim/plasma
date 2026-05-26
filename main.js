/**
 * @fileoverview plasma — Phase 2 entry point.
 *
 * Initializes WebGPU, builds the Phase-2 compute graph (PLM + HLL +
 * forward Euler with dimensional splitting), runs a rAF loop with a
 * fixed-timestep accumulator that delegates each substep to
 * `sim.step()`, and pauses on tab hide via `visibilitychange`.
 *
 * Per substep we run compute_dt internally — PHYSICS_DT here is the
 * accumulator pacing, not the physics dt the sweep sees. The sim's
 * own CFL-derived dt is what actually advances the state.
 *
 * On any WebGPU failure (no adapter, no device, no nav.gpu) the
 * `#no-webgpu` landing element is unhidden and we bail before
 * scheduling any frames.
 */

import { initDevice } from './src/gpu/device.js';
import { Sim } from './src/sim.js';
import { setupUI } from './src/ui.js';

// ── Fixed-timestep config (mirrors geon's accumulator pattern) ──
const PHYSICS_DT     = 1 / 128;   // seconds per physics substep
const MAX_SUBSTEPS   = 8;         // hard cap per frame to avoid stale-dt bursts
const MAX_FRAME_DT   = 1 / 16;    // clamp huge frame deltas (tab refocus, etc.)
const ACCUMULATOR_CAP = PHYSICS_DT * MAX_SUBSTEPS;

class PlasmaSim {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {GPUAdapter} adapter
     * @param {GPUDevice} device
     * @param {GPUTextureFormat} format
     * @param {boolean} [hasTimestamp=false] adapter advertises 'timestamp-query'
     */
    constructor(canvas, adapter, device, format, hasTimestamp = false) {
        this.canvas = canvas;
        this.adapter = adapter;
        this.device = device;
        this.format = format;
        this.hasTimestamp = !!hasTimestamp;

        this.context = canvas.getContext('webgpu');
        this.context.configure({
            device,
            format,
            alphaMode: 'premultiplied',
        });

        this.sim = new Sim(device, this.context, format, { hasTimestamp: this.hasTimestamp });
        this.lastTime = 0;
        this.accumulator = 0;
        this.running = true;
        this._hidden = false;
        this._loopScheduled = false;

        this._sizeCanvas();
        this._resizeHandler = () => this._sizeCanvas();
        window.addEventListener('resize', this._resizeHandler, { passive: true });
    }

    /** Match the canvas backing store to its CSS box (DPR-aware). */
    _sizeCanvas() {
        const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        const w = Math.max(1, Math.floor(window.innerWidth  * dpr));
        const h = Math.max(1, Math.floor(window.innerHeight * dpr));
        if (this.canvas.width !== w)  this.canvas.width  = w;
        if (this.canvas.height !== h) this.canvas.height = h;
    }

    /**
     * Async init — defers to the Sim orchestrator for pipelines,
     * buffers, initial preset, LUT upload, and uniforms.
     */
    async init() {
        await this.sim.init();
    }

    /**
     * One physics substep. dt is the rAF-driven pacing budget; the
     * actual GPU-side dt is CFL-derived inside `sim.step()`.
     */
    step(_dt) {
        this.sim.step();
    }

    /** Encode and submit the render chain. */
    render() {
        this.sim.render();
    }

    /** rAF entry point — wraps `_loopBody` in try/catch and re-schedules. */
    loop(timestamp) {
        this._loopScheduled = false;
        try { this._loopBody(timestamp); }
        catch (e) { console.error('[plasma] loop error:', e); }
        if (!this._hidden) this._scheduleLoop();
    }

    _scheduleLoop() {
        if (this._loopScheduled) return; // prevent duplicate rAF chains
        this._loopScheduled = true;
        requestAnimationFrame((t) => this.loop(t));
    }

    _loopBody(timestamp) {
        const rawDt = Math.min((timestamp - this.lastTime) / 1000, MAX_FRAME_DT);
        this.lastTime = timestamp;

        // Honor sim.running (UI may have paused the sim) and apply the
        // speed scale to the accumulator inflow rate.
        const isRunning = this.sim && this.sim.running !== undefined ? this.sim.running : this.running;
        const speed = (this.sim && this.sim.speedScale) || 1;
        if (isRunning) {
            this.accumulator += rawDt * speed;
            if (this.accumulator > ACCUMULATOR_CAP) this.accumulator = ACCUMULATOR_CAP;
            let substeps = 0;
            while (this.accumulator >= PHYSICS_DT && substeps < MAX_SUBSTEPS) {
                this.step(PHYSICS_DT);
                this.accumulator -= PHYSICS_DT;
                substeps += 1;
            }
        }

        this.render();
    }

    start() {
        this.lastTime = performance.now();
        this._scheduleLoop();
    }
}

/** Reveal the no-WebGPU landing fallback. */
function showNoWebGPU(err) {
    console.warn('[plasma] WebGPU unavailable:', err);
    const fallback = document.getElementById('no-webgpu');
    const canvas = document.getElementById('simCanvas');
    if (fallback) fallback.hidden = false;
    if (canvas) canvas.hidden = true;
}

async function main() {
    const canvas = document.getElementById('simCanvas');
    if (!canvas) {
        console.error('[plasma] #simCanvas not found in DOM');
        return;
    }

    let adapter, device, format, hasTimestamp;
    try {
        ({ adapter, device, format, hasTimestamp } = await initDevice());
    } catch (e) {
        showNoWebGPU(e);
        return;
    }

    const sim = new PlasmaSim(canvas, adapter, device, format, hasTimestamp);
    try {
        await sim.init();
    } catch (e) {
        showNoWebGPU(e);
        return;
    }

    window.plasma = sim; // debug handle

    // Mount Phase-5 UI now that the sim is initialized.
    try {
        setupUI(sim);
        document.body.classList.add('app-ready');
    } catch (e) {
        console.warn('[plasma] UI setup failed:', e);
    }

    // Tab-hide pause: visibilitychange flips `_hidden`, and `loop()` only
    // reschedules an rAF when `!_hidden`. So one more frame runs after
    // hide (the already-queued rAF), then the loop chain dies. On unhide
    // we reset `lastTime` (so the accumulator doesn't see a huge dt) and
    // restart the chain. Verified: physics + render stop on tab hide.
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            sim._hidden = true;
        } else {
            sim._hidden = false;
            sim.lastTime = performance.now();
            sim._scheduleLoop();
        }
    });

    // Optional: react to device loss with a console warning. Real recovery
    // (re-acquire adapter, rebuild pipelines) lands later — Phase 1 is happy
    // to just stop.
    device.lost.then((info) => {
        console.warn('[plasma] WebGPU device lost:', info.reason, info.message);
        sim.running = false;
    });

    sim.start();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
} else {
    main();
}

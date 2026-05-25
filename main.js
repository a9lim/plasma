/**
 * @fileoverview plasma — Phase 1 entry point.
 *
 * Initializes WebGPU, sets up a single fullscreen-quad render pipeline
 * that paints the brand red, runs a rAF loop with a fixed-timestep
 * accumulator (no physics yet — `step()` is a no-op that just drains
 * the accumulator), and pauses on tab hide via `visibilitychange`.
 *
 * On any WebGPU failure (no adapter, no device, no nav.gpu) the
 * `#no-webgpu` landing element is unhidden and we bail before
 * scheduling any frames.
 */

import { initDevice } from './src/gpu/device.js';

// ── Fixed-timestep config (mirrors geon's accumulator pattern) ──
const PHYSICS_DT     = 1 / 128;   // seconds per physics substep
const MAX_SUBSTEPS   = 32;        // hard cap per frame to avoid death spiral
const MAX_FRAME_DT   = 1 / 16;    // clamp huge frame deltas (tab refocus, etc.)
const ACCUMULATOR_CAP = PHYSICS_DT * MAX_SUBSTEPS;

class PlasmaSim {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {GPUAdapter} adapter
     * @param {GPUDevice} device
     * @param {GPUTextureFormat} format
     */
    constructor(canvas, adapter, device, format) {
        this.canvas = canvas;
        this.adapter = adapter;
        this.device = device;
        this.format = format;

        this.context = canvas.getContext('webgpu');
        this.context.configure({
            device,
            format,
            alphaMode: 'premultiplied',
        });

        this.pipeline = null;          // set in init()
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
     * Build the fullscreen-quad render pipeline. Vertex shader emits
     * three vertices from `vertex_index`, no vertex buffer.
     */
    async init() {
        const wgslResponse = await fetch(new URL('./src/gpu/shaders/clear.wgsl', import.meta.url));
        if (!wgslResponse.ok) {
            throw new Error(`Failed to fetch clear.wgsl: ${wgslResponse.status}`);
        }
        const wgsl = await wgslResponse.text();

        const module = this.device.createShaderModule({ code: wgsl, label: 'plasma.clear' });
        this.pipeline = this.device.createRenderPipeline({
            label: 'plasma.clear.pipeline',
            layout: 'auto',
            vertex:   { module, entryPoint: 'vsMain' },
            fragment: { module, entryPoint: 'fsMain', targets: [{ format: this.format }] },
            primitive: { topology: 'triangle-list' },
        });
    }

    /**
     * Phase-1 no-op physics step. Drains the accumulator so the frame
     * loop reads correctly; real RHS lands in Phase 2.
     */
    step(_dt) { /* no-op until Phase 2 */ }

    /** Encode one fullscreen-quad pass and submit. */
    render() {
        const encoder = this.device.createCommandEncoder({ label: 'plasma.frame' });
        const view = this.context.getCurrentTexture().createView();
        const pass = encoder.beginRenderPass({
            label: 'plasma.clearPass',
            colorAttachments: [{
                view,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        pass.setPipeline(this.pipeline);
        pass.draw(3, 1, 0, 0);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
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

        if (this.running) {
            this.accumulator += rawDt;
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

    let adapter, device, format;
    try {
        ({ adapter, device, format } = await initDevice());
    } catch (e) {
        showNoWebGPU(e);
        return;
    }

    const sim = new PlasmaSim(canvas, adapter, device, format);
    try {
        await sim.init();
    } catch (e) {
        showNoWebGPU(e);
        return;
    }

    window.plasma = sim; // debug handle

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

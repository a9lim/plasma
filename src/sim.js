/**
 * @fileoverview Phase 2 orchestrator.
 *
 * Encapsulates the GPU side of the sim: owns buffers, pipelines, the
 * renderer, and the per-step encoding logic. Wired up from main.js,
 * which still owns the rAF loop + accumulator + visibilitychange.
 *
 * Time integration: dimensional-split forward Euler.
 *   per step:
 *     1. compute_dt (reset → reduce → finalize) on current state
 *     2. sweep along axis A: PLM → HLL → update  (ping-pong U)
 *     3. sweep along axis B: PLM → HLL → update  (ping-pong U)
 *   Strang ordering: even step → (X, Y), odd step → (Y, X)
 *
 * Submit boundaries: queue.writeBuffer() is processed by the queue in
 * insertion order BEFORE any subsequent command buffer's contents. If
 * we batched compute_dt + both sweeps into a single encoder + submit,
 * the second and third writeBuffer() calls (for the sweep_dir flip)
 * would land before any pass ran, clobbering the first sweep's
 * uniform values. We therefore submit three command buffers per step:
 * one for dt, one per sweep. The CPU side cost is trivial.
 */

import { PlasmaBuffers } from './gpu/buffers.js';
import { createPipelines } from './gpu/pipelines.js';
import { PlasmaRenderer } from './gpu/render.js';
import { makeSodPreset } from './presets.js';
import { VIRIDIS } from './colormaps.js';
import { GRID_N, DOMAIN_LENGTH, GAMMA_DEFAULT, WORKGROUP } from './config.js';

const WG = WORKGROUP;

export class Sim {
    /**
     * @param {GPUDevice} device
     * @param {GPUCanvasContext} context
     * @param {GPUTextureFormat} format
     */
    constructor(device, context, format) {
        this.device  = device;
        this.context = context;
        this.format  = format;

        this.n  = GRID_N;
        this.dx = DOMAIN_LENGTH / this.n;
        this.gamma   = GAMMA_DEFAULT;
        this.viewMin = 0.05;
        this.viewMax = 1.10;

        this.stepCount = 0;

        this.buffers   = null;
        this.pipelines = null;
        this.renderer  = null;
    }

    /**
     * Async init — fetches WGSL, builds pipelines, allocates buffers,
     * uploads the Sod IC and viridis LUT, primes uniforms + dt.
     */
    async init() {
        this.pipelines = await createPipelines(this.device, this.format);
        this.buffers   = new PlasmaBuffers(this.device, this.n);
        this.renderer  = new PlasmaRenderer(this.device, this.context, this.pipelines, this.buffers);

        this.loadPreset(makeSodPreset(this.n));
        this.buffers.uploadLUT(VIRIDIS);

        // Initial dt floor — finalize() will overwrite this every step.
        const seed = new Float32Array([1e-4]);
        this.device.queue.writeBuffer(this.buffers.dt, 0, seed.buffer);

        this._pushUniformsFor(0);
    }

    /**
     * Load a preset descriptor: replaces U, γ, dx, and view window.
     */
    loadPreset(preset) {
        this.gamma   = preset.gamma   ?? this.gamma;
        this.viewMin = preset.viewMin ?? this.viewMin;
        this.viewMax = preset.viewMax ?? this.viewMax;
        if (preset.domainLength) this.dx = preset.domainLength / this.n;
        this.buffers.uploadInitialState(preset.data);
        this.stepCount = 0;
    }

    _pushUniformsFor(sweepDir) {
        this.buffers.pushUniforms({
            dx: this.dx,
            gamma: this.gamma,
            viewMin: this.viewMin,
            viewMax: this.viewMax,
            gridN: this.n,
            sweepDir,
            stepParity: this.stepCount & 1,
        });
    }

    _sweepBindGroup(srcBuf, dstBuf) {
        return this.device.createBindGroup({
            label: 'plasma.sweep.bg',
            layout: this.pipelines.layouts.sweep,
            entries: [
                { binding: 0, resource: { buffer: this.buffers.uniform } },
                { binding: 1, resource: { buffer: srcBuf } },
                { binding: 2, resource: { buffer: this.buffers.slopes } },
                { binding: 3, resource: { buffer: this.buffers.flux } },
                { binding: 4, resource: { buffer: dstBuf } },
                { binding: 5, resource: { buffer: this.buffers.dt } },
            ],
        });
    }

    _dtBindGroup(srcBuf) {
        return this.device.createBindGroup({
            label: 'plasma.computeDt.bg',
            layout: this.pipelines.layouts.dt,
            entries: [
                { binding: 0, resource: { buffer: this.buffers.uniform } },
                { binding: 1, resource: { buffer: srcBuf } },
                { binding: 2, resource: { buffer: this.buffers.wavespeed } },
                { binding: 3, resource: { buffer: this.buffers.dt } },
            ],
        });
    }

    /** Submit compute_dt as its own command buffer. */
    _submitComputeDt(sweepDir) {
        const { device, pipelines, buffers } = this;
        const groups = Math.ceil(this.n / WG);

        this._pushUniformsFor(sweepDir);
        const bg = this._dtBindGroup(buffers.current);

        const encoder = device.createCommandEncoder({ label: 'plasma.computeDt.enc' });
        const pass = encoder.beginComputePass({ label: 'plasma.computeDt' });
        pass.setBindGroup(0, bg);

        pass.setPipeline(pipelines.pipelines.dtReset);
        pass.dispatchWorkgroups(1, 1, 1);

        pass.setPipeline(pipelines.pipelines.dtReduce);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.setPipeline(pipelines.pipelines.dtFinalize);
        pass.dispatchWorkgroups(1, 1, 1);

        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    /** Submit one sweep (PLM → HLL → update) as its own command buffer. */
    _submitSweep(sweepDir) {
        const { device, pipelines, buffers } = this;
        const groups = Math.ceil(this.n / WG);

        this._pushUniformsFor(sweepDir);
        const bg = this._sweepBindGroup(buffers.current, buffers.next);

        const encoder = device.createCommandEncoder({ label: `plasma.sweep.dir${sweepDir}.enc` });
        const pass = encoder.beginComputePass({ label: `plasma.sweep.dir${sweepDir}` });
        pass.setBindGroup(0, bg);

        pass.setPipeline(pipelines.pipelines.reconstructPlm);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.setPipeline(pipelines.pipelines.riemannHll);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.setPipeline(pipelines.pipelines.updateConserved);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.end();
        device.queue.submit([encoder.finish()]);

        buffers.swap();
    }

    /**
     * One full physics step.
     */
    step() {
        // Strang ordering — flip x↔y every other step.
        const dirs = (this.stepCount & 1) ? [1, 0] : [0, 1];

        this._submitComputeDt(dirs[0]);
        this._submitSweep(dirs[0]);
        this._submitSweep(dirs[1]);

        this.stepCount += 1;
    }

    /**
     * Encode and submit the render chain (view-field → colormap →
     * composite).
     */
    render() {
        // Ensure view_min/view_max are present in uniforms for the
        // colormap pass. The last sweep's pushUniforms already covers
        // this, but a no-op resend keeps state explicit if step()
        // hasn't run yet (e.g. very first render after init).
        this._pushUniformsFor(0);
        this.renderer.render();
    }
}

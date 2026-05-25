/**
 * @fileoverview Phase 3a orchestrator — 2.5D ideal MHD with CT.
 *
 * Pipeline per step:
 *   Submit 1 (compute_dt): reset → reduce → finalize. Sweep_dir = 0.
 *   Submit 2 (x-sweep RC+Riemann): writeBuffer sweep_dir=0; PLM_x → HLL_x.
 *                                  Writes flux_x_0/flux_x_1, slopes_x_*.
 *   Submit 3 (y-sweep RC+Riemann): writeBuffer sweep_dir=1; PLM_y → HLL_y.
 *                                  Writes flux_y_0/flux_y_1, slopes_y_*.
 *   Submit 4 (unsplit CT update): compute_emf → update_conserved → update_b.
 *                                 Writes U0_next, U1_next, Bx_next, By_next.
 *   buffers.swap() flips current/next.
 *
 * Why 4 submits: writeBuffer for the sweep_dir uniform must complete
 * before the bound shader reads it. Within a single submit, prior
 * writeBuffers are visible to the encoded passes — but multiple
 * writeBuffers before one submit all land in the queue and the encoder
 * sees only the last value. Splitting into separate submits is what
 * makes the per-pass uniform difference real.
 *
 * Pre-flight for Phase 3b: RK3 is 3 stages × (dt + x-sweep + y-sweep +
 * unsplit_update) = 12 submits, which is too many. The right shape is
 * to move sweep_dir (and stage_weight) into a storage buffer written
 * by a tiny init pass; that lets a whole step encode in 1-2 submits.
 * For 3a we ship the simpler writeBuffer pattern and document the
 * call-out below.
 */

import { PlasmaBuffers } from './gpu/buffers.js';
import { createPipelines } from './gpu/pipelines.js';
import { PlasmaRenderer } from './gpu/render.js';
import { makeBrioWuPreset } from './presets.js';
import { VIRIDIS } from './colormaps.js';
import { GRID_N, DOMAIN_LENGTH, GAMMA_DEFAULT, WORKGROUP, VIEW_DENSITY } from './config.js';

const WG = WORKGROUP;

export class Sim {
    constructor(device, context, format) {
        this.device  = device;
        this.context = context;
        this.format  = format;

        this.n  = GRID_N;
        this.dx = DOMAIN_LENGTH / this.n;
        this.gamma   = GAMMA_DEFAULT;
        this.viewMin = 0.05;
        this.viewMax = 1.10;
        this.viewMode = VIEW_DENSITY;

        this.stepCount = 0;

        this.buffers   = null;
        this.pipelines = null;
        this.renderer  = null;
    }

    async init() {
        this.pipelines = await createPipelines(this.device, this.format);
        this.buffers   = new PlasmaBuffers(this.device, this.n);
        this.renderer  = new PlasmaRenderer(this.device, this.context, this.pipelines, this.buffers);

        // Phase 3a default preset: Brio-Wu MHD shock tube. Sod still
        // available via loadPreset(makeSodPreset(n)).
        this.loadPreset(makeBrioWuPreset(this.n));
        this.buffers.uploadLUT(VIRIDIS);

        const seed = new Float32Array([1e-4]);
        this.device.queue.writeBuffer(this.buffers.dt, 0, seed.buffer);

        this._pushUniformsFor(0);
    }

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
            viewMode: this.viewMode,
        });
    }

    // ── Bind-group builders ─────────────────────────────────────────
    // We rebuild bind groups every step because the ping-pong current/next
    // pair flips. WebGPU bind-group creation is cheap (well under 50 µs
    // each on a typical adapter).

    _dtBG() {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.computeDt.bg',
            layout: this.pipelines.layouts.dt,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.U0_current } },
                { binding: 2, resource: { buffer: b.U1_current } },
                { binding: 3, resource: { buffer: b.Bx_current } },
                { binding: 4, resource: { buffer: b.By_current } },
                { binding: 5, resource: { buffer: b.wavespeed } },
                { binding: 6, resource: { buffer: b.dt } },
            ],
        });
    }

    _reconstructBG(s0Buf, s1Buf) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.reconstruct.bg',
            layout: this.pipelines.layouts.reconstruct,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.U0_current } },
                { binding: 2, resource: { buffer: b.U1_current } },
                { binding: 3, resource: { buffer: b.Bx_current } },
                { binding: 4, resource: { buffer: b.By_current } },
                { binding: 5, resource: { buffer: s0Buf } },
                { binding: 6, resource: { buffer: s1Buf } },
            ],
        });
    }

    _riemannBG(s0Buf, s1Buf, f0Buf, f1Buf) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.riemann.bg',
            layout: this.pipelines.layouts.riemann,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.U0_current } },
                { binding: 2, resource: { buffer: b.U1_current } },
                { binding: 3, resource: { buffer: b.Bx_current } },
                { binding: 4, resource: { buffer: b.By_current } },
                { binding: 5, resource: { buffer: s0Buf } },
                { binding: 6, resource: { buffer: s1Buf } },
                { binding: 7, resource: { buffer: f0Buf } },
                { binding: 8, resource: { buffer: f1Buf } },
            ],
        });
    }

    _emfBG() {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.emf.bg',
            layout: this.pipelines.layouts.emf,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.flux_x_1 } },
                { binding: 2, resource: { buffer: b.flux_y_1 } },
                { binding: 3, resource: { buffer: b.Ez_edge } },
            ],
        });
    }

    _updateConservedBG() {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.updateU.bg',
            layout: this.pipelines.layouts.updateU,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.U0_current } },
                { binding: 2, resource: { buffer: b.U1_current } },
                { binding: 3, resource: { buffer: b.flux_x_0 } },
                { binding: 4, resource: { buffer: b.flux_x_1 } },
                { binding: 5, resource: { buffer: b.flux_y_0 } },
                { binding: 6, resource: { buffer: b.flux_y_1 } },
                { binding: 7, resource: { buffer: b.dt } },
                { binding: 8, resource: { buffer: b.U0_next } },
                { binding: 9, resource: { buffer: b.U1_next } },
            ],
        });
    }

    _updateBBG() {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.updateB.bg',
            layout: this.pipelines.layouts.updateB,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.Bx_current } },
                { binding: 2, resource: { buffer: b.By_current } },
                { binding: 3, resource: { buffer: b.Ez_edge } },
                { binding: 4, resource: { buffer: b.dt } },
                { binding: 5, resource: { buffer: b.Bx_next } },
                { binding: 6, resource: { buffer: b.By_next } },
            ],
        });
    }

    // ── Submit helpers ──────────────────────────────────────────────
    _submitComputeDt() {
        const { device, pipelines } = this;
        const groups = Math.ceil(this.n / WG);
        this._pushUniformsFor(0);
        const bg = this._dtBG();

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

    _submitDirectionalSweep(axis) {
        const { device, pipelines, buffers } = this;
        const groups = Math.ceil(this.n / WG);
        this._pushUniformsFor(axis);

        const s0Buf = (axis === 0) ? buffers.slopes_x_0 : buffers.slopes_y_0;
        const s1Buf = (axis === 0) ? buffers.slopes_x_1 : buffers.slopes_y_1;
        const f0Buf = (axis === 0) ? buffers.flux_x_0  : buffers.flux_y_0;
        const f1Buf = (axis === 0) ? buffers.flux_x_1  : buffers.flux_y_1;

        const recBG = this._reconstructBG(s0Buf, s1Buf);
        const riBG  = this._riemannBG(s0Buf, s1Buf, f0Buf, f1Buf);

        const encoder = device.createCommandEncoder({ label: `plasma.sweep.axis${axis}.enc` });
        const pass = encoder.beginComputePass({ label: `plasma.sweep.axis${axis}` });

        pass.setPipeline(pipelines.pipelines.reconstructPlm);
        pass.setBindGroup(0, recBG);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.setPipeline(pipelines.pipelines.riemannHll);
        pass.setBindGroup(0, riBG);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    _submitUnsplitUpdate() {
        const { device, pipelines } = this;
        const groups = Math.ceil(this.n / WG);

        const emfBG = this._emfBG();
        const uBG   = this._updateConservedBG();
        const bBG   = this._updateBBG();

        const encoder = device.createCommandEncoder({ label: 'plasma.update.enc' });
        const pass = encoder.beginComputePass({ label: 'plasma.update' });

        pass.setPipeline(pipelines.pipelines.computeEmf);
        pass.setBindGroup(0, emfBG);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.setPipeline(pipelines.pipelines.updateConserved);
        pass.setBindGroup(0, uBG);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.setPipeline(pipelines.pipelines.updateB);
        pass.setBindGroup(0, bBG);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    step() {
        this._submitComputeDt();
        this._submitDirectionalSweep(0);
        this._submitDirectionalSweep(1);
        this._submitUnsplitUpdate();
        this.buffers.swap();
        this.stepCount += 1;
    }

    render() {
        this._pushUniformsFor(0);
        this.renderer.render();
    }

    // ── Debug: ∇·B L2 norm sanity (uncomment in main.js every 100 frames)
    // Add a readback path here if you want to verify CT in production.
    // Skipped by default to keep the hot path GPU-only.
    /*
    async debugDivB() {
        const n = this.n;
        const cells = n * n;
        const staging_bx = this.device.createBuffer({
            size: cells * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const staging_by = this.device.createBuffer({
            size: cells * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.Bx_current, 0, staging_bx, 0, cells * 4);
        enc.copyBufferToBuffer(this.buffers.By_current, 0, staging_by, 0, cells * 4);
        this.device.queue.submit([enc.finish()]);
        await Promise.all([staging_bx.mapAsync(GPUMapMode.READ), staging_by.mapAsync(GPUMapMode.READ)]);
        const bx = new Float32Array(staging_bx.getMappedRange()).slice();
        const by = new Float32Array(staging_by.getMappedRange()).slice();
        staging_bx.unmap(); staging_by.unmap();
        staging_bx.destroy(); staging_by.destroy();
        let s2 = 0;
        for (let j = 0; j < n; j++) {
            for (let i = 0; i < n; i++) {
                const idx  = j * n + i;
                const il   = j * n + ((i - 1 + n) % n);
                const jd   = ((j - 1 + n) % n) * n + i;
                const d    = (bx[idx] - bx[il]) + (by[idx] - by[jd]);
                s2 += d * d;
            }
        }
        return Math.sqrt(s2 / cells) / this.dx;
    }
    */
}

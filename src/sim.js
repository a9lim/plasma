/**
 * @fileoverview Phase 3b orchestrator — RK3 SSP + HLLD + PPM + CT.
 *
 * RK3 SSP scheme (Gottlieb-Shu 1998):
 *     U(1)   = U(n) + dt · L(U(n))
 *     U(2)   = (3/4)U(n) + (1/4)U(1) + (1/4)dt · L(U(1))
 *     U(n+1) = (1/3)U(n) + (2/3)U(2) + (2/3)dt · L(U(2))
 *
 * Applied in lockstep to:
 *     U_cell        (ρ, ρv, E, Bz)
 *     Bx_face       (face-centered)
 *     By_face       (face-centered)
 *
 * dt is computed ONCE at the start of the timestep from U(n)'s state and
 * reused across all three stages — required for SSP. L(U) = -∇·F(U) with
 * F via PPM reconstruction → HLLD Riemann → CT-EMF for the face B field.
 *
 * Submit structure (ONE submit per physics step, two compute passes):
 *   pass 1: compute_dt on U(n) (writes dt_buf).
 *   pass 2: all three RK3 stages chained.
 *
 * Encoder layout (pass 2):
 *   Stage 1 (input U_n; writes U_1):
 *     PPM_x → HLLD_x → PPM_y → HLLD_y → EMF → update_U(w=1) → update_B(w=1)
 *   Stage 2 (input U_1; writes U_2 with U_n combined in):
 *     PPM_x → HLLD_x → PPM_y → HLLD_y → EMF → update_U(w=3/4,1/4,1/4) → update_B
 *   Stage 3 (input U_2; writes U_next with U_n combined in):
 *     PPM_x → HLLD_x → PPM_y → HLLD_y → EMF → update_U(w=1/3,2/3,2/3) → update_B
 *
 * After submit 2, slot_next holds U(n+1). buffers.swap() flips the
 * (slot_n, slot_next) handles so render and the next step's compute_dt
 * see the new state. We need 4 storage slots (A/B for n/n+1 ping-pong,
 * C/D for stage 1/2 scratch) because stage 3's update reads U_n as RO
 * while writing U_out as RW — WebGPU forbids aliasing those.
 *
 * Bind-group rebuild: rebuilt at the start of each step (cheap; <50µs
 * each) because the per-stage "input" slot for L(U) shifts (n→1→2).
 *
 * Single-submit guarantee: writeBuffer for the per-stage parameters is
 * skipped entirely — stage_1/2/3 uniform buffers are pre-populated at
 * init. sweep_dir lives in two separate uniform buffers (uniform_x,
 * uniform_y) also pre-populated. Only dt_buf changes during the encoder,
 * and it's written GPU-side by compute_dt.
 *
 * Transpiler-friendly notes: every dispatch is a straight nested loop
 * over (i,j) cells with no shared-memory dependencies inside the
 * workgroup beyond the standard workgroup-barrier pattern used in
 * compute_dt's reduce.
 */

import { PlasmaBuffers } from './gpu/buffers.js';
import { createPipelines } from './gpu/pipelines.js';
import { PlasmaRenderer } from './gpu/render.js';
import { makeOrszagTangPreset } from './presets.js';
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

        // Phase 3b default preset: Orszag-Tang vortex (the canonical 2D MHD test).
        this.loadPreset(makeOrszagTangPreset(this.n));
        this.buffers.uploadLUT(VIRIDIS);

        const seed = new Float32Array([1e-4]);
        this.device.queue.writeBuffer(this.buffers.dt, 0, seed.buffer);

        this._pushUniforms();
    }

    loadPreset(preset) {
        this.gamma   = preset.gamma   ?? this.gamma;
        this.viewMin = preset.viewMin ?? this.viewMin;
        this.viewMax = preset.viewMax ?? this.viewMax;
        if (preset.domainLength) this.dx = preset.domainLength / this.n;
        this.buffers.uploadInitialState(preset.data);
        this.stepCount = 0;
        this._pushUniforms();
    }

    _pushUniforms() {
        this.buffers.pushUniforms({
            dx: this.dx,
            gamma: this.gamma,
            viewMin: this.viewMin,
            viewMax: this.viewMax,
            gridN: this.n,
            stepParity: this.stepCount & 1,
            viewMode: this.viewMode,
        });
    }

    // ── Bind-group builders ─────────────────────────────────────────
    // All bind groups are rebuilt per step. WebGPU bind-group creation
    // is cheap (< 50 µs each on a typical adapter).

    _dtBG() {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.computeDt.bg',
            layout: this.pipelines.layouts.dt,
            entries: [
                { binding: 0, resource: { buffer: b.uniform_x } },
                { binding: 1, resource: { buffer: b.U0_n } },
                { binding: 2, resource: { buffer: b.U1_n } },
                { binding: 3, resource: { buffer: b.Bx_n } },
                { binding: 4, resource: { buffer: b.By_n } },
                { binding: 5, resource: { buffer: b.wavespeed } },
                { binding: 6, resource: { buffer: b.dt } },
            ],
        });
    }

    /**
     * Build a PPM-reconstruction bind group for sweep direction `axis`,
     * reading from the given (U0, U1, Bx, By) slot.
     */
    _reconstructBG(axis, U0, U1, Bx, By) {
        const b = this.buffers;
        const uniBuf = (axis === 0) ? b.uniform_x : b.uniform_y;
        const eL0 = (axis === 0) ? b.edge_l_x_0 : b.edge_l_y_0;
        const eL1 = (axis === 0) ? b.edge_l_x_1 : b.edge_l_y_1;
        const eR0 = (axis === 0) ? b.edge_r_x_0 : b.edge_r_y_0;
        const eR1 = (axis === 0) ? b.edge_r_x_1 : b.edge_r_y_1;
        return this.device.createBindGroup({
            label: `plasma.reconstruct.axis${axis}.bg`,
            layout: this.pipelines.layouts.reconstruct,
            entries: [
                { binding: 0, resource: { buffer: uniBuf } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: eL0 } },
                { binding: 6, resource: { buffer: eL1 } },
                { binding: 7, resource: { buffer: eR0 } },
                { binding: 8, resource: { buffer: eR1 } },
            ],
        });
    }

    _riemannBG(axis, U0, U1, Bx, By) {
        const b = this.buffers;
        const uniBuf = (axis === 0) ? b.uniform_x : b.uniform_y;
        const eL0 = (axis === 0) ? b.edge_l_x_0 : b.edge_l_y_0;
        const eL1 = (axis === 0) ? b.edge_l_x_1 : b.edge_l_y_1;
        const eR0 = (axis === 0) ? b.edge_r_x_0 : b.edge_r_y_0;
        const eR1 = (axis === 0) ? b.edge_r_x_1 : b.edge_r_y_1;
        const f0  = (axis === 0) ? b.flux_x_0   : b.flux_y_0;
        const f1  = (axis === 0) ? b.flux_x_1   : b.flux_y_1;
        return this.device.createBindGroup({
            label: `plasma.riemann.axis${axis}.bg`,
            layout: this.pipelines.layouts.riemann,
            entries: [
                { binding: 0,  resource: { buffer: uniBuf } },
                { binding: 1,  resource: { buffer: U0 } },
                { binding: 2,  resource: { buffer: U1 } },
                { binding: 3,  resource: { buffer: Bx } },
                { binding: 4,  resource: { buffer: By } },
                { binding: 5,  resource: { buffer: eL0 } },
                { binding: 6,  resource: { buffer: eL1 } },
                { binding: 7,  resource: { buffer: eR0 } },
                { binding: 8,  resource: { buffer: eR1 } },
                { binding: 9,  resource: { buffer: f0 } },
                { binding: 10, resource: { buffer: f1 } },
            ],
        });
    }

    _emfBG() {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.emf.bg',
            layout: this.pipelines.layouts.emf,
            entries: [
                { binding: 0, resource: { buffer: b.uniform_x } },
                { binding: 1, resource: { buffer: b.flux_x_1 } },
                { binding: 2, resource: { buffer: b.flux_y_1 } },
                { binding: 3, resource: { buffer: b.Ez_edge } },
            ],
        });
    }

    /**
     * Build the weighted-update bind group for a stage.
     *   stageBuf  : uniform buffer with (a0, a1, dt_w, _)
     *   U0_other  : second source state for the linear combination
     *   U1_other  : .
     *   U0_out, U1_out : destination
     */
    _updateUBG(stageBuf, U0_other, U1_other, U0_out, U1_out) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.updateU.bg',
            layout: this.pipelines.layouts.updateU,
            entries: [
                { binding: 0,  resource: { buffer: b.uniform_x } },
                { binding: 1,  resource: { buffer: stageBuf } },
                { binding: 2,  resource: { buffer: b.U0_n } },
                { binding: 3,  resource: { buffer: b.U1_n } },
                { binding: 4,  resource: { buffer: U0_other } },
                { binding: 5,  resource: { buffer: U1_other } },
                { binding: 6,  resource: { buffer: b.flux_x_0 } },
                { binding: 7,  resource: { buffer: b.flux_x_1 } },
                { binding: 8,  resource: { buffer: b.flux_y_0 } },
                { binding: 9,  resource: { buffer: b.flux_y_1 } },
                { binding: 10, resource: { buffer: b.dt } },
                { binding: 11, resource: { buffer: U0_out } },
                { binding: 12, resource: { buffer: U1_out } },
            ],
        });
    }

    _updateBBG(stageBuf, Bx_other, By_other, Bx_out, By_out) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.updateB.bg',
            layout: this.pipelines.layouts.updateB,
            entries: [
                { binding: 0, resource: { buffer: b.uniform_x } },
                { binding: 1, resource: { buffer: stageBuf } },
                { binding: 2, resource: { buffer: b.Bx_n } },
                { binding: 3, resource: { buffer: b.By_n } },
                { binding: 4, resource: { buffer: Bx_other } },
                { binding: 5, resource: { buffer: By_other } },
                { binding: 6, resource: { buffer: b.Ez_edge } },
                { binding: 7, resource: { buffer: b.dt } },
                { binding: 8, resource: { buffer: Bx_out } },
                { binding: 9, resource: { buffer: By_out } },
            ],
        });
    }

    /**
     * Encode the dt computation into an open compute pass. compute_dt is
     * a separate pass (vs. inline in the RK3 pass) because dtReset uses
     * @workgroup_size(1) which differs from the 8×8 used by the RK3
     * substages — but more importantly, this keeps the wavespeed atomic
     * isolated from the rest of the pipeline state.
     */
    _encodeComputeDt(encoder) {
        const { pipelines } = this;
        const groups = Math.ceil(this.n / WG);
        const bg = this._dtBG();

        const pass = encoder.beginComputePass({ label: 'plasma.computeDt' });
        pass.setBindGroup(0, bg);
        pass.setPipeline(pipelines.pipelines.dtReset);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.setPipeline(pipelines.pipelines.dtReduce);
        pass.dispatchWorkgroups(groups, groups, 1);
        pass.setPipeline(pipelines.pipelines.dtFinalize);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();
    }

    /**
     * Encode one RK3 stage: PPM_x → HLLD_x → PPM_y → HLLD_y → EMF
     *                       → update_U → update_B
     *
     * @param pass       open compute pass to append into
     * @param srcU0      U0 source for L(U) eval
     * @param srcU1      U1 source for L(U) eval
     * @param srcBx      Bx source for L(U) eval (also used in update for n-stage's a0 weight via stageBuf)
     * @param srcBy      By source for L(U) eval
     * @param otherU0    U0 "other" buffer for the linear combination
     * @param otherU1    U1 "other" buffer
     * @param otherBx    Bx "other" buffer
     * @param otherBy    By "other" buffer
     * @param dstU0      destination
     * @param dstU1      destination
     * @param dstBx      destination
     * @param dstBy      destination
     * @param stageBuf   stage_params uniform buffer (a0, a1, dt_w, _)
     */
    _encodeStage(pass, srcU0, srcU1, srcBx, srcBy,
                 otherU0, otherU1, otherBx, otherBy,
                 dstU0, dstU1, dstBx, dstBy, stageBuf) {
        const { pipelines } = this;
        const groups = Math.ceil(this.n / WG);

        // x-sweep: reconstruct + Riemann
        pass.setPipeline(pipelines.pipelines.reconstructPpm);
        pass.setBindGroup(0, this._reconstructBG(0, srcU0, srcU1, srcBx, srcBy));
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.setPipeline(pipelines.pipelines.riemannHlld);
        pass.setBindGroup(0, this._riemannBG(0, srcU0, srcU1, srcBx, srcBy));
        pass.dispatchWorkgroups(groups, groups, 1);

        // y-sweep: reconstruct + Riemann
        pass.setPipeline(pipelines.pipelines.reconstructPpm);
        pass.setBindGroup(0, this._reconstructBG(1, srcU0, srcU1, srcBx, srcBy));
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.setPipeline(pipelines.pipelines.riemannHlld);
        pass.setBindGroup(0, this._riemannBG(1, srcU0, srcU1, srcBx, srcBy));
        pass.dispatchWorkgroups(groups, groups, 1);

        // CT EMF at corners from x/y face fluxes
        pass.setPipeline(pipelines.pipelines.computeEmf);
        pass.setBindGroup(0, this._emfBG());
        pass.dispatchWorkgroups(groups, groups, 1);

        // Weighted update for cell-centered state
        pass.setPipeline(pipelines.pipelines.updateConservedWeighted);
        pass.setBindGroup(0, this._updateUBG(stageBuf, otherU0, otherU1, dstU0, dstU1));
        pass.dispatchWorkgroups(groups, groups, 1);

        // Weighted update for face-centered B
        pass.setPipeline(pipelines.pipelines.updateBWeighted);
        pass.setBindGroup(0, this._updateBBG(stageBuf, otherBx, otherBy, dstBx, dstBy));
        pass.dispatchWorkgroups(groups, groups, 1);
    }

    /**
     * One physics step — single submit:
     *   pass 1: compute_dt (reads U_n, writes dt_buf).
     *   pass 2: three RK3 SSP stages chained.
     *
     * Storage-buffer writes from earlier passes in the same encoder are
     * visible to later passes by WebGPU's implicit barriers — so dt_buf
     * written by compute_dt is readable by stage 1's update kernels.
     */
    step() {
        const { device } = this;
        const b = this.buffers;

        const encoder = device.createCommandEncoder({ label: 'plasma.step.enc' });

        // Pass 1: compute the timestep from U(n).
        this._encodeComputeDt(encoder);

        // Pass 2: three RK3 SSP stages.
        const pass = encoder.beginComputePass({ label: 'plasma.rk3' });

        // Stage 1: U(1) = U(n) + dt · L(U(n))
        // Weights: a0=1, a1=0, dt_w=1. otherU = U_n itself (a1=0 so unused).
        this._encodeStage(pass,
            b.U0_n, b.U1_n, b.Bx_n, b.By_n,           // source for L(U)
            b.U0_n, b.U1_n, b.Bx_n, b.By_n,           // "other" — unused since a1=0
            b.U0_1, b.U1_1, b.Bx_1, b.By_1,           // destination
            b.stage_1);

        // Stage 2: U(2) = 3/4 U(n) + 1/4 U(1) + 1/4 dt · L(U(1))
        this._encodeStage(pass,
            b.U0_1, b.U1_1, b.Bx_1, b.By_1,           // source for L(U)
            b.U0_1, b.U1_1, b.Bx_1, b.By_1,           // "other" = U(1)
            b.U0_2, b.U1_2, b.Bx_2, b.By_2,           // destination
            b.stage_2);

        // Stage 3: U(n+1) = 1/3 U(n) + 2/3 U(2) + 2/3 dt · L(U(2))
        // Destination MUST be a distinct buffer from slot_n (the shader
        // reads U_n as RO and writes U_out as RW; WebGPU forbids aliasing
        // those in one bind group). We write to slot_next and swap.
        this._encodeStage(pass,
            b.U0_2,    b.U1_2,    b.Bx_2,    b.By_2,        // source for L(U)
            b.U0_2,    b.U1_2,    b.Bx_2,    b.By_2,        // "other" = U(2)
            b.U0_next, b.U1_next, b.Bx_next, b.By_next,     // destination — slot next
            b.stage_3);

        pass.end();
        device.queue.submit([encoder.finish()]);

        // Promote slot_next to slot_n for the next step.
        b.swap();
        this.stepCount += 1;
    }

    render() {
        this._pushUniforms();
        this.renderer.render();
    }
}

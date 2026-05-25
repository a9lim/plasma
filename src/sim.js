/**
 * @fileoverview Phase 4 orchestrator — RK3 SSP + HLLD + PPM + CT
 *                + per-edge BCs + explicit resistivity.
 *
 * RK3 SSP scheme (Gottlieb-Shu 1998):
 *     U(1)   = U(n) + dt · L(U(n))
 *     U(2)   = (3/4)U(n) + (1/4)U(1) + (1/4)dt · L(U(1))
 *     U(n+1) = (1/3)U(n) + (2/3)U(2) + (2/3)dt · L(U(2))
 *
 * Per RK3 stage we run, in order:
 *     1.  apply-bcs               (fill ghost cells from BC modes)
 *     2.  reconstruct-ppm (x)     (PPM L/R edge states, x-sweep)
 *     3.  riemann-hlld    (x)
 *     4.  reconstruct-ppm (y)
 *     5.  riemann-hlld    (y)
 *     6.  compute-emf            (CT corner-Ez from face fluxes)
 *     7.  update-conserved-weighted (cell-centered SSP update)
 *     8.  update-b-weighted      (face-B SSP update)
 *     9.  apply-resistivity      (η ∇²B per stage, linear → SSP)
 *
 * The BC fill at the START of each stage refreshes ghost cells from the
 * CURRENT input state (which differs per stage: U_n / U_1 / U_2). The
 * resistive step at the END uses ghost cells filled BEFORE the stage,
 * which is correct because they haven't been overwritten by the stage's
 * interior writes (only interior indices are touched by the CT and
 * conserved updates).
 *
 * dt is computed ONCE at the start of the step (from U(n)) and reused
 * across all three stages — required for SSP. We include the parabolic
 * resistive CFL inside compute-dt (dt_res = 0.5·dx²/η; min'd with dt_hyp).
 *
 * Submit structure (ONE submit per physics step, two compute passes):
 *   pass 1: compute_dt on U(n).
 *   pass 2: three RK3 stages chained, each with the 9 sub-steps above.
 *
 * Dispatch shapes (interior grid N×N, ghost = 2, n_total = N+4):
 *   apply-bcs:          (n_total+1) × (n_total+1)        — covers Bx/By extra row/col
 *   reconstruct-ppm:    (N+2) × (N+2)                    — extended dispatch
 *   riemann-hlld (x):   (N+1) × (N+2)
 *   riemann-hlld (y):   (N+2) × (N+1)
 *   compute-emf:        (N+1) × (N+1)
 *   update-conserved:   N × N
 *   update-b:           (N+1) × (N+1)                    — covers Bx/By interior
 *   apply-resistivity:  (n_total+1) × (n_total+1)        — covers all face axes
 *   compute-dt reduce:  N × N                            — interior only
 */

import { PlasmaBuffers } from './gpu/buffers.js';
import { createPipelines } from './gpu/pipelines.js';
import { PlasmaRenderer } from './gpu/render.js';
import { makeOrszagTangPreset, PRESETS } from './presets.js';
import { VIRIDIS } from './colormaps.js';
import {
    GRID_N, GHOST_WIDTH, DOMAIN_LENGTH, GAMMA_DEFAULT, WORKGROUP,
    VIEW_JZ, ETA_DEFAULT, BC_PERIODIC, CFL, PRESSURE_FLOOR,
} from './config.js';

const WG = WORKGROUP;

export class Sim {
    constructor(device, context, format) {
        this.device  = device;
        this.context = context;
        this.format  = format;

        this.n        = GRID_N;
        this.ghost    = GHOST_WIDTH;
        this.n_total  = this.n + 2 * this.ghost;
        this.domainLength = DOMAIN_LENGTH;
        this.dx       = DOMAIN_LENGTH / this.n;
        this.gamma    = GAMMA_DEFAULT;
        this.viewMin  = 0.05;
        this.viewMax  = 1.10;
        this.viewMode = VIEW_JZ;
        this.eta      = ETA_DEFAULT;
        this.cfl      = CFL;
        this.pressureFloor = PRESSURE_FLOOR;

        this.stepCount = 0;
        this.simTime   = 0;
        this.lastDt    = 0;

        // UI integration state — owned by Sim so save/load can capture it.
        this.running     = true;
        this.speedScale  = 1;
        this.presetName  = 'orszag-tang';

        this.buffers   = null;
        this.pipelines = null;
        this.renderer  = null;

        // Default BC state: all periodic, neutral driven state.
        this.bcConfig = {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
            driven: { rho: 1, vx: 0, vy: 0, vz: 0, bx: 0, by: 0, bz: 0, p: 1 },
        };
    }

    async init() {
        this.pipelines = await createPipelines(this.device, this.format);
        this.buffers   = new PlasmaBuffers(this.device, this.n);
        this.renderer  = new PlasmaRenderer(this.device, this.context, this.pipelines, this.buffers);

        // Phase 4 default preset still Orszag-Tang for smoke check;
        // Phase 5 wires the dropdown for Harris/etc.
        this.loadPreset(makeOrszagTangPreset(this.n));
        this.buffers.uploadLUT(VIRIDIS);

        const seed = new Float32Array([1e-4]);
        this.device.queue.writeBuffer(this.buffers.dt, 0, seed.buffer);

        this._pushUniforms();
        this.buffers.pushBC(this.bcConfig);
    }

    loadPreset(preset) {
        this.gamma   = preset.gamma   ?? this.gamma;
        this.viewMin = preset.viewMin ?? this.viewMin;
        this.viewMax = preset.viewMax ?? this.viewMax;
        this.eta     = preset.eta     ?? this.eta;
        if (preset.domainLength) {
            this.domainLength = preset.domainLength;
            this.dx = preset.domainLength / this.n;
        }
        if (preset.bc) {
            this.bcConfig = { ...this.bcConfig, ...preset.bc };
        }
        if (preset.id) this.presetName = preset.id;
        this.buffers.uploadInitialState(preset.data);
        this.stepCount = 0;
        this.simTime   = 0;
        this._pushUniforms();
        this.buffers.pushBC(this.bcConfig);
    }

    /**
     * Public API — load a preset by name. Recognized names mirror the
     * keys of `PRESETS` in `presets.js` ('sod', 'brio-wu', 'orszag-tang',
     * 'harris'). Unknown names are a no-op.
     */
    setPreset(name) {
        const fn = PRESETS[name];
        if (!fn) {
            console.warn(`[plasma] setPreset: unknown preset "${name}"`);
            return;
        }
        const preset = fn(this.n);
        this.loadPreset(preset);
    }

    /**
     * Update a single BC edge mode.
     * @param {'N'|'S'|'E'|'W'} edge
     * @param {number} mode  BC_PERIODIC | BC_OUTFLOW | BC_REFLECTING | BC_DRIVEN
     */
    setBC(edge, mode) {
        const key = ({ N: 'modeN', S: 'modeS', E: 'modeE', W: 'modeW' })[edge];
        if (!key) return;
        this.bcConfig = { ...this.bcConfig, [key]: mode };
        this.buffers.pushBC(this.bcConfig);
    }

    /**
     * Update the driven inflow primitive state. Partial — only provided
     * fields are overwritten.
     */
    setDrivenState(state) {
        this.bcConfig = {
            ...this.bcConfig,
            driven: { ...this.bcConfig.driven, ...state },
        };
        this.buffers.pushBC(this.bcConfig);
    }

    /** Update explicit resistivity. UI typically passes 0 for "ideal". */
    setEta(eta) {
        this.eta = eta;
        this._pushUniforms();
    }

    /** Update the view mode enum. Pushed via uniforms on next render. */
    setViewMode(mode) {
        this.viewMode = mode;
        // Pick a sensible default colormap window per view if the caller
        // hasn't overridden. The Sim does not own a colormap-LUT switch
        // yet; that lands in Phase 6 alongside LIC. For now we just set
        // the linear-window endpoints.
        switch (mode) {
            case 0: this.viewMin = 0.05; this.viewMax = 1.10; break; // ρ
            case 1: this.viewMin = 0.01; this.viewMax = 1.00; break; // p
            case 2: this.viewMin = 0.0;  this.viewMax = 1.5;  break; // |v|
            case 3: this.viewMin = 0.0;  this.viewMax = 2.0;  break; // |B|
            case 4: this.viewMin = -3.0; this.viewMax = 3.0;  break; // Jz (signed)
            default: break;
        }
        this._pushUniforms();
    }

    setCFL(cfl)         { this.cfl = cfl; /* used by compute-dt via uniforms in a later wiring */ }
    setGamma(g)         { this.gamma = g; this._pushUniforms(); }
    setPressureFloor(p) { this.pressureFloor = p; /* shader floor is constant in WGSL; UI exposes for future use */ }

    setRunning(r)       { this.running = !!r; }
    setSpeedScale(s)    { this.speedScale = s; }

    /**
     * Re-allocate buffers at a new interior resolution and re-load the
     * current preset. Existing buffers are released by dropping the
     * `PlasmaBuffers` instance; GC handles GPU resource teardown when
     * the device is alive.
     */
    setResolution(n) {
        if (n === this.n) return;
        if (n !== 256 && n !== 512 && n !== 1024) {
            console.warn(`[plasma] setResolution: unsupported n=${n}`);
            return;
        }
        // Destroy existing buffers (best-effort — WebGPU has no formal
        // destroy on storage buffers; the GC will reclaim).
        this.n       = n;
        this.n_total = n + 2 * this.ghost;
        this.dx      = this.domainLength / n;
        this.buffers = new PlasmaBuffers(this.device, n);
        // Re-bind the renderer at the new buffer set.
        this.renderer = new PlasmaRenderer(this.device, this.context, this.pipelines, this.buffers);
        this.buffers.uploadLUT(VIRIDIS);
        const seed = new Float32Array([1e-4]);
        this.device.queue.writeBuffer(this.buffers.dt, 0, seed.buffer);
        this._pushUniforms();
        // Re-load whichever preset is current.
        this.setPreset(this.presetName);
    }

    /**
     * Serialize sim configuration to a JSON string. Excludes the
     * (large) U_n buffer; on load we re-instantiate from the named
     * preset and re-apply parameters. This is intentional — a full
     * buffer snapshot would dwarf localStorage budgets at 512/1024,
     * and the use-case is "save my UI state", not "rewind exactly".
     */
    saveState() {
        return JSON.stringify({
            v: 1,
            preset: this.presetName,
            n: this.n,
            viewMode: this.viewMode,
            eta: this.eta,
            gamma: this.gamma,
            cfl: this.cfl,
            pressureFloor: this.pressureFloor,
            speedScale: this.speedScale,
            running: this.running,
            bc: this.bcConfig,
        });
    }

    /** Restore from `saveState()` output. */
    loadState(s) {
        let obj;
        try { obj = JSON.parse(s); } catch (e) { console.warn('[plasma] loadState parse:', e); return; }
        if (!obj || obj.v !== 1) return;
        if (obj.n && obj.n !== this.n) this.setResolution(obj.n);
        if (obj.preset) this.setPreset(obj.preset);
        if (obj.viewMode !== undefined) this.setViewMode(obj.viewMode);
        if (obj.eta !== undefined)      this.setEta(obj.eta);
        if (obj.gamma !== undefined)    this.setGamma(obj.gamma);
        if (obj.cfl !== undefined)      this.setCFL(obj.cfl);
        if (obj.pressureFloor !== undefined) this.setPressureFloor(obj.pressureFloor);
        if (obj.speedScale !== undefined)    this.setSpeedScale(obj.speedScale);
        if (obj.running !== undefined)       this.setRunning(obj.running);
        if (obj.bc) {
            this.bcConfig = { ...this.bcConfig, ...obj.bc };
            this.buffers.pushBC(this.bcConfig);
        }
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
            eta: this.eta,
        });
    }

    // ── Bind-group builders ─────────────────────────────────────────

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

    _applyBcsBG(U0, U1, Bx, By) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyBcs.bg',
            layout: this.pipelines.layouts.applyBcs,
            entries: [
                { binding: 0, resource: { buffer: b.uniform_x } },
                { binding: 1, resource: { buffer: b.bc_uniforms } },
                { binding: 2, resource: { buffer: U0 } },
                { binding: 3, resource: { buffer: U1 } },
                { binding: 4, resource: { buffer: Bx } },
                { binding: 5, resource: { buffer: By } },
            ],
        });
    }

    _applyResBG(stageBuf, Bx, By, U1_out) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyRes.bg',
            layout: this.pipelines.layouts.applyRes,
            entries: [
                { binding: 0, resource: { buffer: b.uniform_x } },
                { binding: 1, resource: { buffer: stageBuf } },
                { binding: 2, resource: { buffer: Bx } },
                { binding: 3, resource: { buffer: By } },
                { binding: 4, resource: { buffer: U1_out } },
                { binding: 5, resource: { buffer: b.dt } },
            ],
        });
    }

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

    _encodeComputeDt(encoder) {
        const { pipelines } = this;
        const groupsInterior = Math.ceil(this.n / WG);
        const bg = this._dtBG();

        const pass = encoder.beginComputePass({ label: 'plasma.computeDt' });
        pass.setBindGroup(0, bg);
        pass.setPipeline(pipelines.pipelines.dtReset);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.setPipeline(pipelines.pipelines.dtReduce);
        pass.dispatchWorkgroups(groupsInterior, groupsInterior, 1);
        pass.setPipeline(pipelines.pipelines.dtFinalize);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();
    }

    /**
     * Encode one RK3 stage. The stage's source-state slot (srcU0/...) is
     * where apply-bcs writes ghosts; reconstruct-ppm + Riemann read from
     * the same slot; update writes to dstU0/.../dstBy. After the CT
     * update, apply-resistivity adds η∇²B in-place on the destination
     * face/cell buffers.
     */
    _encodeStage(pass, srcU0, srcU1, srcBx, srcBy,
                 otherU0, otherU1, otherBx, otherBy,
                 dstU0, dstU1, dstBx, dstBy, stageBuf) {
        const { pipelines } = this;
        const N         = this.n;
        const N2        = N + 2;
        const N1        = N + 1;
        const Ntotal    = this.n_total;
        const gInterior = Math.ceil(N / WG);
        const gN1       = Math.ceil(N1 / WG);
        const gN2       = Math.ceil(N2 / WG);
        const gTotalP1  = Math.ceil((Ntotal + 1) / WG);

        // 1. apply-bcs (fills ghosts in srcU0/srcU1/srcBx/srcBy).
        pass.setPipeline(pipelines.pipelines.applyBcs);
        pass.setBindGroup(0, this._applyBcsBG(srcU0, srcU1, srcBx, srcBy));
        pass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);

        // 2-5. PPM + Riemann, both axes.
        pass.setPipeline(pipelines.pipelines.reconstructPpm);
        pass.setBindGroup(0, this._reconstructBG(0, srcU0, srcU1, srcBx, srcBy));
        pass.dispatchWorkgroups(gN2, gN2, 1);

        pass.setPipeline(pipelines.pipelines.riemannHlld);
        pass.setBindGroup(0, this._riemannBG(0, srcU0, srcU1, srcBx, srcBy));
        pass.dispatchWorkgroups(gN1, gN2, 1);   // (N+1) × (N+2) for x-sweep

        pass.setPipeline(pipelines.pipelines.reconstructPpm);
        pass.setBindGroup(0, this._reconstructBG(1, srcU0, srcU1, srcBx, srcBy));
        pass.dispatchWorkgroups(gN2, gN2, 1);

        pass.setPipeline(pipelines.pipelines.riemannHlld);
        pass.setBindGroup(0, this._riemannBG(1, srcU0, srcU1, srcBx, srcBy));
        pass.dispatchWorkgroups(gN2, gN1, 1);   // (N+2) × (N+1) for y-sweep

        // 6. CT EMF at corners.
        pass.setPipeline(pipelines.pipelines.computeEmf);
        pass.setBindGroup(0, this._emfBG());
        pass.dispatchWorkgroups(gN1, gN1, 1);   // (N+1)²

        // 7. Weighted update for cell-centered state — interior only.
        pass.setPipeline(pipelines.pipelines.updateConservedWeighted);
        pass.setBindGroup(0, this._updateUBG(stageBuf, otherU0, otherU1, dstU0, dstU1));
        pass.dispatchWorkgroups(gInterior, gInterior, 1);

        // 8. Weighted update for face-centered B — covers (N+1)² combined.
        pass.setPipeline(pipelines.pipelines.updateBWeighted);
        pass.setBindGroup(0, this._updateBBG(stageBuf, otherBx, otherBy, dstBx, dstBy));
        pass.dispatchWorkgroups(gN1, gN1, 1);

        // 9a. Re-fill ghost cells on the DESTINATION buffer so the
        // resistive Laplacian stencil sees BC-consistent values. The
        // update-conserved/update-b kernels only wrote interior cells/
        // faces; without this refill, the dst-buffer ghosts would carry
        // stale data from previous steps (the buffer ping-pongs).
        pass.setPipeline(pipelines.pipelines.applyBcs);
        pass.setBindGroup(0, this._applyBcsBG(dstU0, dstU1, dstBx, dstBy));
        pass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);

        // 9b. Resistivity (η ∇²B) — adds in-place to dstBx, dstBy, and
        // the Bz component of dstU1. Reads neighbours from the same
        // buffers; within a single dispatch, neighbour reads return
        // pre-write values (WebGPU has no cross-invocation memory
        // ordering inside a dispatch), which gives the correct
        // explicit-Euler diffusion semantics.
        pass.setPipeline(pipelines.pipelines.applyResistivity);
        pass.setBindGroup(0, this._applyResBG(stageBuf, dstBx, dstBy, dstU1));
        pass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);
    }

    step() {
        const { device } = this;
        const b = this.buffers;

        const encoder = device.createCommandEncoder({ label: 'plasma.step.enc' });

        // Pass 1: compute dt from U(n).
        this._encodeComputeDt(encoder);

        // Pass 2: three RK3 SSP stages.
        const pass = encoder.beginComputePass({ label: 'plasma.rk3' });

        // Stage 1: U(1) = U(n) + dt · L(U(n))
        this._encodeStage(pass,
            b.U0_n, b.U1_n, b.Bx_n, b.By_n,
            b.U0_n, b.U1_n, b.Bx_n, b.By_n,
            b.U0_1, b.U1_1, b.Bx_1, b.By_1,
            b.stage_1);

        // Stage 2: U(2) = 3/4 U(n) + 1/4 U(1) + 1/4 dt · L(U(1))
        this._encodeStage(pass,
            b.U0_1, b.U1_1, b.Bx_1, b.By_1,
            b.U0_1, b.U1_1, b.Bx_1, b.By_1,
            b.U0_2, b.U1_2, b.Bx_2, b.By_2,
            b.stage_2);

        // Stage 3: U(n+1) = 1/3 U(n) + 2/3 U(2) + 2/3 dt · L(U(2))
        this._encodeStage(pass,
            b.U0_2,    b.U1_2,    b.Bx_2,    b.By_2,
            b.U0_2,    b.U1_2,    b.Bx_2,    b.By_2,
            b.U0_next, b.U1_next, b.Bx_next, b.By_next,
            b.stage_3);

        pass.end();
        device.queue.submit([encoder.finish()]);

        b.swap();
        this.stepCount += 1;
        // simTime is advanced by the actual GPU-computed dt, which we
        // don't read back here; UI presents `stepCount * estimated_dt`
        // when needed, and stats-display.js does its own readback of
        // the dt buffer for accurate clock reporting.
    }

    render() {
        this._pushUniforms();
        this.renderer.render();
    }
}

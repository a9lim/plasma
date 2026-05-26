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
    LIC_INTENSITY_DEFAULT, LIC_DRIFT_X, LIC_DRIFT_Y,
} from './config.js';

const WG = WORKGROUP;

export class Sim {
    constructor(device, context, format, opts = {}) {
        this.device  = device;
        this.context = context;
        this.format  = format;
        this.hasTimestamp = !!opts.hasTimestamp;

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
        // Per-preset η floor coefficient — η_min = etaFloorCoeff · dx
        // enforces a grid magnetic Reynolds limit so current-sheet
        // thinning can't outrun dissipation and NaN-cascade. 0 disables
        // the floor (1D shocks + Harris's thick-sheet geometry don't
        // need it). Set by loadPreset.
        this.etaFloorCoeff = 0;

        this.stepCount = 0;
        this.simTime   = 0;
        this.lastDt    = 0;

        // UI integration state — owned by Sim so save/load can capture it.
        this.running     = true;
        this.speedScale  = 1;
        this.presetName  = 'orszag-tang';

        // LIC state — phase animates per render frame in wall-clock time;
        // intensity is UI-controlled. Drift constants stay fixed for now
        // (locked decisions: ~0.5 cells/sec horizontal).
        this.licIntensity = LIC_INTENSITY_DEFAULT;
        this.licDriftX    = LIC_DRIFT_X;
        this.licDriftY    = LIC_DRIFT_Y;
        this.licPhase     = 0;
        this._lastRenderTime = 0;

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

        // GPU-step timing infrastructure. Allocated only when the adapter
        // advertises `timestamp-query` (see device.js). The query set holds
        // two 64-bit timestamps (start of stage 1 + end of stage 3 of the
        // RK3 compute pass). `_tsResolve` is the buffer we resolve the
        // query values into (16 B, QUERY_RESOLVE | COPY_SRC) — the readback
        // staging buffer comes from StatsDisplay's ReadbackPool so we don't
        // hold a second one here.
        this._tsQuerySet = null;
        this._tsResolve  = null;
        this._tsLastMs   = 0;  // last successfully read GPU step time, in ms
        if (this.hasTimestamp) {
            this._tsQuerySet = this.device.createQuerySet({
                label: 'plasma.timestampQuerySet',
                type:  'timestamp',
                count: 2,
            });
            this._tsResolve = this.device.createBuffer({
                label: 'plasma.timestampResolve',
                size:  16, // 2 × u64
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            });
        }

        // Phase 4 default preset still Orszag-Tang for smoke check;
        // Phase 5 wires the dropdown for Harris/etc.
        this.loadPreset(makeOrszagTangPreset(this.n));
        this.buffers.uploadLUT(VIRIDIS);

        const seed = new Float32Array([1e-4]);
        this.device.queue.writeBuffer(this.buffers.dt, 0, seed.buffer);

        this._pushUniforms();
        this._pushLicUniforms();
        this.buffers.pushBC(this.bcConfig);

        // Pre-bake all step/render bind groups against the current
        // PlasmaBuffers identity. Rebuilt by setResolution() when buffers
        // are re-instantiated.
        this._buildBindGroupCache();
    }

    loadPreset(preset) {
        this.gamma   = preset.gamma   ?? this.gamma;
        this.viewMin = preset.viewMin ?? this.viewMin;
        this.viewMax = preset.viewMax ?? this.viewMax;
        // Floor coeff must be set BEFORE dx (in case the preset changes
        // domainLength) and BEFORE eta is clamped.
        this.etaFloorCoeff = preset.etaFloorCoeff ?? 0;
        if (preset.domainLength) {
            this.domainLength = preset.domainLength;
            this.dx = preset.domainLength / this.n;
        }
        // Apply preset's preferred η, then floor to the grid Reynolds limit.
        // For presets without a floor (etaFloorCoeff = 0) this is a no-op
        // and preset.eta (often 0 = ideal MHD) passes through unchanged.
        this.eta = Math.max(preset.eta ?? this.eta, this.getEtaMin());
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

    /**
     * Smallest stable resistivity at the current preset + resolution.
     * Returns 0 if the active preset doesn't declare a floor.
     *
     * Grid magnetic Reynolds criterion: η ≳ C · v_char · dx. The
     * preset's etaFloorCoeff bakes in the C · v_char product (calibrated
     * empirically per preset); the dx factor adapts the floor to the
     * current grid resolution automatically.
     */
    getEtaMin() {
        return (this.etaFloorCoeff ?? 0) * this.dx;
    }

    /**
     * Update explicit resistivity. UI typically passes 0 for "ideal", but
     * for presets with an active η floor (currently Orszag-Tang) the
     * value is clamped up to getEtaMin() to keep current-sheet thinning
     * within the grid's resolvable scales.
     */
    setEta(eta) {
        this.eta = Math.max(eta, this.getEtaMin());
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

    setCFL(cfl)         { this.cfl = cfl; this._pushUniforms(); }
    setGamma(g)         { this.gamma = g; this._pushUniforms(); }
    setPressureFloor(p) { this.pressureFloor = p; this._pushUniforms(); }

    setRunning(r)       { this.running = !!r; }
    setSpeedScale(s)    { this.speedScale = s; }

    /**
     * LIC modulation strength. 0 = colormap passes through unchanged,
     * 1 = full ±50% luminance swing, 2 = double the swing (slider clamps).
     */
    setLicIntensity(v) {
        this.licIntensity = Math.max(0, Math.min(2, +v || 0));
        this._pushLicUniforms();
    }

    /** Drift speed in noise-pixels/sec — drift_x, drift_y are direction × speed. */
    setLicDrift(dx, dy) {
        this.licDriftX = +dx || 0;
        this.licDriftY = +dy || 0;
        this._pushLicUniforms();
    }

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
        this._pushLicUniforms();
        // Re-load whichever preset is current.
        this.setPreset(this.presetName);
        // All GPU buffers changed identity — rebuild the bind-group cache
        // (and via _buildBindGroupCache, the renderer/LIC side caches).
        this._buildBindGroupCache();
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
            licIntensity: this.licIntensity,
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
        if (obj.licIntensity !== undefined) this.setLicIntensity(obj.licIntensity);
    }

    _pushUniforms() {
        this.buffers.pushUniforms({
            dx: this.dx,
            gamma: this.gamma,
            viewMin: this.viewMin,
            viewMax: this.viewMax,
            gridN: this.n,
            viewMode: this.viewMode,
            eta: this.eta,
            cfl: this.cfl,
            pressureFloor: this.pressureFloor,
        });
    }

    /**
     * Push render-pace LIC state (phase / intensity / drift). Called
     * every render frame and on intensity / drift slider changes. The
     * main physics-state uniform buffer is NOT touched.
     */
    _pushLicUniforms() {
        this.buffers.pushLicUniforms({
            licPhase:     this.licPhase,
            licIntensity: this.licIntensity,
            licDriftX:    this.licDriftX,
            licDriftY:    this.licDriftY,
        });
    }

    // ── Bind-group builders ─────────────────────────────────────────
    //
    // Side-naming convention: when a builder name says "Side", it means
    // the bind group references at least one of the four ping-pong
    // handles (U0_n, U1_n, Bx_n, By_n) or their twin (U0_next, U1_next,
    // Bx_next, By_next). The cache keeps two copies (a/b) for each such
    // bind group, indexed by the value of `buffers._side` AT THE TIME
    // the bind group is consumed. Side-independent bind groups live as
    // single entries.
    //
    // All builders take explicit buffer args (no implicit `this.buffers`
    // reads) so the cache builder can pin them to the A-side or B-side
    // handles deterministically.

    _dtBG(U0_n, U1_n, Bx_n, By_n) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.computeDt.bg',
            layout: this.pipelines.layouts.dt,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0_n } },
                { binding: 2, resource: { buffer: U1_n } },
                { binding: 3, resource: { buffer: Bx_n } },
                { binding: 4, resource: { buffer: By_n } },
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
                { binding: 0, resource: { buffer: b.uniform } },
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
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: stageBuf } },
                { binding: 2, resource: { buffer: Bx } },
                { binding: 3, resource: { buffer: By } },
                { binding: 4, resource: { buffer: U1_out } },
                { binding: 5, resource: { buffer: b.dt } },
                // Snapshot buffers — see apply-resistivity.wgsl header.
                // The snapshot dispatch writes these; the main dispatch
                // reads them (race-free 5-point Laplacian).
                { binding: 6, resource: { buffer: b.Bx_res_snap } },
                { binding: 7, resource: { buffer: b.By_res_snap } },
                { binding: 8, resource: { buffer: b.U1_res_snap } },
            ],
        });
    }

    // Conservation diagnostics reduction (Session 8). Reads the
    // POST-step destination buffers — the bind group is built per
    // (stage 3 dst) side, since stage 3's dst is the next step's U_n.
    // Output: `cons_out` (7 f32 + pad). One bind group per pipeline.
    _conservationTileBG(U0, U1, Bx, By) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.conservationTile.bg',
            layout: this.pipelines.layouts.conservationTile,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: b.cons_tile_partials } },
            ],
        });
    }

    _conservationFinalizeBG() {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.conservationFinalize.bg',
            layout: this.pipelines.layouts.conservationFinalize,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.cons_tile_partials } },
                { binding: 2, resource: { buffer: b.cons_out } },
            ],
        });
    }

    _energyFloorBG(U0_out, U1_out, Bx, By) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.energyFloor.bg',
            layout: this.pipelines.layouts.energyFloor,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0_out } },
                { binding: 2, resource: { buffer: U1_out } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
            ],
        });
    }

    _reconstructBG(axis, U0, U1, Bx, By) {
        const b = this.buffers;
        const sweepBuf = (axis === 0) ? b.sweepDir_x : b.sweepDir_y;
        const eL0 = (axis === 0) ? b.edge_l_x_0 : b.edge_l_y_0;
        const eL1 = (axis === 0) ? b.edge_l_x_1 : b.edge_l_y_1;
        const eR0 = (axis === 0) ? b.edge_r_x_0 : b.edge_r_y_0;
        const eR1 = (axis === 0) ? b.edge_r_x_1 : b.edge_r_y_1;
        return this.device.createBindGroup({
            label: `plasma.reconstruct.axis${axis}.bg`,
            layout: this.pipelines.layouts.reconstruct,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: eL0 } },
                { binding: 6, resource: { buffer: eL1 } },
                { binding: 7, resource: { buffer: eR0 } },
                { binding: 8, resource: { buffer: eR1 } },
                { binding: 9, resource: { buffer: sweepBuf } },
            ],
        });
    }

    _riemannBG(axis, U0, U1, Bx, By) {
        const b = this.buffers;
        const sweepBuf = (axis === 0) ? b.sweepDir_x : b.sweepDir_y;
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
                { binding: 0,  resource: { buffer: b.uniform } },
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
                { binding: 11, resource: { buffer: sweepBuf } },
            ],
        });
    }

    // EMF bind group is now side- and stage-dependent — Gardiner-Stone
    // 2005 upwind CT needs the cell-centered Ez at the four cells
    // around each corner, computed from the SRC U0 and SRC face B
    // (whichever buffer the current stage's PPM read from). Caller
    // passes the stage's (U0_src, Bx_src, By_src) so the cache builds
    // A/B variants per stage.
    _emfBG(U0_src, Bx_src, By_src) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.emf.bg',
            layout: this.pipelines.layouts.emf,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.flux_x_1 } },
                { binding: 2, resource: { buffer: b.flux_y_1 } },
                { binding: 3, resource: { buffer: b.Ez_edge } },
                { binding: 4, resource: { buffer: U0_src } },
                { binding: 5, resource: { buffer: Bx_src } },
                { binding: 6, resource: { buffer: By_src } },
            ],
        });
    }

    _updateUBG(stageBuf, U0_n, U1_n, U0_other, U1_other, U0_out, U1_out) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.updateU.bg',
            layout: this.pipelines.layouts.updateU,
            entries: [
                { binding: 0,  resource: { buffer: b.uniform } },
                { binding: 1,  resource: { buffer: stageBuf } },
                { binding: 2,  resource: { buffer: U0_n } },
                { binding: 3,  resource: { buffer: U1_n } },
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

    _updateBBG(stageBuf, Bx_n, By_n, Bx_other, By_other, Bx_out, By_out) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.updateB.bg',
            layout: this.pipelines.layouts.updateB,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: stageBuf } },
                { binding: 2, resource: { buffer: Bx_n } },
                { binding: 3, resource: { buffer: By_n } },
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
     * Pre-bake every bind group used by `step()` and `_encodeComputeDt`.
     * Called from `init()` and `setResolution()` — anywhere `this.buffers`
     * or the renderers change identity. The cache holds one entry per
     * (stage × kernel × side); side-independent kernels use a single
     * entry under `entry.bg`, side-dependent kernels use `entry.a`/`b`.
     *
     * Side semantics: A means `_side === 'a'` (U0_n = U0_a, U0_next = U0_b).
     * B is the opposite. `swap()` flips `_side` after each step, so
     * lookups happen against the CURRENT side at encode time.
     *
     * What flips per side:
     *   - The four primary handles: U0_n, U1_n, Bx_n, By_n.
     *   - The four destination handles: U0_next, U1_next, Bx_next, By_next.
     * Everything else (U0_1, U1_1, U0_2, U1_2, fluxes, edges, Bx_{1,2},
     * By_{1,2}, snap buffers, dt, uniforms, sweepDir, licUniform, LUT,
     * noise, field, colored, lic_out) is identity-stable across swaps.
     *
     * Subtlety the audit caught: update-conserved/update-b ALWAYS bind
     * U_n / Bx_n / By_n (the SSP "blend-against-U(n)" reference), even
     * in stages 2 and 3 where the src state has moved on to U_1 / U_2.
     * So updateU and updateB at EVERY stage are side-dependent. Same
     * for compute-dt (reads U_n).
     */
    _buildBindGroupCache() {
        const b = this.buffers;
        // Helper: build per-side handles for a hypothetical "_side = which"
        // configuration without mutating buffers._side. Mirrors the swap
        // logic but returns a struct.
        const handles = (which) => (which === 'a')
            ? {
                U0_n: b.U0_a, U1_n: b.U1_a, Bx_n: b.Bx_a, By_n: b.By_a,
                U0_next: b.U0_b, U1_next: b.U1_b, Bx_next: b.Bx_b, By_next: b.By_b,
            }
            : {
                U0_n: b.U0_b, U1_n: b.U1_b, Bx_n: b.Bx_b, By_n: b.By_b,
                U0_next: b.U0_a, U1_next: b.U1_a, Bx_next: b.Bx_a, By_next: b.By_a,
            };
        const sides = { a: handles('a'), b: handles('b') };

        // Build a stage's bind groups for ONE side. Caller invokes twice
        // (a/b) and stores both. Side-INDEPENDENT entries are duplicated
        // (cheap — same handles either way) so the lookup can be uniform.
        const buildStage = (stageIdx, h) => {
            // stage_1: src = U_n, other = U_n, dst = U_1
            // stage_2: src = U_1, other = U_1, dst = U_2
            // stage_3: src = U_2, other = U_2, dst = U_next
            let src, other, dst, stageBuf;
            if (stageIdx === 1) {
                src   = { U0: h.U0_n, U1: h.U1_n, Bx: h.Bx_n, By: h.By_n };
                other = { U0: h.U0_n, U1: h.U1_n, Bx: h.Bx_n, By: h.By_n };
                dst   = { U0: b.U0_1, U1: b.U1_1, Bx: b.Bx_1, By: b.By_1 };
                stageBuf = b.stage_1;
            } else if (stageIdx === 2) {
                src   = { U0: b.U0_1, U1: b.U1_1, Bx: b.Bx_1, By: b.By_1 };
                other = { U0: b.U0_1, U1: b.U1_1, Bx: b.Bx_1, By: b.By_1 };
                dst   = { U0: b.U0_2, U1: b.U1_2, Bx: b.Bx_2, By: b.By_2 };
                stageBuf = b.stage_2;
            } else {
                src   = { U0: b.U0_2,    U1: b.U1_2,    Bx: b.Bx_2,    By: b.By_2    };
                other = { U0: b.U0_2,    U1: b.U1_2,    Bx: b.Bx_2,    By: b.By_2    };
                dst   = { U0: h.U0_next, U1: h.U1_next, Bx: h.Bx_next, By: h.By_next };
                stageBuf = b.stage_3;
            }
            return {
                applyBcsSrc:    this._applyBcsBG(src.U0, src.U1, src.Bx, src.By),
                reconstructX:   this._reconstructBG(0, src.U0, src.U1, src.Bx, src.By),
                reconstructY:   this._reconstructBG(1, src.U0, src.U1, src.Bx, src.By),
                riemannX:       this._riemannBG(0, src.U0, src.U1, src.Bx, src.By),
                riemannY:       this._riemannBG(1, src.U0, src.U1, src.Bx, src.By),
                emf:            this._emfBG(src.U0, src.Bx, src.By),
                updateU:        this._updateUBG(stageBuf,
                                                h.U0_n, h.U1_n,
                                                other.U0, other.U1,
                                                dst.U0, dst.U1),
                energyFloor:    this._energyFloorBG(dst.U0, dst.U1, src.Bx, src.By),
                updateB:        this._updateBBG(stageBuf,
                                                h.Bx_n, h.By_n,
                                                other.Bx, other.By,
                                                dst.Bx, dst.By),
                applyBcsDst:    this._applyBcsBG(dst.U0, dst.U1, dst.Bx, dst.By),
                applyRes:       this._applyResBG(stageBuf, dst.Bx, dst.By, dst.U1),
            };
        };

        // For each stage, build a/b variants. The variants share most
        // entries (only the side-dependent ones differ) but we build the
        // full struct twice for lookup uniformity. The overhead is ~50 µs
        // × 22 = ~1.1 ms one-time at startup — negligible.
        this._bgCache = {
            dt: {
                a: this._dtBG(sides.a.U0_n, sides.a.U1_n, sides.a.Bx_n, sides.a.By_n),
                b: this._dtBG(sides.b.U0_n, sides.b.U1_n, sides.b.Bx_n, sides.b.By_n),
            },
            stage1: { a: buildStage(1, sides.a), b: buildStage(1, sides.b) },
            stage2: { a: buildStage(2, sides.a), b: buildStage(2, sides.b) },
            stage3: { a: buildStage(3, sides.a), b: buildStage(3, sides.b) },
            // Conservation reduction reads the POST-step state — stage
            // 3's dst, which is U_next at encode time. Side-dependent
            // because U_next flips with the ping-pong.
            consTile: {
                a: this._conservationTileBG(sides.a.U0_next, sides.a.U1_next, sides.a.Bx_next, sides.a.By_next),
                b: this._conservationTileBG(sides.b.U0_next, sides.b.U1_next, sides.b.Bx_next, sides.b.By_next),
            },
            consFinalize: this._conservationFinalizeBG(),
        };

        // Renderer / LIC bind groups also depend on the primary handles
        // (view-field reads U_n + face B; lic-advect reads face B). Have
        // the renderer rebuild its A/B cache against the new buffers.
        if (this.renderer)        this.renderer.rebuildSideCache();
        if (this.renderer && this.renderer.lic) this.renderer.lic.rebuildSideCache();
    }

    _encodeComputeDt(encoder) {
        const { pipelines } = this;
        const groupsInterior = Math.ceil(this.n / WG);
        const bg = this._bgCache.dt[this.buffers._side];

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
     * Encode one RK3 stage from the pre-baked bind-group cache.
     *
     * `stageIdx` ∈ {1, 2, 3}; `side` ∈ {'a', 'b'}. The encoder also
     * branches on `this.eta` to skip the resistive triad (9a/9b/9c) at
     * ideal MHD (Change #7) and to skip the leading apply-bcs in stages
     * 2 and 3 when η > 0, since stage k's 9a already filled stage k+1's
     * src-ghosts (Change #6).
     *
     * Asymmetry between paths:
     *   η = 0: every stage runs step 1 (PPM ghosts) + the 7-pass core,
     *          no resistive triad. Stage k+1's step 1 is the only place
     *          stage k+1's src-ghosts get filled.
     *   η > 0: stage 1 runs step 1 (fills U_n's ghosts before PPM). 9a
     *          at the end of stages 1 and 2 fills the dst-ghosts, which
     *          ARE the next stage's src-ghosts (dst_k === src_{k+1} for
     *          k = 1, 2). So stages 2 and 3 skip step 1. Stage 3's own
     *          9a is still needed (for its 9b/9c) but its ghost output
     *          is discarded at the next step's step 1.
     */
    _encodeStage(pass, stageIdx, side) {
        const { pipelines } = this;
        const N         = this.n;
        const gInterior = Math.ceil(N / WG);
        const gN1       = Math.ceil((N + 1) / WG);
        const gN2       = Math.ceil((N + 2) / WG);
        // apply-bcs covers (N_total+1)² (all ghost strips + boundary faces).
        const gTotalP1  = Math.ceil((this.n_total + 1) / WG);
        // apply-resistivity now dispatches over the Laplacian read footprint
        // (interior + 1-cell ghost margin + extra face index) = (N+3)².
        // Shader shifts the index by (ghost - 1u) — see Change #10 in this
        // round and the snapshot/main entry-point comments.
        const gResis    = Math.ceil((N + 3) / WG);

        const bgs = this._bgCache[`stage${stageIdx}`][side];

        // 1. apply-bcs on src — fills ghosts for the upcoming PPM stencil.
        //    Skipped in stages 2 and 3 when η > 0: stage k's 9a fed the
        //    dst-ghosts (which alias stage k+1's src-ghosts for k=1,2).
        if (this.eta <= 0 || stageIdx === 1) {
            pass.setPipeline(pipelines.pipelines.applyBcs);
            pass.setBindGroup(0, bgs.applyBcsSrc);
            pass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);
        }

        // 2-5. PPM + Riemann, both axes.
        pass.setPipeline(pipelines.pipelines.reconstructPpm);
        pass.setBindGroup(0, bgs.reconstructX);
        pass.dispatchWorkgroups(gN2, gN2, 1);

        pass.setPipeline(pipelines.pipelines.riemannHlld);
        pass.setBindGroup(0, bgs.riemannX);
        pass.dispatchWorkgroups(gN1, gN2, 1);   // (N+1) × (N+2) for x-sweep

        pass.setPipeline(pipelines.pipelines.reconstructPpm);
        pass.setBindGroup(0, bgs.reconstructY);
        pass.dispatchWorkgroups(gN2, gN2, 1);

        pass.setPipeline(pipelines.pipelines.riemannHlld);
        pass.setBindGroup(0, bgs.riemannY);
        pass.dispatchWorkgroups(gN2, gN1, 1);   // (N+2) × (N+1) for y-sweep

        // 6. CT EMF at corners.
        pass.setPipeline(pipelines.pipelines.computeEmf);
        pass.setBindGroup(0, bgs.emf);
        pass.dispatchWorkgroups(gN1, gN1, 1);   // (N+1)²

        // 7. Weighted update for cell-centered state — interior only.
        pass.setPipeline(pipelines.pipelines.updateConservedWeighted);
        pass.setBindGroup(0, bgs.updateU);
        pass.dispatchWorkgroups(gInterior, gInterior, 1);

        // 7.5 Energy floor with magnetic-pressure correction. Uses the
        // SOURCE face B (srcBx/srcBy) — at this point update-b hasn't
        // written dstBx/dstBy for this stage yet. The floor is
        // E_min = KE + ½|B|² + p_floor/(γ−1); without the ½|B|² term
        // (omitted by update-conserved due to its 10-binding cap) thin
        // current sheets can clamp E below the magnetic-pressure
        // contribution. Reads U0_out, Bx/By; clamps U1_out.E in place.
        pass.setPipeline(pipelines.pipelines.energyFloor);
        pass.setBindGroup(0, bgs.energyFloor);
        pass.dispatchWorkgroups(gInterior, gInterior, 1);

        // 8. Weighted update for face-centered B — covers (N+1)² combined.
        pass.setPipeline(pipelines.pipelines.updateBWeighted);
        pass.setBindGroup(0, bgs.updateB);
        pass.dispatchWorkgroups(gN1, gN1, 1);

        // 9a/9b/9c — resistivity triad. Skipped entirely at η = 0
        // (Change #7): the shaders no-op internally but we avoid issuing
        // 6 dispatches per step on every ideal-MHD preset.
        if (this.eta > 0) {
            // 9a. Re-fill ghost cells on the DESTINATION buffer so the
            // resistive Laplacian stencil sees BC-consistent values. The
            // update-conserved/update-b kernels only wrote interior
            // cells/faces; without this refill, the dst-buffer ghosts
            // would carry stale data from previous steps.
            pass.setPipeline(pipelines.pipelines.applyBcs);
            pass.setBindGroup(0, bgs.applyBcsDst);
            pass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);

            // 9b. Snapshot dst → snap. Race-free per-cell copy; required
            // because the next pass's 5-point Laplacian reads neighbours,
            // and WebGPU has no cross-invocation memory ordering inside
            // a dispatch — so reading from the same buffer being written
            // racily picks up post-write values at workgroup-tile bounds.
            // Dispatched over (N+3)² with (ghost-1) shift (Change #10).
            pass.setPipeline(pipelines.pipelines.applyResSnapshot);
            pass.setBindGroup(0, bgs.applyRes);
            pass.dispatchWorkgroups(gResis, gResis, 1);

            // 9c. Resistivity (η ∇²B) — reads snap, writes dst. Same
            // dispatch shape as 9b; interior-bounds checks inside the
            // shader filter writes to interior cells/faces only.
            pass.setPipeline(pipelines.pipelines.applyResistivity);
            pass.setBindGroup(0, bgs.applyRes);
            pass.dispatchWorkgroups(gResis, gResis, 1);
        }
    }

    step() {
        const { device } = this;
        const b = this.buffers;
        const side = b._side;  // pinned for this step's encoding

        const encoder = device.createCommandEncoder({ label: 'plasma.step.enc' });

        // Pass 1: compute dt from U(n).
        this._encodeComputeDt(encoder);

        // Pass 2: three RK3 SSP stages. Optionally instrumented with two
        // timestamp queries (start of stage 1, end of stage 3) when the
        // adapter supports the `timestamp-query` feature.
        const passDesc = { label: 'plasma.rk3' };
        if (this._tsQuerySet) {
            passDesc.timestampWrites = {
                querySet: this._tsQuerySet,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex:       1,
            };
        }
        const pass = encoder.beginComputePass(passDesc);

        // Stage 1: U(1) = U(n) + dt · L(U(n))
        this._encodeStage(pass, 1, side);

        // Stage 2: U(2) = 3/4 U(n) + 1/4 U(1) + 1/4 dt · L(U(1))
        this._encodeStage(pass, 2, side);

        // Stage 3: U(n+1) = 1/3 U(n) + 2/3 U(2) + 2/3 dt · L(U(2))
        this._encodeStage(pass, 3, side);

        pass.end();

        // Pass 3: conservation diagnostics reduction over the just-
        // written destination state (stage 3 dst === U_next at this
        // point, before the swap below). Two dispatches in one pass —
        // per-tile partials, then a single-workgroup finalize. Output
        // (cons_out) is pulled by stats-display via the existing
        // readback batch at its own cadence; we just write it here.
        // Folding it inside the RK3 timestamp pass would corrupt the
        // GPU-step measurement, so it gets its own pass.
        {
            const consPass = encoder.beginComputePass({ label: 'plasma.conservationReduce' });
            const tilesPerAxis = Math.ceil(this.n / WG);
            consPass.setPipeline(this.pipelines.pipelines.conservationTile);
            consPass.setBindGroup(0, this._bgCache.consTile[side]);
            consPass.dispatchWorkgroups(tilesPerAxis, tilesPerAxis, 1);
            consPass.setPipeline(this.pipelines.pipelines.conservationFinalize);
            consPass.setBindGroup(0, this._bgCache.consFinalize);
            consPass.dispatchWorkgroups(1, 1, 1);
            consPass.end();
        }

        // Resolve the timestamp queries into the GPU-side resolve buffer.
        // Readback (mapAsync) is owned by StatsDisplay so we don't block
        // the hot path here; it's pull-driven at whatever cadence Stats
        // is running.
        if (this._tsQuerySet) {
            encoder.resolveQuerySet(this._tsQuerySet, 0, 2, this._tsResolve, 0);
        }

        device.queue.submit([encoder.finish()]);

        b.swap();
        this.stepCount += 1;
        // simTime is advanced by the actual GPU-computed dt, which we
        // don't read back here; UI presents `stepCount * estimated_dt`
        // when needed, and stats-display.js does its own readback of
        // the dt buffer for accurate clock reporting.
    }

    render() {
        // Advance LIC phase by wall-clock dt (not simulation dt) — the
        // drift is purely visual. Phase is in seconds; the shader
        // multiplies by lic_drift_{x,y} (noise-pixels/sec) to get a
        // sample offset in noise-pixel coords directly. On the very
        // first render, _lastRenderTime is 0 so we skip the increment.
        const now = performance.now();
        if (this._lastRenderTime > 0) {
            const dt = Math.min(0.1, (now - this._lastRenderTime) / 1000);
            this.licPhase += dt;
        }
        this._lastRenderTime = now;
        // Only the small (16 B) LIC uniform changes per frame now; the
        // main 64 B physics uniform is rewritten only on parameter
        // changes (setEta / setCFL / setViewMode / setGamma / loadPreset).
        this._pushLicUniforms();
        this.renderer.render();
    }
}

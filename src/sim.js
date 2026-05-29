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
 * Submit structure: one primary physics command buffer per step, plus an
 * optional tiny dt-feedback readback at a reduced cadence unless source/RKL2
 * feedback needs per-step reductions.
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
import { ReadbackPool, readbackSlice } from './gpu/readback.js';
import { makeSandboxPreset, PRESETS } from './presets.js';
import { VIRIDIS } from './colormaps.js';
import { MICRO_TRANSPORT_MAX_SCALE } from './microphysics.js';
import {
    GRID_N, GHOST_WIDTH, DOMAIN_LENGTH, GAMMA_DEFAULT, WORKGROUP,
    VIEW_JZ, ETA_DEFAULT, BC_PERIODIC, CFL, PRESSURE_FLOOR,
    LIC_INTENSITY_DEFAULT, LIC_DRIFT_X, LIC_DRIFT_Y, DT_MAX,
    FLAG_COOLING, FLAG_GRAVITY_EXT, FLAG_GRAVITY_SELF, FLAG_HALL, FLAG_CONDUCTION,
    FLAG_AMBIPOLAR, FLAG_BIERMANN, FLAG_VISCOSITY, FLAG_GEOMETRY,
    FLAG_SPONGE, FLAG_HEATING, FLAG_RADIATION, FLAG_ELECTRON_INERTIA,
    EXTENDED_SOURCE_FLAGS,
    GEOMETRY_CYLINDRICAL,
    GRAVITY_SOLVER_MULTIGRID, GRAVITY_SOLVER_JACOBI,
    DEFAULT_PHYSICS_STATE, SOURCE_SUBSTEPS_HARD_MAX, STS_COEFFS_MAX_S, VIEW_RANGES,
    ETA_ANOM_JCRIT_DEFAULT,
} from './config.js';

const WG = WORKGROUP;
const DT_READBACK_IDLE_STRIDE = 16;

function defaultBCConfig() {
    return {
        modeN: BC_PERIODIC, modeS: BC_PERIODIC,
        modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        driven: { rho: 1, vx: 0, vy: 0, vz: 0, bx: 0, by: 0, bz: 0, p: 1 },
    };
}

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
        // Anomalous resistivity coefficients. α = 0 means the constant-η
        // baseline (no boost); α > 0 enables the Birn 2001 GEM-style
        // η(|J|) = η_0 + α · max(0, |J|/J_crit − 1)² closure. J_crit is
        // in code-units (J_z is computed from face B via central diff).
        this.etaAnomAlpha = 0;
        this.etaAnomJcrit = ETA_ANOM_JCRIT_DEFAULT;
        // Per-preset η floor coefficient — η_min = etaFloorCoeff · dx
        // enforces a grid magnetic Reynolds limit so current-sheet
        // thinning can't outrun dissipation and NaN-cascade. 0 disables
        // the floor (1D shocks + Harris's thick-sheet geometry don't
        // need it). Set by loadPreset.
        this.etaFloorCoeff = 0;

        this.stepCount = 0;
        this.simTime   = 0;
        this.lastDt    = 0;

        // RKL2/source feedback state — see _encodeResistivitySuperStep and
        // _prepareSourceSubsteps. GPU reductions arrive asynchronously, so
        // cold/invalid hints are treated conservatively by the sizing paths
        // rather than optimistically taking a single source substep.
        this._lastDtHyp        = 1.0e-4;
        this._lastDtParabolic  = 1.0e30;
        this._lastEtaMax       = 0;
        this._lastEtaMaxValid  = false;
        this._lastHallRateMax = 0;
        this._lastHallRateValid = false;
        this._lastHallSubsteps = 1;
        this._lastCondRateMax = 0;
        this._lastCondRateValid = false;
        this._lastCondSubsteps = 1;
        this._lastViscSubsteps = 1;
        this._lastNonidealSubsteps = 1;
        this._lastRadiationSubsteps = 1;
        this._lastSuperStepS   = 1;
        this._dtReadbackBusy   = false;
        this._dtReadbackPool   = null;  // lazy-init in init()
        this._pendingTimeSteps = 0;
        this._bufferGeneration = 0;

        // ── Extended physics state. Canonical verification presets run the
        // ideal/resistive MHD core with numerical guards only; cooling,
        // gravity, conduction, and Hall are opt-in through preset.physics or
        // the explicit setters.
        this._applyPhysicsConfig();

        // UI integration state — owned by Sim so save/load can capture it.
        this.running     = true;
        this.speedScale  = 1;
        this.presetName  = 'sandbox';

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
        this.bcConfig = defaultBCConfig();
    }

    async init() {
        this.pipelines = await createPipelines(this.device, this.format);
        this.buffers   = new PlasmaBuffers(this.device, this.n);
        this.renderer  = new PlasmaRenderer(this.device, this.context, this.pipelines, this.buffers);
        this._dtReadbackPool = new ReadbackPool(this.device);

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
        this.loadPreset(makeSandboxPreset(this.n));
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

    _applyPhysicsConfig(physics = {}) {
        Object.assign(this, DEFAULT_PHYSICS_STATE);
        if (!physics) return;
        const flags = physics.physicsFlags ?? physics.flags;
        if (flags !== undefined) this.physicsFlags = flags >>> 0;
        if (physics.emfMode !== undefined)             this.emfMode = physics.emfMode | 0;
        if (physics.hallDi !== undefined)              this.hallDi = +physics.hallDi || 0;
        if (physics.hallSubstepsMax !== undefined)     this.hallSubstepsMax = Math.max(1, physics.hallSubstepsMax | 0);
        if (physics.coolingLambda0 !== undefined)      this.coolingLambda0 = Math.max(0, +physics.coolingLambda0 || 0);
        if (physics.coolingTFloor !== undefined)       this.coolingTFloor = Math.max(0, +physics.coolingTFloor || 0);
        if (physics.coolingTRef !== undefined)         this.coolingTRef = Math.max(1e-30, +physics.coolingTRef || 1);
        if (physics.coolingCurveMode !== undefined)    this.coolingCurveMode = Math.max(0, physics.coolingCurveMode | 0);
        if (physics.conductionKappa !== undefined)     this.conductionKappa = Math.max(0, +physics.conductionKappa || 0);
        if (physics.conductionIsoFrac !== undefined)   this.conductionIsoFrac = Math.min(1, Math.max(0, +physics.conductionIsoFrac || 0));
        if (physics.conductionSatFrac !== undefined)   this.conductionSatFrac = Math.max(0, +physics.conductionSatFrac || 0);
        if (physics.gravityGx !== undefined)           this.gravityGx = +physics.gravityGx || 0;
        if (physics.gravityGy !== undefined)           this.gravityGy = +physics.gravityGy || 0;
        if (physics.gravityG !== undefined)            this.gravityG = Math.max(0, +physics.gravityG || 0);
        if (physics.gravityPoissonIters !== undefined) this.gravityPoissonIters = Math.max(1, physics.gravityPoissonIters | 0);
        if (physics.gravityBoundaryMode !== undefined) this.gravityBoundaryMode = Math.max(0, physics.gravityBoundaryMode | 0);
        if (physics.gravitySolverMode !== undefined)   this.gravitySolverMode = physics.gravitySolverMode | 0;
        if (physics.hallElectronPressureFrac !== undefined) {
            this.hallElectronPressureFrac = Math.min(1, Math.max(0, +physics.hallElectronPressureFrac || 0));
        }
        if (physics.coolingMetallicity !== undefined) this.coolingMetallicity = Math.max(0, +physics.coolingMetallicity || 0);
        if (physics.heatingGamma0 !== undefined)      this.heatingGamma0 = Math.max(0, +physics.heatingGamma0 || 0);
        if (physics.heatingDensityExp !== undefined)  this.heatingDensityExp = Math.max(0, +physics.heatingDensityExp || 0);
        if (physics.heatingTCut !== undefined)        this.heatingTCut = Math.max(0, +physics.heatingTCut || 0);
        if (physics.ambipolarEta !== undefined)       this.ambipolarEta = Math.max(0, +physics.ambipolarEta || 0);
        if (physics.biermannCoeff !== undefined)      this.biermannCoeff = +physics.biermannCoeff || 0;
        if (physics.neutralFrac !== undefined)        this.neutralFrac = Math.min(1, Math.max(0, +physics.neutralFrac || 0));
        if (physics.ionizationT0 !== undefined)       this.ionizationT0 = Math.max(1e-30, +physics.ionizationT0 || 1);
        if (physics.viscosityNu !== undefined)        this.viscosityNu = Math.max(0, +physics.viscosityNu || 0);
        if (physics.viscosityBulk !== undefined)      this.viscosityBulk = Math.max(0, +physics.viscosityBulk || 0);
        if (physics.viscosityAnisoFrac !== undefined) this.viscosityAnisoFrac = Math.min(1, Math.max(0, +physics.viscosityAnisoFrac || 0));
        if (physics.viscosityShock !== undefined)     this.viscosityShock = Math.max(0, +physics.viscosityShock || 0);
        if (physics.sourceSubstepsMax !== undefined)  this.sourceSubstepsMax = Math.max(1, physics.sourceSubstepsMax | 0);
        if (physics.geometryMode !== undefined)       this.geometryMode = Math.max(0, physics.geometryMode | 0);
        if (physics.geometryRMin !== undefined)       this.geometryRMin = Math.max(0, +physics.geometryRMin || 0);
        if (physics.gravitySoftening !== undefined)   this.gravitySoftening = Math.max(0, +physics.gravitySoftening || 0);
        if (physics.gravityPoissonOmega !== undefined) this.gravityPoissonOmega = Math.min(1.95, Math.max(0.05, +physics.gravityPoissonOmega || 1));
        if (physics.spongeWidth !== undefined)        this.spongeWidth = Math.max(0, +physics.spongeWidth || 0);
        if (physics.spongeStrength !== undefined)     this.spongeStrength = Math.max(0, +physics.spongeStrength || 0);
        if (physics.coolingTableMix !== undefined)    this.coolingTableMix = Math.max(0, +physics.coolingTableMix || 0);
        if (physics.radiationC !== undefined)          this.radiationC = Math.max(0, +physics.radiationC || 0);
        if (physics.radiationKappaAbs !== undefined)   this.radiationKappaAbs = Math.max(0, +physics.radiationKappaAbs || 0);
        if (physics.radiationKappaScat !== undefined)  this.radiationKappaScat = Math.max(0, +physics.radiationKappaScat || 0);
        if (physics.radiationConst !== undefined)      this.radiationConst = Math.max(0, +physics.radiationConst || 0);
        if (physics.radiationFloor !== undefined)      this.radiationFloor = Math.max(0, +physics.radiationFloor || 0);
        if (physics.electronInertiaLength !== undefined)  this.electronInertiaLength = Math.max(0, +physics.electronInertiaLength || 0);
        if (physics.electronInertiaDamping !== undefined) this.electronInertiaDamping = Math.max(0, +physics.electronInertiaDamping || 0);
        this._invalidateSourceRateHints();
        this._invalidateEtaMaxHint();
    }

    loadPreset(preset) {
        this.gamma   = preset.gamma   ?? this.gamma;
        if (preset.viewMode !== undefined) this.viewMode = preset.viewMode | 0;
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
        this.bcConfig = { ...defaultBCConfig(), ...(preset.bc || {}) };
        this._applyPhysicsConfig(preset.physics);
        if (preset.id) this.presetName = preset.id;
        this.buffers.uploadInitialState(preset.data);
        this.stepCount = 0;
        this.simTime   = 0;
        this.lastDt    = 0;
        this._lastDtHyp       = 1.0e-4;
        this._lastDtParabolic = 1.0e30;
        this._lastEtaMax      = 0;
        this._lastEtaMaxValid = false;
        this._lastHallRateMax = 0;
        this._lastHallRateValid = false;
        this._lastHallSubsteps = 1;
        this._lastCondRateMax = 0;
        this._lastCondRateValid = false;
        this._lastCondSubsteps = 1;
        this._lastViscSubsteps = 1;
        this._lastNonidealSubsteps = 1;
        this._lastRadiationSubsteps = 1;
        this._pendingTimeSteps = 0;
        this._pushUniforms();
        this.buffers.pushBC(this.bcConfig);
    }

    _invalidateHallRateHint() {
        this._lastHallRateValid = false;
        this._lastHallRateMax = 0;
    }

    _invalidateCondRateHint() {
        this._lastCondRateValid = false;
        this._lastCondRateMax = 0;
    }

    _invalidateEtaMaxHint() {
        this._lastEtaMaxValid = false;
        this._lastEtaMax = 0;
        this._lastDtParabolic = 1.0e30;
    }

    _invalidateSourceRateHints() {
        this._invalidateHallRateHint();
        this._invalidateCondRateHint();
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
    setDrivenState(state, edge = null) {
        const edgeKey = edge ? ({ N: 'drivenN', S: 'drivenS', E: 'drivenE', W: 'drivenW' })[edge] : null;
        if (edgeKey) {
            this.bcConfig = {
                ...this.bcConfig,
                [edgeKey]: { ...(this.bcConfig[edgeKey] || this.bcConfig.driven), ...state },
            };
        } else {
            this.bcConfig = {
                ...this.bcConfig,
                driven: { ...this.bcConfig.driven, ...state },
            };
        }
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
     * Update explicit resistivity (the base η_0). UI typically passes 0
     * for "ideal", but for presets with an active η floor (currently
     * Orszag-Tang) the value is clamped up to getEtaMin() to keep
     * current-sheet thinning within the grid's resolvable scales.
     */
    setEta(eta) {
        this.eta = Math.max(eta, this.getEtaMin());
        if (this.etaAnomAlpha > 0) this._invalidateEtaMaxHint();
        this._pushUniforms();
    }

    /**
     * Anomalous-η boost coefficient α. α = 0 disables the boost (sim
     * runs with constant η_0). α > 0 activates the |J|>J_crit-triggered
     * boost — see `anomalous_eta` in shared-helpers.wgsl. Birn 2001
     * GEM closure.
     */
    setEtaAnomAlpha(alpha) {
        this.etaAnomAlpha = Math.max(0, +alpha || 0);
        this._invalidateEtaMaxHint();
        this._pushUniforms();
    }

    /**
     * Anomalous-η critical current density |J_crit|. Below this the
     * boost is zero; above it grows quadratically in (|J|/J_crit − 1).
     */
    setEtaAnomJcrit(jcrit) {
        this.etaAnomJcrit = Math.max(1e-6, +jcrit || 1e-6);
        this._invalidateEtaMaxHint();
        this._pushUniforms();
    }

    /** Update the view mode enum. Pushed via uniforms on next render. */
    setViewMode(mode) {
        this.viewMode = mode;
        // Pick a sensible default colormap window per view if the caller
        // hasn't overridden. Windows live in config.js VIEW_RANGES, keyed
        // by the VIEW_* enum. Unknown modes leave the current window.
        const range = VIEW_RANGES[mode];
        if (range) {
            this.viewMin = range.min;
            this.viewMax = range.max;
        }
        this._pushUniforms();
    }

    setCFL(cfl)         { this.cfl = cfl; this._pushUniforms(); }
    setGamma(g)         { this.gamma = g; this._invalidateCondRateHint(); this._pushUniforms(); }
    setPressureFloor(p) { this.pressureFloor = p; this._pushUniforms(); }

    // ── Extended physics setters (breadth pass) ────────────────────
    setPhysicsFlag(flag, on) {
        if (on) this.physicsFlags |=  flag;
        else    this.physicsFlags &= ~flag;
        if ((flag & FLAG_HALL) !== 0) this._invalidateHallRateHint();
        if ((flag & FLAG_CONDUCTION) !== 0) this._invalidateCondRateHint();
        this._pushUniforms();
    }
    setEmfMode(mode)             { this.emfMode = mode | 0; this._pushUniforms(); }
    setHallDi(d)                 { this.hallDi = +d || 0; this._invalidateHallRateHint(); this._pushUniforms(); }
    setHallSubstepsMax(n)        { this.hallSubstepsMax = Math.max(1, n | 0); this._pushUniforms(); }
    setCoolingLambda0(v)         { this.coolingLambda0 = Math.max(0, +v || 0); this._pushUniforms(); }
    setCoolingTFloor(v)          { this.coolingTFloor = Math.max(0, +v || 0); this._pushUniforms(); }
    setCoolingTRef(v)            { this.coolingTRef = Math.max(1e-30, +v || 1); this._invalidateCondRateHint(); this._pushUniforms(); }
    setCoolingCurveMode(mode)    { this.coolingCurveMode = Math.max(0, mode | 0); this._pushUniforms(); }
    setConductionKappa(v)        { this.conductionKappa = Math.max(0, +v || 0); this._invalidateCondRateHint(); this._pushUniforms(); }
    setConductionIsoFrac(v)      { this.conductionIsoFrac = Math.min(1, Math.max(0, +v || 0)); this._pushUniforms(); }
    setConductionSatFrac(v)      { this.conductionSatFrac = Math.max(0, +v || 0); this._pushUniforms(); }
    setHallElectronPressureFrac(v) { this.hallElectronPressureFrac = Math.min(1, Math.max(0, +v || 0)); this._pushUniforms(); }
    setGravityG(v)               { this.gravityG = Math.max(0, +v || 0); this._pushUniforms(); }
    setGravityVec(gx, gy)        { this.gravityGx = +gx || 0; this.gravityGy = +gy || 0; this._pushUniforms(); }
    setGravityPoissonIters(n)    { this.gravityPoissonIters = Math.max(1, n | 0); this._pushUniforms(); }
    setGravityBoundaryMode(mode) { this.gravityBoundaryMode = Math.max(0, mode | 0); this._pushUniforms(); }
    setGravitySolverMode(mode)   { this.gravitySolverMode = mode === GRAVITY_SOLVER_JACOBI ? GRAVITY_SOLVER_JACOBI : GRAVITY_SOLVER_MULTIGRID; }
    setCoolingMetallicity(v)     { this.coolingMetallicity = Math.max(0, +v || 0); this._pushUniforms(); }
    setHeatingGamma0(v)          { this.heatingGamma0 = Math.max(0, +v || 0); this._pushUniforms(); }
    setHeatingDensityExp(v)      { this.heatingDensityExp = Math.max(0, +v || 0); this._pushUniforms(); }
    setHeatingTCut(v)            { this.heatingTCut = Math.max(0, +v || 0); this._pushUniforms(); }
    setAmbipolarEta(v)           { this.ambipolarEta = Math.max(0, +v || 0); this._pushUniforms(); }
    setBiermannCoeff(v)          { this.biermannCoeff = +v || 0; this._pushUniforms(); }
    setNeutralFrac(v)            { this.neutralFrac = Math.min(1, Math.max(0, +v || 0)); this._pushUniforms(); }
    setIonizationT0(v)           { this.ionizationT0 = Math.max(1e-30, +v || 1); this._pushUniforms(); }
    setViscosityNu(v)            { this.viscosityNu = Math.max(0, +v || 0); this._pushUniforms(); }
    setViscosityBulk(v)          { this.viscosityBulk = Math.max(0, +v || 0); this._pushUniforms(); }
    setViscosityAnisoFrac(v)     { this.viscosityAnisoFrac = Math.min(1, Math.max(0, +v || 0)); this._pushUniforms(); }
    setViscosityShock(v)         { this.viscosityShock = Math.max(0, +v || 0); this._pushUniforms(); }
    setSourceSubstepsMax(n)      { this.sourceSubstepsMax = Math.max(1, n | 0); this._pushUniforms(); }
    setGeometryMode(mode)        { this.geometryMode = Math.max(0, mode | 0); this._pushUniforms(); }
    setGeometryRMin(v)           { this.geometryRMin = Math.max(0, +v || 0); this._pushUniforms(); }
    setGravitySoftening(v)       { this.gravitySoftening = Math.max(0, +v || 0); this._pushUniforms(); }
    setGravityPoissonOmega(v)    { this.gravityPoissonOmega = Math.min(1.95, Math.max(0.05, +v || 1)); this._pushUniforms(); }
    setSpongeWidth(v)            { this.spongeWidth = Math.max(0, +v || 0); this._pushUniforms(); }
    setSpongeStrength(v)         { this.spongeStrength = Math.max(0, +v || 0); this._pushUniforms(); }
    setRadiationC(v)             { this.radiationC = Math.max(0, +v || 0); this._pushUniforms(); }
    setRadiationKappaAbs(v)      { this.radiationKappaAbs = Math.max(0, +v || 0); this._pushUniforms(); }
    setRadiationKappaScat(v)     { this.radiationKappaScat = Math.max(0, +v || 0); this._pushUniforms(); }
    setRadiationConst(v)         { this.radiationConst = Math.max(0, +v || 0); this._pushUniforms(); }
    setRadiationFloor(v)         { this.radiationFloor = Math.max(0, +v || 0); this._pushUniforms(); }
    setElectronInertiaLength(v)  { this.electronInertiaLength = Math.max(0, +v || 0); this._pushUniforms(); }
    setElectronInertiaDamping(v) { this.electronInertiaDamping = Math.max(0, +v || 0); this._pushUniforms(); }

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
     * current preset. Existing GPU buffers are retired after already
     * submitted work drains so repeated resolution toggles do not rely on GC.
     */
    setResolution(n) {
        if (n === this.n) return true;
        if (n !== 256 && n !== 512 && n !== 1024) {
            console.warn(`[plasma] setResolution: unsupported n=${n}`);
            return false;
        }
        const old = {
            n: this.n,
            n_total: this.n_total,
            dx: this.dx,
            buffers: this.buffers,
            renderer: this.renderer,
            bgCache: this._bgCache,
            dynamicBGCache: this._dynamicBGCache,
            rkl2BGCache: this._rkl2BGCache,
            sourceBGCache: this._sourceBGCache,
            bufferGeneration: this._bufferGeneration,
        };
        let nextBuffers = null;
        try {
            nextBuffers = new PlasmaBuffers(this.device, n);
            const nextRenderer = new PlasmaRenderer(this.device, this.context, this.pipelines, nextBuffers);
            nextBuffers.uploadLUT(VIRIDIS);
            const seed = new Float32Array([1e-4]);
            this.device.queue.writeBuffer(nextBuffers.dt, 0, seed.buffer);

            this.n       = n;
            this.n_total = n + 2 * this.ghost;
            this.dx      = this.domainLength / n;
            this.buffers = nextBuffers;
            this.renderer = nextRenderer;
            this._bufferGeneration += 1;

            this._pushUniforms();
            this._pushLicUniforms();
            // All GPU buffers changed identity — rebuild the bind-group cache
            // (and via _buildBindGroupCache, the renderer/LIC side caches).
            this._buildBindGroupCache();
            // Re-load whichever preset is current.
            this.setPreset(this.presetName);
        } catch (e) {
            if (nextBuffers && nextBuffers !== old.buffers && typeof nextBuffers.destroy === 'function') {
                nextBuffers.destroy();
            }
            this.n = old.n;
            this.n_total = old.n_total;
            this.dx = old.dx;
            this.buffers = old.buffers;
            this.renderer = old.renderer;
            this._bgCache = old.bgCache;
            this._dynamicBGCache = old.dynamicBGCache;
            this._rkl2BGCache = old.rkl2BGCache;
            this._sourceBGCache = old.sourceBGCache;
            this._bufferGeneration = old.bufferGeneration;
            console.warn(`[plasma] setResolution failed for n=${n}:`, e);
            return false;
        }
        if (old.buffers && typeof old.buffers.destroy === 'function') {
            this.device.queue.onSubmittedWorkDone()
                .then(() => old.buffers.destroy())
                .catch(() => old.buffers.destroy());
        }
        return true;
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
            etaAnomAlpha: this.etaAnomAlpha,
            etaAnomJcrit: this.etaAnomJcrit,
            gamma: this.gamma,
            cfl: this.cfl,
            pressureFloor: this.pressureFloor,
            speedScale: this.speedScale,
            running: this.running,
            bc: this.bcConfig,
            licIntensity: this.licIntensity,
            physics: {
                physicsFlags: this.physicsFlags,
                emfMode: this.emfMode,
                hallDi: this.hallDi,
                hallSubstepsMax: this.hallSubstepsMax,
                coolingLambda0: this.coolingLambda0,
                coolingTFloor: this.coolingTFloor,
                coolingTRef: this.coolingTRef,
                coolingCurveMode: this.coolingCurveMode,
                conductionKappa: this.conductionKappa,
                conductionIsoFrac: this.conductionIsoFrac,
                conductionSatFrac: this.conductionSatFrac,
                gravityGx: this.gravityGx,
                gravityGy: this.gravityGy,
                gravityG: this.gravityG,
                gravityPoissonIters: this.gravityPoissonIters,
                gravityBoundaryMode: this.gravityBoundaryMode,
                gravitySolverMode: this.gravitySolverMode,
                hallElectronPressureFrac: this.hallElectronPressureFrac,
                coolingMetallicity: this.coolingMetallicity,
                heatingGamma0: this.heatingGamma0,
                heatingDensityExp: this.heatingDensityExp,
                heatingTCut: this.heatingTCut,
                ambipolarEta: this.ambipolarEta,
                biermannCoeff: this.biermannCoeff,
                neutralFrac: this.neutralFrac,
                ionizationT0: this.ionizationT0,
                viscosityNu: this.viscosityNu,
                viscosityBulk: this.viscosityBulk,
                viscosityAnisoFrac: this.viscosityAnisoFrac,
                viscosityShock: this.viscosityShock,
                sourceSubstepsMax: this.sourceSubstepsMax,
                geometryMode: this.geometryMode,
                geometryRMin: this.geometryRMin,
                gravitySoftening: this.gravitySoftening,
                gravityPoissonOmega: this.gravityPoissonOmega,
                spongeWidth: this.spongeWidth,
                spongeStrength: this.spongeStrength,
                coolingTableMix: this.coolingTableMix,
                radiationC: this.radiationC,
                radiationKappaAbs: this.radiationKappaAbs,
                radiationKappaScat: this.radiationKappaScat,
                radiationConst: this.radiationConst,
                radiationFloor: this.radiationFloor,
                electronInertiaLength: this.electronInertiaLength,
                electronInertiaDamping: this.electronInertiaDamping,
            },
        });
    }

    /** Restore from `saveState()` output. */
    loadState(s) {
        let obj;
        try { obj = JSON.parse(s); } catch (e) { console.warn('[plasma] loadState parse:', e); return { ok: false, buffersChanged: false }; }
        if (!obj || obj.v !== 1) return { ok: false, buffersChanged: false };
        const oldBuffers = this.buffers;
        if (obj.n && obj.n !== this.n && this.setResolution(obj.n) === false) {
            return { ok: false, buffersChanged: false };
        }
        if (obj.preset) this.setPreset(obj.preset);
        if (obj.viewMode !== undefined) this.setViewMode(obj.viewMode);
        if (obj.eta !== undefined)          this.setEta(obj.eta);
        if (obj.etaAnomAlpha !== undefined) this.setEtaAnomAlpha(obj.etaAnomAlpha);
        if (obj.etaAnomJcrit !== undefined) this.setEtaAnomJcrit(obj.etaAnomJcrit);
        if (obj.gamma !== undefined)    this.setGamma(obj.gamma);
        if (obj.cfl !== undefined)      this.setCFL(obj.cfl);
        if (obj.pressureFloor !== undefined) this.setPressureFloor(obj.pressureFloor);
        if (obj.speedScale !== undefined)    this.setSpeedScale(obj.speedScale);
        if (obj.running !== undefined)       this.setRunning(obj.running);
        if (obj.bc) {
            this.bcConfig = { ...this.bcConfig, ...obj.bc };
            this.buffers.pushBC(this.bcConfig);
        }
        if (obj.physics) {
            this._applyPhysicsConfig(obj.physics);
            this._pushUniforms();
        }
        if (obj.licIntensity !== undefined) this.setLicIntensity(obj.licIntensity);
        return { ok: true, buffersChanged: this.buffers !== oldBuffers };
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
            etaAnomAlpha: this.etaAnomAlpha,
            etaAnomJcrit: this.etaAnomJcrit,
            // Extended physics
            hallDi:              this.hallDi,
            hallSubstepsMax:     this.hallSubstepsMax,
            coolingLambda0:      this.coolingLambda0,
            coolingTFloor:       this.coolingTFloor,
            coolingTRef:         this.coolingTRef,
            coolingCurveMode:    this.coolingCurveMode,
            conductionKappa:     this.conductionKappa,
            conductionIsoFrac:   this.conductionIsoFrac,
            conductionSatFrac:   this.conductionSatFrac,
            gravityGx:           this.gravityGx,
            gravityGy:           this.gravityGy,
            gravityG:            this.gravityG,
            gravityPoissonIters: this.gravityPoissonIters,
            gravityBoundaryMode: this.gravityBoundaryMode,
            hallElectronPressureFrac: this.hallElectronPressureFrac,
            coolingMetallicity: this.coolingMetallicity,
            heatingGamma0: this.heatingGamma0,
            heatingDensityExp: this.heatingDensityExp,
            heatingTCut: this.heatingTCut,
            ambipolarEta: this.ambipolarEta,
            biermannCoeff: this.biermannCoeff,
            neutralFrac: this.neutralFrac,
            ionizationT0: this.ionizationT0,
            viscosityNu: this.viscosityNu,
            viscosityBulk: this.viscosityBulk,
            viscosityAnisoFrac: this.viscosityAnisoFrac,
            viscosityShock: this.viscosityShock,
            sourceSubstepsMax: this.sourceSubstepsMax,
            geometryMode: this.geometryMode,
            geometryRMin: this.geometryRMin,
            gravitySoftening: this.gravitySoftening,
            gravityPoissonOmega: this.gravityPoissonOmega,
            spongeWidth: this.spongeWidth,
            spongeStrength: this.spongeStrength,
            coolingTableMix: this.coolingTableMix,
            radiationC: this.radiationC,
            radiationKappaAbs: this.radiationKappaAbs,
            radiationKappaScat: this.radiationKappaScat,
            radiationConst: this.radiationConst,
            radiationFloor: this.radiationFloor,
            electronInertiaLength: this.electronInertiaLength,
            electronInertiaDamping: this.electronInertiaDamping,
            physicsFlags:        this.physicsFlags,
            emfMode:             this.emfMode,
        });
        // Mirror the host-side cache so _encodeResistivitySuperStep
        // can skip-test alpha without re-reading the GPU uniform.
        this.buffers._etaAnomAlpha = this.etaAnomAlpha;
        this.buffers._etaAnomJcrit = this.etaAnomJcrit;
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
                { binding: 7, resource: { buffer: b.eta_max_buf } },
                { binding: 8, resource: { buffer: b.hall_speed_buf } },
                { binding: 9, resource: { buffer: b.cond_speed_buf } },
            ],
        });
    }

    _memoBG(key, build) {
        if (!this._dynamicBGCache) return build();
        let bg = this._dynamicBGCache.get(key);
        if (!bg) {
            bg = build();
            this._dynamicBGCache.set(key, bg);
        }
        return bg;
    }

    _bufKey(buffer) {
        return buffer?.label ?? 'null';
    }

    _scaleDtBG(srcDt, dstDt) {
        return this.device.createBindGroup({
            label: 'plasma.scaleDt.bg',
            layout: this.pipelines.layouts.scaleDt,
            entries: [
                { binding: 0, resource: { buffer: srcDt } },
                { binding: 1, resource: { buffer: dstDt } },
            ],
        });
    }

    _sourceDtBG() {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.sourceDt.bg',
            layout: this.pipelines.layouts.sourceDt,
            entries: [
                { binding: 0, resource: { buffer: b.dt_half } },
                { binding: 1, resource: { buffer: b.source_dt_params } },
                { binding: 2, resource: { buffer: b.hall_dt } },
                { binding: 3, resource: { buffer: b.cond_dt } },
                { binding: 4, resource: { buffer: b.visc_dt } },
                { binding: 5, resource: { buffer: b.nonideal_dt } },
                { binding: 6, resource: { buffer: b.rad_dt } },
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

    // ── RKL2 super-time-stepping bind groups ────────────────────────
    // The RKL2 implementation uses 5 buffer roles (dst, init, pprev,
    // prev, tmp) × 3 components (Bx, By, U1), bound per substep through
    // 3 different pipelines (snapshot, apply-resistivity-init,
    // apply-resistivity-prev). `_buildRkl2BindGroupCache()` pre-bakes
    // the side × role-rotation × substep bind groups so the hot path only
    // selects them.
    _applyResSnapBG(srcBx, srcBy, srcU1, dstBx, dstBy, dstU1) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyResSnap.bg',
            layout: this.pipelines.layouts.applyResSnap,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: srcBx } },
                { binding: 2, resource: { buffer: srcBy } },
                { binding: 3, resource: { buffer: srcU1 } },
                { binding: 4, resource: { buffer: dstBx } },
                { binding: 5, resource: { buffer: dstBy } },
                { binding: 6, resource: { buffer: dstU1 } },
            ],
        });
    }

    _applyResInitBG(stsMetaBuf, initBx, initBy, initU1, pprevBx, pprevBy, pprevU1,
                    tmpBx, tmpBy, tmpU1) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyResInit.bg',
            layout: this.pipelines.layouts.applyResInit,
            entries: [
                { binding: 0,  resource: { buffer: b.uniform } },
                { binding: 1,  resource: { buffer: stsMetaBuf } },
                { binding: 2,  resource: { buffer: b.sts_coeffs } },
                { binding: 3,  resource: { buffer: initBx } },
                { binding: 4,  resource: { buffer: initBy } },
                { binding: 5,  resource: { buffer: initU1 } },
                { binding: 6,  resource: { buffer: pprevBx } },
                { binding: 7,  resource: { buffer: pprevBy } },
                { binding: 8,  resource: { buffer: pprevU1 } },
                { binding: 9,  resource: { buffer: tmpBx } },
                { binding: 10, resource: { buffer: tmpBy } },
                { binding: 11, resource: { buffer: tmpU1 } },
                // Session 10: shader reads fresh dt_super from dt_buf[0].
                { binding: 12, resource: { buffer: b.dt } },
            ],
        });
    }

    _applyResPrevBG(stsMetaBuf, initBx, initBy, prevBx, prevBy, prevU1,
                    tmpBx, tmpBy, tmpU1) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyResPrev.bg',
            layout: this.pipelines.layouts.applyResPrev,
            entries: [
                { binding: 0,  resource: { buffer: b.uniform } },
                { binding: 1,  resource: { buffer: stsMetaBuf } },
                { binding: 2,  resource: { buffer: b.sts_coeffs } },
                { binding: 3,  resource: { buffer: initBx } },
                { binding: 4,  resource: { buffer: initBy } },
                { binding: 5,  resource: { buffer: prevBx } },
                { binding: 6,  resource: { buffer: prevBy } },
                { binding: 7,  resource: { buffer: prevU1 } },
                { binding: 8,  resource: { buffer: tmpBx } },
                { binding: 9,  resource: { buffer: tmpBy } },
                { binding: 10, resource: { buffer: tmpU1 } },
                // Session 10: shader reads fresh dt_super from dt_buf[0].
                { binding: 11, resource: { buffer: b.dt } },
            ],
        });
    }

    // Conservation diagnostics reduction (Session 8). Reads the
    // POST-step destination buffers — the bind group is built per
    // (stage 3 dst) side, since stage 3's dst is the next step's U_n.
    // Output: `cons_out` (24-scalar stats/conservation packet). One bind
    // group per pipeline.
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

    // Pointer perturbation. Binds the CURRENT primary handles so the
    // dispatch mutates U_n / face B directly — the next step() consumes
    // the perturbed state. Two side variants (a/b) baked at startup.
    _perturbBG(U0, U1, Bx, By) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.perturb.bg',
            layout: this.pipelines.layouts.perturb,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: b.perturbUniform } },
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
        this._dynamicBGCache = new Map();
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
            scaleDtHalf: this._scaleDtBG(b.dt, b.dt_half),
            sourceDt: this._sourceDtBG(),
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
            // Pointer perturbation operates on the CURRENT primary buffers.
            // Build A/B variants so applyPerturbation can pick by side at
            // call time without rebuilding bind groups per event.
            perturb: {
                a: this._perturbBG(sides.a.U0_n, sides.a.U1_n, sides.a.Bx_n, sides.a.By_n),
                b: this._perturbBG(sides.b.U0_n, sides.b.U1_n, sides.b.Bx_n, sides.b.By_n),
            },
        };
        this._buildRkl2BindGroupCache();
        this._buildSourceBindGroupCache();

        // Renderer / LIC bind groups also depend on the primary handles
        // (view-field reads U_n + face B; lic-advect reads face B). Have
        // the renderer rebuild its A/B cache against the new buffers.
        if (this.renderer)        this.renderer.rebuildSideCache();
        if (this.renderer && this.renderer.lic) this.renderer.lic.rebuildSideCache();
    }

    _buildSourceBindGroupCache() {
        const b = this.buffers;
        const sides = ['a', 'b'];
        const targets = ['src', 'dst'];
        this._sourceBGCache = { a: {}, b: {} };
        for (const side of sides) {
            for (const target of targets) {
                const h = this._sourceHandles(side, target);
                const timed = (dtBuffer) => ({
                    cooling: this._coolingBG(h.U0, h.U1, h.Bx, h.By, dtBuffer),
                    gravityPhi: this._gravityBG(h.U0, h.U1, b.phi, h.Bx, h.By, dtBuffer),
                    gravityPhiNext: this._gravityBG(h.U0, h.U1, b.phi_next, h.Bx, h.By, dtBuffer),
                    geometry: this._geometryBG(h.U0, h.U1, h.Bx, h.By, dtBuffer),
                });
                this._sourceBGCache[side][target] = {
                    handles: h,
                    applyBcs: this._applyBcsBG(h.U0, h.U1, h.Bx, h.By),
                    conduction: this._conductionBG(h.U0, h.U1, h.Bx, h.By),
                    ohm: this._ohmBG(h.U0, h.U1, h.Bx, h.By),
                    viscosity: this._viscosityBG(h.U0, h.U1, h.Bx, h.By),
                    radiation: this._radiationBG(h.U0, h.U1, h.Bx, h.By),
                    energyFloor: this._energyFloorBG(h.U0, h.U1, h.Bx, h.By),
                    dt: timed(b.dt),
                    dtHalf: timed(b.dt_half),
                };
            }
        }
    }

    _buildRkl2BindGroupCache() {
        const b = this.buffers;
        const initSet = { Bx: b.Bx_res_init,  By: b.By_res_init,  U1: b.U1_res_init  };
        const sets = [
            { Bx: b.Bx_res_pprev, By: b.By_res_pprev, U1: b.U1_res_pprev },
            { Bx: b.Bx_res_prev,  By: b.By_res_prev,  U1: b.U1_res_prev  },
            { Bx: b.Bx_res_tmp,   By: b.By_res_tmp,   U1: b.U1_res_tmp   },
        ];
        const sideHandles = {
            a: { U0: b.U0_b, Bx: b.Bx_b, By: b.By_b, U1: b.U1_b },
            b: { U0: b.U0_a, Bx: b.Bx_a, By: b.By_a, U1: b.U1_a },
        };

        this._rkl2BGCache = {};
        for (const side of ['a', 'b']) {
            const h = sideHandles[side];
            const cache = {
                seed: [initSet, ...sets].map(dest => this._applyResSnapBG(
                    h.Bx, h.By, h.U1,
                    dest.Bx, dest.By, dest.U1,
                )),
                init: [],
                prev: [],
                finalFromSet: sets.map(src => this._applyResSnapBG(
                    src.Bx, src.By, src.U1,
                    h.Bx,  h.By,  h.U1,
                )),
                energyFloor: this._energyFloorBG(h.U0, h.U1, h.Bx, h.By),
            };

            let pprev = 0, prev = 1, tmp = 2;
            for (let j = 1; j <= b.sts_coeffs_max_s; j++) {
                const stsMetaBuf = b.sts_meta_per_j[j - 1];
                cache.init[j - 1] = this._applyResInitBG(
                    stsMetaBuf,
                    initSet.Bx, initSet.By, initSet.U1,
                    sets[pprev].Bx, sets[pprev].By, sets[pprev].U1,
                    sets[tmp].Bx,   sets[tmp].By,   sets[tmp].U1,
                );
                cache.prev[j - 1] = this._applyResPrevBG(
                    stsMetaBuf,
                    initSet.Bx, initSet.By,
                    sets[prev].Bx, sets[prev].By, sets[prev].U1,
                    sets[tmp].Bx,  sets[tmp].By,  sets[tmp].U1,
                );
                const oldPprev = pprev;
                pprev = prev;
                prev = tmp;
                tmp = oldPprev;
            }
            this._rkl2BGCache[side] = cache;
        }
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

    _encodeScaleDtHalf(encoder) {
        const pass = encoder.beginComputePass({ label: 'plasma.scaleDtHalf' });
        pass.setPipeline(this.pipelines.pipelines.scaleDtHalf);
        pass.setBindGroup(0, this._bgCache.scaleDtHalf);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();
    }

    _encodeSourceDt(encoder) {
        const pass = encoder.beginComputePass({ label: 'plasma.sourceDt' });
        pass.setPipeline(this.pipelines.pipelines.sourceDt);
        pass.setBindGroup(0, this._bgCache.sourceDt);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();
    }

    /**
     * Encode one RK3 stage from the pre-baked bind-group cache.
     *
     * `stageIdx` ∈ {1, 2, 3}; `side` ∈ {'a', 'b'}.
     *
     * Session 8 — resistive diffusion moved OUT of the per-stage path
     * into a single RKL2 super-step at the END of the RK3 macro-step
     * (Lie split). This drops the per-stage resistivity triad
     * (9a / 9b / 9c) — and with it the post-CT apply-bcs that used to
     * fill dst-ghosts. The apply-bcs at the START of each stage now
     * always runs (no more η>0 skip), since stage-to-stage ghost
     * carry-over is gone.
     */
    _encodeStage(pass, stageIdx, side) {
        const { pipelines } = this;
        const N         = this.n;
        const gInterior = Math.ceil(N / WG);
        const gN1       = Math.ceil((N + 1) / WG);
        const gN2       = Math.ceil((N + 2) / WG);
        // apply-bcs covers (N_total+1)² (all ghost strips + boundary faces).
        const gTotalP1  = Math.ceil((this.n_total + 1) / WG);

        const bgs = this._bgCache[`stage${stageIdx}`][side];

        // 1. apply-bcs on src — fills ghosts for the upcoming PPM stencil.
        //    Always runs now (Session 8): the per-stage resistivity triad
        //    that used to fill dst-ghosts was retired in favour of a
        //    single end-of-step RKL2 super-step.
        pass.setPipeline(pipelines.pipelines.applyBcs);
        pass.setBindGroup(0, bgs.applyBcsSrc);
        pass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);

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

        // Resistivity moved OUT of the per-stage path (Session 8).
        // RKL2 super-step encoded once per RK3 macro-step — see
        // `_encodeResistivitySuperStep` in `step()`.
    }

    // ── RKL2 coefficient computation ────────────────────────────────
    //
    // Meyer, Diehl & Kupka (2014) §3, Algorithm 1 / Eq 13. Coefficients
    //   b_0 = b_1 = 1/3,    b_j  = (j² + j − 2) / (2 j (j + 1)),  j ≥ 2
    //   a_j = 1 − b_j
    //   w_1 = 4 / (s² + s − 2)        (s ≥ 2; s = 1 special case)
    //   μ̃_1 = b_1 · w_1,   ν_1 = 0,   μ_1 = 0,   γ̃_1 = 0           (j = 1)
    //   For j ≥ 2:
    //     μ_j         = (2j − 1)/j · b_j / b_{j-1}
    //     ν_j         = -(j − 1)/j · b_j / b_{j-2}
    //     μ̃_j         = μ_j · w_1
    //     γ̃_j         = -a_{j-1} · μ̃_j
    //
    // Substep count from Δt_super and dt_parabolic:
    //   stability bound (MDK14 eq 18):
    //     Δt_super ≤ ((s² + s − 2) / 2) · Δt_FE
    //   with Δt_FE = dt_parabolic (the 2D 5-point FE bound: dx²/(4η);
    //   see compute-dt.wgsl). Solving for the minimum integer s:
    //     s² + s − 2 ≥ 2·r,   r ≡ Δt_super / dt_parabolic
    //     s ≥ ½·(−1 + √(8r + 9))
    //     s = ceil(½·(√(8r + 9) − 1))
    // capped at sts_coeffs_max_s (100; brief-specified safety cap).
    //
    // Session 13 retrospective — the prior formula was
    //     s = ceil(½·(√(1 + 8r) − 1))
    // which solves s² + s = 2r (MISSING the `-2` stability margin). It
    // picks an s that's one short whenever the true sRaw straddles an
    // integer:  at r ∈ (s_max_true, s_max_true + 0.5] it returns s,
    // but s is only stable for r ≤ (s² + s − 2)/2 — so RKL2 ran with
    // a substep count too small to damp the highest-k modes. Combined
    // with the 2× error in dt_parabolic (Session 8 set it to the 1D FE
    // bound rather than the 2D one — see compute-dt.wgsl header), the
    // effective stability ratio was off by 2-3× for typical operating
    // points. At Orszag-Tang N=1024 η ~5e-1 (true ratio ~13), the buggy
    // code picked s=5 while true required s=7; the highest-k modes
    // amplified each macro step until resistivity caught up, giving
    // the "blobs fade in and out" visual.
    //
    // For s = 1 we want pure forward Euler (Y_1 = U^n + Δt · L(U^n)) —
    // achieved with the special-case μ̃_1 = 1 (the MDK formula divides
    // by (s² + s − 2) which is zero at s = 1, hence the special case).
    // FE is stable for r ≤ 1/2 (i.e., Δt_super ≤ Δt_FE = dt_parabolic/2…
    // wait, that's not right since dt_parabolic IS Δt_FE here. FE is
    // stable for r ≤ 1.). The formula's lower limit handles this: at
    // r = 1 sRaw = ½·(√17 − 1) ≈ 1.56, ceil → 2, so we over-substep
    // at the FE/RKL2 boundary by one — negligible cost, zero risk.
    _computeRKL2Coeffs(dt_super, dt_parabolic, sMax) {
        let s;
        if (dt_parabolic >= 1e29 || dt_super <= 0) {
            s = 1;
        } else {
            const ratio = dt_super / dt_parabolic;
            const sRaw = 0.5 * (Math.sqrt(8 * ratio + 9) - 1);
            s = Math.max(1, Math.ceil(sRaw));
            if (s > sMax) s = sMax;
        }

        const coeffs = new Float32Array(sMax * 4);
        if (s === 0) return { s, coeffsArr: coeffs };

        const b = new Float64Array(s + 1);
        const a = new Float64Array(s + 1);
        b[0] = 1.0 / 3.0;
        if (s >= 1) b[1] = 1.0 / 3.0;
        for (let j = 2; j <= s; j++) {
            b[j] = (j * j + j - 2) / (2.0 * j * (j + 1));
        }
        for (let j = 0; j <= s; j++) a[j] = 1.0 - b[j];

        const w1 = (s >= 2) ? (4.0 / (s * s + s - 2)) : 1.0;
        // j = 1: μ_1 = ν_1 = γ̃_1 = 0; μ̃_1 = b_1·w_1 for s ≥ 2, or 1 for s = 1.
        const mu1tilde = (s === 1) ? 1.0 : b[1] * w1;
        coeffs[0] = 0.0;
        coeffs[1] = 0.0;
        coeffs[2] = mu1tilde;
        coeffs[3] = 0.0;

        for (let j = 2; j <= s; j++) {
            const mu   = (2.0 * j - 1) / j * b[j] / b[j - 1];
            const nu   = -(j - 1) / j * b[j] / b[j - 2];
            const muT  = mu * w1;
            const gamT = -a[j - 1] * muT;
            const base = (j - 1) * 4;
            coeffs[base + 0] = mu;
            coeffs[base + 1] = nu;
            coeffs[base + 2] = muT;
            coeffs[base + 3] = gamT;
        }
        return { s, coeffsArr: coeffs };
    }

    /**
     * Encode the RKL2 super-step that diffuses face B + Bz by η over
     * Δt = dt_hyp. Called once per RK3 macro-step, AFTER stage 3 has
     * written the final hyperbolic state into the side's `next` buffers
     * (which alias the stage-3 destination — see `_buildBindGroupCache`).
     *
     * Operator splitting: Lie (1st-order). The hyperbolic + ideal-MHD
     * update completes first; resistivity applies on top.
     *
     * Skip-paths:
     *   - eta == 0 AND alpha == 0: no resistivity at all. Skip.
     *   - eta_max ≈ 0 (dt_parabolic huge): s = 1 forward Euler with
     *     negligible Δt·L term. Still encoded — single substep is
     *     cheap.
     */
    _encodeResistivitySuperStep(encoder, side, dt_super, dt_parabolic) {
        const b = this.buffers;
        const { pipelines } = this;
        const alpha = b._etaAnomAlpha ?? 0;
        if (this.eta <= 0 && alpha <= 0) return;

        const { s, coeffsArr } = this._computeRKL2Coeffs(
            dt_super, dt_parabolic, b.sts_coeffs_max_s,
        );
        if (s <= 0) return;
        this._lastSuperStepS = s;

        b.pushStsCoeffs(coeffsArr);

        const N = this.n;
        const gResis    = Math.ceil((N + 3) / WG);
        const gInterior = Math.ceil(N / WG);
        const gTotalP1  = Math.ceil((this.n_total + 1) / WG);

        // dst buffers — stage 3 wrote into the side's `next` slot
        // (before the post-step swap).
        const handles = (side === 'a')
            ? { U0: b.U0_b, Bx: b.Bx_b, By: b.By_b, U1: b.U1_b }
            : { U0: b.U0_a, Bx: b.Bx_a, By: b.By_a, U1: b.U1_a };
        const rkl2Cache = this._rkl2BGCache?.[side] ?? null;

        const initSet = { Bx: b.Bx_res_init,  By: b.By_res_init,  U1: b.U1_res_init  };
        const setA    = { Bx: b.Bx_res_pprev, By: b.By_res_pprev, U1: b.U1_res_pprev };
        const setB    = { Bx: b.Bx_res_prev,  By: b.By_res_prev,  U1: b.U1_res_prev  };
        const setC    = { Bx: b.Bx_res_tmp,   By: b.By_res_tmp,   U1: b.U1_res_tmp   };

        // Refresh dst-side ghost cells before RKL2's substep loop reads
        // them at boundary-face corners. Session 12 retrospective (fifth
        // Harris bug, part 1 of 2): stage 3 writes fresh INTERIOR state
        // to dst, but the dst-side ghosts haven't seen apply-bcs since
        // this side was last a stage source — 1 step lagged. RKL2's
        // curl(η J) stencil at boundary faces (ix == ghost or ix ==
        // ghost+n_interior; iy mirror for By) reads cell-centered and
        // face-B values one cell INSIDE the dispatch range, which lands
        // in ghost storage. Stale ghosts break the periodic-wrap
        // invariance for RKL2's update of boundary faces — observed as
        // slow divB drift at i=255 in Harris (the last interior column
        // under periodic E/W). Without this dispatch, divB at the
        // periodic boundary grows ~5× faster and eventually destabilizes
        // the hyperbolic step. Run apply-bcs in its OWN compute pass so
        // the writes complete + synchronize before the seed-snapshot
        // dispatches read them. The `applyBcsDst` bind group was built
        // for each stage by `_buildBindGroupCache` (line ~822) but never
        // dispatched after Session 8 retired the per-stage resistivity
        // triad — this restores the dispatch in its only remaining
        // legitimate slot. It matters for mixed-BC presets (Harris plus
        // the x-outflow shock tubes). For all-periodic presets, the dispatch
        // is still cheap and keeps the post-step face convention explicit.
        {
            const bcPass = encoder.beginComputePass({ label: 'plasma.rkl2.applyBcsDst' });
            bcPass.setPipeline(pipelines.pipelines.applyBcs);
            bcPass.setBindGroup(0, this._bgCache.stage3[side].applyBcsDst);
            bcPass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);
            bcPass.end();
        }

        const pass = encoder.beginComputePass({ label: 'plasma.rkl2' });

        // Seed: dst → init, dst → A (pprev), dst → B (prev), dst → C (tmp).
        //
        // tmp must be seeded too even though the substep init/prev passes
        // OVERWRITE its interior values — the snapshot copy at the END of
        // the super-step (prev → dst) covers the full (N+3)² window
        // including ghost cells, and ghosts never get touched by init or
        // prev (their `in_*_interior` checks gate them off). After substep
        // 1's rotation, `prev` points at what was previously `tmp`; if
        // tmp's ghost strip was zero-initialized (buffer allocation
        // default), the final snapshot zeros dst's ghost strip, which
        // then drives massive ∇²B at the wall in the NEXT step's
        // hyperbolic update and detonates Harris within ~50 steps
        // (Session 9). Seeding tmp from dst keeps its ghost equal to
        // whatever apply-bcs wrote when this side was last source.
        pass.setPipeline(pipelines.pipelines.applyResSnapshot);
        const seedDests = [initSet, setA, setB, setC];
        for (let k = 0; k < seedDests.length; k++) {
            const dest = seedDests[k];
            const bg = rkl2Cache?.seed?.[k]
                ?? this._applyResSnapBG(
                    handles.Bx, handles.By, handles.U1,
                    dest.Bx,    dest.By,    dest.U1,
                );
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(gResis, gResis, 1);
        }

        // s substeps with role rotation. The substep_idx that the init /
        // prev shaders read lives in a per-j pre-baked uniform buffer
        // (`b.sts_meta_per_j[j-1]`), NOT in a single shared buffer
        // rewritten per iteration. Rewriting a shared buffer here would
        // be a no-op race: WebGPU orders every `queue.writeBuffer` before
        // the next `queue.submit`, so all s writes collapse to the last
        // value before any of the dispatches in this compute pass
        // actually run, and every substep would see substep_idx = s.
        // See buffers.js `sts_meta_per_j` comment for the full history.
        let pprev = setA, prev = setB, tmp = setC;
        let pprevIdx = 0, prevIdx = 1, tmpIdx = 2;
        for (let j = 1; j <= s; j++) {
            const stsMetaBuf = b.sts_meta_per_j[j - 1];

            const bgInit = rkl2Cache?.init?.[j - 1]
                ?? this._applyResInitBG(
                    stsMetaBuf,
                    initSet.Bx, initSet.By, initSet.U1,
                    pprev.Bx,   pprev.By,   pprev.U1,
                    tmp.Bx,     tmp.By,     tmp.U1,
                );
            pass.setPipeline(pipelines.pipelines.applyResInit);
            pass.setBindGroup(0, bgInit);
            pass.dispatchWorkgroups(gResis, gResis, 1);

            const bgPrev = rkl2Cache?.prev?.[j - 1]
                ?? this._applyResPrevBG(
                    stsMetaBuf,
                    initSet.Bx, initSet.By,
                    prev.Bx,    prev.By,    prev.U1,
                    tmp.Bx,     tmp.By,     tmp.U1,
                );
            pass.setPipeline(pipelines.pipelines.applyResPrev);
            pass.setBindGroup(0, bgPrev);
            pass.dispatchWorkgroups(gResis, gResis, 1);

            // Rotate: new_pprev = old prev, new_prev = tmp,
            //         new_tmp = old pprev (free to overwrite).
            const oldPprev = pprev;
            pprev = prev;
            prev  = tmp;
            tmp   = oldPprev;
            const oldPprevIdx = pprevIdx;
            pprevIdx = prevIdx;
            prevIdx  = tmpIdx;
            tmpIdx   = oldPprevIdx;
        }

        // Copy Y_s back into dst. Y_s ended up in `prev` after substep s's
        // rotation (substep s wrote into tmp, then rotation made it prev).
        const bgFinal = rkl2Cache?.finalFromSet?.[prevIdx]
            ?? this._applyResSnapBG(
                prev.Bx, prev.By, prev.U1,
                handles.Bx, handles.By, handles.U1,
            );
        pass.setPipeline(pipelines.pipelines.applyResSnapshot);
        pass.setBindGroup(0, bgFinal);
        pass.dispatchWorkgroups(gResis, gResis, 1);

        // RKL2 updates face B after the RK3-stage energy floor has already
        // run. Clamp total energy once more against the post-diffusion B so
        // the next primitive recovery cannot start with negative pressure in
        // a cell where diffusion locally strengthened |B|.
        pass.setPipeline(pipelines.pipelines.energyFloor);
        pass.setBindGroup(0, rkl2Cache?.energyFloor
            ?? this._energyFloorBG(handles.U0, handles.U1, handles.Bx, handles.By));
        pass.dispatchWorkgroups(gInterior, gInterior, 1);

        pass.end();

        // Canonicalize the destination ghosts/boundary faces after the
        // resistive copy-back. This keeps post-step diagnostics, rendering,
        // and the next step's source state on the same boundary convention
        // instead of exposing RKL2's scratch-buffer ghost state for one frame.
        {
            const bcPass = encoder.beginComputePass({ label: 'plasma.rkl2.postApplyBcsDst' });
            bcPass.setPipeline(pipelines.pipelines.applyBcs);
            bcPass.setBindGroup(0, this._bgCache.stage3[side].applyBcsDst);
            bcPass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);
            bcPass.end();
        }
    }

    // ── Extended physics bind-group factories ─────────────────────
    // `_buildSourceBindGroupCache()` pre-bakes the hot-path source groups;
    // these factories remain for cache construction and unusual test paths.
    _coolingBG(U0, U1, Bx, By, dtBuffer = this.buffers.dt) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyCooling.bg',
            layout: this.pipelines.layouts.cooling,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: dtBuffer } },
                { binding: 6, resource: { buffer: b.microphysics } },
            ],
        });
    }

    _poissonBG(U0, phi_in, phi_out) {
        const b = this.buffers;
        const key = `poisson:${this._bufKey(U0)}:${this._bufKey(phi_in)}:${this._bufKey(phi_out)}`;
        return this._memoBG(key, () => this.device.createBindGroup({
            label: 'plasma.solvePoisson.bg',
            layout: this.pipelines.layouts.poisson,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: phi_in } },
                { binding: 3, resource: { buffer: phi_out } },
                { binding: 4, resource: { buffer: b.rho_mean } },
                { binding: 5, resource: { buffer: b.rho_mean_partials } },
            ],
        }));
    }

    _poissonMgBG(level, U0, phiMain, phiIn, phiOut, rhsIn, rhsOut, phiCoarse) {
        const b = this.buffers;
        const phiMainBuf = phiMain   ?? b.poisson_mg_dummy_a;
        const phiInBuf   = phiIn     ?? b.poisson_mg_dummy_ro;
        const phiOutBuf  = phiOut    ?? b.poisson_mg_dummy_b;
        const rhsInBuf   = rhsIn     ?? b.poisson_mg_dummy_ro;
        const rhsOutBuf  = rhsOut    ?? b.poisson_mg_dummy_c;
        const coarseBuf  = phiCoarse ?? b.poisson_mg_dummy_ro;
        const key = [
            'poissonMg', level.n, this._bufKey(U0), this._bufKey(phiMainBuf),
            this._bufKey(phiInBuf), this._bufKey(phiOutBuf), this._bufKey(rhsInBuf),
            this._bufKey(rhsOutBuf), this._bufKey(coarseBuf),
        ].join(':');
        return this._memoBG(key, () => this.device.createBindGroup({
            label: `plasma.solvePoissonMg.${level.n}.bg`,
            layout: this.pipelines.layouts.poissonMg,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: level.uniform } },
                { binding: 2, resource: { buffer: U0 } },
                { binding: 3, resource: { buffer: b.rho_mean } },
                { binding: 4, resource: { buffer: phiMainBuf } },
                { binding: 5, resource: { buffer: phiInBuf } },
                { binding: 6, resource: { buffer: phiOutBuf } },
                { binding: 7, resource: { buffer: rhsInBuf } },
                { binding: 8, resource: { buffer: rhsOutBuf } },
                { binding: 9, resource: { buffer: coarseBuf } },
            ],
        }));
    }

    _poissonUsesCylindricalOperator() {
        return (this.physicsFlags & FLAG_GEOMETRY) !== 0
            && this.geometryMode === GEOMETRY_CYLINDRICAL;
    }

    _canUsePoissonMultigrid() {
        return this.gravitySolverMode !== GRAVITY_SOLVER_JACOBI
            && !this._poissonUsesCylindricalOperator();
    }

    _encodePoissonMean(encoder, U0, gInterior) {
        const pass = encoder.beginComputePass({ label: 'plasma.poisson.mean' });
        pass.setPipeline(this.pipelines.pipelines.solvePoissonReduceMean);
        pass.setBindGroup(0, this._poissonBG(U0, this.buffers.phi, this.buffers.phi_next));
        pass.dispatchWorkgroups(gInterior, gInterior, 1);
        pass.setPipeline(this.pipelines.pipelines.solvePoissonFinalizeMean);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();
    }

    _encodePoissonJacobi(encoder, U0, gInterior, iters) {
        const b = this.buffers;
        if (iters <= 0) return b.phi;
        const pass = encoder.beginComputePass({ label: 'plasma.poisson.jacobi' });
        pass.setPipeline(this.pipelines.pipelines.solvePoissonIterate);
        const bgAB = this._poissonBG(U0, b.phi,      b.phi_next);
        const bgBA = this._poissonBG(U0, b.phi_next, b.phi);
        for (let it = 0; it < iters; it++) {
            pass.setBindGroup(0, (it & 1) ? bgBA : bgAB);
            pass.dispatchWorkgroups(gInterior, gInterior, 1);
        }
        pass.end();
        return (iters & 1) ? b.phi_next : b.phi;
    }

    _encodePoissonMultigrid(encoder, U0, iters) {
        const b = this.buffers;
        const levels = b.ensurePoissonMultigridLevels();
        if (!levels || levels.length < 2 || iters <= 0) return b.phi;

        const pass = encoder.beginComputePass({ label: 'plasma.poisson.multigrid' });
        const pp = this.pipelines.pipelines;
        const groups = (n) => Math.ceil(n / WG);
        const current = levels.map(level => level.phiA);
        const spare   = levels.map(level => level.phiB);
        const swapLevel = (li) => {
            const tmp = current[li];
            current[li] = spare[li];
            spare[li] = tmp;
        };

        pass.setPipeline(pp.solvePoissonMgInit);
        pass.setBindGroup(0, this._poissonMgBG(
            levels[0], U0,
            b.phi,
            null,
            levels[0].phiA,
            null,
            levels[0].rhs,
            null,
        ));
        pass.dispatchWorkgroups(groups(levels[0].n), groups(levels[0].n), 1);

        const finestIters = Math.max(1, iters | 0);
        const cycles = Math.max(1, Math.min(8, Math.ceil(finestIters / 16)));
        const preSweeps = 2;
        const postSweeps = 2;
        const coarseSweeps = Math.max(12, Math.min(64, Math.ceil(finestIters / 2)));

        const smoothLevel = (li, sweeps) => {
            const level = levels[li];
            pass.setPipeline(pp.solvePoissonMgSmooth);
            for (let s = 0; s < sweeps; s++) {
                pass.setBindGroup(0, this._poissonMgBG(
                    level, U0,
                    null,
                    current[li],
                    spare[li],
                    level.rhs,
                    null,
                    null,
                ));
                pass.dispatchWorkgroups(groups(level.n), groups(level.n), 1);
                swapLevel(li);
            }
        };

        const last = levels.length - 1;
        for (let cycle = 0; cycle < cycles; cycle++) {
            for (let li = 0; li < last; li++) {
                smoothLevel(li, preSweeps);
                const coarse = levels[li + 1];
                pass.setPipeline(pp.solvePoissonMgRestrict);
                pass.setBindGroup(0, this._poissonMgBG(
                    coarse, U0,
                    null,
                    current[li],
                    coarse.phiA,
                    levels[li].rhs,
                    coarse.rhs,
                    null,
                ));
                pass.dispatchWorkgroups(groups(coarse.n), groups(coarse.n), 1);
                current[li + 1] = coarse.phiA;
                spare[li + 1] = coarse.phiB;
            }

            smoothLevel(last, coarseSweeps);

            for (let li = last - 1; li >= 0; li--) {
                const fine = levels[li];
                pass.setPipeline(pp.solvePoissonMgProlongate);
                pass.setBindGroup(0, this._poissonMgBG(
                    fine, U0,
                    null,
                    current[li],
                    spare[li],
                    fine.rhs,
                    null,
                    current[li + 1],
                ));
                pass.dispatchWorkgroups(groups(fine.n), groups(fine.n), 1);
                swapLevel(li);
                smoothLevel(li, postSweeps);
            }
        }

        pass.setPipeline(pp.solvePoissonMgCopyToMain);
        pass.setBindGroup(0, this._poissonMgBG(
            levels[0], U0,
            b.phi,
            current[0],
            null,
            levels[0].rhs,
            null,
            null,
        ));
        pass.dispatchWorkgroups(groups(levels[0].n), groups(levels[0].n), 1);
        pass.end();
        return b.phi;
    }

    _gravityBG(U0, U1, phi, Bx, By, dtBuffer = this.buffers.dt) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyGravity.bg',
            layout: this.pipelines.layouts.gravity,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: phi } },
                { binding: 4, resource: { buffer: dtBuffer } },
                { binding: 5, resource: { buffer: Bx } },
                { binding: 6, resource: { buffer: By } },
            ],
        });
    }

    _conductionBG(U0, U1, Bx, By) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyConduction.bg',
            layout: this.pipelines.layouts.conduction,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                // Session 15: conduction reads its own sub-step dt buffer
                // so _encodeExtendedPhysics can loop the conduction
                // compute_delta/apply_delta pair N_cond times within a
                // single macro Δt. Mirror of the Hall sub-cycle pattern.
                { binding: 5, resource: { buffer: b.cond_dt } },
                { binding: 6, resource: { buffer: b.conduction_dE } },
                { binding: 7, resource: { buffer: b.microphysics } },
            ],
        });
    }

    _hallBG(U0, Bx, By, U1) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyHall.bg',
            layout: this.pipelines.layouts.hall,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: Bx } },
                { binding: 3, resource: { buffer: By } },
                { binding: 4, resource: { buffer: U1 } },
                // Session 15: Hall reads its own sub-step dt buffer so
                // _encodeExtendedPhysics can run N_hall iterations within
                // a single macro Δt without rewriting dt_buf.
                { binding: 5, resource: { buffer: b.hall_dt } },
                { binding: 6, resource: { buffer: b.hall_E } },
                { binding: 7, resource: { buffer: b.hall_mb0 } },
            ],
        });
    }

    _nonidealBG(U0, U1, Bx, By) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyNonideal.bg',
            layout: this.pipelines.layouts.nonideal,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: b.nonideal_dt } },
                { binding: 6, resource: { buffer: b.nonideal_E } },
            ],
        });
    }

    _ohmBG(U0, U1, Bx, By) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyOhm.bg',
            layout: this.pipelines.layouts.ohm,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: b.nonideal_dt } },
                { binding: 6, resource: { buffer: b.nonideal_E } },
                { binding: 7, resource: { buffer: b.hall_E } },
                { binding: 8, resource: { buffer: b.hall_mb0 } },
                { binding: 9, resource: { buffer: b.microphysics } },
            ],
        });
    }

    _viscosityBG(U0, U1, Bx, By) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyViscosity.bg',
            layout: this.pipelines.layouts.viscosity,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: b.visc_dt } },
                { binding: 6, resource: { buffer: b.viscosity_dU } },
                { binding: 7, resource: { buffer: b.microphysics } },
            ],
        });
    }

    _geometryBG(U0, U1, Bx, By, dtBuffer = this.buffers.dt) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyGeometry.bg',
            layout: this.pipelines.layouts.geometry,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: dtBuffer } },
            ],
        });
    }

    _radiationBG(U0, U1, Bx, By) {
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.applyRadiation.bg',
            layout: this.pipelines.layouts.radiation,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0 } },
                { binding: 2, resource: { buffer: U1 } },
                { binding: 3, resource: { buffer: Bx } },
                { binding: 4, resource: { buffer: By } },
                { binding: 5, resource: { buffer: b.rad_dt } },
                { binding: 6, resource: { buffer: b.radiation_E } },
                { binding: 7, resource: { buffer: b.radiation_dE } },
                { binding: 8, resource: { buffer: b.microphysics } },
                { binding: 9, resource: { buffer: b.bc_uniforms } },
            ],
        });
    }

    _encodeApplyBcsDst(encoder, side, label) {
        const pass = encoder.beginComputePass({ label });
        const gTotalP1 = Math.ceil((this.n_total + 1) / WG);
        pass.setPipeline(this.pipelines.pipelines.applyBcs);
        pass.setBindGroup(0, this._bgCache.stage3[side].applyBcsDst);
        pass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);
        pass.end();
    }

    _sourceHandles(side, target = 'dst') {
        const b = this.buffers;
        if (side === 'a') {
            return target === 'src'
                ? { U0: b.U0_a, U1: b.U1_a, Bx: b.Bx_a, By: b.By_a }
                : { U0: b.U0_b, U1: b.U1_b, Bx: b.Bx_b, By: b.By_b };
        }
        return target === 'src'
            ? { U0: b.U0_b, U1: b.U1_b, Bx: b.Bx_b, By: b.By_b }
            : { U0: b.U0_a, U1: b.U1_a, Bx: b.Bx_a, By: b.By_a };
    }

    _encodeApplyBcsHandles(encoder, handles, label, cachedBG = null) {
        const pass = encoder.beginComputePass({ label });
        const gTotalP1 = Math.ceil((this.n_total + 1) / WG);
        pass.setPipeline(this.pipelines.pipelines.applyBcs);
        pass.setBindGroup(0, cachedBG ?? this._applyBcsBG(handles.U0, handles.U1, handles.Bx, handles.By));
        pass.dispatchWorkgroups(gTotalP1, gTotalP1, 1);
        pass.end();
    }

    /**
     * Encode the extended-physics passes after the RKL2 super-step.
     * All operate on the side's stage-3 destination buffers (which
     * alias U_next before the post-step swap). Each pass is gated by
     * the corresponding bit in `physicsFlags` — when nothing is
     * enabled this returns false without encoding. Numerical-only flags
     * (positivity, upwind EMF) are not treated as source physics here.
     */
    _encodeExtendedPhysics(encoder, side,
                           hallSubsteps = 1,
                           condSubsteps = 1,
                           viscSubsteps = 1,
                           nonidealSubsteps = 1,
                           opts = {}) {
        if ((this.physicsFlags & EXTENDED_SOURCE_FLAGS) === 0) return false;
        const b = this.buffers;
        const { pipelines } = this;

        const target = opts.target ?? 'dst';
        const dtBuffer = opts.dtBuffer ?? b.dt;
        const labelPrefix = opts.labelPrefix ?? `plasma.extended.${target}`;
        const handles = this._sourceHandles(side, target);
        const cache = this._sourceBGCache?.[side]?.[target] ?? null;
        const timedCache = dtBuffer === b.dt_half
            ? cache?.dtHalf
            : (dtBuffer === b.dt ? cache?.dt : null);

        const N         = this.n;
        const gInterior = Math.ceil(N / WG);
        const gN1       = Math.ceil((N + 1) / WG);

        const flags = this.physicsFlags;
        const coolingOn = ((flags & FLAG_COOLING) !== 0 && this.coolingLambda0 > 0)
                       || ((flags & FLAG_HEATING) !== 0 && this.heatingGamma0 > 0);
        const selfGravOn = (flags & FLAG_GRAVITY_SELF) !== 0 && this.gravityG > 0;
        const externalGravOn = (flags & FLAG_GRAVITY_EXT) !== 0
            && (this.gravityGx !== 0 || this.gravityGy !== 0);
        const gravityOn = selfGravOn || externalGravOn;
        const conductionOn = (flags & FLAG_CONDUCTION) !== 0 && this.conductionKappa > 0;
        const radiationOn = (flags & FLAG_RADIATION) !== 0
            && this.radiationC > 0
            && (this.radiationKappaAbs > 0 || this.radiationKappaScat > 0);
        const viscosityOn = (flags & FLAG_VISCOSITY) !== 0
            && ((this.viscosityNu || 0) > 0
             || (this.viscosityBulk || 0) > 0
             || (this.viscosityShock || 0) > 0);
        const ohmOn = ((flags & FLAG_HALL) !== 0 && this.hallDi > 0)
                   || ((flags & FLAG_AMBIPOLAR) !== 0 && this.ambipolarEta > 0)
                   || ((flags & FLAG_BIERMANN) !== 0 && this.biermannCoeff !== 0)
                   || ((flags & FLAG_ELECTRON_INERTIA) !== 0
                       && this.electronInertiaLength > 0
                       && this.electronInertiaDamping > 0);
        const geometryOn = ((flags & FLAG_GEOMETRY) !== 0 && this.geometryMode === GEOMETRY_CYLINDRICAL)
                        || ((flags & FLAG_SPONGE) !== 0
                            && this.spongeWidth > 0
                            && this.spongeStrength > 0);
        const anySource = gravityOn || coolingOn || conductionOn || radiationOn
                       || viscosityOn || ohmOn || geometryOn;
        if (!anySource) return false;

        const applyBcs = (label) => {
            this._encodeApplyBcsHandles(encoder, handles, label, cache?.applyBcs ?? null);
        };
        const computePass = (label, body) => {
            const pass = encoder.beginComputePass({ label });
            body(pass);
            pass.end();
        };
        const gravityBG = (phi) => {
            if (timedCache && phi === b.phi) return timedCache.gravityPhi;
            if (timedCache && phi === b.phi_next) return timedCache.gravityPhiNext;
            return this._gravityBG(handles.U0, handles.U1, phi, handles.Bx, handles.By, dtBuffer);
        };
        const coolingBG = timedCache?.cooling
            ?? this._coolingBG(handles.U0, handles.U1, handles.Bx, handles.By, dtBuffer);
        const geometryBG = timedCache?.geometry
            ?? this._geometryBG(handles.U0, handles.U1, handles.Bx, handles.By, dtBuffer);
        const conductionBG = cache?.conduction
            ?? this._conductionBG(handles.U0, handles.U1, handles.Bx, handles.By);
        const radiationBG = cache?.radiation
            ?? this._radiationBG(handles.U0, handles.U1, handles.Bx, handles.By);
        const viscosityBG = cache?.viscosity
            ?? this._viscosityBG(handles.U0, handles.U1, handles.Bx, handles.By);
        const ohmBG = cache?.ohm
            ?? this._ohmBG(handles.U0, handles.U1, handles.Bx, handles.By);
        const energyFloorBG = cache?.energyFloor
            ?? this._energyFloorBG(handles.U0, handles.U1, handles.Bx, handles.By);

        applyBcs(`${labelPrefix}.preApplyBcs`);

        // ── Self-gravity Poisson solve ─────────────────────────────
        // Runs FIRST so gravity acceleration uses the freshly-updated φ.
        // Cooling / conduction / Hall don't depend on φ so their order
        // among themselves doesn't matter for the breadth pass.
        let gravityPhi = b.phi;
        if (selfGravOn) {
            const iters = Math.max(0, this.gravityPoissonIters | 0);
            this._encodePoissonMean(encoder, handles.U0, gInterior);
            if (iters > 0) {
                gravityPhi = this._canUsePoissonMultigrid()
                    ? this._encodePoissonMultigrid(encoder, handles.U0, iters)
                    : this._encodePoissonJacobi(encoder, handles.U0, gInterior, iters);
            }
        }

        if (gravityOn || coolingOn) {
            computePass(`${labelPrefix}.localSources`, (pass) => {
                if (gravityOn) {
                    pass.setPipeline(pipelines.pipelines.applyGravity);
                    pass.setBindGroup(0, gravityBG(gravityPhi));
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                }
                if (coolingOn) {
                    pass.setPipeline(pipelines.pipelines.applyCooling);
                    pass.setBindGroup(0, coolingBG);
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                }
            });
            if (conductionOn || radiationOn || viscosityOn || ohmOn) {
                applyBcs(`${labelPrefix}.afterLocalApplyBcs`);
            }
        }

        if (conductionOn) {
            const nCond = Math.max(1, condSubsteps | 0);
            for (let c = 0; c < nCond; c++) {
                computePass(`${labelPrefix}.conduction.${c + 1}`, (pass) => {
                    pass.setPipeline(pipelines.pipelines.computeConductionDelta);
                    pass.setBindGroup(0, conductionBG);
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                    pass.setPipeline(pipelines.pipelines.applyConductionDelta);
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                });
                if (c + 1 < nCond || radiationOn || viscosityOn || ohmOn) {
                    applyBcs(`${labelPrefix}.afterConduction${c + 1}ApplyBcs`);
                }
            }
        }

        if (radiationOn) {
            const nRad = Math.max(1, opts.radiationSubsteps ?? 1);
            for (let r = 0; r < nRad; r++) {
                computePass(`${labelPrefix}.radiation.${r + 1}`, (pass) => {
                    pass.setPipeline(pipelines.pipelines.computeRadiationDelta);
                    pass.setBindGroup(0, radiationBG);
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                    pass.setPipeline(pipelines.pipelines.applyRadiationDelta);
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                });
            }
            if (viscosityOn || ohmOn) {
                applyBcs(`${labelPrefix}.afterRadiationApplyBcs`);
            }
        }

        if (viscosityOn) {
            const nVisc = Math.max(1, viscSubsteps | 0);
            for (let v = 0; v < nVisc; v++) {
                computePass(`${labelPrefix}.viscosity.${v + 1}`, (pass) => {
                    pass.setPipeline(pipelines.pipelines.computeViscosityDelta);
                    pass.setBindGroup(0, viscosityBG);
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                    pass.setPipeline(pipelines.pipelines.applyViscosityDelta);
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                });
                if (v + 1 < nVisc || ohmOn) {
                    applyBcs(`${labelPrefix}.afterViscosity${v + 1}ApplyBcs`);
                }
            }
        }

        if (ohmOn) {
            const ohmSubsteps = Math.max(1, hallSubsteps | 0, nonidealSubsteps | 0);
            for (let o = 0; o < ohmSubsteps; o++) {
                computePass(`${labelPrefix}.ohm.${o + 1}`, (pass) => {
                    pass.setPipeline(pipelines.pipelines.computeOhmEmf);
                    pass.setBindGroup(0, ohmBG);
                    pass.dispatchWorkgroups(gN1, gN1, 1);
                    pass.setPipeline(pipelines.pipelines.applyOhmHall);
                    pass.dispatchWorkgroups(gN1, gN1, 1);
                    pass.setPipeline(pipelines.pipelines.repairOhmHallEnergy);
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                    pass.setPipeline(pipelines.pipelines.applyOhmDissipative);
                    pass.dispatchWorkgroups(gN1, gN1, 1);
                    // Separate energy-repair dispatch: recompute cell pressure
                    // from the now-fully-written faces (cross-dispatch ordering
                    // makes the dissipative face writes visible here), avoiding
                    // the in-dispatch read-after-write race on neighbour faces.
                    pass.setPipeline(pipelines.pipelines.repairOhmDissipativeEnergy);
                    pass.dispatchWorkgroups(gInterior, gInterior, 1);
                });
                if (o + 1 < ohmSubsteps) {
                    applyBcs(`${labelPrefix}.afterOhm${o + 1}ApplyBcs`);
                }
            }
        }

        if (geometryOn) {
            computePass(`${labelPrefix}.geometry`, (pass) => {
                pass.setPipeline(pipelines.pipelines.applyGeometry);
                pass.setBindGroup(0, geometryBG);
                pass.dispatchWorkgroups(gInterior, gInterior, 1);
            });
        }

        computePass(`${labelPrefix}.energyFloor`, (pass) => {
            pass.setPipeline(pipelines.pipelines.energyFloor);
            pass.setBindGroup(0, energyFloorBG);
            pass.dispatchWorkgroups(gInterior, gInterior, 1);
        });
        applyBcs(`${labelPrefix}.postApplyBcs`);
        return true;
    }

    _resistiveSizingBounds() {
        const alpha = this.buffers?._etaAnomAlpha ?? 0;
        if (this.eta <= 0 && alpha <= 0) {
            return { dtSuper: 0, dtParabolic: 1.0e30 };
        }
        if (alpha > 0 && !this._lastEtaMaxValid) {
            const sMax = this.buffers?.sts_coeffs_max_s ?? STS_COEFFS_MAX_S;
            const maxRatio = Math.max(1, (sMax * sMax + sMax - 2) / 2);
            return { dtSuper: DT_MAX, dtParabolic: DT_MAX / maxRatio };
        }
        let etaMax = Math.max(this.eta || 0, this._lastEtaMax || 0);
        // Anomalous eta ~ |J|^2 can spike within a single step during a current-
        // sheet collapse, but _lastEtaMax is one step lagged — so RKL2 could
        // briefly under-size its substep count s and let the highest-k resistive
        // mode grow for one step. Inflate the estimate by a growth margin when
        // alpha > 0 so s covers a one-step anomalous jump. Costs ~sqrt(margin)
        // extra substeps; a no-op at alpha = 0 (constant eta cannot spike).
        const ETA_ANOM_RKL2_MARGIN = 4.0;
        if (alpha > 0) etaMax *= ETA_ANOM_RKL2_MARGIN;
        const hostDtPar = etaMax > 1.0e-30
            ? 0.25 * this.dx * this.dx / etaMax
            : 1.0e30;
        // RKL2's dispatch count is chosen on the CPU before the fresh GPU dt
        // can be read back. Size for the shader-side upper bound instead of
        // the previous-step dt so tight render loops cannot under-substep.
        return {
            dtSuper: DT_MAX,
            dtParabolic: Math.min(this._lastDtParabolic, hostDtPar),
        };
    }

    /**
     * Hall sub-cycling sizing (Session 15 — Tóth 2008 spirit).
     *
     * The Hall term is dispersive (whistler waves) with explicit stability bound
     *   Δt_sub ≤ O(dx² / (v_A · d_i)) = 1 / hall_rate_max
     * The macro Δt only respects the hyperbolic CFL; we run the Hall
     * 3-pass sequence N_hall times within each Strang half-step with
     * dt_sub = dt_half / N_hall to maintain stability.
     *
     * Inputs:
     *   - `_lastHallRateMax` from the previous step's `compute-dt`
     *     reduction. Cold or invalidated hints use the configured soft cap
     *     for one step so fresh GPU dt source-cap feedback is respected.
     *   - `_lastDtHyp` from same readback.
     *   - `hallSubstepsMax` user-supplied soft cap (uniform). The GPU dt
     *     reducer now shrinks future macro steps when that soft cap would be
     *     exceeded; this host path may temporarily exceed it up to a hard
     *     safety cap while dt feedback catches up.
     *
     * Returns:
     *   { nSubsteps, dtSub } where nSubsteps ≥ 1.
     */
    _hallSizing(dtMacro) {
        if ((this.physicsFlags & FLAG_HALL) === 0) {
            return { nSubsteps: 1, dtSub: dtMacro };
        }
        if (this.hallDi <= 0) {
            return { nSubsteps: 1, dtSub: dtMacro };
        }
        const safety = 0.5;
        const dtHalf = 0.5 * Math.max(dtMacro, 0);
        const softCap = Math.max(1, this.hallSubstepsMax | 0);
        if (!this._lastHallRateValid) {
            const n = Math.min(this._sourceSubstepHardCap(), softCap);
            return { nSubsteps: n, dtSub: dtHalf / n };
        }
        if (this._lastHallRateMax <= 0) {
            return { nSubsteps: 1, dtSub: dtMacro };
        }
        // Each Strang half-step sub-step must satisfy dt_sub · rate ≤ safety.
        // Add a 50% safety margin and cap by user-configured max.
        const ideal = dtHalf * this._lastHallRateMax / safety;
        const required = Math.max(1, Math.ceil(ideal));
        const n = Math.min(this._sourceSubstepHardCap(), required);
        this._lastSourceSoftCapExceeded ||= required > softCap;
        this._lastSourceHardCapExceeded ||= required > n;
        return { nSubsteps: n, dtSub: dtHalf / n };
    }

    /**
     * Conduction sub-cycling sizing (Session 15).
     *
     * Anisotropic conduction is parabolic with explicit-FE stability
     * bound dt_sub ≤ dx² / (4χ) = 1 / cond_rate_max, where χ = (γ-1)
     * κ/ρ. Sub-cycling has the same shape as the Hall pattern — the
     * compute_delta/apply_delta pair runs N_cond times within each
     * Strang half-step with dt_sub = dt_half / N_cond.
     *
     * A future iteration could fold the conduction operator into the
     * existing RKL2 super-step alongside resistivity (Phase 9 #4 in
     * docs/HANDOFF.md), which would give an O(√N_cond) rather than
     * O(N_cond) cost scaling. For N_cond ≤ 20 — the typical range —
     * the constant-factor win from sub-cycling competes with RKL2 and
     * avoids the 10-binding cap reshuffling that proper RKL2 folding
     * would require.
     *
     * Cap: the shared source sub-step ceiling is a performance target, not
     * a stability override. If the previous-step source rate says more
     * substeps are needed, we take them up to SOURCE_SUBSTEPS_HARD_MAX; the
     * compute-dt shader adds a source-cap macro limiter so steady workloads
     * come back under the configured soft cap. Cold/invalidated rate hints
     * use the soft cap immediately instead of a single optimistic substep.
     */
    _conductionSizing(dtMacro) {
        if ((this.physicsFlags & FLAG_CONDUCTION) === 0) {
            return { nSubsteps: 1, dtSub: dtMacro };
        }
        if (this.conductionKappa <= 0) {
            return { nSubsteps: 1, dtSub: dtMacro };
        }
        const safety = 0.5;
        const dtHalf = 0.5 * Math.max(dtMacro, 0);
        const softCap = this._sourceSubstepCap();
        if (!this._lastCondRateValid) {
            const n = Math.min(this._sourceSubstepHardCap(), softCap);
            return { nSubsteps: n, dtSub: dtHalf / n };
        }
        if (this._lastCondRateMax <= 0) {
            return { nSubsteps: 1, dtSub: dtMacro };
        }
        const ideal = dtHalf * this._lastCondRateMax / safety;
        const required = Math.max(1, Math.ceil(ideal));
        const n = Math.min(this._sourceSubstepHardCap(), required);
        this._lastSourceSoftCapExceeded ||= required > softCap;
        this._lastSourceHardCapExceeded ||= required > n;
        return { nSubsteps: n, dtSub: dtHalf / n };
    }

    _sourceSubstepCap() {
        return Math.max(1, (this.sourceSubstepsMax ?? this.hallSubstepsMax ?? 8) | 0);
    }

    _sourceSubstepHardCap() {
        return SOURCE_SUBSTEPS_HARD_MAX;
    }

    _viscositySizing(dtMacro) {
        if ((this.physicsFlags & FLAG_VISCOSITY) === 0) {
            return { nSubsteps: 1, dtSub: dtMacro };
        }
        const tscaleMax = MICRO_TRANSPORT_MAX_SCALE;
        const nuEff = Math.max((this.viscosityNu || 0) * tscaleMax,
                               (this.viscosityBulk || 0) * tscaleMax,
                               this.viscosityShock || 0);
        if (nuEff <= 0 || dtMacro <= 0) return { nSubsteps: 1, dtSub: dtMacro };
        const safety = 0.45;
        const dtHalf = 0.5 * Math.max(dtMacro, 0);
        const parabolicRate = 4.0 * nuEff / Math.max(this.dx * this.dx, 1e-30);
        const ideal = dtHalf * parabolicRate / safety;
        const softCap = this._sourceSubstepCap();
        const required = Math.max(1, Math.ceil(ideal));
        const n = Math.min(this._sourceSubstepHardCap(), required);
        this._lastSourceSoftCapExceeded ||= required > softCap;
        this._lastSourceHardCapExceeded ||= required > n;
        return { nSubsteps: n, dtSub: dtHalf / n };
    }

    _nonidealSizing(dtMacro) {
        const ambiOn = (this.physicsFlags & FLAG_AMBIPOLAR) !== 0
                    && this.ambipolarEta > 0
                    && this.neutralFrac > 0;
        const biermannOn = (this.physicsFlags & FLAG_BIERMANN) !== 0
                        && this.biermannCoeff !== 0
                        && this.hallElectronPressureFrac > 0;
        const electronInertiaOn = (this.physicsFlags & FLAG_ELECTRON_INERTIA) !== 0
                               && this.electronInertiaLength > 0
                               && this.electronInertiaDamping > 0;
        if (!ambiOn && !biermannOn && !electronInertiaOn) return { nSubsteps: 1, dtSub: dtMacro };

        const ambiRate = ambiOn
            ? 4.0 * this.ambipolarEta * Math.max(this.neutralFrac, 0)
                / Math.max(this.dx * this.dx, 1e-30)
            : 0.0;
        // Biermann is a source rather than diffusion; use a conservative
        // gradient-scale proxy so large coefficients get smaller explicit
        // substeps without adding another storage reduction to compute-dt.
        // This proxy is blind to the 1/rho^2 amplification at low density, so
        // the hard stability backstop is the per-substep dB_z cap in
        // apply-ohm.wgsl (BIERMANN_DBZ_CAP_FRAC), which bounds the kick where
        // the live density is actually known.
        const batteryRate = biermannOn
            ? Math.abs(this.biermannCoeff) / Math.max(this.dx, 1e-30)
            : 0.0;
        const eta4 = electronInertiaOn
            ? this.electronInertiaDamping * this.electronInertiaLength * this.electronInertiaLength
            : 0.0;
        // Hyper-resistive E_ei = -eta4 del^2 J gives dB/dt = -eta4 del^4 B.
        // The 2D composed biharmonic has spectral radius ~64/dx^4 (= (8/dx^2)^2),
        // so the forward-Euler bound is dt <= dx^4/(32 eta4) — the rate is
        // 32*eta4/dx^4, NOT the 1D 16/dx^4 (which left only ~11% margin and could
        // under-substep the highest-k current mode in 2D).
        const hyperRate = eta4 > 0
            ? 32.0 * eta4 / Math.max(this.dx ** 4, 1e-30)
            : 0.0;
        const rate = ambiRate + batteryRate + hyperRate;
        if (rate <= 0 || dtMacro <= 0) return { nSubsteps: 1, dtSub: dtMacro };

        const safety = 0.45;
        const dtHalf = 0.5 * Math.max(dtMacro, 0);
        const ideal = dtHalf * rate / safety;
        const softCap = this._sourceSubstepCap();
        const required = Math.max(1, Math.ceil(ideal));
        const n = Math.min(this._sourceSubstepHardCap(), required);
        this._lastSourceSoftCapExceeded ||= required > softCap;
        this._lastSourceHardCapExceeded ||= required > n;
        return { nSubsteps: n, dtSub: dtHalf / n };
    }

    _radiationSizing(dtMacro) {
        const radOn = (this.physicsFlags & FLAG_RADIATION) !== 0
            && this.radiationC > 0
            && (this.radiationKappaAbs > 0 || this.radiationKappaScat > 0);
        if (!radOn || dtMacro <= 0) return { nSubsteps: 1, dtSub: dtMacro };
        const kappa = Math.max(this.radiationKappaAbs || 0, 0)
                    + Math.max(this.radiationKappaScat || 0, 0);
        const dx2 = Math.max(this.dx * this.dx, 1e-30);
        const opacityMin = 0.01;
        // Only the explicit FLD diffusion term constrains the substep count.
        // The gas<->radiation absorption exchange is now integrated exactly
        // (implicit/exponential) in apply-radiation.wgsl, so it is
        // unconditionally stable and no longer enters the stiffness budget.
        // (This also retires the old exchange-rate estimate, which incorrectly
        // dropped the rho factor and under-substepped dense cells.)
        const diffusionRate = kappa > 0
            ? 4.0 * this.radiationC / Math.max(kappa * opacityMin * dx2, 1e-30)
            : 0.0;
        const rate = diffusionRate;
        if (rate <= 0) return { nSubsteps: 1, dtSub: dtMacro };
        const safety = 0.35;
        const dtHalf = 0.5 * Math.max(dtMacro, 0);
        const ideal = dtHalf * rate / safety;
        const softCap = this._sourceSubstepCap();
        const required = Math.max(1, Math.ceil(ideal));
        const n = Math.min(this._sourceSubstepHardCap(), required);
        this._lastSourceSoftCapExceeded ||= required > softCap;
        this._lastSourceHardCapExceeded ||= required > n;
        return { nSubsteps: n, dtSub: dtHalf / n };
    }

    _prepareSourceSubsteps(dtMacro) {
        this._lastSourceSoftCapExceeded = false;
        this._lastSourceHardCapExceeded = false;
        const hallSizing = this._hallSizing(dtMacro);
        const condSizing = this._conductionSizing(dtMacro);
        const viscSizing = this._viscositySizing(dtMacro);
        const nonidealSizing = this._nonidealSizing(dtMacro);
        const radiationSizing = this._radiationSizing(dtMacro);

        const ohmOn = (this.physicsFlags & (FLAG_HALL | FLAG_AMBIPOLAR | FLAG_BIERMANN | FLAG_ELECTRON_INERTIA)) !== 0;
        const ohmSubsteps = ohmOn
            ? Math.max(1, hallSizing.nSubsteps, nonidealSizing.nSubsteps)
            : 1;
        const params = new Float32Array([
            1.0 / Math.max(1, ohmSubsteps),
            1.0 / Math.max(1, condSizing.nSubsteps),
            1.0 / Math.max(1, viscSizing.nSubsteps),
            1.0 / Math.max(1, ohmSubsteps),
            1.0 / Math.max(1, radiationSizing.nSubsteps),
            0.0, 0.0, 0.0,
        ]);
        this.device.queue.writeBuffer(this.buffers.source_dt_params, 0, params.buffer);

        if ((this.physicsFlags & FLAG_HALL) !== 0 && this.hallDi > 0) {
            this._lastHallSubsteps = ohmSubsteps;
        } else {
            this._lastHallSubsteps = hallSizing.nSubsteps;
        }
        this._lastCondSubsteps = condSizing.nSubsteps;
        this._lastViscSubsteps = viscSizing.nSubsteps;
        this._lastNonidealSubsteps = ohmSubsteps;
        this._lastRadiationSubsteps = radiationSizing.nSubsteps;

        return {
            hallSubsteps: ohmSubsteps,
            condSubsteps: condSizing.nSubsteps,
            viscSubsteps: viscSizing.nSubsteps,
            nonidealSubsteps: ohmSubsteps,
            radiationSubsteps: radiationSizing.nSubsteps,
        };
    }

    step() {
        const { device } = this;
        const b = this.buffers;
        const side = b._side;  // pinned for this step's encoding

        const encoder = device.createCommandEncoder({ label: 'plasma.step.enc' });

        // Pass 1: compute dt from U(n).
        this._encodeComputeDt(encoder);
        this._encodeScaleDtHalf(encoder);

        const hasExtendedSources = (this.physicsFlags & EXTENDED_SOURCE_FLAGS) !== 0;
        const sourceSizing = hasExtendedSources
            ? this._prepareSourceSubsteps(this._lastDtHyp)
            : { hallSubsteps: 1, condSubsteps: 1, viscSubsteps: 1, nonidealSubsteps: 1, radiationSubsteps: 1 };
        if (hasExtendedSources) {
            this._encodeSourceDt(encoder);
        }

        // Pass 1.5: Strang-style source half-step on U(n). dt_half is
        // produced on-GPU from the freshly reduced macro dt, avoiding a CPU
        // readback stall before the hot path. Explicit sub-cycled sources
        // (conduction / viscosity / generalized Ohm) use GPU-divided dt
        // buffers from the same fresh half step; only their integer substep
        // counts are conservatively based on last step's source speeds.
        if (hasExtendedSources) {
            this._encodeExtendedPhysics(encoder, side,
                                        sourceSizing.hallSubsteps,
                                        sourceSizing.condSubsteps,
                                        sourceSizing.viscSubsteps,
                                        sourceSizing.nonidealSubsteps,
                                        {
                                            target: 'src',
                                            dtBuffer: b.dt_half,
                                            radiationSubsteps: sourceSizing.radiationSubsteps,
                                            labelPrefix: 'plasma.extended.preStrang',
                                        });
        }

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

        // Pass 3 (Session 8): RKL2 super-step for resistive diffusion.
        // Runs ON the side's stage-3 dst buffers (which alias U_next
        // before the swap). Substep count `s` is conservatively sized from
        // the shader-side dt upper bound plus the best host eta bound, so
        // fresh GPU dt feedback cannot under-substep a tight render loop.
        // Skipped entirely at η = 0 AND α = 0.
        const rkl2Bounds = this._resistiveSizingBounds();
        this._encodeResistivitySuperStep(
            encoder, side,
            rkl2Bounds.dtSuper,
            rkl2Bounds.dtParabolic,
        );

        // Pass 3.5: matching source half-step on U(n+1). Source physics is
        // still split from the hyperbolic RK3 operator, but the old Lie split
        // is replaced by S(dt/2) H(dt) S(dt/2), which is the right next rung
        // for stiff cooling/transport/Ohm/gravity coupling without moving the
        // whole integrator to an IMEX method.
        if (hasExtendedSources) {
            this._encodeExtendedPhysics(encoder, side,
                                        sourceSizing.hallSubsteps,
                                        sourceSizing.condSubsteps,
                                        sourceSizing.viscSubsteps,
                                        sourceSizing.nonidealSubsteps,
                                        {
                                            target: 'dst',
                                            dtBuffer: b.dt_half,
                                            radiationSubsteps: sourceSizing.radiationSubsteps,
                                            labelPrefix: 'plasma.extended.postStrang',
                                        });
        }

        // Pass 4: conservation diagnostics reduction over the just-
        // written destination state (stage 3 dst === U_next at this
        // point, before the swap below). Two dispatches in one pass —
        // per-tile partials, then a single-workgroup finalize. Output
        // (cons_out) is pulled by stats-display as a tiny scalar packet at
        // its own cadence; we just write it here.
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

        // Track approximate physical time for any skipped/busy dt readbacks.
        // Base-MHD frames do not need a copy/map after every step; source and
        // RKL2 feedback still ask for per-step reductions when active.
        this._pendingTimeSteps += 1;
        if (this._shouldReadbackDt()) this._maybeReadbackDt();

        b.swap();
        this.stepCount += 1;
        // simTime advances when the next dt feedback packet lands. If idle
        // readback is cadence-limited, _pendingTimeSteps folds the skipped
        // steps into that later update.
    }

    /**
     * Merge a dt_buf readback into the host-side feedback cache. StatsDisplay
     * also calls this when it already has the dt packet, which lets idle
     * base-MHD runs avoid a dedicated copy submit on every physics step.
     */
    syncDtFeedback(dt) {
        const pending = this._pendingTimeSteps;
        if (Number.isFinite(dt[0]) && dt[0] > 0) {
            this._lastDtHyp = dt[0];
            this.lastDt = dt[0];
            if (pending > 0) this.simTime += dt[0] * pending;
            this._pendingTimeSteps = 0;
        } else if (pending > 0 && Number.isFinite(this._lastDtHyp) && this._lastDtHyp > 0) {
            this.simTime += this._lastDtHyp * pending;
            this._pendingTimeSteps = 0;
        }
        if (Number.isFinite(dt[1]) && dt[1] > 0) this._lastDtParabolic = dt[1];
        if (Number.isFinite(dt[2])) {
            this._lastEtaMax = dt[2];
            this._lastEtaMaxValid = true;
        }
        if (Number.isFinite(dt[3]) && dt[3] >= 0) {
            this._lastHallRateMax = dt[3];
            this._lastHallRateValid = true;
        }
        if (Number.isFinite(dt[4]) && dt[4] >= 0) {
            this._lastCondRateMax = dt[4];
            this._lastCondRateValid = true;
        }
    }

    _needsPerStepDtFeedback() {
        const flags = this.physicsFlags || 0;
        if ((flags & FLAG_HALL) !== 0 && this.hallDi > 0) return true;
        if ((flags & FLAG_CONDUCTION) !== 0 && this.conductionKappa > 0) return true;
        if ((this.eta || 0) > 0 || (this.etaAnomAlpha || 0) > 0) return true;
        return false;
    }

    _shouldReadbackDt() {
        if (this._needsPerStepDtFeedback()) return true;
        return (this.stepCount % DT_READBACK_IDLE_STRIDE) === 0;
    }

    /**
     * Fire an async readback of dt_buf (5 f32 slots: dt_hyp, dt_parabolic,
     * eta_max, hall_rate_max, cond_rate_max). Skips silently if a previous
     * readback hasn't completed.
     */
    _maybeReadbackDt() {
        if (this._dtReadbackBusy) return;
        if (!this._dtReadbackPool) return;
        this._dtReadbackBusy = true;
        const generation = this._bufferGeneration;
        const dtBuffer = this.buffers.dt;
        readbackSlice(this.device, this._dtReadbackPool,
                      dtBuffer, 0, 20)
            .then(ab => {
                if (generation !== this._bufferGeneration || dtBuffer !== this.buffers.dt) return;
                const arr = new Float32Array(ab);
                this.syncDtFeedback(arr);
            })
            .catch(e => {
                // Validation errors on stale buffers can fire during
                // setResolution; drop silently and let the next call
                // re-establish a healthy readback.
                console.warn('[plasma] dt readback:', e);
            })
            .finally(() => { this._dtReadbackBusy = false; });
    }

    /**
     * Apply a user-driven pointer perturbation to the CURRENT primary
     * buffers (U0_n / U1_n / Bx_n / By_n at the side that will be read
     * by the next `step()`). The dispatch goes in its own command buffer
     * — WebGPU queue ordering guarantees it lands before the next step's
     * compute-dt reads U_n.
     *
     * Args (all in interior-frame domain coords / units):
     *   kind:    'drag' | 'excite'
     *   cx, cy:  Gaussian center, in domain units (0 = interior left edge)
     *   dvec_x, dvec_y: drag-vector magnitude × direction (code velocity
     *           for drag, code-B for excite)
     *   sigma:  Gaussian σ in domain units
     *   amplitude: scalar pre-multiplier on the deposited δ(ρv) or δAz
     *
     * For drag, the deposited momentum at the center cell is
     *   δm = amplitude · ρ · (dvec_x, dvec_y),
     * so amplitude=1 means "drag this cell at exactly (dvec_x, dvec_y)".
     *
     * For excite, the curl-of-Az evaluated at the center gives δB =
     * amplitude · (dvec_x, dvec_y). Smaller amplitudes for non-disruptive
     * perturbations; the default scaling lives in ui.js.
     */
    applyPerturbation({ kind, cx, cy, dvec_x, dvec_y, sigma, amplitude = 1 }) {
        if (!this.buffers || !this.pipelines) return;
        if (!Number.isFinite(cx) || !Number.isFinite(cy)) return;
        if (!Number.isFinite(dvec_x) || !Number.isFinite(dvec_y)) return;
        if (!Number.isFinite(sigma) || sigma <= 0) return;
        if (kind !== 'drag' && kind !== 'excite') return;

        const b = this.buffers;
        b.pushPerturbUniforms({
            cx, cy,
            dvec_x, dvec_y,
            sigma,
            amplitude,
        });

        const side = b._side;
        const bg = this._bgCache.perturb[side];
        const { pipelines } = this.pipelines;
        const N = this.n;
        const gInterior = Math.ceil(N / WG);
        const gN1       = Math.ceil((N + 1) / WG);

        const encoder = this.device.createCommandEncoder({ label: 'plasma.perturb.enc' });
        const pass = encoder.beginComputePass({ label: 'plasma.perturb.pass' });
        pass.setBindGroup(0, bg);

        if (kind === 'drag') {
            pass.setPipeline(pipelines.perturbDrag);
            pass.dispatchWorkgroups(gInterior, gInterior, 1);
        } else {
            // excite: face B update (one extra row+col so every interior
            // face has an owning invocation), then cell E re-sync. The
            // two dispatches share one pass; WebGPU guarantees the second
            // sees the first's writes.
            pass.setPipeline(pipelines.perturbExciteB);
            pass.dispatchWorkgroups(gN1, gN1, 1);
            pass.setPipeline(pipelines.perturbExciteEnergy);
            pass.dispatchWorkgroups(gInterior, gInterior, 1);
        }
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    render() {
        const licEnabled = this.licIntensity > 1.0e-6;
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
        this.renderer.render({ licEnabled });
    }
}

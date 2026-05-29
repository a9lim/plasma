/**
 * @fileoverview Simulation constants — Phase 4.
 *
 * Locked here so they're easy to grep and easy to expose later as
 * advanced-tab sliders without touching shaders. Values mirror the
 * Phase-4 spec in ~/.claude/plans/geon-currently-uses-cpu-abstract-cat.md.
 */

// Grid resolution — square. 256² default per locked decision. This is the
// INTERIOR resolution; storage buffers expand by GHOST_WIDTH per side.
export const GRID_N = 256;

// Ghost-cell width per side. PPM's 5-cell stencil reads [i-2, i+2], so we
// need 2 layers on every edge. Cell-centered storage is (N+4)×(N+4); the
// interior range is [GHOST_WIDTH, GHOST_WIDTH+N) in both axes.
export const GHOST_WIDTH = 2;

// Total cell-centered storage width (one axis). Mirror in shaders via the
// `grid_n_total` uniform field — see shared-helpers.wgsl.
export const GRID_N_TOTAL = GRID_N + 2 * GHOST_WIDTH;

// Workgroup tile size. 8×8 = 64 invocations per dispatch group; fits the
// hardware-typical 64-thread wavefront and keeps PPM stencil access local.
export const WORKGROUP = 8;

// CFL number for RK3 SSP with PPM+HLLD. 0.4 leaves headroom above the
// 0.5 textbook ceiling for 2D dimensional splitting.
export const CFL = 0.4;

// Adiabatic index. Sod preset overrides this to 1.4 in presets.js.
export const GAMMA_DEFAULT = 5.0 / 3.0;

// Domain extent in simulation units. dx is derived; we keep a square box
// per the default unless a preset overrides it.
export const DOMAIN_LENGTH = 1.0;

// Pressure floor on every primitive recovery — guards HLL flux + sound-speed
// computations against negative-pressure edge cases.
export const PRESSURE_FLOOR = 1e-6;

// Density floor — analogous safety for ρ ≤ 0 (shouldn't happen, but cheap).
export const DENSITY_FLOOR = 1e-6;

// Hard cap on the per-step dt. Defends against initial-condition transients
// where the wave-speed reduction reports something pathologically tiny.
export const DT_MAX = 1e-2;

// Soft floor on dt so a stalled simulation doesn't grind frames to zero.
export const DT_MIN = 1e-8;

// Default explicit resistivity. Harris reconnection acid-test runs at this
// value. UI exposure is Phase 5; the snap-to-0 logic also lives there.
export const ETA_DEFAULT = 1e-3;

// View-mode enum. Phase 3a adds |B| and Jz for the MHD view; Phase 5+ adds
// the remaining options (β, vorticity, Schlieren). Session 15 adds three
// extended-physics views — temperature, heat-flux magnitude, gravitational
// potential — so cooling / conduction / self-gravity are directly visible.
export const VIEW_DENSITY  = 0;
export const VIEW_PRESSURE = 1;
export const VIEW_VMAG     = 2;
export const VIEW_BMAG     = 3;
export const VIEW_JZ       = 4;
export const VIEW_T        = 5;  // T = p / ρ
export const VIEW_QMAG     = 6;  // |q| heat flux magnitude (anisotropic Spitzer)
export const VIEW_PHI      = 7;  // gravitational potential φ
export const VIEW_ENTROPY  = 8;  // entropy proxy K = p / ρ^γ

// Normalization window for the default density view. Sod expects ρ ∈ [0.125,
// 1.0] initially; we give a small margin in both directions.
export const VIEW_DENSITY_MIN = 0.05;
export const VIEW_DENSITY_MAX = 1.10;

// Boundary-condition modes (per-edge). Folded into one shader via switch
// on a `bc_uniforms` storage buffer holding 4 u32 mode IDs (N, S, E, W).
export const BC_PERIODIC   = 0;
export const BC_OUTFLOW    = 1;
export const BC_REFLECTING = 2;
export const BC_DRIVEN     = 3;

// Edge indices into the bc_uniforms storage buffer's mode array.
export const EDGE_N = 0;  // top
export const EDGE_S = 1;  // bottom
export const EDGE_E = 2;  // right
export const EDGE_W = 3;  // left

// Uniform-buffer layout (Extended physics — slots 16-63 added for Hall,
// cooling, conduction, gravity, non-ideal MHD, transport, EMF mode, physics flags). See `Uniforms`
// struct in shared-helpers.wgsl for the canonical layout.
//   Slots 0-15  (original 64 B): dx, gamma, view_min, view_max, eta,
//                eta_anom_alpha, _pad×2, grid_n, grid_n_total, ghost_w,
//                pressure_floor, cfl, view_mode, eta_anom_jcrit, noise_n.
//   Slots 16-31 (Session 14/15 64 B): hall_di, hall_substeps_max, cooling_lambda0,
//                cooling_T_floor, cooling_T_ref, conduction_kappa,
//                conduction_iso_frac, conduction_sat_frac, gravity_gx,
//                gravity_gy, gravity_G, gravity_poisson_iters,
//                physics_flags, emf_mode, cooling_curve_mode,
//                hall_electron_pressure_frac.
//   Slots 32-63 (Session 17+ 128 B): cooling/heating shape, ambipolar and
//                Biermann terms, viscosity, shared source substep cap,
//                cylindrical geometry, softened/relaxed Poisson, sponge
//                layer, grey radiation transfer, electron-inertia smoothing,
//                Poisson boundary mode,
//                and reserved headroom.
// 64 × 4B = 256B.
//
// Sweep direction is in two static SweepDir uniform buffers (16 B each)
// bound by reconstruct-ppm + riemann-hlld. LIC render-pace state
// (phase, intensity, drift_x, drift_y) is in a separate 16 B
// LicUniforms buffer rewritten per render frame.
export const UNIFORM_BUFFER_SIZE = 256;

// ── Extended physics flags (slot 28: physics_flags bitfield) ───────────
// Each feature is OFF by default — extended physics is opt-in so the
// existing presets behave exactly as before unless the user enables them.
export const FLAG_COOLING      = 1 << 0;
export const FLAG_GRAVITY_EXT  = 1 << 1;  // constant external g vector
export const FLAG_GRAVITY_SELF = 1 << 2;  // Poisson-solved self-gravity
export const FLAG_CONDUCTION   = 1 << 3;  // anisotropic thermal conduction
export const FLAG_HALL         = 1 << 4;  // Hall MHD correction to face B
export const FLAG_POSITIVITY   = 1 << 5;  // stronger positivity guard
export const FLAG_EMF_UPWIND   = 1 << 6;  // GS upwind EMF (vs BS arithmetic mean)
export const FLAG_AMBIPOLAR    = 1 << 7;  // ion-neutral ambipolar magnetic diffusion
export const FLAG_BIERMANN     = 1 << 8;  // Biermann battery from ∇ρ × ∇p_e
export const FLAG_VISCOSITY    = 1 << 9;  // physical/shear/bulk viscosity
export const FLAG_GEOMETRY     = 1 << 10; // axisymmetric cylindrical source terms
export const FLAG_SPONGE       = 1 << 11; // boundary sponge layer
export const FLAG_HEATING      = 1 << 12; // volumetric heating balancing radiative losses
export const FLAG_RADIATION    = 1 << 13; // grey radiation energy diffusion/coupling
export const FLAG_ELECTRON_INERTIA = 1 << 14; // kinetic-scale hyper-resistive Ohm closure

// Baseline flags used by the canonical verification presets. Positivity and
// upwind CT are numerical guards for the ideal/resistive MHD core; cooling,
// gravity, conduction, and Hall are opt-in source physics.
export const BASE_PHYSICS_FLAGS = FLAG_POSITIVITY | FLAG_EMF_UPWIND;
export const EXTENDED_SOURCE_FLAGS = FLAG_COOLING
                                   | FLAG_GRAVITY_EXT
                                   | FLAG_GRAVITY_SELF
                                   | FLAG_CONDUCTION
                                   | FLAG_HALL
                                   | FLAG_AMBIPOLAR
                                   | FLAG_BIERMANN
                                   | FLAG_VISCOSITY
                                   | FLAG_GEOMETRY
                                   | FLAG_SPONGE
                                   | FLAG_HEATING
                                   | FLAG_RADIATION
                                   | FLAG_ELECTRON_INERTIA;
export const EXTENDED_PHYSICS_FLAGS = BASE_PHYSICS_FLAGS | EXTENDED_SOURCE_FLAGS;

// EMF mode enum (slot 29: emf_mode).
export const EMF_MODE_BS_MEAN  = 0;  // Balsara-Spicer arithmetic mean (legacy fallback)
export const EMF_MODE_GS_UPWIND = 1; // Gardiner-Stone 2005 upwind default

// Optically-thin cooling curve selector (slot 30). BREMS preserves the old
// single √T exact integrator; TABLE uses the Session-16 compact power-law
// curve. CIE uses a broader collisional-ionization-equilibrium-inspired
// solar-metallicity table with metallicity scaling and heating support.
// TABULATED samples the uploaded microphysics storage table, letting the
// physics layer swap curves without shader edits.
export const COOLING_CURVE_BREMS = 0;
export const COOLING_CURVE_TABLE = 1;
export const COOLING_CURVE_CIE   = 2;
export const COOLING_CURVE_TABULATED = 3;

// Geometry source selector (slot 45).
export const GEOMETRY_CARTESIAN   = 0;
export const GEOMETRY_CYLINDRICAL = 1; // x is cylindrical radius r, y is z

// Self-gravity Poisson boundary selector (slot 59).
export const GRAVITY_BOUNDARY_PERIODIC = 0; // ∇²φ = 4πG(ρ-ρ̄), periodic box
export const GRAVITY_BOUNDARY_ISOLATED = 1; // ∇²φ = 4πGρ, zero-φ exterior

// Host-side self-gravity solver selector. This is intentionally not a
// Uniforms slot: shaders are selected by the orchestrator so the older
// Jacobi path can remain the cylindrical/fallback implementation.
export const GRAVITY_SOLVER_MULTIGRID = 0;
export const GRAVITY_SOLVER_JACOBI    = 1;

// bc_uniforms storage-buffer layout:
//   u32 mode[4]  — N, S, E, W
//   f32 driven_{edge}_{rho,vx,vy,vz,bx,by,bz,p} for N, S, E, W
// 4*4 + 4*8*4 = 144B, padded to 160B for alignment headroom.
export const BC_UNIFORM_BUFFER_SIZE = 160;

// ── LIC (line integral convolution) — Phase 6 ──────────────────────────
// Animated noise advection along B-field. The noise base is resolution-
// independent (LIC samples it with bilinear interpolation), so we ship a
// single 1024×1024 white-noise texture once at init and reuse across
// resolution changes.
export const LIC_NOISE_N = 1024;

// PRNG seed for the deterministic white-noise base. Any 32-bit integer;
// the actual value doesn't matter — pick something memorable for the
// build to be reproducible. (Mulberry32 PRNG, see buffers.js.)
export const LIC_NOISE_SEED = 0xC0FFEE;

// Number of backward-trace steps per pixel. 20 × 0.5 cells = ~10 cells
// of trace length — long enough to resolve magnetic islands and short
// enough to stay performant at 1024².
export const LIC_STEPS = 20;

// Backward-trace step length in cell widths. RK2 midpoint integration.
export const LIC_STEP_SIZE = 0.5;

// Default UI slider value for LIC intensity. 0 = no LIC modulation,
// 1 = full strength. Composite blends colormap by `mix(1, L, intensity*0.5+0.5)`,
// so intensity=1 means a ±50% luminance modulation.
export const LIC_INTENSITY_DEFAULT = 1.0;

// Drift direction in noise-pixel units per second of wall-clock time.
// The shader adds `lic_phase * drift_{x,y}` to the noise sample
// position, so `drift` is the rate at which the noise pattern
// translates relative to traced positions.
// Phase 6 default: horizontal drift at 0.5 noise-pixel/sec — slow and
// directional rather than chaotic. At a 1024² noise base with a 256²
// grid, 0.5 noise-pixel/sec corresponds to ~0.125 cell/sec of apparent
// motion — perceptible without being distracting.
export const LIC_DRIFT_X = 0.5;
export const LIC_DRIFT_Y = 0.0;

// Epsilon below which |B| is treated as effectively zero (trace halts).
// Prevents runaway sampling in field-free regions.
export const LIC_B_EPS = 1.0e-8;

// ── View colormap windows ──────────────────────────────────────────────
// Default linear (view_min, view_max) normalization endpoints per view
// mode, keyed by the VIEW_* enum above. Presets may override via
// preset.viewMin / preset.viewMax; sim.setViewMode falls back to these.
// Signed fields (Jz, φ) use symmetric windows. Moved out of a switch in
// sim.js so all visualization clamps live next to the enum they key on.
export const VIEW_RANGES = {
    [VIEW_DENSITY]:  { min: 0.05,  max: 1.10 },   // ρ
    [VIEW_PRESSURE]: { min: 0.01,  max: 1.00 },   // p
    [VIEW_VMAG]:     { min: 0.0,   max: 1.5 },    // |v|
    [VIEW_BMAG]:     { min: 0.0,   max: 2.0 },    // |B|
    [VIEW_JZ]:       { min: -3.0,  max: 3.0 },    // J_z (signed)
    [VIEW_T]:        { min: 0.0,   max: 1.0 },    // T = p/ρ
    [VIEW_QMAG]:     { min: 0.0,   max: 1.0e-3 }, // |q| (κ_∥·dT/dx scale)
    [VIEW_PHI]:      { min: -5e-3, max: 5e-3 },   // φ (signed)
    [VIEW_ENTROPY]:  { min: 0.0,   max: 1.0 },    // K = p/ρ^γ
};

// ── Solver safety caps ──────────────────────────────────────────────────
// Hard ceiling on explicit source sub-cycles within one macro Δt. The
// user-facing soft caps (hallSubstepsMax / sourceSubstepsMax) are
// performance targets; this is the stability backstop the host clamps to
// when an estimate would exceed it. See sim.js _sourceSubsteps.
export const SOURCE_SUBSTEPS_HARD_MAX = 128;

// Max RKL2 super-step length we will allocate coefficient/meta buffers
// for. Caps a single resistive super-step so it never blocks the UI
// thread; the host clamps s to this. See gpu/buffers.js.
export const STS_COEFFS_MAX_S = 100;

// ── Extended-physics default state ──────────────────────────────────────
// Fallback scalars + flags applied on every loadPreset for fields a preset
// doesn't override. Canonical verification presets keep these values but
// leave the source-flag bits clear (extended physics is opt-in). Lives here
// (not sim.js) so every tunable default sits in one file; references the
// flag / enum constants defined above. The `physics` block is part of the
// save/load schema — see sim.js _applyPhysicsConfig.
export const DEFAULT_PHYSICS_STATE = Object.freeze({
    physicsFlags: BASE_PHYSICS_FLAGS,
    emfMode: EMF_MODE_GS_UPWIND,
    hallDi: 0.02,
    hallSubstepsMax: 8,
    coolingLambda0: 0.01,
    coolingTFloor: 1.0e-4,
    coolingTRef: 1.0,
    coolingCurveMode: COOLING_CURVE_TABULATED,
    conductionKappa: 1.0e-3,
    conductionIsoFrac: 0.1,
    conductionSatFrac: 0.0,
    gravityGx: 0.0,
    gravityGy: 0.0,
    gravityG: 1.0e-3,
    gravityPoissonIters: 30,
    gravityBoundaryMode: GRAVITY_BOUNDARY_PERIODIC,
    gravitySolverMode: GRAVITY_SOLVER_MULTIGRID,
    hallElectronPressureFrac: 0.0,
    coolingMetallicity: 1.0,
    heatingGamma0: 0.0,
    heatingDensityExp: 1.0,
    heatingTCut: 0.0,
    ambipolarEta: 0.0,
    biermannCoeff: 0.0,
    neutralFrac: 0.0,
    ionizationT0: 1.0,
    viscosityNu: 0.0,
    viscosityBulk: 0.0,
    viscosityAnisoFrac: 0.0,
    viscosityShock: 0.0,
    sourceSubstepsMax: 8,
    geometryMode: GEOMETRY_CARTESIAN,
    geometryRMin: 0.0,
    gravitySoftening: 0.0,
    gravityPoissonOmega: 1.0,
    spongeWidth: 0.0,
    spongeStrength: 0.0,
    coolingTableMix: 0.0,
    radiationC: 0.0,
    radiationKappaAbs: 0.0,
    radiationKappaScat: 0.0,
    radiationConst: 1.0,
    radiationFloor: 1.0e-12,
    electronInertiaLength: 0.0,
    electronInertiaDamping: 0.0,
});

// Anomalous-resistivity default activation threshold J_crit (code units).
// α = 0 (the default) keeps the constant-η baseline; J_crit only matters
// once the η-anomalous slider arms α > 0. See sim.js.
export const ETA_ANOM_JCRIT_DEFAULT = 10.0;

// ── Pointer perturbation tuning ─────────────────────────────────────────
// Left-drag deposits Gaussian momentum (δv ≈ DRAG_VSCALE × Δcell at the
// center cell); right-drag deposits a divergence-free B rotation
// (δB ≈ EXCITE_BSCALE × Δcell). σ defaults to PERTURB_SIGMA_CELLS cells at
// the live resolution so the footprint is visible without dominating.
// PERTURB_MAX_DCELL clamps per-event drag so fast flicks don't blow up a
// cell. Consumed by ui.js wirePointerPerturbation.
export const PERTURB_SIGMA_CELLS = 4.0;
export const DRAG_VSCALE         = 0.25;  // cell-velocity per cell-of-drag
export const EXCITE_BSCALE       = 0.20;  // code-B per cell-of-drag
export const PERTURB_MAX_DCELL   = 6.0;   // per-event drag clamp (cells)

// ── Playback + diagnostics cadence ──────────────────────────────────────
// Topbar speed-cycle multipliers (steps per displayed frame). Index 0 is
// the startup value.
export const SPEED_OPTIONS = [1, 2, 4, 8, 16];

// Stats sparkline ring-buffer capacity (≈ 20 s at 12 Hz). See stats-display.js.
export const STATS_SPARK_CAP = 240;

// Stats readback cadence in render frames between GPU→CPU stat copies,
// resolution-adaptive (≈12 / 6 / 3 Hz at 60 fps). Coarser at high res so
// the readback never dominates the frame budget. See stats-display.js.
export const READBACK_CADENCE = Object.freeze({
    lo:  5,   // n < 512
    mid: 10,  // n >= 512
    hi:  20,  // n >= 1024
});

// Probe (hover-sample) readback frequency, independent of render cadence.
export const PROBE_HZ = 10;  // 100 ms interval

// Dual-energy switchover: trust the independently-advected internal energy
// for pressure only when it exceeds this fraction of total E (else the
// E − KE − B² subtraction is well-conditioned). See probe.js / shaders.
export const DUAL_ENERGY_FRACTION = 1e-3;

// ── UI slider tuning ────────────────────────────────────────────────────
// η log-slider snap-to-0 floor: values at or below this read as "0" (ideal
// MHD) when the active preset has no grid-Reynolds η floor. See ui.js.
export const ETA_SLIDER_MIN_FLOOR = -6.5;

// Every interactive slider's geometry in one table so ranges can be
// retuned without hunting through ui.js / physics-panel.js. Two shapes:
//   • linear sliders        → { min, max, step }
//   • log10 / snap-off sliders → { lo, hi, step }  (the builder extends the
//     range 0.5 decade below `lo` for the off zone; off boundary = lo−0.05)
// Linear sliders that snap to 0 at the bottom (heatingTCut, radiation κ's,
// viscosity bulk/shock, electron damping) use { min, max, step } and treat
// `min + 0.05` as their off boundary. Keys are grouped by the panel section
// they render in.
export const SLIDER_BOUNDS = Object.freeze({
    // Settings ▸ Numerics
    cfl:                { min: 0.1, max: 0.8, step: 0.05 },
    gamma:              { min: 1.1, max: 2.0, step: 0.05 },
    pressureFloorLog:   { min: -8,  max: -3,  step: 0.5 },   // log10(p_floor)
    sourceSubstepsCap:  { min: 1,   max: 64,  step: 1 },

    // Settings ▸ Resistivity η
    etaAnomAlpha:       { lo: -6, hi: 0, step: 0.25 },        // log10; snap-to-0, no flag
    etaAnomJcrit:       { min: 1, max: 100, step: 1 },

    // Settings ▸ Render
    licIntensity:       { min: 0, max: 2, step: 0.05 },
    licDrift:           { min: 0, max: 4, step: 0.1 },

    // Physics ▸ Hall
    hallDi:             { lo: -4, hi: 0, step: 0.25 },        // log10; FLAG_HALL
    hallElectronPe:     { min: 0, max: 1, step: 0.05 },       // p_e/p

    // Physics ▸ Cooling & Heating
    coolingLambda0:     { lo: -4, hi: 0, step: 0.25 },        // log10; FLAG_COOLING
    coolingMetallicity: { min: 0, max: 3, step: 0.1 },
    heatingGamma0:      { lo: -6, hi: 0, step: 0.25 },        // log10; FLAG_HEATING
    heatingDensityExp:  { min: 0, max: 2, step: 0.1 },
    heatingTCut:        { min: -4.5, max: 2, step: 0.25 },    // log10; snap-to-0

    // Physics ▸ Conduction
    conductionKappa:    { lo: -6, hi: 0, step: 0.25 },        // log10; FLAG_CONDUCTION
    conductionIsoFrac:  { min: 0, max: 1, step: 0.05 },
    conductionSatFrac:  { min: 0, max: 1, step: 0.05 },

    // Physics ▸ Radiation
    radiationC:         { lo: -2, hi: 2, step: 0.25 },        // log10; FLAG_RADIATION
    radiationKappaAbs:  { min: -4.5, max: 2, step: 0.25 },    // log10; snap-to-0
    radiationKappaScat: { min: -4.5, max: 2, step: 0.25 },    // log10; snap-to-0
    radiationConst:     { min: -4, max: 1, step: 0.25 },      // log10 a_r

    // Physics ▸ Viscosity
    viscosityNu:        { lo: -7, hi: -1, step: 0.25 },       // log10; FLAG_VISCOSITY
    viscosityBulk:      { min: -7.5, max: -1, step: 0.25 },   // log10; snap-to-0
    viscosityAnisoFrac: { min: 0, max: 1, step: 0.05 },
    viscosityShock:     { min: -7.5, max: -1, step: 0.25 },   // log10; snap-to-0

    // Physics ▸ Non-ideal Ohm
    ambipolarEta:       { lo: -7, hi: -1, step: 0.25 },       // log10; FLAG_AMBIPOLAR
    neutralFrac:        { min: 0, max: 1, step: 0.05 },
    ionizationT0:       { min: -4, max: 2, step: 0.25 },      // log10
    biermannCoeff:      { lo: -8, hi: -1, step: 0.25 },       // log10; FLAG_BIERMANN
    electronInertiaLen: { lo: -5, hi: -1, step: 0.25 },       // log10; FLAG_ELECTRON_INERTIA
    electronDamping:    { min: -4.5, max: 0, step: 0.25 },    // log10; snap-to-0

    // Physics ▸ Gravity
    gravityG:           { lo: -4, hi: 2, step: 0.25 },        // log10; FLAG_GRAVITY_SELF
    gravityPoissonIters:{ min: 8, max: 128, step: 1 },   // >=1 enforced in sim.js so the solve never silently skips (stale-phi-as-force)
    gravitySoftening:   { min: 0, max: 0.2, step: 0.005 },
    gravityPoissonOmega:{ min: 0.2, max: 1.8, step: 0.05 },

    // Physics ▸ Geometry & sponge
    geometryRMin:       { min: 0, max: 0.25, step: 0.005 },
    spongeWidth:        { min: 0, max: 32, step: 1 },
    spongeStrength:     { lo: -3, hi: 1, step: 0.25 },        // log10; FLAG_SPONGE
});

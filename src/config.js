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
// the remaining options (β, vorticity, Schlieren).
export const VIEW_DENSITY  = 0;
export const VIEW_PRESSURE = 1;
export const VIEW_VMAG     = 2;
export const VIEW_BMAG     = 3;
export const VIEW_JZ       = 4;

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

// Uniform-buffer layout (Round 2): see `Uniforms` struct in shared-helpers.wgsl.
//   f32 dx, gamma, view_min, view_max, eta, _pad_lic_0, _pad_lic_1, _pad_lic_2
//   u32 grid_n, grid_n_total, ghost_w, _pad_sweep
//   f32 cfl                                  (slot 12)
//   u32 view_mode                            (slot 13)
//   f32 _pad_lic_3                           (slot 14)
//   u32 noise_n                              (slot 15)
// 16 × 4B = 64B.
//
// Sweep direction is in two static SweepDir uniform buffers (16 B each)
// bound by reconstruct-ppm + riemann-hlld. LIC render-pace state
// (phase, intensity, drift_x, drift_y) is in a separate 16 B
// LicUniforms buffer rewritten per render frame.
export const UNIFORM_BUFFER_SIZE = 64;

// bc_uniforms storage-buffer layout:
//   u32 mode[4]  — N, S, E, W
//   f32 driven_rho, driven_vx, driven_vy, driven_vz, driven_bx, driven_by, driven_bz, driven_p
// 4*4 + 8*4 = 48B, padded to 64B for alignment headroom.
export const BC_UNIFORM_BUFFER_SIZE = 64;

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

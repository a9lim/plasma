/**
 * @fileoverview Phase 2 simulation constants.
 *
 * Locked here so they're easy to grep and easy to expose later as
 * advanced-tab sliders without touching shaders. Values mirror the
 * Phase-2 spec in ~/.claude/plans/geon-currently-uses-cpu-abstract-cat.md.
 */

// Grid resolution — square. 256² default per locked decision.
export const GRID_N = 256;

// Workgroup tile size. 8×8 = 64 invocations per dispatch group; fits the
// hardware-typical 64-thread wavefront and keeps PLM stencil access local.
export const WORKGROUP = 8;

// CFL number for forward Euler with HLL+PLM. 0.4 leaves headroom above the
// 0.5 textbook ceiling for 2D dimensional splitting.
export const CFL = 0.4;

// Adiabatic index. Sod preset overrides this to 1.4 in presets.js.
export const GAMMA_DEFAULT = 5.0 / 3.0;

// Domain extent in simulation units. dx is derived; we keep a square box.
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

// Uniform-buffer layout (Phase 3a): see `Uniforms` struct in shared-helpers.wgsl.
//   f32 dx, gamma, view_min, view_max,
//   u32 grid_n, sweep_dir, step_parity, view_mode
// 8 × 4B = 32B, padded to 64B for future room.
export const UNIFORM_BUFFER_SIZE = 64;

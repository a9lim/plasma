// ─── shared-helpers.wgsl ─────────────────────────────────────────────
// Phase 4: 2.5D resistive MHD on a Yee-style staggered grid with
// 2-layer ghost cells per side and per-edge boundary conditions.
//
//   • PPM reconstruction (Colella & Woodward 1984) — produces per-cell,
//     per-direction L/R primitive face states (no slopes).
//   • HLLD Riemann (Miyoshi & Kusano 2005) — 5-wave structure, with
//     HLLC and HLL fallbacks for the three documented degenerate branches.
//   • RK3 SSP — three storage slots (n/1/2) for U_cell + face B,
//     updated in lockstep. Stage weights live in three small uniform
//     buffers (stage_1, stage_2, stage_3), written once at init.
//   • Explicit resistivity — η ∇²B added per RK3 stage AFTER the CT
//     update. SSP-compatible by linearity (η ∇² is a linear operator on B).
//   • Per-edge BCs filled by apply-bcs.wgsl at the start of each stage,
//     into 2 ghost-cell layers per side. The main shaders drop wrap_idx
//     in favour of direct indexing.
//
// ── Transpiler contract (vanilla WebGPU only) ───────────────────────
// The codebase is written so a future WebGPU→CPU JS transpiler can map
// each compute dispatch onto a clean nested loop:
//   • No subgroup ops, no indirect dispatch, no push constants.
//   • All bind-group layouts are static: one uniform + N storage buffers,
//     no dynamic offsets.
//   • Workgroup-shared memory only in compute-dt's reduction (a textbook
//     tile-max pattern; trivially maps to a per-workgroup loop on CPU).
//   • Atomics confined to compute-dt (`atomic<u32>` over float bits via
//     bitcast); easy CPU emulation.
//   • Per-stage parameters are immutable uniform buffers; per-sweep
//     direction is a second immutable uniform buffer. The transpiler
//     just binds the right pair per dispatch.
//   • BC dispatch is a single shader with a switch over 4 modes × 4
//     edges = 16 cases — no shader permutations, no specialization
//     constants.
//
// ── Ghost-cell convention (Phase 4) ─────────────────────────────────
// All buffers expand by GHOST_W = 2 cells per side. With interior size
// N×N, cell-centered buffer size is (N+2*GHOST_W) × (N+2*GHOST_W) =
// (N+4)×(N+4). The INTERIOR index range in both axes is:
//     [GHOST_W, GHOST_W + N) = [2, N+2)
// Ghost cells occupy:
//     i ∈ [0, GHOST_W)              → west / left ghost  (W edge)
//     i ∈ [GHOST_W+N, N+2*GHOST_W)  → east / right ghost (E edge)
//     j ∈ [0, GHOST_W)              → south / bottom ghost (S edge)
//     j ∈ [GHOST_W+N, N+2*GHOST_W)  → north / top    ghost (N edge)
// apply-bcs.wgsl runs at the start of each RK3 stage and fills these
// strips per BC mode for that edge. Corners use whichever non-periodic
// edge owns them, falling back to either side if both are periodic.
//
// ── Face / edge convention (Yee staggered, Phase 4 — LEFT/DOWN owner) ─
// Cell (i,j) is bounded by:
//   Bx_face[i,   j]  on its LEFT  (x-face at i-½)
//   Bx_face[i+1, j]  on its RIGHT (x-face at i+½)
//   By_face[i, j  ]  on its BOTTOM (y-face at j-½)
//   By_face[i, j+1]  on its TOP    (y-face at j+½)
// Bx_face dimensions: (N+2*GHOST_W + 1) × (N+2*GHOST_W) = (N+5) × (N+4).
// By_face dimensions: (N+2*GHOST_W)     × (N+2*GHOST_W + 1) = (N+4) × (N+5).
// Ez_edge sits at cell corners (i-½, j-½), dim (N+5) × (N+5).
//
// Discrete CT update (forward Euler):
//   Bx_face[i,j] += -(dt/dy) · (Ez_edge[i, j+1] - Ez_edge[i, j])
//   By_face[i,j] += +(dt/dx) · (Ez_edge[i+1,j] - Ez_edge[i, j])
// where the Ez differences are taken at the two endpoints of the
// face. Ez_corner is computed by the Gardiner & Stone 2005 upwind CT
// formula (see compute-emf.wgsl) — a Balsara-Spicer-style arithmetic
// base at the four neighbouring face fluxes plus four upwind-biased
// quarter-derivative corrections driven by the HLLD contact velocity.
// Discrete ∇·B is preserved exactly to machine precision regardless of
// the corner-Ez recipe: corner contributions cancel in pairs around
// every cell whenever the same Ez_corner value is shared by all four
// faces it touches (which the staggered update enforces).
//
// Face fluxes carry the SAME convention:
//   flux_x[i, j] = flux through the LEFT face of cell (i, j) at i-½.
//   flux_y[i, j] = flux through the BOTTOM face of cell (i, j) at j-½.
//
// ── Bind-group layouts (transpiler reads these as nested-loop kernels) ─
// All compute pipelines use one bind group (group 0) shaped as:
//   binding 0: uniform Uniforms        (single shared physics-state uniform)
//   binding 1..N: storage buffers      (read-only or read-write)
// Weighted-update pipelines add an additional uniform StageParams at
// binding 1. The BC shader uses one extra storage buffer `bc_uniforms`
// to branch over per-edge modes + driven inflow state. Per-axis sweeps
// (reconstruct-ppm, riemann-hlld) take an extra uniform SweepDir at the
// highest binding index — the only field that varies between x and
// y sweeps. Render-pace LIC parameters (phase, intensity, drift) live
// in a separate LicUniforms buffer pushed per frame.
//
// ── State layout ────────────────────────────────────────────────────
// Cell-centered conserved state, packed as TWO vec4<f32> arrays per
// ping-pong slot:
//   U0[idx] = (ρ, ρ·vx, ρ·vy, ρ·vz)
//   U1[idx] = (E,  Bz,    _pad,  _pad)
// where E is total energy density:
//   E = p/(γ-1) + ½·ρ·|v|² + ½·|B|²
//
// ── γ & floors ──────────────────────────────────────────────────────
// γ comes from the Uniforms; pressure & density floors match config.js.

// Main physics-state Uniforms (64 B). Written when a physics parameter
// changes (preset/eta/cfl/gamma/view_mode/resolution/pressure_floor). Slots
// 5,6,7,14 are reserved (previously held LIC fields — now in a separate
// LicUniforms buffer rewritten per render frame). Slot 11 holds the
// pressure floor (was the dead _pad_sweep slot reclaimed for live UI).
// Slot 12 holds the CFL number (was the unused step_parity slot).
struct Uniforms {
    dx:            f32,
    gamma:         f32,
    view_min:      f32,
    view_max:      f32,
    eta:           f32,
    _pad_lic_0:    f32,  // reserved (was lic_phase — now in LicUniforms)
    _pad_lic_1:    f32,  // reserved (was lic_intensity — now in LicUniforms)
    _pad_lic_2:    f32,  // reserved (was lic_drift_x — now in LicUniforms)
    grid_n:        u32,  // INTERIOR grid resolution per axis
    grid_n_total:  u32,  // grid_n + 2*ghost_w (total storage width per axis)
    ghost_w:       u32,  // ghost-cell width per side (= 2 in Phase 4)
    pressure_floor:f32,  // minimum p in cons→prim recovery (UI slider)
    cfl:           f32,  // hyperbolic CFL number — consumed by compute-dt
    view_mode:     u32,  // 0=ρ, 1=p, 2=|v|, 3=|B|, 4=Jz
    _pad_lic_3:    f32,  // reserved (was lic_drift_y — now in LicUniforms)
    noise_n:       u32,  // noise-buffer side length (square, default 1024)
};

// Sweep-direction uniform — 16 B. Two of these (sweepDir_x = 0u,
// sweepDir_y = 1u) are pre-written at construction; reconstruct-ppm and
// riemann-hlld are the only shaders that read this.
struct SweepDir {
    sweep_dir: u32,    // 0 = x-sweep, 1 = y-sweep
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

// LIC render-pace parameters — 16 B. Rewritten by the host every render
// frame (per-frame phase advance + slider-controlled intensity/drift).
// Bound by lic-advect (compute) and composite (fragment).
struct LicUniforms {
    lic_phase:     f32,  // animated noise-sample phase offset (seconds)
    lic_intensity: f32,  // 0 = no LIC modulation, 1 = full strength (2 = doubled)
    lic_drift_x:   f32,  // noise drift x (noise-pixels/sec)
    lic_drift_y:   f32,  // noise drift y (noise-pixels/sec)
};

// BC modes — match config.js / enum values.
const BC_PERIODIC:   u32 = 0u;
const BC_OUTFLOW:    u32 = 1u;
const BC_REFLECTING: u32 = 2u;
const BC_DRIVEN:     u32 = 3u;

// Per-edge BC config + driven inflow state. Single storage buffer bound
// only by apply-bcs.wgsl. Driven state is in primitive form; the shader
// converts to conservative on write.
//
// Index convention for `mode[]`:
//   mode[0] = N (top,    j-ghost-strip at the top)
//   mode[1] = S (bottom, j-ghost-strip at the bottom)
//   mode[2] = E (right,  i-ghost-strip at the right)
//   mode[3] = W (left,   i-ghost-strip at the left)
struct BcUniforms {
    mode_n: u32,
    mode_s: u32,
    mode_e: u32,
    mode_w: u32,
    driven_rho: f32,
    driven_vx:  f32,
    driven_vy:  f32,
    driven_vz:  f32,
    driven_bx:  f32,
    driven_by:  f32,
    driven_bz:  f32,
    driven_p:   f32,
};

// PRESSURE_FLOOR is now live-controlled via U.pressure_floor (Uniforms
// slot 11) — pass it explicitly to helpers below. The shader-side
// default (matching config.js) is 1e-6.
const DENSITY_FLOOR:  f32 = 1.0e-6;

// ── Indexing helpers ────────────────────────────────────────────────
// Phase 4 drops wrap_idx in favour of direct indexing into ghost-padded
// buffers. The interior range is [ghost_w, ghost_w+grid_n) in both axes;
// PPM's 5-point stencil [i-2, i+2] is always in-bounds for interior i.

// Cell-centered linear index into an (N_total × N_total) buffer.
fn cell_idx_total(ix: u32, iy: u32, n_total: u32) -> u32 {
    return iy * n_total + ix;
}

// Backwards-compat alias used by view-field/colormap. In Phase 4 callers
// use the n_total form; we keep `cell_index` as a thin alias.
fn cell_index(ix: u32, iy: u32, n_total: u32) -> u32 {
    return iy * n_total + ix;
}

// Bx_face owns one extra column (N_total+1 wide), still N_total tall.
//   Bx_face[i,j] sits on the LEFT face of cell (i,j) at x = (i-ghost-½)·dx.
fn bx_face_idx(ix: u32, iy: u32, n_total: u32) -> u32 {
    return iy * (n_total + 1u) + ix;
}

// By_face owns one extra row (N_total tall + 1), still N_total wide.
//   By_face[i,j] sits on the BOTTOM face of cell (i,j) at y = (j-ghost-½)·dx.
fn by_face_idx(ix: u32, iy: u32, n_total: u32) -> u32 {
    return iy * n_total + ix;
}

// Ez_edge sits at corners (i-½, j-½), so (N_total+1) × (N_total+1).
fn ez_edge_idx(ix: u32, iy: u32, n_total: u32) -> u32 {
    return iy * (n_total + 1u) + ix;
}

// Legacy helper preserved for any shaders that still want it (none should
// in Phase 4; the function is kept only for compute-dt's fallback path if
// needed). Direct indexing replaces wrap.
fn wrap_idx(i: i32, n: i32) -> u32 {
    return u32(((i % n) + n) % n);
}

// Cell-centered Bx average from the two adjacent x-faces (LEFT-face owner):
//   bx_c = 0.5 · (Bx_face[i, j] + Bx_face[i+1, j])
// Callers inline this — we expose only the per-face indices so we don't
// need to pass storage-buffer pointers across function calls (WGSL
// pointer-to-storage parameters are not universally supported).
fn bx_face_left_idx(ix: u32, iy: u32, n_total: u32)  -> u32 { return bx_face_idx(ix,      iy, n_total); }
fn bx_face_right_idx(ix: u32, iy: u32, n_total: u32) -> u32 { return bx_face_idx(ix + 1u, iy, n_total); }
fn by_face_down_idx(ix: u32, iy: u32, n_total: u32)  -> u32 { return by_face_idx(ix, iy,      n_total); }
fn by_face_up_idx(ix: u32, iy: u32, n_total: u32)    -> u32 { return by_face_idx(ix, iy + 1u, n_total); }

// ── Cons / prim conversion ──────────────────────────────────────────
struct MhdPrim {
    rho: f32,
    vx:  f32,
    vy:  f32,
    vz:  f32,
    p:   f32,
    bx:  f32,
    by:  f32,
    bz:  f32,
};

struct MhdCons {
    rho:  f32,
    mx:   f32,
    my:   f32,
    mz:   f32,
    E:    f32,
    bx:   f32,
    by:   f32,
    bz:   f32,
};

fn cons_to_prim_mhd(U0: vec4<f32>, U1: vec4<f32>, bx_c: f32, by_c: f32, gamma: f32, p_floor: f32) -> MhdPrim {
    var P: MhdPrim;
    P.rho = max(U0.x, DENSITY_FLOOR);
    P.vx  = U0.y / P.rho;
    P.vy  = U0.z / P.rho;
    P.vz  = U0.w / P.rho;
    P.bx  = bx_c;
    P.by  = by_c;
    P.bz  = U1.y;
    let ke = 0.5 * P.rho * (P.vx*P.vx + P.vy*P.vy + P.vz*P.vz);
    let mb = 0.5 * (P.bx*P.bx + P.by*P.by + P.bz*P.bz);
    P.p   = max((gamma - 1.0) * (U1.x - ke - mb), p_floor);
    return P;
}

fn fast_mag_speed(P: MhdPrim, gamma: f32, axis: u32, p_floor: f32) -> f32 {
    // c_fast² = ½(c_s² + c_A²) + ½√((c_s² + c_A²)² − 4 c_s² c_An²)
    let rho = max(P.rho, DENSITY_FLOOR);
    let p   = max(P.p,   p_floor);
    let cs2 = gamma * p / rho;
    let b2  = P.bx*P.bx + P.by*P.by + P.bz*P.bz;
    let ca2 = b2 / rho;
    var can2: f32;
    if (axis == 0u) { can2 = P.bx * P.bx / rho; }
    else            { can2 = P.by * P.by / rho; }
    let sum = cs2 + ca2;
    let disc = max(sum * sum - 4.0 * cs2 * can2, 0.0);
    let cf2  = 0.5 * (sum + sqrt(disc));
    return sqrt(max(cf2, 0.0));
}

// 1D MHD flux along the sweep axis given full primitive state.
struct MhdFlux {
    f0: vec4<f32>,
    f1: vec4<f32>,
    f_bt1: f32,
    f_bt2: f32,
};

fn mhd_flux(P: MhdPrim, gamma: f32, axis: u32, p_floor: f32) -> MhdFlux {
    let rho = max(P.rho, DENSITY_FLOOR);
    let p   = max(P.p,   p_floor);
    let ke  = 0.5 * rho * (P.vx*P.vx + P.vy*P.vy + P.vz*P.vz);
    let mb  = 0.5 * (P.bx*P.bx + P.by*P.by + P.bz*P.bz);
    let E   = p / (gamma - 1.0) + ke + mb;
    let p_t = p + mb;
    let vdotb = P.vx*P.bx + P.vy*P.by + P.vz*P.bz;

    var F: MhdFlux;
    if (axis == 0u) {
        F.f0 = vec4<f32>(
            rho * P.vx,
            rho * P.vx * P.vx + p_t - P.bx * P.bx,
            rho * P.vx * P.vy       - P.bx * P.by,
            rho * P.vx * P.vz       - P.bx * P.bz,
        );
        F.f1 = vec4<f32>(
            (E + p_t) * P.vx - P.bx * vdotb,
            P.vx * P.bz - P.vz * P.bx,
            0.0, 0.0,
        );
        F.f_bt1 = P.vx * P.by - P.vy * P.bx;
        F.f_bt2 = P.vx * P.bz - P.vz * P.bx;
    } else {
        F.f0 = vec4<f32>(
            rho * P.vy,
            rho * P.vy * P.vx       - P.by * P.bx,
            rho * P.vy * P.vy + p_t - P.by * P.by,
            rho * P.vy * P.vz       - P.by * P.bz,
        );
        F.f1 = vec4<f32>(
            (E + p_t) * P.vy - P.by * vdotb,
            P.vy * P.bz - P.vz * P.by,
            0.0, 0.0,
        );
        F.f_bt1 = P.vy * P.bx - P.vx * P.by;
        F.f_bt2 = P.vy * P.bz - P.vz * P.by;
    }
    return F;
}

struct ConsPair {
    U0: vec4<f32>,
    U1: vec4<f32>,
};

fn prim_to_cons_pair(P: MhdPrim, gamma: f32, p_floor: f32) -> ConsPair {
    let rho = max(P.rho, DENSITY_FLOOR);
    let p   = max(P.p,   p_floor);
    let ke  = 0.5 * rho * (P.vx*P.vx + P.vy*P.vy + P.vz*P.vz);
    let mb  = 0.5 * (P.bx*P.bx + P.by*P.by + P.bz*P.bz);
    let E   = p / (gamma - 1.0) + ke + mb;
    var R: ConsPair;
    R.U0 = vec4<f32>(rho, rho*P.vx, rho*P.vy, rho*P.vz);
    R.U1 = vec4<f32>(E, P.bz, 0.0, 0.0);
    return R;
}

struct PrimPair {
    p0: vec4<f32>,
    p1: vec4<f32>,
};

fn normal_velocity_mhd(P: MhdPrim, axis: u32) -> f32 {
    if (axis == 0u) { return P.vx; } else { return P.vy; }
}

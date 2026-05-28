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
//   U1[idx] = (E,  Bz,    e_int, K)
// where E is total energy density:
//   E = p/(γ-1) + ½·ρ·|v|² + ½·|B|²
// and e_int/K are the dual-energy auxiliaries:
//   e_int = p/(γ-1),  K = p / ρ^γ.
//
// ── γ & floors ──────────────────────────────────────────────────────
// γ comes from the Uniforms; pressure & density floors match config.js.

// Main physics-state Uniforms (256 B). Written when a physics parameter
// changes (preset/eta/cfl/gamma/view_mode/resolution/pressure_floor). Slots
// 6,7,14 are reserved (previously held LIC fields — now in a separate
// LicUniforms buffer rewritten per render frame). Slot 5 was reclaimed
// in Session 8 for the anomalous-resistivity α coefficient. Slot 11 holds
// the pressure floor (was the dead _pad_sweep slot reclaimed for live UI).
// Slot 12 holds the CFL number (was the unused step_parity slot). Slot 14
// was reclaimed in Session 8 for the anomalous-resistivity J_crit threshold.
//
// Resistivity model:
//   η_local(J) = eta + eta_anom_alpha · ((|J|/eta_anom_jcrit − 1)_+)²
// where (·)_+ is the positive part. eta = base (Spitzer/uniform) resistivity;
// eta_anom_alpha = 0 disables anomalous boost entirely (constant-η baseline).
// |J| = |J_z| in 2.5D (only Jz is nonzero from in-plane fields).
struct Uniforms {
    // ── Original 64 B (slots 0-15) ────────────────────────────────────
    dx:              f32,
    gamma:           f32,
    view_min:        f32,
    view_max:        f32,
    eta:             f32,  // base resistivity (η_0)
    eta_anom_alpha:  f32,  // anomalous boost coefficient (slot 5; was _pad_lic_0)
    _pad_lic_1:      f32,  // reserved (was lic_intensity — now in LicUniforms)
    _pad_lic_2:      f32,  // reserved (was lic_drift_x — now in LicUniforms)
    grid_n:          u32,  // INTERIOR grid resolution per axis
    grid_n_total:    u32,  // grid_n + 2*ghost_w (total storage width per axis)
    ghost_w:         u32,  // ghost-cell width per side (= 2 in Phase 4)
    pressure_floor:  f32,  // minimum p in cons→prim recovery (UI slider)
    cfl:             f32,  // hyperbolic CFL number — consumed by compute-dt
    view_mode:       u32,  // 0=ρ, 1=p, 2=|v|, 3=|B|, 4=Jz
    eta_anom_jcrit:  f32,  // anomalous activation threshold |J_crit| (slot 14)
    noise_n:         u32,  // noise-buffer side length (square, default 1024)
    // ── Extended physics (slots 16-31, added for breadth pass) ───────
    hall_di:               f32,  // Hall ion inertial length d_i (code units; 0 = no Hall)
    hall_substeps_max:     u32,  // Max Hall sub-cycles per macro step
    cooling_lambda0:       f32,  // Cooling rate scale Λ_0 (0 = no cooling)
    cooling_T_floor:       f32,  // Below this T, Λ → 0
    cooling_T_ref:         f32,  // Reference temperature for Λ(T) normalization
    conduction_kappa:      f32,  // Parallel thermal conductivity κ_∥ (0 = no conduction)
    conduction_iso_frac:   f32,  // κ_⊥ / κ_∥ (0 = fully anisotropic, 1 = isotropic)
    conduction_sat_frac:   f32,  // Saturated heat-flux fraction (0 = unlimited)
    gravity_gx:            f32,  // External gravity x (constant)
    gravity_gy:            f32,  // External gravity y (constant; e.g., -1 for "down")
    gravity_G:             f32,  // Newton's G for self-gravity (0 = no self-gravity)
    gravity_poisson_iters: u32,  // Jacobi iterations per macro step
    physics_flags:               u32,  // Bitfield: COOLING|GRAV_EXT|GRAV_SELF|COND|HALL|POSITIVITY|EMF_UPWIND
    emf_mode:                    u32,  // 0 = BS arithmetic mean, 1 = GS upwind
    cooling_curve_mode:          u32,  // 0 = √T brems, 1 = piecewise power-law table
    hall_electron_pressure_frac: f32,  // p_e / p for generalized Ohm's law pressure term
    // ── Higher-fidelity source physics (slots 32-63) ───────────────
    cooling_metallicity:      f32,  // CIE/table metallicity multiplier (solar = 1)
    heating_gamma0:           f32,  // volumetric heating scale (0 = off)
    heating_density_exp:      f32,  // heating ∝ ρ^a
    heating_T_cut:            f32,  // optional hot-gas heating cutoff; <=0 disables
    ambipolar_eta:            f32,  // η_A coefficient for ion-neutral drift diffusion
    biermann_coeff:           f32,  // Biermann battery strength
    neutral_frac:             f32,  // base neutral fraction multiplier
    ionization_T0:            f32,  // neutral fraction turnover temperature
    viscosity_nu:             f32,  // kinematic shear viscosity
    viscosity_bulk:           f32,  // bulk viscosity coefficient
    viscosity_aniso_frac:     f32,  // Braginskii-like B-aligned fraction
    viscosity_shock:          f32,  // extra compression-triggered artificial viscosity
    source_substeps_max:      u32,  // shared cap for explicit source sub-cycles
    geometry_mode:            u32,  // 0 = Cartesian, 1 = cylindrical axisymmetry
    geometry_r_min:           f32,  // radial origin offset / axis guard
    gravity_softening:        f32,  // screened-Poisson softening length
    gravity_poisson_omega:    f32,  // weighted-Jacobi relaxation parameter
    sponge_width:             f32,  // sponge width in grid cells
    sponge_strength:          f32,  // damping rate for boundary sponge
    cooling_table_mix:        f32,  // reserved blend/scale for external tables
    radiation_c:              f32,  // reduced speed of light for grey radiation transfer
    radiation_kappa_abs:      f32,  // absorption/thermal coupling opacity
    radiation_kappa_scat:     f32,  // scattering opacity for flux-limited diffusion
    radiation_const:          f32,  // radiation constant a_r in E_rad = a_r T^4
    radiation_floor:          f32,  // floor for positive radiation energy density
    electron_inertia_length:  f32,  // kinetic regularization length d_e for hyper-resistive Ohm term
    electron_inertia_damping: f32,  // coefficient for d_e^2 ∇²J high-k smoothing
    gravity_boundary_mode:    u32,  // 0 = periodic mean-subtracted, 1 = isolated zero-φ exterior
    _pad_ext_60:              f32,
    _pad_ext_61:              f32,
    _pad_ext_62:              f32,
    _pad_ext_63:              f32,
};

// Flag bits — keep in sync with FLAG_* constants in config.js.
const FLAG_COOLING:      u32 = 1u << 0u;
const FLAG_GRAVITY_EXT:  u32 = 1u << 1u;
const FLAG_GRAVITY_SELF: u32 = 1u << 2u;
const FLAG_CONDUCTION:   u32 = 1u << 3u;
const FLAG_HALL:         u32 = 1u << 4u;
const FLAG_POSITIVITY:   u32 = 1u << 5u;
const FLAG_EMF_UPWIND:   u32 = 1u << 6u;
const FLAG_AMBIPOLAR:    u32 = 1u << 7u;
const FLAG_BIERMANN:     u32 = 1u << 8u;
const FLAG_VISCOSITY:    u32 = 1u << 9u;
const FLAG_GEOMETRY:     u32 = 1u << 10u;
const FLAG_SPONGE:       u32 = 1u << 11u;
const FLAG_HEATING:      u32 = 1u << 12u;
const FLAG_RADIATION:    u32 = 1u << 13u;
const FLAG_ELECTRON_INERTIA: u32 = 1u << 14u;

fn flag_set(flags: u32, bit: u32) -> bool {
    return (flags & bit) != 0u;
}

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
const EDGE_N_BC: u32 = 0u;
const EDGE_S_BC: u32 = 1u;
const EDGE_E_BC: u32 = 2u;
const EDGE_W_BC: u32 = 3u;

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
    driven_n_rho: f32,
    driven_n_vx:  f32,
    driven_n_vy:  f32,
    driven_n_vz:  f32,
    driven_n_bx:  f32,
    driven_n_by:  f32,
    driven_n_bz:  f32,
    driven_n_p:   f32,
    driven_s_rho: f32,
    driven_s_vx:  f32,
    driven_s_vy:  f32,
    driven_s_vz:  f32,
    driven_s_bx:  f32,
    driven_s_by:  f32,
    driven_s_bz:  f32,
    driven_s_p:   f32,
    driven_e_rho: f32,
    driven_e_vx:  f32,
    driven_e_vy:  f32,
    driven_e_vz:  f32,
    driven_e_bx:  f32,
    driven_e_by:  f32,
    driven_e_bz:  f32,
    driven_e_p:   f32,
    driven_w_rho: f32,
    driven_w_vx:  f32,
    driven_w_vy:  f32,
    driven_w_vz:  f32,
    driven_w_bx:  f32,
    driven_w_by:  f32,
    driven_w_bz:  f32,
    driven_w_p:   f32,
};

// PRESSURE_FLOOR is now live-controlled via U.pressure_floor (Uniforms
// slot 11) — pass it explicitly to helpers below. The shader-side
// default (matching config.js) is 1e-6.
const DENSITY_FLOOR:  f32 = 1.0e-6;
const DUAL_ENERGY_FRACTION: f32 = 1.0e-3;

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

fn entropy_proxy(rho: f32, p: f32, gamma: f32, p_floor: f32) -> f32 {
    return max(p, p_floor) / pow(max(rho, DENSITY_FLOOR), gamma);
}

fn pressure_from_dual_energy(U0: vec4<f32>, U1: vec4<f32>, bx_c: f32, by_c: f32, gamma: f32, p_floor: f32) -> f32 {
    let rho = max(U0.x, DENSITY_FLOOR);
    let vx = U0.y / rho;
    let vy = U0.z / rho;
    let vz = U0.w / rho;
    let ke = 0.5 * rho * (vx*vx + vy*vy + vz*vz);
    let mb = 0.5 * (bx_c*bx_c + by_c*by_c + U1.y*U1.y);
    let eth_total = U1.x - ke - mb;
    let eth_floor = p_floor / max(gamma - 1.0, 1.0e-6);
    let total_ok = eth_total > max(eth_floor, DUAL_ENERGY_FRACTION * max(abs(U1.x), eth_floor))
                && eth_total == eth_total;
    let dual_eth = max(U1.z, eth_floor);
    let eth = select(dual_eth, eth_total, total_ok);
    return max((gamma - 1.0) * eth, p_floor);
}

fn pack_u1_aux(E: f32, bz: f32, rho: f32, p: f32, gamma: f32, p_floor: f32) -> vec4<f32> {
    let p_safe = max(p, p_floor);
    let eth = p_safe / max(gamma - 1.0, 1.0e-6);
    return vec4<f32>(E, bz, eth, entropy_proxy(rho, p_safe, gamma, p_floor));
}

fn cons_to_prim_mhd(U0: vec4<f32>, U1: vec4<f32>, bx_c: f32, by_c: f32, gamma: f32, p_floor: f32) -> MhdPrim {
    var P: MhdPrim;
    P.rho = max(U0.x, DENSITY_FLOOR);
    P.vx  = U0.y / P.rho;
    P.vy  = U0.z / P.rho;
    P.vz  = U0.w / P.rho;
    P.bx  = bx_c;
    P.by  = by_c;
    P.bz  = U1.y;
    P.p   = pressure_from_dual_energy(U0, U1, bx_c, by_c, gamma, p_floor);
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
    R.U1 = pack_u1_aux(E, P.bz, rho, p, gamma, p_floor);
    return R;
}

struct PrimPair {
    p0: vec4<f32>,
    p1: vec4<f32>,
};

fn normal_velocity_mhd(P: MhdPrim, axis: u32) -> f32 {
    if (axis == 0u) { return P.vx; } else { return P.vy; }
}

// ── Anomalous resistivity ──────────────────────────────────────────
// Ad-hoc closure used by both the Hall-MHD and PIC-validated MRX
// literature (e.g., Birn et al. 2001 GEM challenge; Schumlak 2017;
// Trintchouk & Yamada 2003). Triggers enhanced (fast) reconnection
// only where local current density exceeds a critical value, leaving
// bulk plasma at the base Spitzer-like resistivity.
//
// Form:  η(|J|) = η_0 + α · max(0, |J|/J_crit − 1)²
// With α = 0 this reduces exactly to constant resistivity. The (·)²
// makes the boost smooth at the threshold; squared form is the
// default used by most reconnection models.
fn anomalous_eta(j_mag: f32, eta0: f32, alpha: f32, jcrit: f32) -> f32 {
    if (alpha <= 0.0) { return eta0; }
    let jcrit_safe = max(jcrit, 1.0e-12);
    let r = j_mag / jcrit_safe;
    let excess = max(0.0, r - 1.0);
    return eta0 + alpha * excess * excess;
}

// J_z recipe (inlined by callers — WGSL pointer-to-storage parameters
// aren't universally supported across implementations and add no value
// here):
//   bx_c(ix, iy) = 0.5 · (Bx_face[ix, iy] + Bx_face[ix+1, iy])
//   by_c(ix, iy) = 0.5 · (By_face[ix, iy] + By_face[ix, iy+1])
//   J_z(ix, iy)  = (by_c(ix+1, iy) − by_c(ix-1, iy)) / (2·dx)
//                − (bx_c(ix,   iy+1) − bx_c(ix,   iy-1)) / (2·dx)
// (∂By/∂x − ∂Bx/∂y, central differences). Used by view-field, apply-
// resistivity, and compute-dt's η-max reduction.

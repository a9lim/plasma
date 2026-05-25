// ─── shared-helpers.wgsl ─────────────────────────────────────────────
// Conservative ↔ primitive conversion, sound-speed helper, periodic
// index wrap, and the shared Uniforms struct.
//
// Conservative state per cell (vec4):
//   U.x = ρ
//   U.y = ρ·vx
//   U.z = ρ·vy
//   U.w = E  (total energy density: thermal + kinetic)
//
// Primitive state (vec4):
//   prim.x = ρ
//   prim.y = vx
//   prim.z = vy
//   prim.w = p  (pressure)
//
// γ comes from the Uniforms; pressure floor is hard-coded to match
// config.js PRESSURE_FLOOR. Keep in sync if that value moves.
//
// Floors avoid divide-by-zero in cons→prim conversion (ρ → 0) and avoid
// NaN sound speeds from a transiently negative thermal energy.

struct Uniforms {
    dx:           f32,
    gamma:        f32,
    view_min:     f32,
    view_max:     f32,
    grid_n:       u32,
    sweep_dir:    u32,  // 0 = x-sweep, 1 = y-sweep
    step_parity:  u32,  // 0 = even (X→Y), 1 = odd (Y→X) — informational, driven by JS
    _pad0:        u32,
};

const PRESSURE_FLOOR: f32 = 1.0e-6;
const DENSITY_FLOOR:  f32 = 1.0e-6;

fn cons_to_prim(U: vec4<f32>, gamma: f32) -> vec4<f32> {
    let rho = max(U.x, DENSITY_FLOOR);
    let vx  = U.y / rho;
    let vy  = U.z / rho;
    let ke  = 0.5 * rho * (vx * vx + vy * vy);
    let p   = max((gamma - 1.0) * (U.w - ke), PRESSURE_FLOOR);
    return vec4<f32>(rho, vx, vy, p);
}

fn prim_to_cons(prim: vec4<f32>, gamma: f32) -> vec4<f32> {
    let rho = max(prim.x, DENSITY_FLOOR);
    let p   = max(prim.w, PRESSURE_FLOOR);
    let ke  = 0.5 * rho * (prim.y * prim.y + prim.z * prim.z);
    let E   = p / (gamma - 1.0) + ke;
    return vec4<f32>(rho, rho * prim.y, rho * prim.z, E);
}

fn sound_speed(prim: vec4<f32>, gamma: f32) -> f32 {
    let rho = max(prim.x, DENSITY_FLOOR);
    let p   = max(prim.w, PRESSURE_FLOOR);
    return sqrt(gamma * p / rho);
}

// Periodic wrap for index arithmetic. WGSL u32 underflow wraps modulo 2³²
// which corrupts indices, so we add `n` before the % to keep things in
// [0, n).  n must be ≥ 1.
fn wrap_idx(i: i32, n: i32) -> u32 {
    return u32(((i % n) + n) % n);
}

fn cell_index(ix: u32, iy: u32, n: u32) -> u32 {
    return iy * n + ix;
}

// Wrap a 2D pair and compute the linear cell index in one go.
fn cell_index_wrapped(ix: i32, iy: i32, n: i32) -> u32 {
    let wx = wrap_idx(ix, n);
    let wy = wrap_idx(iy, n);
    return wy * u32(n) + wx;
}

// Compute the 1D Euler flux given a primitive state along the sweep axis.
// `axis` = 0 → x-flux (vx is the normal velocity).
// `axis` = 1 → y-flux (vy is the normal velocity).
fn euler_flux(prim: vec4<f32>, gamma: f32, axis: u32) -> vec4<f32> {
    let rho = max(prim.x, DENSITY_FLOOR);
    let vx  = prim.y;
    let vy  = prim.z;
    let p   = max(prim.w, PRESSURE_FLOOR);
    let ke  = 0.5 * rho * (vx * vx + vy * vy);
    let E   = p / (gamma - 1.0) + ke;

    if (axis == 0u) {
        return vec4<f32>(
            rho * vx,
            rho * vx * vx + p,
            rho * vx * vy,
            (E + p) * vx,
        );
    } else {
        return vec4<f32>(
            rho * vy,
            rho * vy * vx,
            rho * vy * vy + p,
            (E + p) * vy,
        );
    }
}

// Normal velocity along the sweep axis.
fn normal_velocity(prim: vec4<f32>, axis: u32) -> f32 {
    if (axis == 0u) { return prim.y; } else { return prim.z; }
}

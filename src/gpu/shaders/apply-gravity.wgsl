// ─── apply-gravity.wgsl ───────────────────────────────────────────────
// Source-term application for gravitational acceleration. Two modes
// (FLAG_GRAVITY_EXT and FLAG_GRAVITY_SELF) — they compose additively:
//
//   External:    g = (gravity_gx, gravity_gy, 0)         constant.
//   Self:        g = -∇φ                                  from solve-poisson.
//
// Conservation-form source terms (Stone & Norman 1992 / Tóth & Roe 2002):
//
//   d(ρv)/dt |_grav = ρ g
//   dE   /dt |_grav = ρ v · g
//
// (No source on ρ itself; mass is conserved.)
//
// Integration: explicit source kick over dt_hyp. Momentum is forward Euler;
// the energy work uses the time-centered velocity for this kick, so a uniform
// acceleration preserves the kinetic-energy change implied by the momentum
// update instead of creating/destroying thermal energy.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0       (rw) — momentum gets ρ·g·dt added
//   2 U1       (rw) — energy gets ρ·v·g·dt added
//   3 phi      (ro) — self-gravity potential (zero if FLAG_GRAVITY_SELF off)
//   4 dt_buf   (uniform)
//   5 Bx_face  (ro)
//   6 By_face  (ro)

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       phi:        array<f32>;
@group(0) @binding(4) var<uniform>             dt_buf:     DtUniform;
@group(0) @binding(5) var<storage, read>       Bx_face:    array<f32>;
@group(0) @binding(6) var<storage, read>       By_face:    array<f32>;

fn phi_at(gx_in: i32, gy_in: i32, n_interior: u32, n_total: u32, ghost: u32) -> f32 {
    let n = i32(n_interior);
    var gx = gx_in;
    var gy = gy_in;
    if (U_uniforms.gravity_boundary_mode == 1u) {
        if (gx < 0 || gx >= n || gy < 0 || gy >= n) {
            return 0.0;
        }
    } else {
        gx = ((gx % n) + n) % n;
        gy = ((gy % n) + n) % n;
    }
    let ix = ghost + u32(gx);
    let iy = ghost + u32(gy);
    return phi[cell_idx_total(ix, iy, n_total)];
}

fn cell_bx_grav(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
}

fn cell_by_grav(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                + By_face[by_face_idx(ix, iy + 1u, n_total)]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flags = U_uniforms.physics_flags;
    let do_ext  = flag_set(flags, FLAG_GRAVITY_EXT);
    let do_self = flag_set(flags, FLAG_GRAVITY_SELF) && (U_uniforms.gravity_G > 0.0);
    if (!do_ext && !do_self) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    let u0  = U0[c];
    let u1  = U1[c];
    let rho = max(u0.x, DENSITY_FLOOR);
    let vx  = u0.y / rho;
    let vy  = u0.z / rho;

    var gx: f32 = 0.0;
    var gy: f32 = 0.0;
    if (do_ext) {
        gx = gx + U_uniforms.gravity_gx;
        gy = gy + U_uniforms.gravity_gy;
    }
    if (do_self) {
        // g = -∇φ. Fourth-order periodic central differences on the
        // cell-centered potential. This is still cheap, but it substantially
        // reduces the force-phase error in smooth Jeans-mode tests compared
        // with the 2-point stencil, without changing the Jacobi buffer
        // topology or relying on φ ghost cells.
        let dx = U_uniforms.dx;
        let gx0 = i32(gid.x);
        let gy0 = i32(gid.y);
        let dphi_dx = (-phi_at(gx0 + 2, gy0, n_interior, n_total, ghost)
                       + 8.0 * phi_at(gx0 + 1, gy0, n_interior, n_total, ghost)
                       - 8.0 * phi_at(gx0 - 1, gy0, n_interior, n_total, ghost)
                       + phi_at(gx0 - 2, gy0, n_interior, n_total, ghost)) / (12.0 * dx);
        let dphi_dy = (-phi_at(gx0, gy0 + 2, n_interior, n_total, ghost)
                       + 8.0 * phi_at(gx0, gy0 + 1, n_interior, n_total, ghost)
                       - 8.0 * phi_at(gx0, gy0 - 1, n_interior, n_total, ghost)
                       + phi_at(gx0, gy0 - 2, n_interior, n_total, ghost)) / (12.0 * dx);
        gx = gx - dphi_dx;
        gy = gy - dphi_dy;
    }

    let dt = dt_buf.dt;
    let dpx = rho * gx * dt;
    let dpy = rho * gy * dt;
    let vx_mid = vx + 0.5 * gx * dt;
    let vy_mid = vy + 0.5 * gy * dt;
    let dE = rho * (vx_mid * gx + vy_mid * gy) * dt;

    let u0_new = vec4<f32>(u0.x, u0.y + dpx, u0.z + dpy, u0.w);
    let E_new = u1.x + dE;
    let bx = cell_bx_grav(ix, iy, n_total);
    let by = cell_by_grav(ix, iy, n_total);
    let p_new = pressure_from_dual_energy(
        u0_new,
        vec4<f32>(E_new, u1.y, u1.z, u1.w),
        bx,
        by,
        U_uniforms.gamma,
        U_uniforms.pressure_floor,
    );

    U0[c] = u0_new;
    U1[c] = pack_u1_aux(E_new, u1.y, max(u0_new.x, DENSITY_FLOOR), p_new,
                         U_uniforms.gamma, U_uniforms.pressure_floor);
}

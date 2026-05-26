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
// Integration: explicit forward Euler over dt_hyp. Stable for any
// reasonable g — gravitational dynamics is generally well-conditioned.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0       (rw) — momentum gets ρ·g·dt added
//   2 U1       (rw) — energy gets ρ·v·g·dt added
//   3 phi      (ro) — self-gravity potential (zero if FLAG_GRAVITY_SELF off)
//   4 dt_buf   (uniform)

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       phi:        array<f32>;
@group(0) @binding(4) var<uniform>             dt_buf:     DtUniform;

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
        // g = -∇φ. Central differences on the cell-centered potential.
        let dx = U_uniforms.dx;
        let phi_l = phi[cell_idx_total(ix - 1u, iy,      n_total)];
        let phi_r = phi[cell_idx_total(ix + 1u, iy,      n_total)];
        let phi_d = phi[cell_idx_total(ix,      iy - 1u, n_total)];
        let phi_u = phi[cell_idx_total(ix,      iy + 1u, n_total)];
        gx = gx - (phi_r - phi_l) / (2.0 * dx);
        gy = gy - (phi_u - phi_d) / (2.0 * dx);
    }

    let dt = dt_buf.dt;
    let dpx = rho * gx * dt;
    let dpy = rho * gy * dt;
    // Energy work: ρ(v·g) dt. Use the OLD v (forward Euler).
    let dE = rho * (vx * gx + vy * gy) * dt;

    U0[c] = vec4<f32>(u0.x, u0.y + dpx, u0.z + dpy, u0.w);
    U1[c] = vec4<f32>(u1.x + dE, u1.y, u1.z, u1.w);
}

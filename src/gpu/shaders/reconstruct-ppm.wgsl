// ─── reconstruct-ppm.wgsl ────────────────────────────────────────────
// Colella & Woodward 1984 PPM reconstruction on MHD primitive variables.
// Produces per-cell, per-direction *left* and *right* face primitive
// states (the two endpoints of the limited parabola through cell j).
// At face (j+½), the Riemann solver pairs:
//     QL = edge_R[j]    (right edge of cell j)
//     QR = edge_L[j+1]  (left edge of cell j+1)
//
// Phase 4: drops wrap_idx in favour of direct indexing into ghost-padded
// arrays. We dispatch over cells [ghost-1, ghost+N+1) along the sweep
// axis — one cell wider on each side than the interior — so the Riemann
// solver sees valid PPM edges on BOTH cells adjacent to every boundary
// x-face. For the outermost cells (i = ghost-1 and i = ghost+N), the
// full 5-point PPM stencil would extend past the available ghost band
// (ghost = 2 only covers ±2 from the interior, leaving 1 cell short for
// the 5-point stencil at the outermost dispatch cell). Those cells fall
// back to piecewise-constant reconstruction: edge_l = edge_r = q_c.
// This is the standard "graceful PPM degradation at the buffer edge"
// approach — for the boundary face physics it matches the BC-derived
// ghost values, which are already lower order.
//
// Algorithm per primitive variable q, per direction:
//   1. 4th-order edge interpolant:
//      q_{j+½} = (7/12)(q_j + q_{j+1}) - (1/12)(q_{j-1} + q_{j+2})
//      Compute at j-½ and j+½ to get q_L (left edge) and q_R (right edge).
//   2. Monotonicity: if (q_R - q_j)(q_j - q_L) ≤ 0 → q_L = q_R = q_j.
//   3. CW limiter: see header — canonical overshoot detection.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0_in   (ro)
//   2 U1_in   (ro)
//   3 Bx_face (ro)
//   4 By_face (ro)
//   5 edge_l_0 (rw)
//   6 edge_l_1 (rw)
//   7 edge_r_0 (rw)
//   8 edge_r_1 (rw)
//   9 sweep   (uniform SweepDir) — sweep_dir = 0 (x) or 1 (y)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:     array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:   array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:   array<f32>;
@group(0) @binding(5) var<storage, read_write> edge_l_0:  array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> edge_l_1:  array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> edge_r_0:  array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> edge_r_1:  array<vec4<f32>>;
@group(0) @binding(9) var<uniform>             sweep:     SweepDir;

// Pack one cell's primitive state into the two vec4 we feed to PPM.
//   p0 = (ρ, vx, vy, vz)
//   p1 = (p, Bt1, Bz, _)    Bt1 = By for x-sweep, Bx for y-sweep
fn cell_primitive_pair_ppm(ix: u32, iy: u32, n_total: u32, gamma: f32, axis: u32) -> PrimPair {
    let idx = cell_idx_total(ix, iy, n_total);
    let bx  = 0.5 * (Bx_face[bx_face_left_idx(ix, iy, n_total)]
                   + Bx_face[bx_face_right_idx(ix, iy, n_total)]);
    let by  = 0.5 * (By_face[by_face_down_idx(ix, iy, n_total)]
                   + By_face[by_face_up_idx(ix, iy, n_total)]);
    let P   = cons_to_prim_mhd(U0_in[idx], U1_in[idx], bx, by, gamma);
    var R: PrimPair;
    R.p0    = vec4<f32>(P.rho, P.vx, P.vy, P.vz);
    if (axis == 0u) {
        R.p1 = vec4<f32>(P.p, P.by, P.bz, 0.0);
    } else {
        R.p1 = vec4<f32>(P.p, P.bx, P.bz, 0.0);
    }
    return R;
}

fn ppm_limit_scalar(q_c: f32, q_L_raw: f32, q_R_raw: f32) -> vec2<f32> {
    var qL = q_L_raw;
    var qR = q_R_raw;
    let dL = q_c - qL;
    let dR = qR - q_c;
    if (dL * dR <= 0.0) {
        return vec2<f32>(q_c, q_c);
    }
    let dq = qR - qL;
    let q6 = 6.0 * (q_c - 0.5 * (qL + qR));
    let dq2 = dq * dq;
    if (dq * q6 > dq2) {
        qL = 3.0 * q_c - 2.0 * qR;
    } else if (dq * q6 < -dq2) {
        qR = 3.0 * q_c - 2.0 * qL;
    }
    return vec2<f32>(qL, qR);
}

struct PpmLR {
    L: vec4<f32>,
    R: vec4<f32>,
};

fn ppm_limit_vec4(q_c: vec4<f32>, q_L: vec4<f32>, q_R: vec4<f32>) -> PpmLR {
    let x = ppm_limit_scalar(q_c.x, q_L.x, q_R.x);
    let y = ppm_limit_scalar(q_c.y, q_L.y, q_R.y);
    let z = ppm_limit_scalar(q_c.z, q_L.z, q_R.z);
    let w = ppm_limit_scalar(q_c.w, q_L.w, q_R.w);
    var out: PpmLR;
    out.L = vec4<f32>(x.x, y.x, z.x, w.x);
    out.R = vec4<f32>(x.y, y.y, z.y, w.y);
    return out;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    // Dispatch over the extended interior range: [ghost-1, ghost+N+1).
    // Width n_interior + 2.
    let extent = n_interior + 2u;
    if (gid.x >= extent || gid.y >= extent) { return; }
    let ix = gid.x + ghost - 1u;
    let iy = gid.y + ghost - 1u;

    let axis = sweep.sweep_dir;
    let g    = U_uniforms.gamma;
    let idx  = cell_idx_total(ix, iy, n_total);

    // ── Check if the full 5-point stencil fits in storage ──────────
    var stencil_ok = true;
    if (axis == 0u) {
        stencil_ok = (ix >= 2u) && (ix + 2u < n_total);
    } else {
        stencil_ok = (iy >= 2u) && (iy + 2u < n_total);
    }
    let p_c = cell_primitive_pair_ppm(ix, iy, n_total, g, axis);

    if (!stencil_ok) {
        // Piecewise-constant fallback: both edges equal cell value.
        var l0 = p_c.p0; var r0 = p_c.p0;
        var l1 = p_c.p1; var r1 = p_c.p1;
        l0.x = max(l0.x, DENSITY_FLOOR);
        r0.x = max(r0.x, DENSITY_FLOOR);
        l1.x = max(l1.x, PRESSURE_FLOOR);
        r1.x = max(r1.x, PRESSURE_FLOOR);
        edge_l_0[idx] = l0;
        edge_l_1[idx] = l1;
        edge_r_0[idx] = r0;
        edge_r_1[idx] = r1;
        return;
    }

    // 5-point stencil along the sweep axis: m2, m1, c, p1, p2.
    var im2_x: u32 = ix; var im1_x: u32 = ix; var ip1_x: u32 = ix; var ip2_x: u32 = ix;
    var im2_y: u32 = iy; var im1_y: u32 = iy; var ip1_y: u32 = iy; var ip2_y: u32 = iy;
    if (axis == 0u) {
        im2_x = ix - 2u; im1_x = ix - 1u;
        ip1_x = ix + 1u; ip2_x = ix + 2u;
    } else {
        im2_y = iy - 2u; im1_y = iy - 1u;
        ip1_y = iy + 1u; ip2_y = iy + 2u;
    }

    let p_m2 = cell_primitive_pair_ppm(im2_x, im2_y, n_total, g, axis);
    let p_m1 = cell_primitive_pair_ppm(im1_x, im1_y, n_total, g, axis);
    let p_p1 = cell_primitive_pair_ppm(ip1_x, ip1_y, n_total, g, axis);
    let p_p2 = cell_primitive_pair_ppm(ip2_x, ip2_y, n_total, g, axis);

    let c7 = 7.0 / 12.0;
    let c1 = 1.0 / 12.0;
    let qL0_raw = c7 * (p_m1.p0 + p_c.p0)  - c1 * (p_m2.p0 + p_p1.p0);
    let qR0_raw = c7 * (p_c.p0  + p_p1.p0) - c1 * (p_m1.p0 + p_p2.p0);
    let qL1_raw = c7 * (p_m1.p1 + p_c.p1)  - c1 * (p_m2.p1 + p_p1.p1);
    let qR1_raw = c7 * (p_c.p1  + p_p1.p1) - c1 * (p_m1.p1 + p_p2.p1);

    let lim0 = ppm_limit_vec4(p_c.p0, qL0_raw, qR0_raw);
    let lim1 = ppm_limit_vec4(p_c.p1, qL1_raw, qR1_raw);

    var l0 = lim0.L; var r0 = lim0.R;
    var l1 = lim1.L; var r1 = lim1.R;
    l0.x = max(l0.x, DENSITY_FLOOR);
    r0.x = max(r0.x, DENSITY_FLOOR);
    l1.x = max(l1.x, PRESSURE_FLOOR);
    r1.x = max(r1.x, PRESSURE_FLOOR);

    edge_l_0[idx] = l0;
    edge_l_1[idx] = l1;
    edge_r_0[idx] = r0;
    edge_r_1[idx] = r1;
}

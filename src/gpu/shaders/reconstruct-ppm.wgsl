// ─── reconstruct-ppm.wgsl ────────────────────────────────────────────
// Colella & Woodward 1984 PPM reconstruction on MHD primitive variables.
// Produces per-cell, per-direction *left* and *right* face primitive
// states (the two endpoints of the limited parabola through cell j).
// At face (j+½), the Riemann solver pairs:
//     QL = edge_R[j]    (right edge of cell j)
//     QR = edge_L[j+1]  (left edge of cell j+1)
//
// Algorithm per primitive variable q, per direction:
//   1. 4th-order edge interpolant:
//      q_{j+½} = (7/12)(q_j + q_{j+1}) - (1/12)(q_{j-1} + q_{j+2})
//      Compute at j-½ and j+½ to get q_L (left edge) and q_R (right edge).
//   2. Monotonicity: if (q_R - q_j)(q_j - q_L) ≤ 0 → q_L = q_R = q_j (flatten).
//   3. CW limiter (overshoot): with Δq = q_R - q_L, q6 = 6(q_j - ½(q_L + q_R)),
//      if Δq · q6 >  Δq²  → q_L = 3 q_j - 2 q_R
//      if Δq · q6 < -Δq²  → q_R = 3 q_j - 2 q_L
//      Equivalent to the |q_R - q_j| ≥ 2|q_L - q_j| (and symmetric) form.
//   4. No contact-discontinuity sharpening (CW §1.5 skipped; basic limited
//      PPM is already strong).
//
// We process all 7 active primitive components packed in two vec4s per
// edge state, matching the PLM slope layout but writing absolute values
// instead of slopes:
//   edge_l_0[idx] = (ρ_L, vx_L, vy_L, vz_L)
//   edge_l_1[idx] = (p_L, Bt1_L, Bz_L, _)
//   edge_r_0[idx] = (ρ_R, vx_R, vy_R, vz_R)
//   edge_r_1[idx] = (p_R, Bt1_R, Bz_R, _)
// where Bt1 = By for x-sweep, Bx for y-sweep. The normal-direction B is
// taken directly from the staggered face value at the Riemann step
// (continuous across the face by CT — never PPM-reconstructed).
//
// Bindings (same shape as PLM, doubled outputs):
//   0 uniforms (uniform)
//   1 U0_in   (ro)
//   2 U1_in   (ro)
//   3 Bx_face (ro)
//   4 By_face (ro)
//   5 edge_l_0 (rw)
//   6 edge_l_1 (rw)
//   7 edge_r_0 (rw)
//   8 edge_r_1 (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:     array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:   array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:   array<f32>;
@group(0) @binding(5) var<storage, read_write> edge_l_0:  array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> edge_l_1:  array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> edge_r_0:  array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> edge_r_1:  array<vec4<f32>>;

// Pack a cell's primitive state into the two vec4s used for reconstruction.
fn cell_primitive_pair_ppm(ix: u32, iy: u32, n: u32, gamma: f32, axis: u32) -> PrimPair {
    let idx = cell_index(ix, iy, n);
    let bx  = 0.5 * (Bx_face[bx_face_left_index(ix, iy, n)] + Bx_face[bx_face_right_index(ix, iy, n)]);
    let by  = 0.5 * (By_face[by_face_down_index(ix, iy, n)] + By_face[by_face_up_index(ix, iy, n)]);
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

// Scalar PPM limiter applied to one component.
//   q_m2, q_m1, q_c, q_p1 → returns (q_left_edge, q_right_edge).
// q_left_edge  = value at face (c - ½) using the (q_m2, q_m1, q_c, q_p1) stencil
// q_right_edge = value at face (c + ½) using the (q_m1, q_c, q_p1, q_p2) stencil.
// To avoid passing q_p2 around for every variable, we accept BOTH edge
// estimates pre-computed by the caller and just apply the limiter here.
fn ppm_limit_scalar(q_c: f32, q_L_raw: f32, q_R_raw: f32) -> vec2<f32> {
    var qL = q_L_raw;
    var qR = q_R_raw;
    // Step 2: local extremum → flatten.
    let dL = q_c - qL;
    let dR = qR - q_c;
    if (dL * dR <= 0.0) {
        return vec2<f32>(q_c, q_c);
    }
    // Step 3: CW overshoot limiter (canonical form).
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
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i  = i32(n);
    let ix   = i32(gid.x);
    let iy   = i32(gid.y);
    let axis = U_uniforms.sweep_dir;
    let g    = U_uniforms.gamma;

    // 5-point stencil along the sweep axis: m2, m1, c, p1, p2.
    var im2_x: u32; var im1_x: u32; var ip1_x: u32; var ip2_x: u32;
    var im2_y: u32; var im1_y: u32; var ip1_y: u32; var ip2_y: u32;
    if (axis == 0u) {
        im2_x = wrap_idx(ix - 2, n_i); im1_x = wrap_idx(ix - 1, n_i);
        ip1_x = wrap_idx(ix + 1, n_i); ip2_x = wrap_idx(ix + 2, n_i);
        im2_y = gid.y; im1_y = gid.y; ip1_y = gid.y; ip2_y = gid.y;
    } else {
        im2_x = gid.x; im1_x = gid.x; ip1_x = gid.x; ip2_x = gid.x;
        im2_y = wrap_idx(iy - 2, n_i); im1_y = wrap_idx(iy - 1, n_i);
        ip1_y = wrap_idx(iy + 1, n_i); ip2_y = wrap_idx(iy + 2, n_i);
    }

    let p_m2 = cell_primitive_pair_ppm(im2_x, im2_y, n, g, axis);
    let p_m1 = cell_primitive_pair_ppm(im1_x, im1_y, n, g, axis);
    let p_c  = cell_primitive_pair_ppm(gid.x, gid.y, n, g, axis);
    let p_p1 = cell_primitive_pair_ppm(ip1_x, ip1_y, n, g, axis);
    let p_p2 = cell_primitive_pair_ppm(ip2_x, ip2_y, n, g, axis);

    // 4th-order edge interpolation:
    //   q_{j-½} from (q_{j-2}, q_{j-1}, q_j, q_{j+1})
    //   q_{j+½} from (q_{j-1}, q_j,    q_{j+1}, q_{j+2})
    // q_{j+½} = (7/12)(q_j + q_{j+1}) - (1/12)(q_{j-1} + q_{j+2})
    let c7 = 7.0 / 12.0;
    let c1 = 1.0 / 12.0;
    let qL0_raw = c7 * (p_m1.p0 + p_c.p0)  - c1 * (p_m2.p0 + p_p1.p0);
    let qR0_raw = c7 * (p_c.p0  + p_p1.p0) - c1 * (p_m1.p0 + p_p2.p0);
    let qL1_raw = c7 * (p_m1.p1 + p_c.p1)  - c1 * (p_m2.p1 + p_p1.p1);
    let qR1_raw = c7 * (p_c.p1  + p_p1.p1) - c1 * (p_m1.p1 + p_p2.p1);

    let lim0 = ppm_limit_vec4(p_c.p0, qL0_raw, qR0_raw);
    let lim1 = ppm_limit_vec4(p_c.p1, qL1_raw, qR1_raw);

    let idx = cell_index(gid.x, gid.y, n);

    // Apply floors on density and pressure to the reconstructed edges.
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

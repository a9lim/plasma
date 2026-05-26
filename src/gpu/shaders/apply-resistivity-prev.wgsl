// ─── apply-resistivity-prev.wgsl ─────────────────────────────────────
// Pass 2 of one RKL2 substep. Adds
//     μ_j · Y_{j-1} + μ̃_j · Δt · L(Y_{j-1})
// to the accumulator written by apply-resistivity-init.wgsl, completing
// Y_j in the tmp slot. L(Y_{j-1}) uses the SAME frozen-init η_local as
// pass 1 (sampled from U^n via face B in init slots).
//
// Bindings (10 storage + 2 uniforms):
//   0  uniforms
//   1  sts_meta        (uniform)
//   2  sts_coeffs      (ro storage)
//   3  Bx_init  (ro)             — only for η_local (frozen)
//   4  By_init  (ro)             — only for η_local (frozen)
//   5  Bx_prev  (ro)
//   6  By_prev  (ro)
//   7  U1_prev  (ro)
//   8  Bx_tmp   (rw)             — read accumulator + write Y_j
//   9  By_tmp   (rw)
//   10 U1_tmp   (rw)
//
// Storage bindings: 9. Under the 10 cap.

struct StsMeta {
    substep_idx: u32,
    s_total:     u32,
    dt_super:    f32,
    _pad:        f32,
};

@group(0) @binding(0)  var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1)  var<uniform> sts_meta:   StsMeta;
@group(0) @binding(2)  var<storage, read>       sts_coeffs: array<f32>;
@group(0) @binding(3)  var<storage, read>       Bx_init:    array<f32>;
@group(0) @binding(4)  var<storage, read>       By_init:    array<f32>;
@group(0) @binding(5)  var<storage, read>       Bx_prev:    array<f32>;
@group(0) @binding(6)  var<storage, read>       By_prev:    array<f32>;
@group(0) @binding(7)  var<storage, read>       U1_prev:    array<vec4<f32>>;
@group(0) @binding(8)  var<storage, read_write> Bx_tmp:     array<f32>;
@group(0) @binding(9)  var<storage, read_write> By_tmp:     array<f32>;
@group(0) @binding(10) var<storage, read_write> U1_tmp:     array<vec4<f32>>;

fn jz_mag_init(ix: u32, iy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let by_R = 0.5 * (By_init[by_face_down_idx(ix + 1u, iy, n_total)]
                    + By_init[by_face_up_idx  (ix + 1u, iy, n_total)]);
    let by_L = 0.5 * (By_init[by_face_down_idx(ix - 1u, iy, n_total)]
                    + By_init[by_face_up_idx  (ix - 1u, iy, n_total)]);
    let bx_U = 0.5 * (Bx_init[bx_face_left_idx (ix, iy + 1u, n_total)]
                    + Bx_init[bx_face_right_idx(ix, iy + 1u, n_total)]);
    let bx_D = 0.5 * (Bx_init[bx_face_left_idx (ix, iy - 1u, n_total)]
                    + Bx_init[bx_face_right_idx(ix, iy - 1u, n_total)]);
    let dby_dx = (by_R - by_L) * 0.5 * dx_inv;
    let dbx_dy = (bx_U - bx_D) * 0.5 * dx_inv;
    return abs(dby_dx - dbx_dy);
}

fn eta_local_init(ix: u32, iy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let alpha = U_uniforms.eta_anom_alpha;
    if (alpha <= 0.0) { return U_uniforms.eta; }
    let jmag = jz_mag_init(ix, iy, n_total, dx_inv);
    return anomalous_eta(jmag, U_uniforms.eta, alpha, U_uniforms.eta_anom_jcrit);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_total    = U_uniforms.grid_n_total;
    let n_interior = U_uniforms.grid_n;
    let ghost      = U_uniforms.ghost_w;
    let dx_inv     = 1.0 / U_uniforms.dx;
    let dx2_inv    = dx_inv * dx_inv;
    let dt_super   = sts_meta.dt_super;
    if (sts_meta.s_total == 0u) { return; }

    let j_idx       = sts_meta.substep_idx;
    let base        = (j_idx - 1u) * 4u;
    let mu_j        = sts_coeffs[base + 0u];
    let mu_tilde_j  = sts_coeffs[base + 2u];

    let ix = gid.x + ghost - 1u;
    let iy = gid.y + ghost - 1u;

    let in_cell_interior =
        ix >= ghost && ix < ghost + n_interior &&
        iy >= ghost && iy < ghost + n_interior;
    let in_bx_interior =
        ix > ghost && ix < ghost + n_interior &&
        iy >= ghost && iy < ghost + n_interior;
    let in_by_interior =
        ix >= ghost && ix < ghost + n_interior &&
        iy > ghost && iy < ghost + n_interior;

    if (!(in_cell_interior || in_bx_interior || in_by_interior)) { return; }

    // ── Bz ─────────────────────────────────────────────────────────
    if (in_cell_interior) {
        let c  = cell_idx_total(ix,      iy,      n_total);
        let xl = cell_idx_total(ix - 1u, iy,      n_total);
        let xr = cell_idx_total(ix + 1u, iy,      n_total);
        let yd = cell_idx_total(ix,      iy - 1u, n_total);
        let yu = cell_idx_total(ix,      iy + 1u, n_total);

        let eta_c = eta_local_init(ix, iy, n_total, dx_inv);
        let bz_p  = U1_prev[c].y;
        let lap_p = U1_prev[xr].y + U1_prev[xl].y
                  + U1_prev[yu].y + U1_prev[yd].y - 4.0 * bz_p;
        let L_p   = eta_c * dx2_inv * lap_p;

        var u1 = U1_tmp[c];
        u1.y = u1.y + mu_j * bz_p + dt_super * mu_tilde_j * L_p;
        U1_tmp[c] = u1;
    }

    // ── Bx_face ────────────────────────────────────────────────────
    if (in_bx_interior) {
        let c  = bx_face_idx(ix,      iy,      n_total);
        let xl = bx_face_idx(ix - 1u, iy,      n_total);
        let xr = bx_face_idx(ix + 1u, iy,      n_total);
        let yd = bx_face_idx(ix,      iy - 1u, n_total);
        let yu = bx_face_idx(ix,      iy + 1u, n_total);

        let eta_l = eta_local_init(ix - 1u, iy, n_total, dx_inv);
        let eta_r = eta_local_init(ix,      iy, n_total, dx_inv);
        let eta_f = 0.5 * (eta_l + eta_r);

        let v_p   = Bx_prev[c];
        let lap_p = Bx_prev[xr] + Bx_prev[xl] + Bx_prev[yu] + Bx_prev[yd] - 4.0 * v_p;
        let L_p   = eta_f * dx2_inv * lap_p;

        Bx_tmp[c] = Bx_tmp[c] + mu_j * v_p + dt_super * mu_tilde_j * L_p;
    }

    // ── By_face ────────────────────────────────────────────────────
    if (in_by_interior) {
        let c  = by_face_idx(ix,      iy,      n_total);
        let xl = by_face_idx(ix - 1u, iy,      n_total);
        let xr = by_face_idx(ix + 1u, iy,      n_total);
        let yd = by_face_idx(ix,      iy - 1u, n_total);
        let yu = by_face_idx(ix,      iy + 1u, n_total);

        let eta_d = eta_local_init(ix, iy - 1u, n_total, dx_inv);
        let eta_u = eta_local_init(ix, iy,      n_total, dx_inv);
        let eta_f = 0.5 * (eta_d + eta_u);

        let v_p   = By_prev[c];
        let lap_p = By_prev[xr] + By_prev[xl] + By_prev[yu] + By_prev[yd] - 4.0 * v_p;
        let L_p   = eta_f * dx2_inv * lap_p;

        By_tmp[c] = By_tmp[c] + mu_j * v_p + dt_super * mu_tilde_j * L_p;
    }
}

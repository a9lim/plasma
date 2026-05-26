// ─── apply-resistivity-prev.wgsl ─────────────────────────────────────
// Pass 2 of one RKL2 substep. Adds
//     μ_j · Y_{j-1} + μ̃_j · Δt · L(Y_{j-1})
// to the accumulator written by apply-resistivity-init.wgsl, completing
// Y_j in the tmp slot.
//
// L is the **curl(η J)** form for face-centered B:
//     ∂Bx/∂t |_res = −∂_y(η J_z)
//     ∂By/∂t |_res = +∂_x(η J_z)
//     ∂Bz/∂t |_res = η ∇²Bz   (cell-centered, kept as component Laplacian)
// J_z is sampled at corners from the PREV face B (substep-dependent).
// η is sampled at the same corner from the FROZEN init face B —
// "frozen-coefficient" RKL2 contract (Meyer-Diehl-Kupka 2014). This
// matches the Athena++/PLUTO canonical resistivity discretization and
// is identically ∇·B-preserving on the Yee staggered grid.
//
// See apply-resistivity-init.wgsl header for the full derivation and
// motivation. The two shaders share L's recipe; only the source field
// (init vs prev) differs.
//
// Bindings (10 storage + 2 uniforms — at per-pipeline cap after the
// Session 10 dt-feedback fix added dt_buf):
//   0  uniforms
//   1  sts_meta        (uniform — dt_super field is unused; see init shader)
//   2  sts_coeffs      (ro storage)
//   3  Bx_init  (ro)             — only for η_local (frozen anomalous closure)
//   4  By_init  (ro)             — only for η_local (frozen anomalous closure)
//   5  Bx_prev  (ro)
//   6  By_prev  (ro)
//   7  U1_prev  (ro)
//   8  Bx_tmp   (rw)             — read accumulator + write Y_j
//   9  By_tmp   (rw)
//   10 U1_tmp   (rw)
//   11 dt_buf   (uniform)        — Session 10: fresh per-step dt_hyp.

struct StsMeta {
    substep_idx: u32,
    s_total:     u32,
    dt_super:    f32,    // unused by shader; kept for CPU diagnostic + layout
    _pad:        f32,
};

struct DtUniform {
    dt_hyp:       f32,
    dt_parabolic: f32,
    eta_max:      f32,
    _pad:         f32,
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
@group(0) @binding(11) var<uniform>             dt_buf:     DtUniform;

// J_z at corner from PREV face B.
fn jz_prev_corner(cx: u32, cy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let by_R = By_prev[by_face_idx(cx,      cy, n_total)];
    let by_L = By_prev[by_face_idx(cx - 1u, cy, n_total)];
    let bx_U = Bx_prev[bx_face_idx(cx, cy,      n_total)];
    let bx_D = Bx_prev[bx_face_idx(cx, cy - 1u, n_total)];
    return ((by_R - by_L) - (bx_U - bx_D)) * dx_inv;
}

// J_z at corner from FROZEN init face B — used only for the anomalous
// η(|J|) closure when α > 0. RKL2's frozen-coefficient contract: η is
// evaluated at U^n and held constant across the substep loop.
fn jz_init_corner_for_eta(cx: u32, cy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let by_R = By_init[by_face_idx(cx,      cy, n_total)];
    let by_L = By_init[by_face_idx(cx - 1u, cy, n_total)];
    let bx_U = Bx_init[bx_face_idx(cx, cy,      n_total)];
    let bx_D = Bx_init[bx_face_idx(cx, cy - 1u, n_total)];
    return ((by_R - by_L) - (bx_U - bx_D)) * dx_inv;
}

// η · J_z at corner. J_z is from PREV (substep-current); η is from
// FROZEN init.
fn ez_res_prev_corner(cx: u32, cy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let jz_p = jz_prev_corner(cx, cy, n_total, dx_inv);
    let alpha = U_uniforms.eta_anom_alpha;
    var eta_c = U_uniforms.eta;
    if (alpha > 0.0) {
        let jz_i = jz_init_corner_for_eta(cx, cy, n_total, dx_inv);
        eta_c = anomalous_eta(abs(jz_i), U_uniforms.eta, alpha,
                              U_uniforms.eta_anom_jcrit);
    }
    return eta_c * jz_p;
}

// Cell-centered η for the Bz Laplacian. Same recipe as init (frozen).
fn jz_mag_cell_init(ix: u32, iy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let by_R = 0.5 * (By_init[by_face_idx(ix + 1u, iy, n_total)]
                    + By_init[by_face_idx(ix + 1u, iy + 1u, n_total)]);
    let by_L = 0.5 * (By_init[by_face_idx(ix - 1u, iy, n_total)]
                    + By_init[by_face_idx(ix - 1u, iy + 1u, n_total)]);
    let bx_U = 0.5 * (Bx_init[bx_face_idx(ix, iy + 1u, n_total)]
                    + Bx_init[bx_face_idx(ix + 1u, iy + 1u, n_total)]);
    let bx_D = 0.5 * (Bx_init[bx_face_idx(ix, iy - 1u, n_total)]
                    + Bx_init[bx_face_idx(ix + 1u, iy - 1u, n_total)]);
    let dby_dx = (by_R - by_L) * 0.5 * dx_inv;
    let dbx_dy = (bx_U - bx_D) * 0.5 * dx_inv;
    return abs(dby_dx - dbx_dy);
}

fn eta_cell_init(ix: u32, iy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let alpha = U_uniforms.eta_anom_alpha;
    if (alpha <= 0.0) { return U_uniforms.eta; }
    let jmag = jz_mag_cell_init(ix, iy, n_total, dx_inv);
    return anomalous_eta(jmag, U_uniforms.eta, alpha, U_uniforms.eta_anom_jcrit);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_total    = U_uniforms.grid_n_total;
    let n_interior = U_uniforms.grid_n;
    let ghost      = U_uniforms.ghost_w;
    let dx_inv     = 1.0 / U_uniforms.dx;
    let dx2_inv    = dx_inv * dx_inv;
    let dt_super   = dt_buf.dt_hyp;
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
    let in_bx_face =
        ix >= ghost && ix <= ghost + n_interior &&
        iy >= ghost && iy < ghost + n_interior;
    let in_by_face =
        ix >= ghost && ix < ghost + n_interior &&
        iy >= ghost && iy <= ghost + n_interior;

    if (!(in_cell_interior || in_bx_face || in_by_face)) { return; }

    // ── Bz (cell-centered) ─────────────────────────────────────────
    if (in_cell_interior) {
        let c  = cell_idx_total(ix,      iy,      n_total);
        let xl = cell_idx_total(ix - 1u, iy,      n_total);
        let xr = cell_idx_total(ix + 1u, iy,      n_total);
        let yd = cell_idx_total(ix,      iy - 1u, n_total);
        let yu = cell_idx_total(ix,      iy + 1u, n_total);

        let eta_c = eta_cell_init(ix, iy, n_total, dx_inv);
        let bz_p  = U1_prev[c].y;
        let lap_p = U1_prev[xr].y + U1_prev[xl].y
                  + U1_prev[yu].y + U1_prev[yd].y - 4.0 * bz_p;
        let L_p   = eta_c * dx2_inv * lap_p;

        // E preservation: init wrote U^n.E into u1.x; prev only adds
        // to u1.y. Read-modify-write preserves u1.x by leaving it
        // untouched. Don't trust U1_tmp.x against an init regression —
        // if init's explicit-x write breaks, this preservation chain
        // breaks too (Session 9 fix).
        var u1 = U1_tmp[c];
        u1.y = u1.y + mu_j * bz_p + dt_super * mu_tilde_j * L_p;
        U1_tmp[c] = u1;
    }

    // ── Bx_face: −∂_y(η J_z) on Y_{j-1} ─────────────────────────────
    if (in_bx_face) {
        let c = bx_face_idx(ix, iy, n_total);
        let ez_bot = ez_res_prev_corner(ix, iy,      n_total, dx_inv);
        let ez_top = ez_res_prev_corner(ix, iy + 1u, n_total, dx_inv);
        let L_p    = -(ez_top - ez_bot) * dx_inv;

        let v_p    = Bx_prev[c];
        Bx_tmp[c]  = Bx_tmp[c] + mu_j * v_p + dt_super * mu_tilde_j * L_p;
    }

    // ── By_face: +∂_x(η J_z) on Y_{j-1} ─────────────────────────────
    if (in_by_face) {
        let c = by_face_idx(ix, iy, n_total);
        let ez_lft = ez_res_prev_corner(ix,      iy, n_total, dx_inv);
        let ez_rgt = ez_res_prev_corner(ix + 1u, iy, n_total, dx_inv);
        let L_p    = (ez_rgt - ez_lft) * dx_inv;

        let v_p    = By_prev[c];
        By_tmp[c]  = By_tmp[c] + mu_j * v_p + dt_super * mu_tilde_j * L_p;
    }
}

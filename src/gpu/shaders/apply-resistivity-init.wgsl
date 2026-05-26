// ─── apply-resistivity-init.wgsl ─────────────────────────────────────
// Pass 1 of one RKL2 substep. Computes
//     Y_tmp = (1 − μ_j − ν_j) · U^n + ν_j · Y_{j-2}
//             + γ̃_j · Δt · L(U^n)
// L is the resistive operator. For the FACE-CENTERED magnetic field
// (Bx, By), L is the **curl(η J)** form on the Yee staggered grid:
//     ∂Bx/∂t |_res = −∂_y(η J_z)              (Bx lives on x-faces)
//     ∂By/∂t |_res = +∂_x(η J_z)              (By lives on y-faces)
//     ∂Bz/∂t |_res = η ∇²Bz                   (cell-centered)
// where J_z lives at cell corners (same staggering as Ez_edge) and
// η_corner is sampled at the same corners. This is the canonical
// Athena++/PLUTO resistivity recipe (Athena++ `src/diffusion/
// diffusion_b.cpp`, PLUTO `Src/MHD/Resistive/res_flux.c`). On the
// discrete Yee grid the curl form is **identically ∇·B-preserving** by
// the same divergence-cleaning argument as ideal-MHD CT: every face
// receives a curl-of-an-edge-centered quantity, so summing the four
// face updates of any cell telescopes to zero contribution to ∇·B.
//
// For uniform η on a divergence-free B the curl form reduces
// algebraically to η ∇²B (proven by the discrete identity
// ∂²x Bx + ∂x∂y By = 0 under discrete ∂x Bx + ∂y By = 0). The previous
// component-wise η ∇²B implementation (Session 8 → 10) was correct in
// continuous math but accumulated a non-zero ∇·B term at the discrete
// level whenever the input field had any divergence (fp32 noise + BC
// inconsistency), and the leak compounded across RKL2 substeps. The
// curl form is structurally immune to this. See HANDOFF Session 10's
// "remaining fourth issue (next session) — divB leak at corner cells"
// for the bisect that surfaced this.
//
// For non-uniform η (anomalous resistivity), the curl form is the
// ONLY discretization that preserves ∇·B — η ∇²B picks up an
// uncontrolled ∇η × J cross term. The anomalous closure η(|J|) is
// evaluated at the corner using the corner J_z directly (more accurate
// than the cell-centered approximation the old code used).
//
// Cell-centered Bz keeps the per-component η ∇²Bz Laplacian. Bz has
// no associated face-staggered Bz (B is the (Bx_face, By_face, Bz_cell)
// triple in 2.5D), so the curl-on-Yee argument doesn't bind it; Bz is
// a passive scalar advected by the resistive operator (∇·B is unaware
// of Bz in 2D).
//
// E (u1.x) has no Laplacian operator (L_E = 0) — the RKL2 recurrence
// applied to E collapses to Y_j.E = U^n.E for all j by induction. The
// kernel writes U^n.E directly from U1_init to bypass cancellation-
// prone arithmetic and to defend against U1_tmp.x stale-buffer
// contamination (Session 9 retrospective).
//
// For j = 1: ν_j = 0, μ_j = 0, γ̃_j = 0  → Y_tmp = U^n. The follow-on
// `apply-resistivity-prev` then adds the μ̃_1 · Δt · L(Y_0) term.
//
// Bindings (one group, 10 storage + 2 uniforms — at the per-pipeline
// storage-binding cap):
//   0  uniforms
//   1  sts_meta        (uniform — substep_idx, s_total, _pad, _pad)
//                       (dt_super field is RETAINED for layout compatibility
//                        and as a CPU-side diagnostic of the lagged value,
//                        but the shader reads the FRESH dt_super from
//                        dt_buf[0] instead — see binding 12 comment.)
//   2  sts_coeffs      (ro storage — packed (μ, ν, μ̃, γ̃) per substep)
//   3  Bx_init  (ro)   — frozen U^n face B (value + J_z for L(U^n) + η anomalous)
//   4  By_init  (ro)
//   5  U1_init  (ro)
//   6  Bx_pprev (ro)
//   7  By_pprev (ro)
//   8  U1_pprev (ro)
//   9  Bx_tmp   (rw)
//   10 By_tmp   (rw)
//   11 U1_tmp   (rw)
//   12 dt_buf   (uniform) — Session 10 RKL2 dt-feedback fix. compute-dt
//                       writes the fresh per-step dt_hyp into dt_buf[0]
//                       at the start of EACH macro step (before any
//                       RKL2 substep runs). Reading from here instead of
//                       sts_meta.dt_super eliminates the CPU-side
//                       _lastDtHyp staleness path that detonated Harris
//                       in tight render loops.

struct StsMeta {
    substep_idx: u32,
    s_total:     u32,
    dt_super:    f32,    // unused by shader; kept for CPU diagnostic + layout
    _pad:        f32,
};

// dt_buf is bound as a UNIFORM struct here (rather than storage<read>)
// for two reasons: (1) keeps this shader's storage-binding count at 9
// (under the 10-per-stage cap), (2) matches the established pattern in
// update-conserved-weighted.wgsl which faces the same cap constraint.
// The buffer is created with both STORAGE and UNIFORM usage flags in
// buffers.js so the same buffer can be bound either way.
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
@group(0) @binding(5)  var<storage, read>       U1_init:    array<vec4<f32>>;
@group(0) @binding(6)  var<storage, read>       Bx_pprev:   array<f32>;
@group(0) @binding(7)  var<storage, read>       By_pprev:   array<f32>;
@group(0) @binding(8)  var<storage, read>       U1_pprev:   array<vec4<f32>>;
@group(0) @binding(9)  var<storage, read_write> Bx_tmp:     array<f32>;
@group(0) @binding(10) var<storage, read_write> By_tmp:     array<f32>;
@group(0) @binding(11) var<storage, read_write> U1_tmp:     array<vec4<f32>>;
@group(0) @binding(12) var<uniform>             dt_buf:     DtUniform;

// J_z at cell corner (cx, cy) from FROZEN init face B. The corner
// (cx, cy) sits at the BOTTOM-LEFT of cell (cx, cy), exactly co-located
// with Ez_edge. Yee-natural stencil:
//   J_z = (∂_x By − ∂_y Bx)
//       = ((By[cx, cy] − By[cx-1, cy]) − (Bx[cx, cy] − Bx[cx, cy-1])) / dx
fn jz_init_corner(cx: u32, cy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let by_R = By_init[by_face_idx(cx,      cy, n_total)];
    let by_L = By_init[by_face_idx(cx - 1u, cy, n_total)];
    let bx_U = Bx_init[bx_face_idx(cx, cy,      n_total)];
    let bx_D = Bx_init[bx_face_idx(cx, cy - 1u, n_total)];
    return ((by_R - by_L) - (bx_U - bx_D)) * dx_inv;
}

// η · J_z at cell corner (cx, cy). η_corner is sampled from the
// anomalous closure evaluated AT THE CORNER using the corner J_z
// directly. For uniform η (α = 0) this just multiplies by U.eta.
fn ez_res_init_corner(cx: u32, cy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let jz = jz_init_corner(cx, cy, n_total, dx_inv);
    let alpha = U_uniforms.eta_anom_alpha;
    var eta_c = U_uniforms.eta;
    if (alpha > 0.0) {
        eta_c = anomalous_eta(abs(jz), U_uniforms.eta, alpha,
                              U_uniforms.eta_anom_jcrit);
    }
    return eta_c * jz;
}

// Cell-centered η for the Bz Laplacian. The Bz update keeps the
// per-component η ∇²Bz form; in 2D Bz doesn't couple to ∇·B so the
// curl-on-Yee argument doesn't bind it. η at the cell center uses the
// face-averaged J_z magnitude (same recipe as Session 8's anomalous
// implementation) — cell-centered is the natural staggering for a
// cell-centered Laplacian.
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
    // Read fresh Δt_super straight from the GPU dt buffer (Session 10
    // RKL2 dt-feedback fix).
    let dt_super   = dt_buf.dt_hyp;
    if (sts_meta.s_total == 0u) { return; }

    let j_idx       = sts_meta.substep_idx;        // 1..s
    let base        = (j_idx - 1u) * 4u;
    let mu_j        = sts_coeffs[base + 0u];
    let nu_j        = sts_coeffs[base + 1u];
    let gam_tilde_j = sts_coeffs[base + 3u];
    let one_minus   = 1.0 - mu_j - nu_j;

    let ix = gid.x + ghost - 1u;
    let iy = gid.y + ghost - 1u;

    // Bz update: strictly interior cells.
    let in_cell_interior =
        ix >= ghost && ix < ghost + n_interior &&
        iy >= ghost && iy < ghost + n_interior;
    // Face updates: include the boundary faces (ix == ghost and
    // ix == ghost + n_interior for Bx; iy mirror for By). Updating the
    // boundary face with the curl form is what makes ∇·B identically
    // preserved at boundary cells too — the four-face telescoping
    // argument needs every face of every interior cell to receive a
    // curl contribution. (HANDOFF Session 10 documented that excluding
    // boundary faces lets ∇·B leak at boundary cells per substep.)
    let in_bx_face =
        ix >= ghost && ix <= ghost + n_interior &&
        iy >= ghost && iy < ghost + n_interior;
    let in_by_face =
        ix >= ghost && ix < ghost + n_interior &&
        iy >= ghost && iy <= ghost + n_interior;

    if (!(in_cell_interior || in_bx_face || in_by_face)) { return; }

    // ── Bz (cell-centered, η ∇²Bz) ─────────────────────────────────
    if (in_cell_interior) {
        let c  = cell_idx_total(ix,      iy,      n_total);
        let xl = cell_idx_total(ix - 1u, iy,      n_total);
        let xr = cell_idx_total(ix + 1u, iy,      n_total);
        let yd = cell_idx_total(ix,      iy - 1u, n_total);
        let yu = cell_idx_total(ix,      iy + 1u, n_total);

        let eta_c = eta_cell_init(ix, iy, n_total, dx_inv);
        let bz_0  = U1_init[c].y;
        let lap_0 = U1_init[xr].y + U1_init[xl].y
                  + U1_init[yu].y + U1_init[yd].y - 4.0 * bz_0;
        let L_0   = eta_c * dx2_inv * lap_0;
        let pprev_bz = U1_pprev[c].y;

        let bz_new = one_minus * bz_0 + nu_j * pprev_bz + dt_super * gam_tilde_j * L_0;

        // E (u1.x) has L_E = 0 ⇒ Y_j.E = U^n.E for all j. Write U^n.E
        // directly from frozen U1_init; ALSO defends against U1_tmp's
        // stale-buffer contamination on substep 1 (Session 9 fix).
        U1_tmp[c] = vec4<f32>(U1_init[c].x, bz_new, 0.0, 0.0);
    }

    // ── Bx_face: ∂_t Bx = −∂_y(η J_z) ──────────────────────────────
    if (in_bx_face) {
        let c = bx_face_idx(ix, iy, n_total);
        // Ez_res at the two corners that bracket this x-face in y.
        // Corner (ix, iy)   is the BOTTOM-LEFT of cell (ix, iy)
        //                      = bottom corner of Bx_face[ix, iy].
        // Corner (ix, iy+1) is its top corner.
        let ez_bot = ez_res_init_corner(ix, iy,      n_total, dx_inv);
        let ez_top = ez_res_init_corner(ix, iy + 1u, n_total, dx_inv);
        let L_0    = -(ez_top - ez_bot) * dx_inv;

        let v_0    = Bx_init[c];
        let pprev  = Bx_pprev[c];
        Bx_tmp[c]  = one_minus * v_0 + nu_j * pprev + dt_super * gam_tilde_j * L_0;
    }

    // ── By_face: ∂_t By = +∂_x(η J_z) ──────────────────────────────
    if (in_by_face) {
        let c = by_face_idx(ix, iy, n_total);
        // Corner (ix,   iy) is the BOTTOM-LEFT of cell (ix, iy)
        //                      = left  corner of By_face[ix, iy].
        // Corner (ix+1, iy) is its right corner.
        let ez_lft = ez_res_init_corner(ix,      iy, n_total, dx_inv);
        let ez_rgt = ez_res_init_corner(ix + 1u, iy, n_total, dx_inv);
        let L_0    = (ez_rgt - ez_lft) * dx_inv;

        let v_0    = By_init[c];
        let pprev  = By_pprev[c];
        By_tmp[c]  = one_minus * v_0 + nu_j * pprev + dt_super * gam_tilde_j * L_0;
    }
}

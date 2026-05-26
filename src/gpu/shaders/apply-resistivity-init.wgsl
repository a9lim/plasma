// ─── apply-resistivity-init.wgsl ─────────────────────────────────────
// Pass 1 of one RKL2 substep. Computes
//     Y_tmp = (1 − μ_j − ν_j) · U^n + ν_j · Y_{j-2}
//             + γ̃_j · Δt · L(U^n)
// L(U^n) is recomputed each substep (cheap 5-point stencil; one extra
// pass over Y_init per cell). η_local is sampled from the FROZEN init
// state J_z — for anomalous resistivity, U_uniforms.eta_anom_alpha > 0
// activates the |J|>J_crit boost; α = 0 returns the uniform base η.
//
// Companion to apply-resistivity-prev.wgsl. See apply-resistivity.wgsl
// for the full RKL2 method overview, recurrence, and host orchestration.
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
//   3  Bx_init  (ro)
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
//                       _lastDtHyp staleness path that detonates Harris
//                       in tight render loops: when consecutive step()
//                       calls have no microtask break, the async
//                       readback of dt_buf into _lastDtHyp never
//                       completes, so sts_meta.dt_super stays at its
//                       1e-4 initial value while the actual dt_hyp has
//                       grown to ~1e-3. RKL2's recurrence then applies
//                       only ~1/10 of the needed resistive diffusion
//                       per macro step → current sheet thins
//                       unphysically → NaN cascade within ~50 steps.
//                       Per-substep diffusion contract: Δt_super in
//                       the recurrence equals dt_hyp (the hyperbolic
//                       step). The substep count s is still CPU-sized
//                       from the lagged _lastDtParabolic ratio — when
//                       the parabolic CFL has actually tightened, the
//                       CPU s is OVER-sized relative to the strict
//                       requirement, which is the safe-fail direction
//                       (RKL2 is unconditionally stable for s ≥
//                       s_critical). See Session 10 / 9 retrospective.

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

// |J_z(ix, iy)| from frozen init face B — central differences on
// face-averaged cell-centered B. Reads (ix±1, iy±1) face cells.
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
    // Read fresh Δt_super straight from the GPU dt buffer (Session 10
    // RKL2 dt-feedback fix). compute-dt populates dt_buf[0] at the start
    // of every macro step before this kernel runs. sts_meta.dt_super
    // would be the lagged value from the host's last successful readback
    // of dt_buf — under tight render loops without microtask breaks that
    // readback never completes and dt_super stays stuck at its initial
    // value, dramatically under-applying η ∇²B over the macro step and
    // detonating Harris.
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
        let bz_0  = U1_init[c].y;
        let lap_0 = U1_init[xr].y + U1_init[xl].y
                  + U1_init[yu].y + U1_init[yd].y - 4.0 * bz_0;
        let L_0   = eta_c * dx2_inv * lap_0;
        let pprev_bz = U1_pprev[c].y;

        let bz_new = one_minus * bz_0 + nu_j * pprev_bz + dt_super * gam_tilde_j * L_0;

        // E (u1.x) has no Laplacian operator (L_E = 0), so the RKL2
        // recurrence collapses to Y_j.E = U^n.E for all j by induction
        // — see the algebra in the file header. Bypass the (cancellation-
        // prone) linear combination math and write U^n.E directly from
        // the frozen init buffer. The pad slots (z, w) match the
        // canonical (E, Bz, 0, 0) layout produced by update-conserved-
        // weighted.
        //
        // This also matters because `U1_tmp` is NOT snapshotted at
        // super-step boot (only init/pprev/prev are seeded from dst), so
        // on the first substep its u1.x slot is whatever stale or
        // zero-initialized data the buffer held. A read-modify-write
        // through `var u1 = U1_tmp[c]; u1.y = bz_new; U1_tmp[c] = u1`
        // would silently propagate that garbage into u1.x and corrupt E
        // across the whole interior — masking it with the energy floor
        // downstream and breaking Harris at η > 0 (Session 9).
        U1_tmp[c] = vec4<f32>(U1_init[c].x, bz_new, 0.0, 0.0);
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

        let v_0   = Bx_init[c];
        let lap_0 = Bx_init[xr] + Bx_init[xl] + Bx_init[yu] + Bx_init[yd] - 4.0 * v_0;
        let L_0   = eta_f * dx2_inv * lap_0;
        let pprev = Bx_pprev[c];

        Bx_tmp[c] = one_minus * v_0 + nu_j * pprev + dt_super * gam_tilde_j * L_0;
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

        let v_0   = By_init[c];
        let lap_0 = By_init[xr] + By_init[xl] + By_init[yu] + By_init[yd] - 4.0 * v_0;
        let L_0   = eta_f * dx2_inv * lap_0;
        let pprev = By_pprev[c];

        By_tmp[c] = one_minus * v_0 + nu_j * pprev + dt_super * gam_tilde_j * L_0;
    }
}

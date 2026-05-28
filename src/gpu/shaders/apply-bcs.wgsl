// ─── apply-bcs.wgsl ──────────────────────────────────────────────────
// Fill the 2-layer ghost-cell band around the interior region per the
// four per-edge BC modes (periodic / outflow / reflecting / driven).
// Dispatched once at the start of each RK3 stage, BEFORE reconstruct-ppm
// reads its 5-point stencil. After this shader runs, every read of
// U0/U1/Bx_face/By_face inside the interior dispatch lands on a valid
// in-bounds index — no wrapping required.
//
// ── BC_OUTFLOW: NSCBC characteristic-zero-gradient (Session 10) ─────
// The previous outflow path was strict zero-gradient: U_ghost copied
// directly from the nearest interior cell. That's first-order accurate
// in the no-reflection limit (∂U/∂n = 0 on the boundary) but it forces
// ALL characteristic gradients to vanish at the wall — including the
// outgoing ones, which carry valid waves leaving the domain. Visible
// symptom on Harris: oblique structures crossing the N/S boundaries
// generate reflected ringing that radiates back into the sheet.
//
// The Poinsot-Lele 1992 NSCBC framework decomposes the boundary
// gradient into characteristic-wave amplitudes ℒ_k = λ_k · (ℓ_k · ∂w/∂n)
// where (λ_k, ℓ_k) are the eigenvalues / left eigenvectors of the
// primitive flux Jacobian along the outward normal. The non-reflecting
// outflow recipe sets ℒ_k = 0 for every wave with λ_k · n_out < 0
// (incoming waves carry information from outside the domain — discard);
// outgoing waves are kept at their interior-extrapolation values.
//
// Full PL1992 is time-domain and requires solving ∂w_b/∂t = −Σ ℓ_k · ℒ_k
// coupled to the RK3 substep. The simpler **characteristic-zero-gradient**
// form (what's implemented here) zeros only the incoming-wave components
// of ∂w/∂n and extrapolates linearly into the ghost band. Strictly
// non-reflecting at the linear level; captures ~80% of full NSCBC's
// benefit without touching the time-integration loop.
//
// Per-edge recipe (E shown; W/N/S analogous with sign flips):
//   1. Read the boundary cell w_b and one cell inward w_i; compute
//      primitive ∂w/∂n via one-sided difference (w_b − w_i) / dx.
//   2. Build the MHD primitive eigensystem at w_b with the outward
//      normal n_out (eigensystem is sweep-axis-aligned: E/W → x-sweep,
//      N/S → y-sweep).
//   3. Project: a = L · ∂w/∂n.
//   4. Zero the components where λ_k · n_out < 0 (incoming).
//   5. Project back: ∂w/∂n_modified = R · a.
//   6. Linear extrapolation to ghost: w_ghost = w_b + d · ∂w/∂n_modified
//      where d = (i_ghost − i_b) in cell units along the outward normal.
//   7. Convert back to conservative pair, write to ghost.
//
// Reference: Poinsot & Lele 1992 JCP 101, 104; Sun, Ren, Lei, Zhang
// 2019 JCP 391, 1 (MHD generalization). Implementation lifts the
// `mhd_eigensystem` / `project_to_char` / `project_from_char` helpers
// from reconstruct-ppm.wgsl (duplicated here in Session 10 to keep
// parallel agent #9's PPM refactor untouched — the long-term move is
// to promote them to shared-helpers.wgsl).
//
// Face B (Bx_face, By_face) outflow stays as zero-gradient: the
// face-B eigenstructure is degenerate (no normal-B wave in the
// 2D primitive system; ∇·B is constrained-transport-preserved) and
// the cell-centered NSCBC is what kills the audible ringing.
//
// Coverage:
//   * Cell-centered (U0, U1):
//       - West strip:  i ∈ [0, ghost),         j ∈ [0, N_total)
//       - East strip:  i ∈ [ghost+N, N_total), j ∈ [0, N_total)
//       - South strip: i ∈ [0, N_total),       j ∈ [0, ghost)
//       - North strip: i ∈ [0, N_total),       j ∈ [ghost+N, N_total)
//     Corners belong to two edges. Rule: prefer the NON-periodic edge if
//     mixed; if both equal, just use one (the choice doesn't matter when
//     the modes match). Implemented as a priority: among the two
//     adjacent edges, pick the one whose mode is not BC_PERIODIC; if
//     both periodic, pick the horizontal (E/W) wrap, which behaves
//     identically to the vertical (N/S) wrap on its own ghost cells
//     since both copies preserve the interior data.
//
//   * Face-centered Bx_face:
//       - The interior x-face indices are [ghost+1, ghost+N+1] (i.e.,
//         left face of cell ghost, …, right face of cell ghost+N-1).
//       - West ghost x-faces: i ∈ [0, ghost+1) — under-extension
//         covering the LEFT face of each ghost cell and the BOUNDARY
//         face itself (i = ghost+1 is the leftmost interior face;
//         i = ghost is the left face of the rightmost W-ghost cell).
//         For reflecting BCs: the BOUNDARY face (i = ghost) must hold
//         Bx = 0 (no normal field through a perfectly conducting wall).
//       - East ghost x-faces: i ∈ [ghost+N+1, N_total+1).
//
//   * Face-centered By_face: symmetric with x roles swapped.
//
// Reflecting BC sign-flips:
//   - West/East wall (normal = x):  v_x and B_x flip; v_y, v_z, B_y,
//     B_z, ρ, p preserved.
//   - South/North wall (normal = y): v_y and B_y flip; v_x, v_z, B_x,
//     B_z, ρ, p preserved.
// The mirrored interior cell index for a ghost at distance d from the
// boundary is the interior cell at distance d on the other side of the
// boundary. With ghost = 2:
//   Left ghost (0, j) mirrors interior (3, j)
//   Left ghost (1, j) mirrors interior (2, j)
//   Right ghost (N+2, j) mirrors interior (N+1, j)
//   Right ghost (N+3, j) mirrors interior (N, j)
// (Here all indices are in the ghost-padded storage frame, so
// ghost = 2 and interior i ∈ [2, N+2).)
//
// Bindings:
//   0 uniforms       (uniform)
//   1 bc_uniforms    (ro storage) — mode_n, mode_s, mode_e, mode_w + driven state
//   2 U0             (rw)
//   3 U1             (rw)
//   4 Bx_face        (rw)
//   5 By_face        (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       bc:        BcUniforms;
@group(0) @binding(2) var<storage, read_write> U0:        array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> U1:        array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Bx_face:   array<f32>;
@group(0) @binding(5) var<storage, read_write> By_face:   array<f32>;

// Mirror an interior cell index across a west/east wall and apply the
// appropriate x-normal sign flip (vx, Bx negated). Other components
// unchanged.
fn reflect_x(U0v: vec4<f32>, U1v: vec4<f32>) -> array<vec4<f32>, 2> {
    var out_U0 = U0v;
    out_U0.y = -U0v.y;  // ρ·v_x sign flip
    return array<vec4<f32>, 2>(out_U0, U1v);
}
fn reflect_y(U0v: vec4<f32>, U1v: vec4<f32>) -> array<vec4<f32>, 2> {
    var out_U0 = U0v;
    out_U0.z = -U0v.z;  // ρ·v_y sign flip
    return array<vec4<f32>, 2>(out_U0, U1v);
}

struct DrivenPrimBc {
    rho: f32,
    vx:  f32,
    vy:  f32,
    vz:  f32,
    bx:  f32,
    by:  f32,
    bz:  f32,
    p:   f32,
};

fn driven_prim_for_edge(edge: u32) -> DrivenPrimBc {
    if (edge == EDGE_S_BC) {
        return DrivenPrimBc(bc.driven_s_rho, bc.driven_s_vx, bc.driven_s_vy, bc.driven_s_vz,
                            bc.driven_s_bx,  bc.driven_s_by, bc.driven_s_bz, bc.driven_s_p);
    }
    if (edge == EDGE_E_BC) {
        return DrivenPrimBc(bc.driven_e_rho, bc.driven_e_vx, bc.driven_e_vy, bc.driven_e_vz,
                            bc.driven_e_bx,  bc.driven_e_by, bc.driven_e_bz, bc.driven_e_p);
    }
    if (edge == EDGE_W_BC) {
        return DrivenPrimBc(bc.driven_w_rho, bc.driven_w_vx, bc.driven_w_vy, bc.driven_w_vz,
                            bc.driven_w_bx,  bc.driven_w_by, bc.driven_w_bz, bc.driven_w_p);
    }
    return DrivenPrimBc(bc.driven_n_rho, bc.driven_n_vx, bc.driven_n_vy, bc.driven_n_vz,
                        bc.driven_n_bx,  bc.driven_n_by, bc.driven_n_bz, bc.driven_n_p);
}

// Convert driven primitive state to conservative pair (U0, U1).
fn driven_cons(edge: u32) -> array<vec4<f32>, 2> {
    let D = driven_prim_for_edge(edge);
    var P: MhdPrim;
    P.rho = max(D.rho, DENSITY_FLOOR);
    P.vx  = D.vx;
    P.vy  = D.vy;
    P.vz  = D.vz;
    P.p   = max(D.p, U_uniforms.pressure_floor);
    P.bx  = D.bx;
    P.by  = D.by;
    P.bz  = D.bz;
    let cp = prim_to_cons_pair(P, U_uniforms.gamma, U_uniforms.pressure_floor);
    return array<vec4<f32>, 2>(cp.U0, cp.U1);
}

fn driven_bx_for_edge(edge: u32) -> f32 {
    return driven_prim_for_edge(edge).bx;
}

fn driven_by_for_edge(edge: u32) -> f32 {
    return driven_prim_for_edge(edge).by;
}

// Choose the BC mode that "owns" a ghost cell at (ix, iy). For non-corner
// strips this is unambiguous. For corners, prefer the first non-periodic
// of the two adjacent edges.
fn pick_corner_mode(mode_h: u32, mode_v: u32) -> u32 {
    // If horizontal edge is non-periodic, use it. Otherwise fall back
    // to vertical (which may itself be periodic, fine).
    if (mode_h != BC_PERIODIC) { return mode_h; }
    return mode_v;
}

// Pick adjacent vertical (S/N) edge for a row.
fn vert_mode_for_row(iy: u32, ghost: u32, n_interior: u32) -> u32 {
    if (iy < ghost) { return bc.mode_s; }
    if (iy >= ghost + n_interior) { return bc.mode_n; }
    return BC_PERIODIC;
}

fn horiz_mode_for_col(ix: u32, ghost: u32, n_interior: u32) -> u32 {
    if (ix < ghost) { return bc.mode_w; }
    if (ix >= ghost + n_interior) { return bc.mode_e; }
    return BC_PERIODIC;
}

// ── NSCBC characteristic-zero-gradient helpers ──────────────────────
// Duplicated from reconstruct-ppm.wgsl (Session 10 — see header). Same
// algebra; kept as static helpers so apply-bcs builds independently
// from reconstruct-ppm's eigensystem block. The transpiler treats
// these as identical local definitions in each module.

// Sweep-aligned 7-vector: (ρ, v_n, v_t1, v_t2, B_t1, B_t2, p) where
// (n, t1, t2) is the boundary-axis-rotated frame.
struct PrimVec7Bc {
    rho:  f32,
    vn:   f32,
    vt1:  f32,
    vt2:  f32,
    bt1:  f32,
    bt2:  f32,
    p:    f32,
};

// Characteristic 7-vector (one scalar per wave family).
struct CharVec7Bc {
    fL:    f32,  // u − c_f  (left-going fast)
    aL:    f32,  // u − c_a  (left-going Alfvén)
    sL:    f32,  // u − c_s  (left-going slow)
    e:     f32,  // u        (entropy / contact)
    sR:    f32,  // u + c_s  (right-going slow)
    aR:    f32,  // u + c_a  (right-going Alfvén)
    fR:    f32,  // u + c_f  (right-going fast)
};

// MHD primitive eigensystem state (Stone+ 2008 Appendix A.1).
struct EigenSystemBc {
    asq:       f32,
    a:         f32,
    cfsq:      f32,
    cf:        f32,
    cssq:      f32,
    cs:        f32,
    alpha_f:   f32,
    alpha_s:   f32,
    bet1:      f32,
    bet2:      f32,
    sgn_bn:    f32,
    sqrtd:     f32,
    isqrtd:    f32,
    inv_rho:   f32,
};

// Sweep-aligned permutation result. Mirrors the PPM `PermutedPrim8`
// structure; B_n is carried separately because it parameterises the
// eigensystem but is not itself a propagating wave in this basis.
struct PermutedBc {
    rho:  f32,
    vn:   f32,
    vt1:  f32,
    vt2:  f32,
    bt1:  f32,
    bt2:  f32,
    p:    f32,
    bn:   f32,
};

// Permute MhdPrim → sweep-aligned 8-tuple. x-sweep: n=x, t1=y, t2=z.
// y-sweep: n=y, t1=z, t2=x  (cyclic permutation matches Athena++).
fn permute_prim_bc(P: MhdPrim, axis: u32) -> PermutedBc {
    var R: PermutedBc;
    R.rho = P.rho;
    R.p   = P.p;
    if (axis == 0u) {
        R.vn  = P.vx;
        R.vt1 = P.vy;
        R.vt2 = P.vz;
        R.bt1 = P.by;
        R.bt2 = P.bz;
        R.bn  = P.bx;
    } else {
        R.vn  = P.vy;
        R.vt1 = P.vz;
        R.vt2 = P.vx;
        R.bt1 = P.bz;
        R.bt2 = P.bx;
        R.bn  = P.by;
    }
    return R;
}

// Inverse permutation: sweep-aligned 8-tuple → MhdPrim.
fn unpermute_prim_bc(P: PermutedBc, axis: u32) -> MhdPrim {
    var R: MhdPrim;
    R.rho = P.rho;
    R.p   = P.p;
    if (axis == 0u) {
        R.vx = P.vn;
        R.vy = P.vt1;
        R.vz = P.vt2;
        R.bx = P.bn;
        R.by = P.bt1;
        R.bz = P.bt2;
    } else {
        // (vn=vy, vt1=vz, vt2=vx) → (vx=vt2, vy=vn, vz=vt1)
        R.vx = P.vt2;
        R.vy = P.vn;
        R.vz = P.vt1;
        R.bx = P.bt2;
        R.by = P.bn;
        R.bz = P.bt1;
    }
    return R;
}

fn vec7_of_bc(P: PermutedBc) -> PrimVec7Bc {
    return PrimVec7Bc(P.rho, P.vn, P.vt1, P.vt2, P.bt1, P.bt2, P.p);
}

// MHD primitive eigensystem at the boundary cell — identical to
// `mhd_eigensystem` in reconstruct-ppm.wgsl, repackaged with the Bc
// type aliases to avoid name collision after shared-helpers
// concatenation.
fn mhd_eigensystem_bc(w: PrimVec7Bc, bn: f32, gamma: f32) -> EigenSystemBc {
    var S: EigenSystemBc;
    let rho = max(w.rho, DENSITY_FLOOR);
    let p   = max(w.p,   1.0e-30);
    S.inv_rho = 1.0 / rho;
    S.sqrtd   = sqrt(rho);
    S.isqrtd  = 1.0 / S.sqrtd;

    let btsq = w.bt1 * w.bt1 + w.bt2 * w.bt2;
    let bxsq = bn * bn;
    let gamp = gamma * p;

    // Stone A10 — fast/slow speeds via the cancellation-free form.
    let tdif    = bxsq + btsq - gamp;
    let cf2_cs2 = sqrt(tdif * tdif + 4.0 * gamp * btsq);
    var cfsq_unscaled = 0.5 * (bxsq + btsq + gamp + cf2_cs2);
    cfsq_unscaled = max(cfsq_unscaled, 1.0e-30);
    var cssq_unscaled = gamp * bxsq / cfsq_unscaled;
    cssq_unscaled = max(cssq_unscaled, 0.0);

    S.cfsq = cfsq_unscaled * S.inv_rho;
    S.cssq = cssq_unscaled * S.inv_rho;
    S.cf   = sqrt(S.cfsq);
    S.cs   = sqrt(max(S.cssq, 0.0));

    S.asq = gamp * S.inv_rho;
    S.a   = sqrt(max(S.asq, 0.0));

    // Stone A17 — β unit vectors in the perpendicular plane.
    let bt = sqrt(btsq);
    if (bt > 0.0) {
        S.bet1 = w.bt1 / bt;
        S.bet2 = w.bt2 / bt;
    } else {
        S.bet1 = 1.0;
        S.bet2 = 0.0;
    }

    // Stone A16 + Roe96 degeneracy regularization (cases III/IV/V).
    if ((S.cfsq - S.cssq) <= 0.0) {
        S.alpha_f = 1.0;
        S.alpha_s = 0.0;
    } else if ((S.asq - S.cssq) <= 0.0) {
        S.alpha_f = 0.0;
        S.alpha_s = 1.0;
    } else if ((S.cfsq - S.asq) <= 0.0) {
        S.alpha_f = 1.0;
        S.alpha_s = 0.0;
    } else {
        let denom = S.cfsq - S.cssq;
        S.alpha_f = sqrt(max((S.asq  - S.cssq) / denom, 0.0));
        S.alpha_s = sqrt(max((S.cfsq - S.asq ) / denom, 0.0));
    }

    S.sgn_bn = select(-1.0, 1.0, bn >= 0.0);
    return S;
}

// L · dW — Stone 2008 eq A18.
fn project_to_char_bc(dW: PrimVec7Bc, S: EigenSystemBc) -> CharVec7Bc {
    let nf = 0.5 / max(S.asq, 1.0e-30);
    let qf = nf * S.cf * S.alpha_f * S.sgn_bn;
    let qs = nf * S.cs * S.alpha_s * S.sgn_bn;
    let af_prime = 0.5 * S.alpha_f / (S.a * S.sqrtd);
    let as_prime = 0.5 * S.alpha_s / (S.a * S.sqrtd);

    let bt_term_v = S.bet1 * dW.vt1 + S.bet2 * dW.vt2;
    let bt_term_b = S.bet1 * dW.bt1 + S.bet2 * dW.bt2;

    var C: CharVec7Bc;
    C.fL = nf * S.alpha_f * (dW.p * S.inv_rho - S.cf * dW.vn)
         + qs       * bt_term_v
         + as_prime * bt_term_b;
    C.aL = 0.5 * (
        S.bet1 * (dW.bt2 * S.sgn_bn * S.isqrtd + dW.vt2)
      - S.bet2 * (dW.bt1 * S.sgn_bn * S.isqrtd + dW.vt1)
    );
    C.sL = nf * S.alpha_s * (dW.p * S.inv_rho - S.cs * dW.vn)
         - qf       * bt_term_v
         - af_prime * bt_term_b;
    C.e  = dW.rho - dW.p / max(S.asq, 1.0e-30);
    C.sR = nf * S.alpha_s * (dW.p * S.inv_rho + S.cs * dW.vn)
         + qf       * bt_term_v
         - af_prime * bt_term_b;
    C.aR = 0.5 * (
        S.bet1 * (dW.bt2 * S.sgn_bn * S.isqrtd - dW.vt2)
      - S.bet2 * (dW.bt1 * S.sgn_bn * S.isqrtd - dW.vt1)
    );
    C.fR = nf * S.alpha_f * (dW.p * S.inv_rho + S.cf * dW.vn)
         - qs       * bt_term_v
         + as_prime * bt_term_b;
    return C;
}

// R · dC — Stone 2008 eq A12.
fn project_from_char_bc(C: CharVec7Bc, S: EigenSystemBc) -> PrimVec7Bc {
    let qf = S.cf * S.alpha_f * S.sgn_bn;
    let qs = S.cs * S.alpha_s * S.sgn_bn;
    let af = S.a  * S.alpha_f * S.sqrtd;
    let as_ = S.a * S.alpha_s * S.sqrtd;
    let rho = 1.0 / max(S.inv_rho, 1.0e-30);

    let af_sum = S.alpha_f * (C.fL + C.fR);
    let as_sum = S.alpha_s * (C.sL + C.sR);
    let af_dif = S.alpha_f * (C.fR - C.fL);
    let as_dif = S.alpha_s * (C.sR - C.sL);

    let qs_fdif = qs * (C.fL - C.fR);
    let qf_sdif = qf * (C.sR - C.sL);
    let aL_sum  = C.aL + C.aR;

    var W: PrimVec7Bc;
    W.rho = rho * (af_sum + as_sum) + C.e;
    W.vn  = S.cf * af_dif + S.cs * as_dif;
    W.vt1 = S.bet1 * (qs_fdif + qf_sdif)
          + S.bet2 * (C.aR - C.aL);
    W.vt2 = S.bet2 * (qs_fdif + qf_sdif)
          + S.bet1 * (C.aL - C.aR);
    W.p   = rho * S.asq * (af_sum + as_sum);
    W.bt1 = S.bet1 * (as_ * (C.fL + C.fR) - af * (C.sL + C.sR))
          - S.bet2 * S.sgn_bn * S.sqrtd * aL_sum;
    W.bt2 = S.bet2 * (as_ * (C.fL + C.fR) - af * (C.sL + C.sR))
          + S.bet1 * S.sgn_bn * S.sqrtd * aL_sum;
    return W;
}

// Zero the characteristic-amplitude components whose eigenvalue points
// INTO the domain (incoming waves). `vn` is the boundary-cell normal
// velocity in the sweep-aligned frame; `n_out` is +1 if the outward
// normal points in the +n direction (E or N edge), −1 otherwise (W or
// S edge). A wave with eigenvalue λ_k is incoming iff λ_k · n_out < 0.
fn zero_incoming_chars(C: CharVec7Bc, S: EigenSystemBc, vn: f32, n_out: f32) -> CharVec7Bc {
    var out = C;
    // Wave eigenvalues (sweep-aligned): {vn−cf, vn−ca, vn−cs, vn,
    // vn+cs, vn+ca, vn+cf} where c_a = |Bn|/√ρ is the Alfvén speed.
    // Derive c_a from the MHD identity c_f² · c_s² = a² · c_a²
    // (Stone+ 2008 A6) → c_a = √(cfsq·cssq / asq). This avoids needing
    // to thread bn through the helper signature.
    let casq = select(0.0, S.cfsq * S.cssq / max(S.asq, 1.0e-30), S.asq > 0.0);
    let ca2 = sqrt(max(casq, 0.0));
    let lam_fL = vn - S.cf;
    let lam_aL = vn - ca2;
    let lam_sL = vn - S.cs;
    let lam_e  = vn;
    let lam_sR = vn + S.cs;
    let lam_aR = vn + ca2;
    let lam_fR = vn + S.cf;
    if (lam_fL * n_out < 0.0) { out.fL = 0.0; }
    if (lam_aL * n_out < 0.0) { out.aL = 0.0; }
    if (lam_sL * n_out < 0.0) { out.sL = 0.0; }
    if (lam_e  * n_out < 0.0) { out.e  = 0.0; }
    if (lam_sR * n_out < 0.0) { out.sR = 0.0; }
    if (lam_aR * n_out < 0.0) { out.aR = 0.0; }
    if (lam_fR * n_out < 0.0) { out.fR = 0.0; }
    return out;
}

// Read one cell-centered MHD primitive state directly from the storage
// buffers, using the existing face-B averaging recipe. Mirrors the
// load done by reconstruct-ppm's tile cache but inlined here so
// apply-bcs doesn't need to bind the face-B buffers in shared memory.
fn read_prim_at(ix: u32, iy: u32, n_total: u32, gamma: f32, p_floor: f32) -> MhdPrim {
    let idx = cell_idx_total(ix, iy, n_total);
    let bx  = 0.5 * (Bx_face[bx_face_left_idx(ix, iy, n_total)]
                   + Bx_face[bx_face_right_idx(ix, iy, n_total)]);
    let by  = 0.5 * (By_face[by_face_down_idx(ix, iy, n_total)]
                   + By_face[by_face_up_idx(ix, iy, n_total)]);
    return cons_to_prim_mhd(U0[idx], U1[idx], bx, by, gamma, p_floor);
}

// PrimVec7 difference helpers (local).
fn prim_sub_bc(a: PrimVec7Bc, b: PrimVec7Bc) -> PrimVec7Bc {
    return PrimVec7Bc(a.rho - b.rho, a.vn - b.vn, a.vt1 - b.vt1, a.vt2 - b.vt2,
                      a.bt1 - b.bt1, a.bt2 - b.bt2, a.p - b.p);
}

fn prim_scale_add_bc(base: PrimVec7Bc, delta: PrimVec7Bc, s: f32) -> PrimVec7Bc {
    return PrimVec7Bc(
        base.rho + s * delta.rho,
        base.vn  + s * delta.vn,
        base.vt1 + s * delta.vt1,
        base.vt2 + s * delta.vt2,
        base.bt1 + s * delta.bt1,
        base.bt2 + s * delta.bt2,
        base.p   + s * delta.p,
    );
}

// Compute the NSCBC characteristic-zero-gradient ghost cell at signed
// offset `d` from the boundary cell `(ib, jb)`, with outward normal
// pointing along the sweep axis. `axis` selects x-sweep (E/W) or
// y-sweep (N/S); `n_out` ∈ {−1, +1} selects W/S vs E/N.
//
// Returns the ghost conservative pair to write.
fn nscbc_outflow_ghost(
    ib: u32, jb: u32,         // boundary cell (last interior cell on the wall side)
    ii: u32, ji: u32,         // one cell INTERIOR of the boundary (for the gradient)
    d:    f32,                // signed cell-offset from boundary cell to ghost cell
    axis: u32,                // 0 = x-sweep (E/W), 1 = y-sweep (N/S)
    n_out: f32,               // +1 if outward normal = +axis, −1 otherwise
    n_total: u32, gamma: f32, p_floor: f32,
) -> ConsPair {
    let pb = read_prim_at(ib, jb, n_total, gamma, p_floor);
    let pi = read_prim_at(ii, ji, n_total, gamma, p_floor);

    let perm_b = permute_prim_bc(pb, axis);
    let perm_i = permute_prim_bc(pi, axis);
    let wb = vec7_of_bc(perm_b);
    let wi = vec7_of_bc(perm_i);
    // perm_i is consumed only via vec7_of_bc; bn at the inward cell
    // isn't part of the wave structure we project, so we don't need it
    // separately. The eigensystem is built at the boundary cell.

    // One-sided primitive gradient (per cell). `n_out` carries the
    // outward direction sign so that `(wb − wi)` is positive for
    // outflow on the +axis side and reverses on the −axis side.
    // ∂w/∂n in cell-units (per cell) = (w at boundary cell) − (w one cell IN).
    // For +axis outward (E/N): "one in" = at ib−1 / jb−1, gradient = wb − wi.
    // For −axis outward (W/S): "one in" = at ib+1 / jb+1, gradient = wi − wb
    //   when expressed along the outward normal. We absorb the sign into n_out
    //   by computing the axis-aligned (positive-axis) difference and treating
    //   n_out as the projection onto the outward direction.
    // We always compute gradient as (boundary − inward) = wb − wi which points
    // in the +n_out direction by construction below: the caller passes (ii, ji)
    // such that (ib − ii) · n_out = +1 along the boundary axis.
    let dw_axis = prim_sub_bc(wb, wi);  // ∂w/∂(+n_out · axis) per cell

    let eig = mhd_eigensystem_bc(wb, perm_b.bn, gamma);
    let a   = project_to_char_bc(dw_axis, eig);
    let a_nr = zero_incoming_chars(a, eig, perm_b.vn, n_out);
    let dw_mod = project_from_char_bc(a_nr, eig);

    // Linear extrapolation in primitive space. `d` is the signed cell
    // offset from boundary cell to ghost (along the +n_out direction);
    // since dw_mod is the per-cell gradient in the +n_out direction,
    // the ghost primitive is wb + d · dw_mod.
    let w_ghost_vec = prim_scale_add_bc(wb, dw_mod, d);

    // Reconstruct an MhdPrim, restore B_n (parameter — NSCBC doesn't
    // touch it; the face-B outflow path keeps the wall-normal B
    // constant). Apply floors so the cons-conversion stays sane.
    var w_ghost: PermutedBc;
    w_ghost.rho = max(w_ghost_vec.rho, DENSITY_FLOOR);
    w_ghost.vn  = w_ghost_vec.vn;
    w_ghost.vt1 = w_ghost_vec.vt1;
    w_ghost.vt2 = w_ghost_vec.vt2;
    w_ghost.bt1 = w_ghost_vec.bt1;
    w_ghost.bt2 = w_ghost_vec.bt2;
    w_ghost.p   = max(w_ghost_vec.p,   p_floor);
    w_ghost.bn  = perm_b.bn;

    let prim_ghost = unpermute_prim_bc(w_ghost, axis);
    return prim_to_cons_pair(prim_ghost, gamma, p_floor);
}

// Fill ONE cell-centered ghost (i, j) based on the appropriate edge mode.
fn fill_cell_ghost(ix: u32, iy: u32, ghost: u32, n_interior: u32, n_total: u32) {
    let h_mode = horiz_mode_for_col(ix, ghost, n_interior);
    let v_mode = vert_mode_for_row(iy, ghost, n_interior);
    let h_edge = select(EDGE_E_BC, EDGE_W_BC, ix < ghost);
    let v_edge = select(EDGE_N_BC, EDGE_S_BC, iy < ghost);
    let in_h_ghost = (ix < ghost) || (ix >= ghost + n_interior);
    let in_v_ghost = (iy < ghost) || (iy >= ghost + n_interior);
    if (!in_h_ghost && !in_v_ghost) { return; }   // interior — never touch.

    var mode: u32;
    var owner_edge: u32;
    if (in_h_ghost && in_v_ghost) {
        // Corner. Prefer non-periodic among horizontal vs vertical.
        mode = pick_corner_mode(h_mode, v_mode);
        owner_edge = select(v_edge, h_edge, h_mode != BC_PERIODIC);
    } else if (in_h_ghost) {
        mode = h_mode;
        owner_edge = h_edge;
    } else {
        mode = v_mode;
        owner_edge = v_edge;
    }

    let dst = cell_idx_total(ix, iy, n_total);

    if (mode == BC_PERIODIC) {
        // Copy from the wrapped interior cell on the opposite side.
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) { src_i = ix + n_interior; }
        else if (ix >= ghost + n_interior) { src_i = ix - n_interior; }
        if (iy < ghost) { src_j = iy + n_interior; }
        else if (iy >= ghost + n_interior) { src_j = iy - n_interior; }
        let src = cell_idx_total(src_i, src_j, n_total);
        U0[dst] = U0[src];
        U1[dst] = U1[src];
        return;
    }

    if (mode == BC_OUTFLOW) {
        // NSCBC characteristic-zero-gradient (Session 10). Identify
        // the boundary cell + one cell INWARD along the dominant
        // outward normal for this ghost, compute the primitive
        // gradient with incoming-wave amplitudes zeroed, and linearly
        // extrapolate to the ghost cell. See header.
        //
        // Strategy: pick a single dominant outward normal axis per
        // ghost based on whether it lies in the horizontal-only,
        // vertical-only, or corner band. Corners use whichever
        // adjacent edge owns the cell (the same priority pick_corner
        // applies); the ghost is then extrapolated along that axis
        // from the matching boundary cell. The "other" axis index is
        // either clamped to the boundary cell (when the corner mode
        // is owned by the horizontal edge) or wrapped if the other
        // edge is periodic.
        //
        // Determine the BC-owning axis and the corresponding boundary
        // cell (ib, jb), inward cell (ii, ji), ghost offset d, and
        // outward-normal sign n_out.
        var axis: u32   = 0u;
        var n_out: f32  = 1.0;
        var ib: u32     = ix;
        var jb: u32     = iy;
        var ii: u32     = ix;
        var ji: u32     = iy;
        var d:  f32     = 0.0;

        // h_axis chosen iff this ghost sits in a horizontal-edge band
        // owned by h_mode. Else v_axis.
        var use_horiz = false;
        if (in_h_ghost && in_v_ghost) {
            // Corner. Same priority as the mode pick: non-periodic wins;
            // if both equal, h owns.
            if (h_mode != BC_PERIODIC) { use_horiz = true; }
            else if (v_mode != BC_PERIODIC) { use_horiz = false; }
            else { use_horiz = true; }
        } else if (in_h_ghost) {
            use_horiz = true;
        } else {
            use_horiz = false;
        }

        if (use_horiz) {
            axis = 0u;
            // West vs East. West: ix < ghost → outward normal = −x.
            // East: ix ≥ ghost+N → outward normal = +x.
            if (ix < ghost) {
                ib = ghost;
                ii = ghost + 1u;
                n_out = -1.0;
                // d is the signed offset (in +n_out cell-units) from
                // ib to the ghost cell ix. n_out = −1, so the ghost at
                // ix lies at (ib − ix) cells in the +n_out direction.
                d = f32(i32(ib) - i32(ix));
            } else {
                ib = ghost + n_interior - 1u;
                ii = ghost + n_interior - 2u;
                n_out = 1.0;
                d = f32(i32(ix) - i32(ib));
            }
            // The orthogonal-axis (y) coordinate: if this row is in the
            // interior y-range, just sample iy. If it's in a y-ghost
            // band, clamp to the boundary y-cell so the gradient is
            // taken from the right column. (Periodic in y is handled by
            // outflow-on-h not touching that axis — corner OK.)
            jb = iy;
            ji = iy;
            if (iy < ghost) {
                if (v_mode == BC_PERIODIC) {
                    jb = iy + n_interior;
                    ji = iy + n_interior;
                } else {
                    jb = ghost;
                    ji = ghost;
                }
            } else if (iy >= ghost + n_interior) {
                if (v_mode == BC_PERIODIC) {
                    jb = iy - n_interior;
                    ji = iy - n_interior;
                } else {
                    jb = ghost + n_interior - 1u;
                    ji = ghost + n_interior - 1u;
                }
            }
        } else {
            axis = 1u;
            if (iy < ghost) {
                jb = ghost;
                ji = ghost + 1u;
                n_out = -1.0;
                d = f32(i32(jb) - i32(iy));
            } else {
                jb = ghost + n_interior - 1u;
                ji = ghost + n_interior - 2u;
                n_out = 1.0;
                d = f32(i32(iy) - i32(jb));
            }
            ib = ix;
            ii = ix;
            if (ix < ghost) {
                if (h_mode == BC_PERIODIC) {
                    ib = ix + n_interior;
                    ii = ix + n_interior;
                } else {
                    ib = ghost;
                    ii = ghost;
                }
            } else if (ix >= ghost + n_interior) {
                if (h_mode == BC_PERIODIC) {
                    ib = ix - n_interior;
                    ii = ix - n_interior;
                } else {
                    ib = ghost + n_interior - 1u;
                    ii = ghost + n_interior - 1u;
                }
            }
        }

        let cp = nscbc_outflow_ghost(
            ib, jb, ii, ji, d, axis, n_out,
            n_total, U_uniforms.gamma, U_uniforms.pressure_floor,
        );
        U0[dst] = cp.U0;
        U1[dst] = cp.U1;
        return;
    }

    if (mode == BC_REFLECTING) {
        // Mirror across the boundary. The wall sits between the
        // outermost ghost cell (i = ghost-1) and the first interior
        // cell (i = ghost). Index formulas:
        //     src_i = 2*ghost - 1 - i      for W ghost
        //     src_i = 2*(ghost+n) - 1 - i  for E ghost
        // Same shape vertically.
        //
        // Corner rule: the corner mode is owned by the first NON-periodic
        // adjacent edge (see pick_corner_mode). If the OTHER axis is
        // periodic, we still need to map its index into the interior
        // before sampling, otherwise the mirror source lands on stale
        // ghost data. We compose: reflect on the owning axis, periodic-
        // wrap on the other axis if it's periodic.
        var src_i = ix;
        var src_j = iy;
        var flip_x = false;
        var flip_y = false;
        let h_is_reflect = in_h_ghost && (h_mode == BC_REFLECTING);
        let v_is_reflect = in_v_ghost && (v_mode == BC_REFLECTING);
        if (h_is_reflect) {
            if (ix < ghost) { src_i = 2u * ghost - 1u - ix; }
            else            { src_i = 2u * (ghost + n_interior) - 1u - ix; }
            flip_x = true;
        } else if (in_h_ghost && h_mode == BC_PERIODIC) {
            if (ix < ghost) { src_i = ix + n_interior; }
            else            { src_i = ix - n_interior; }
        }
        if (v_is_reflect) {
            if (iy < ghost) { src_j = 2u * ghost - 1u - iy; }
            else            { src_j = 2u * (ghost + n_interior) - 1u - iy; }
            flip_y = true;
        } else if (in_v_ghost && v_mode == BC_PERIODIC) {
            if (iy < ghost) { src_j = iy + n_interior; }
            else            { src_j = iy - n_interior; }
        }
        let src = cell_idx_total(src_i, src_j, n_total);
        var u0 = U0[src];
        var u1 = U1[src];
        if (flip_x) { u0.y = -u0.y; }    // flip ρ·v_x
        if (flip_y) { u0.z = -u0.z; }    // flip ρ·v_y
        U0[dst] = u0;
        U1[dst] = u1;
        return;
    }

    // BC_DRIVEN
    let cons = driven_cons(owner_edge);
    U0[dst] = cons[0];
    U1[dst] = cons[1];
}

// Fill one Bx_face entry (index space (N_total+1) × N_total). The
// shape of the strips is different from cell-centered: x-faces have an
// extra column at i = N_total.
//   x-face index i corresponds to position i - ghost - 0.5 within the
//   interior. Interior x-faces (between interior cells, plus the two
//   boundary faces) are i ∈ [ghost, ghost+n_interior]. Ghost x-faces:
//     i ∈ [0, ghost)            → west ghost
//     i ∈ (ghost+n_interior, N_total]   → east ghost
fn fill_bx_face(ix: u32, iy: u32, ghost: u32, n_interior: u32, n_total: u32) {
    // Determine if this is a ghost x-face. The boundary faces (i = ghost
    // and i = ghost + n_interior) are TOUCHED ONLY for reflecting and
    // driven BCs — for periodic and outflow they're part of the interior
    // physics dispatch's writes (or just left alone).
    let in_h_ghost = (ix < ghost) || (ix > ghost + n_interior);
    let on_w_wall  = (ix == ghost);
    let on_e_wall  = (ix == ghost + n_interior);
    let in_v_ghost = (iy < ghost) || (iy >= ghost + n_interior);
    if (!in_h_ghost && !on_w_wall && !on_e_wall && !in_v_ghost) { return; }

    let h_mode = horiz_mode_for_col(ix, ghost, n_interior);
    let v_mode = vert_mode_for_row(iy, ghost, n_interior);
    let h_edge = select(EDGE_E_BC, EDGE_W_BC, ix <= ghost);
    let v_edge = select(EDGE_N_BC, EDGE_S_BC, iy < ghost);

    // Choose mode. If on a boundary face (W or E wall), the horizontal
    // mode owns it unconditionally. Otherwise, corner logic.
    var mode: u32;
    var owner_edge: u32;
    if (on_w_wall) { mode = bc.mode_w; owner_edge = EDGE_W_BC; }
    else if (on_e_wall) { mode = bc.mode_e; owner_edge = EDGE_E_BC; }
    else if (in_h_ghost && in_v_ghost) {
        mode = pick_corner_mode(h_mode, v_mode);
        owner_edge = select(v_edge, h_edge, h_mode != BC_PERIODIC);
    }
    else if (in_h_ghost) { mode = h_mode; owner_edge = h_edge; }
    else                 { mode = v_mode; owner_edge = v_edge; }

    let dst = bx_face_idx(ix, iy, n_total);

    if (mode == BC_PERIODIC) {
        // The mode picker forces PERIODIC here when (a) both axes are
        // periodic (non-corner or both-periodic corner — wrap both),
        // OR (b) on_w/e_wall is true at the X boundary face. In case
        // (b) the X axis is periodic by construction (mode_w/e
        // picked it), but the Y axis BC is independent — at a corner
        // (iy in v-ghost) we must compose with v_mode. Pre-Session-11
        // this branch wrapped Y unconditionally, which corrupted the
        // boundary-face ghost stripe at the four corner cells where
        // periodic-x met outflow-y in Harris (see HANDOFF Session 10
        // "remaining fourth issue").
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) { src_i = ix + n_interior; }
        else if (ix > ghost + n_interior) { src_i = ix - n_interior; }
        let on_wall = on_w_wall || on_e_wall;
        if (iy < ghost) {
            if (on_wall && v_mode != BC_PERIODIC) {
                // Compose Y outflow (zero-gradient) with X periodic.
                src_j = ghost;
            } else {
                src_j = iy + n_interior;
            }
        } else if (iy >= ghost + n_interior) {
            if (on_wall && v_mode != BC_PERIODIC) {
                src_j = ghost + n_interior - 1u;
            } else {
                src_j = iy - n_interior;
            }
        }
        // Boundary faces canonicalize to W boundary under periodic
        // (W and E boundary x-faces are the SAME physical face).
        if (on_wall) { src_i = ghost; }
        Bx_face[dst] = Bx_face[bx_face_idx(src_i, src_j, n_total)];
        return;
    }

    if (mode == BC_OUTFLOW) {
        // Compose per-axis BC at corner ghost cells. The picked OUTFLOW
        // mode applies to the chosen-owning axis; the orthogonal axis
        // may be PERIODIC (in which case we wrap, not clamp). Without
        // this composition, periodic-x on a y-outflow boundary at corner
        // ghost cells reads from the WRONG column, breaking ∇·B
        // preservation in the resistive update (HANDOFF Session 10).
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) {
            if (h_mode == BC_PERIODIC) { src_i = ix + n_interior; }
            else { src_i = ghost; }
        } else if (ix > ghost + n_interior) {
            if (h_mode == BC_PERIODIC) { src_i = ix - n_interior; }
            else { src_i = ghost + n_interior; }
        }
        // boundary face stays put (it IS the interior boundary).
        if (iy < ghost) {
            if (v_mode == BC_PERIODIC) { src_j = iy + n_interior; }
            else { src_j = ghost; }
        } else if (iy >= ghost + n_interior) {
            if (v_mode == BC_PERIODIC) { src_j = iy - n_interior; }
            else { src_j = ghost + n_interior - 1u; }
        }
        Bx_face[dst] = Bx_face[bx_face_idx(src_i, src_j, n_total)];
        return;
    }

    if (mode == BC_REFLECTING) {
        // Perfectly conducting wall: B normal to the wall is zero.
        // On a W/E wall, normal is x → Bx = 0 on that boundary face.
        // For x-ghost faces (away from the boundary), mirror across the
        // wall: face at distance d outside ↔ face at distance d inside,
        // with Bx negated.
        if (on_w_wall || on_e_wall) {
            Bx_face[dst] = 0.0;
            return;
        }
        var src_i = ix;
        var src_j = iy;
        var flip = false;
        if (ix < ghost) {
            // Mirror about the W boundary face at i = ghost.
            // i ∈ {0, 1, …, ghost-1} → src ∈ {2*ghost - i, …}
            src_i = 2u * ghost - ix;
            flip = true;
        } else if (ix > ghost + n_interior) {
            // Mirror about the E boundary face at i = ghost + n_interior.
            src_i = 2u * (ghost + n_interior) - ix;
            flip = true;
        }
        if (iy < ghost) {
            // S-wall reflection for v-ghost rows doesn't flip Bx (normal
            // is y). Just mirror the index.
            src_j = 2u * ghost - 1u - iy;
        } else if (iy >= ghost + n_interior) {
            src_j = 2u * (ghost + n_interior) - 1u - iy;
        }
        var v = Bx_face[bx_face_idx(src_i, src_j, n_total)];
        if (flip) { v = -v; }
        Bx_face[dst] = v;
        return;
    }

    // BC_DRIVEN — set Bx to this edge's driven inflow Bx on the owned strip.
    Bx_face[dst] = driven_bx_for_edge(owner_edge);
}

fn fill_by_face(ix: u32, iy: u32, ghost: u32, n_interior: u32, n_total: u32) {
    let in_v_ghost = (iy < ghost) || (iy > ghost + n_interior);
    let on_s_wall  = (iy == ghost);
    let on_n_wall  = (iy == ghost + n_interior);
    let in_h_ghost = (ix < ghost) || (ix >= ghost + n_interior);
    if (!in_v_ghost && !on_s_wall && !on_n_wall && !in_h_ghost) { return; }

    let h_mode = horiz_mode_for_col(ix, ghost, n_interior);
    let v_mode = vert_mode_for_row(iy, ghost, n_interior);
    let h_edge = select(EDGE_E_BC, EDGE_W_BC, ix < ghost);
    let v_edge = select(EDGE_N_BC, EDGE_S_BC, iy <= ghost);

    var mode: u32;
    var owner_edge: u32;
    if (on_s_wall) { mode = bc.mode_s; owner_edge = EDGE_S_BC; }
    else if (on_n_wall) { mode = bc.mode_n; owner_edge = EDGE_N_BC; }
    else if (in_h_ghost && in_v_ghost) {
        mode = pick_corner_mode(h_mode, v_mode);
        owner_edge = select(v_edge, h_edge, h_mode != BC_PERIODIC);
    }
    else if (in_v_ghost) { mode = v_mode; owner_edge = v_edge; }
    else                 { mode = h_mode; owner_edge = h_edge; }

    let dst = by_face_idx(ix, iy, n_total);

    if (mode == BC_PERIODIC) {
        // Symmetric to fill_bx_face's PERIODIC branch — compose with
        // h_mode at S/N boundary face corners when X is non-periodic.
        var src_i = ix;
        var src_j = iy;
        if (iy < ghost) { src_j = iy + n_interior; }
        else if (iy > ghost + n_interior) { src_j = iy - n_interior; }
        let on_wall = on_s_wall || on_n_wall;
        if (ix < ghost) {
            if (on_wall && h_mode != BC_PERIODIC) {
                src_i = ghost;
            } else {
                src_i = ix + n_interior;
            }
        } else if (ix >= ghost + n_interior) {
            if (on_wall && h_mode != BC_PERIODIC) {
                src_i = ghost + n_interior - 1u;
            } else {
                src_i = ix - n_interior;
            }
        }
        // Canonicalize boundary y-faces: S wall is authoritative.
        if (on_wall) { src_j = ghost; }
        By_face[dst] = By_face[by_face_idx(src_i, src_j, n_total)];
        return;
    }

    if (mode == BC_OUTFLOW) {
        // Compose per-axis BC at corner cells — for Harris this is the
        // primary fix: at By boundary faces in the E-ghost column (or
        // W-ghost), the on_s/n_wall priority picks OUTFLOW from
        // mode_s/n, but the X axis is PERIODIC and the src column
        // index must be wrapped, not clamped to the rightmost/leftmost
        // interior column. See HANDOFF Session 10 "remaining fourth
        // issue" + Session 11.
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) {
            if (h_mode == BC_PERIODIC) { src_i = ix + n_interior; }
            else { src_i = ghost; }
        } else if (ix >= ghost + n_interior) {
            if (h_mode == BC_PERIODIC) { src_i = ix - n_interior; }
            else { src_i = ghost + n_interior - 1u; }
        }
        if (iy < ghost) {
            if (v_mode == BC_PERIODIC) { src_j = iy + n_interior; }
            else { src_j = ghost; }
        } else if (iy > ghost + n_interior) {
            if (v_mode == BC_PERIODIC) { src_j = iy - n_interior; }
            else { src_j = ghost + n_interior; }
        }
        By_face[dst] = By_face[by_face_idx(src_i, src_j, n_total)];
        return;
    }

    if (mode == BC_REFLECTING) {
        if (on_s_wall || on_n_wall) {
            By_face[dst] = 0.0;
            return;
        }
        var src_i = ix;
        var src_j = iy;
        var flip = false;
        if (iy < ghost) {
            src_j = 2u * ghost - iy;
            flip = true;
        } else if (iy > ghost + n_interior) {
            src_j = 2u * (ghost + n_interior) - iy;
            flip = true;
        }
        if (ix < ghost) {
            src_i = 2u * ghost - 1u - ix;
        } else if (ix >= ghost + n_interior) {
            src_i = 2u * (ghost + n_interior) - 1u - ix;
        }
        var v = By_face[by_face_idx(src_i, src_j, n_total)];
        if (flip) { v = -v; }
        By_face[dst] = v;
        return;
    }

    // BC_DRIVEN
    By_face[dst] = driven_by_for_edge(owner_edge);
}

// Single-pass kernel. We dispatch over the FULL (N_total+1, N_total+1)
// box and let each invocation handle whichever buffers its index is valid
// for: cell-centered for (ix < N_total, iy < N_total), Bx_face for
// (ix < N_total+1, iy < N_total), By_face for (ix < N_total, iy < N_total+1).
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_total    = U_uniforms.grid_n_total;
    let n_interior = U_uniforms.grid_n;
    let ghost      = U_uniforms.ghost_w;
    let ix         = gid.x;
    let iy         = gid.y;

    // Cell-centered (U0, U1)
    if (ix < n_total && iy < n_total) {
        fill_cell_ghost(ix, iy, ghost, n_interior, n_total);
    }

    // Bx_face: (n_total+1) × n_total
    if (ix < n_total + 1u && iy < n_total) {
        fill_bx_face(ix, iy, ghost, n_interior, n_total);
    }

    // By_face: n_total × (n_total+1)
    if (ix < n_total && iy < n_total + 1u) {
        fill_by_face(ix, iy, ghost, n_interior, n_total);
    }
}

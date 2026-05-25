// ─── reconstruct-ppm.wgsl ────────────────────────────────────────────
// Colella & Woodward 1984 PPM reconstruction on MHD primitive variables
// with characteristic-variable limiting (Stone+ 2008 §3.4.2 — the
// Athena/Athena++ default for MHD). Produces per-cell, per-direction
// *left* and *right* face primitive states (the two endpoints of the
// limited parabola through cell j). At face (j+½), the Riemann solver
// pairs:
//     QL = edge_R[j]    (right edge of cell j)
//     QR = edge_L[j+1]  (left edge of cell j+1)
//
// ── Characteristic limiting (Session 6 — Round 3 of P0/P1 polish) ───
// Previously the limiter ran per primitive variable independently, which
// can overshoot at strong MHD shocks because the limiter doesn't respect
// the wave structure of the hyperbolic system. The current path projects
// primitive differences (cell → face) onto the 7 left-eigenvectors of
// the linearized MHD primitive flux Jacobian at the cell, applies the
// standard PPM monotonicity limiter component-wise in characteristic
// space (each wave family limited independently), then projects back to
// primitive via the right-eigenvector matrix. Mathematically correct
// for the hyperbolic system; standard in production MHD codes.
//
// The 7-wave MHD primitive eigensystem (sweep-aligned permutation
// (ρ, v_n, v_t1, v_t2, B_t1, B_t2, p); B_n is a parameter, not a wave):
//   0: u−c_f   (left-going fast magnetosonic)
//   1: u−c_a   (left-going Alfvén)
//   2: u−c_s   (left-going slow magnetosonic)
//   3: u       (entropy / contact)
//   4: u+c_s   (right-going slow magnetosonic)
//   5: u+c_a   (right-going Alfvén)
//   6: u+c_f   (right-going fast magnetosonic)
//
// References:
//   Stone, Gardiner, Teuben, Hawley, Simon (2008), ApJS 178, 137 —
//     Athena++ canonical reference. Appendix A.1 gives the primitive
//     eigenvectors (eqs A12, A14–A18); §3.4.2 gives the characteristic
//     limiting recipe.
//   Roe & Balsara (1996), SIAM J. Appl. Math. 56, 57 — normalization
//     conventions for the α_f / α_s factors (Roe96 cases III/IV/V
//     degeneracy regularizations).
//   Brio & Wu (1988) — the perpendicular-field renormalization
//     β_t1 = 1, β_t2 = 0 when B_⊥ = 0 (BW88 eq 45).
//   Colella & Woodward (1984), JCP 54, 174 — original PPM with the
//     monotonicity check this file applies in characteristic space.
//
// ── PPM workgroup-shared primitive cache (landed Round 2) ───────────
// The 5-point sweep stencil reads primitives from a 12×12 workgroup-
// shared `MhdPrim` tile (8×8 interior + 2-cell halo per side) instead
// of recomputing `cons_to_prim_mhd` five times per output cell.
// Characteristic limiting builds on top of this — it reads cached
// primitives, computes the eigensystem per output cell, and projects
// without touching the cache shape. (We deliberately did NOT switch
// the tile to caching the projected characteristic state: the
// eigenmatrices are local to the center cell, so the 4 neighbors'
// projections must use that same center's L matrix, which means
// every cell still has to do its own projection and we'd save
// nothing — the simpler `MhdPrim` tile is the right call.)
//
// ── Boundary degradation ────────────────────────────────────────────
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
// Algorithm per output cell, per direction:
//   1. Phase A: every thread runs cons_to_prim_mhd ONCE into the
//      workgroup tile (center + halo for outer-ring threads). Single
//      top-level workgroupBarrier().
//   2. Phase B: stencil_ok check → fall back to piecewise constant if
//      the 5-point sweep stencil hangs off storage.
//   3. Read W_{-2..+2} from the tile.
//   4. 4th-order edge interpolants in primitive space:
//        q_{j-½} = (7/12)(q_{j-1}+q_j) − (1/12)(q_{j-2}+q_{j+1})
//        q_{j+½} = (7/12)(q_j+q_{j+1}) − (1/12)(q_{j-1}+q_{j+2})
//   5. Form primitive differences dL = q_c − q_{j-½}, dR = q_{j+½} − q_c.
//   6. Compute MHD primitive eigensystem at the cell center W_c.
//   7. Project to characteristic space: a_L = L·dL, a_R = L·dR.
//   8. Apply PPM monotonicity limiter per characteristic component
//      (same algebra as before, just operates on a single wave family
//      at a time): if a_L·a_R ≤ 0 → a_L = a_R = 0; else apply the
//      CW 1984 overshoot check (2·dL² − dL·dR − 2·dR² > 0 etc.).
//   9. Project back: dL' = R·a_L, dR' = R·a_R.
//  10. Recover face states: q_L = q_c − dL', q_R = q_c + dR'.
//  11. Floor density and pressure to keep downstream cons-recovery sane.
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

// 12×12 workgroup-shared primitive cache (8 interior + 2 halo per side).
var<workgroup> tile : array<array<MhdPrim, 12>, 12>;

// ── Sweep-aligned 7-vector ──────────────────────────────────────────
// Components: (ρ, v_n, v_t1, v_t2, B_t1, B_t2, p) where (n, t1, t2) is
// the sweep-axis-rotated frame:
//   x-sweep: n=x, t1=y, t2=z → (ρ, vx, vy, vz, By, Bz, p)
//   y-sweep: n=y, t1=z, t2=x → (ρ, vy, vz, vx, Bz, Bx, p)
// The cyclic (n,t1,t2) ordering matches Athena++'s ivy/ivz permutation
// in characteristic.cpp; it's only a labelling convention — the
// eigensystem is invariant under any orthonormal basis for the
// transverse plane (the β_t1 / β_t2 components absorb that choice).
// B_n is carried separately because it's a parameter of the
// eigensystem, not a wave.
struct PrimVec7 {
    rho:  f32,
    vn:   f32,
    vt1:  f32,
    vt2:  f32,
    bt1:  f32,
    bt2:  f32,
    p:    f32,
};

// Characteristic 7-vector (one scalar per wave family).
struct CharVec7 {
    fL:    f32,  // u − c_f
    aL:    f32,  // u − c_a
    sL:    f32,  // u − c_s
    e:     f32,  // u (entropy)
    sR:    f32,  // u + c_s
    aR:    f32,  // u + c_a
    fR:    f32,  // u + c_f
};

fn prim_zero() -> PrimVec7 {
    return PrimVec7(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

fn prim_add(a: PrimVec7, b: PrimVec7) -> PrimVec7 {
    return PrimVec7(a.rho + b.rho, a.vn + b.vn, a.vt1 + b.vt1, a.vt2 + b.vt2,
                    a.bt1 + b.bt1, a.bt2 + b.bt2, a.p + b.p);
}

fn prim_sub(a: PrimVec7, b: PrimVec7) -> PrimVec7 {
    return PrimVec7(a.rho - b.rho, a.vn - b.vn, a.vt1 - b.vt1, a.vt2 - b.vt2,
                    a.bt1 - b.bt1, a.bt2 - b.bt2, a.p - b.p);
}

fn prim_scaled_diff(a: PrimVec7, b: PrimVec7, c7: f32, m: PrimVec7, n: PrimVec7, c1: f32) -> PrimVec7 {
    // (7/12)(a + b) − (1/12)(m + n) — the 4th-order edge interpolant.
    return PrimVec7(
        c7 * (a.rho + b.rho) - c1 * (m.rho + n.rho),
        c7 * (a.vn  + b.vn ) - c1 * (m.vn  + n.vn ),
        c7 * (a.vt1 + b.vt1) - c1 * (m.vt1 + n.vt1),
        c7 * (a.vt2 + b.vt2) - c1 * (m.vt2 + n.vt2),
        c7 * (a.bt1 + b.bt1) - c1 * (m.bt1 + n.bt1),
        c7 * (a.bt2 + b.bt2) - c1 * (m.bt2 + n.bt2),
        c7 * (a.p   + b.p  ) - c1 * (m.p   + n.p  ),
    );
}

// Compute one cell's primitive state from U0/U1 and adjacent face B,
// the same recipe `cell_primitive_pair_ppm` previously inlined. Pulled
// out as a function so the cache load is a single call site.
fn cell_primitive_cache(ix: u32, iy: u32, n_total: u32, gamma: f32, p_floor: f32) -> MhdPrim {
    let idx = cell_idx_total(ix, iy, n_total);
    let bx  = 0.5 * (Bx_face[bx_face_left_idx(ix, iy, n_total)]
                   + Bx_face[bx_face_right_idx(ix, iy, n_total)]);
    let by  = 0.5 * (By_face[by_face_down_idx(ix, iy, n_total)]
                   + By_face[by_face_up_idx(ix, iy, n_total)]);
    return cons_to_prim_mhd(U0_in[idx], U1_in[idx], bx, by, gamma, p_floor);
}

// Permute an MhdPrim into the sweep-aligned PrimVec7. The sweep-axis
// B component (B_n) is returned as the 8th field — it's a parameter of
// the eigensystem, not part of the wave structure. We carry it inside
// PermutedPrim8 to keep the return type a flat scalar struct (the
// transpiler's SROA pass doesn't currently scalarize struct-of-struct
// returns, so PermutedPrim cannot contain a nested PrimVec7 field).
struct PermutedPrim8 {
    rho:  f32,
    vn:   f32,
    vt1:  f32,
    vt2:  f32,
    bt1:  f32,
    bt2:  f32,
    p:    f32,
    bn:   f32,
};

fn permute_prim(P: MhdPrim, axis: u32) -> PermutedPrim8 {
    var R: PermutedPrim8;
    R.rho = P.rho;
    R.p   = P.p;
    if (axis == 0u) {
        // x-sweep: n=x, t1=y, t2=z
        R.vn  = P.vx;
        R.vt1 = P.vy;
        R.vt2 = P.vz;
        R.bt1 = P.by;
        R.bt2 = P.bz;
        R.bn  = P.bx;
    } else {
        // y-sweep: n=y, t1=z, t2=x (cyclic permutation, matches
        // Athena++ ivy/ivz). The Riemann solver downstream pairs
        // (Bt1, Bt2) with whatever its own sweep convention expects;
        // we unpermute on output (pack_prim_pair) to match the
        // existing PrimPair layout, so consumers see no change.
        R.vn  = P.vy;
        R.vt1 = P.vz;
        R.vt2 = P.vx;
        R.bt1 = P.bz;
        R.bt2 = P.bx;
        R.bn  = P.by;
    }
    return R;
}

fn vec7_of(P: PermutedPrim8) -> PrimVec7 {
    return PrimVec7(P.rho, P.vn, P.vt1, P.vt2, P.bt1, P.bt2, P.p);
}

// Convert a sweep-aligned PrimVec7 back into the (p0, p1) PrimPair the
// downstream PPM math + Riemann solver consume. Must match the
// pre-Round-3 packing exactly.
//   p0 = (ρ, vx, vy, vz)
//   p1 = (p, Bt1_along_sweep_convention, Bt2, 0)
// Existing pack convention for x-sweep: p1.y = By, p1.z = Bz.
// Existing pack convention for y-sweep: p1.y = Bx, p1.z = Bz.
// For y-sweep we need to reverse the (t1, t2) cycle: our PrimVec7
// has (bt1=Bz, bt2=Bx), so the PrimPair wants p1.y = bt2 = Bx,
// p1.z = bt1 = Bz.
fn pack_prim_pair_from_vec7(w: PrimVec7, bn: f32, axis: u32) -> PrimPair {
    var R: PrimPair;
    if (axis == 0u) {
        R.p0 = vec4<f32>(w.rho, w.vn,  w.vt1, w.vt2);
        R.p1 = vec4<f32>(w.p,   w.bt1, w.bt2, 0.0);
    } else {
        // Unpermute: vn=vy, vt1=vz, vt2=vx → (rho, vx, vy, vz) =
        // (rho, vt2, vn, vt1). Same for (bt1=Bz, bt2=Bx) → p1.y = Bx
        // = bt2, p1.z = Bz = bt1.
        R.p0 = vec4<f32>(w.rho, w.vt2, w.vn, w.vt1);
        R.p1 = vec4<f32>(w.p,   w.bt2, w.bt1, 0.0);
    }
    return R;
}

// ── MHD primitive eigensystem (Stone+ 2008 Appendix A.1, adiabatic) ─
// Mirrors Athena++ `LeftEigenmatrixDotVector` /
// `RightEigenmatrixDotVector` in src/reconstruct/characteristic.cpp.
// Computed at the cell center; reused for both dL and dR projection.
struct EigenSystem {
    // Wave-speed-derived intermediates (Stone A10–A18).
    asq:       f32,   // sound speed² = γ p / ρ
    a:         f32,   // sound speed
    cfsq:      f32,   // fast magnetosonic speed² (cell-frame)
    cf:        f32,
    cssq:      f32,   // slow magnetosonic speed²
    cs:        f32,
    alpha_f:   f32,   // (A16) normalization
    alpha_s:   f32,   // (A16) normalization
    bet1:      f32,   // (A17) B-perp direction
    bet2:      f32,
    sgn_bn:    f32,   // sign(B_n)
    sqrtd:     f32,   // √ρ
    isqrtd:    f32,   // 1/√ρ
    inv_rho:   f32,   // 1/ρ
};

fn mhd_eigensystem(w: PrimVec7, bn: f32, gamma: f32) -> EigenSystem {
    var S: EigenSystem;
    let rho = max(w.rho, DENSITY_FLOOR);
    let p   = max(w.p,   1.0e-30);   // floor only to keep speeds finite
    S.inv_rho = 1.0 / rho;
    S.sqrtd   = sqrt(rho);
    S.isqrtd  = 1.0 / S.sqrtd;

    let btsq = w.bt1 * w.bt1 + w.bt2 * w.bt2;
    let bxsq = bn * bn;
    let gamp = gamma * p;                       // = γ p (Stone uses gamp directly)

    // Stone A10 — fast/slow speeds in form that avoids cancellation.
    // cf² + cs² = (a² + b²), cf²·cs² = a²·B_n²/ρ. Compute cf² first by
    // the additive form, then cs² = γp·bxsq / (ρ·cfsq) using the
    // identity. Avoids subtraction near c_a ≈ a.
    let tdif    = bxsq + btsq - gamp;
    let cf2_cs2 = sqrt(tdif * tdif + 4.0 * gamp * btsq);
    var cfsq_unscaled = 0.5 * (bxsq + btsq + gamp + cf2_cs2);
    // Guard cfsq against zero (only happens for ρ→0, p→0, B→0 — the
    // floors above keep gamp > 0 so cfsq ≥ ½ γ p > 0, but defensive).
    cfsq_unscaled = max(cfsq_unscaled, 1.0e-30);
    var cssq_unscaled = gamp * bxsq / cfsq_unscaled;
    // cssq_unscaled ≥ 0 by construction; floor at 0 against round-off.
    cssq_unscaled = max(cssq_unscaled, 0.0);

    S.cfsq = cfsq_unscaled * S.inv_rho;
    S.cssq = cssq_unscaled * S.inv_rho;
    S.cf   = sqrt(S.cfsq);
    S.cs   = sqrt(max(S.cssq, 0.0));

    S.asq = gamp * S.inv_rho;
    S.a   = sqrt(max(S.asq, 0.0));

    // Stone A17 — β unit vectors in the perpendicular plane.
    // When B_⊥ → 0, both transverse components vanish in every
    // eigenvector slot they appear in EXCEPT the Alfvén pair, so
    // the choice of (β_t1, β_t2) is arbitrary so long as
    // β_t1² + β_t2² = 1 and they're not both zero. Brio-Wu 1988 eq
    // 45 / Roe96 pg 60: pick (1, 0). Athena++ matches.
    let bt = sqrt(btsq);
    if (bt > 0.0) {
        S.bet1 = w.bt1 / bt;
        S.bet2 = w.bt2 / bt;
    } else {
        S.bet1 = 1.0;
        S.bet2 = 0.0;
    }

    // Stone A16 + Roe96 cases III/IV/V — α_f, α_s normalization with
    // degeneracy regularization. Each case picks the regular eigenvector
    // pair that survives the limit.
    if ((S.cfsq - S.cssq) <= 0.0) {
        // Roe96 case V (triple umbilic — c_f² == c_s² == a²).
        // Pick α_f = 1, α_s = 0 (fast wave carries acoustic).
        S.alpha_f = 1.0;
        S.alpha_s = 0.0;
    } else if ((S.asq - S.cssq) <= 0.0) {
        // Roe96 case IV — low-β (a < c_s). Slow waves degenerate to
        // acoustic; α_f = 0, α_s = 1.
        S.alpha_f = 0.0;
        S.alpha_s = 1.0;
    } else if ((S.cfsq - S.asq) <= 0.0) {
        // Roe96 case III — high-β (a > c_f). Fast waves degenerate to
        // acoustic; α_f = 1, α_s = 0.
        S.alpha_f = 1.0;
        S.alpha_s = 0.0;
    } else {
        let denom = S.cfsq - S.cssq;
        S.alpha_f = sqrt(max((S.asq  - S.cssq) / denom, 0.0));
        S.alpha_s = sqrt(max((S.cfsq - S.asq ) / denom, 0.0));
    }

    // Stone uses sign(0) = +1 (Athena++ SIGN macro convention).
    S.sgn_bn = select(-1.0, 1.0, bn >= 0.0);
    return S;
}

// L · dW — project primitive difference onto characteristic basis.
// Mirrors Athena++ `LeftEigenmatrixDotVector` MHD-adiabatic branch
// line-for-line (component indices match the v_0…v_6 there). Stone
// 2008 eq A18.
fn project_to_char(dW: PrimVec7, S: EigenSystem) -> CharVec7 {
    let nf = 0.5 / max(S.asq, 1.0e-30);
    let qf = nf * S.cf * S.alpha_f * S.sgn_bn;
    let qs = nf * S.cs * S.alpha_s * S.sgn_bn;
    let af_prime = 0.5 * S.alpha_f / (S.a * S.sqrtd);
    let as_prime = 0.5 * S.alpha_s / (S.a * S.sqrtd);

    let bt_term_v = S.bet1 * dW.vt1 + S.bet2 * dW.vt2;
    let bt_term_b = S.bet1 * dW.bt1 + S.bet2 * dW.bt2;

    var C: CharVec7;
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

// R · dC — project characteristic vector back to primitive space.
// Mirrors Athena++ `RightEigenmatrixDotVector` MHD-adiabatic branch.
// Stone 2008 eq A12.
fn project_from_char(C: CharVec7, S: EigenSystem) -> PrimVec7 {
    let qf = S.cf * S.alpha_f * S.sgn_bn;
    let qs = S.cs * S.alpha_s * S.sgn_bn;
    let af = S.a  * S.alpha_f * S.sqrtd;
    let as_ = S.a * S.alpha_s * S.sqrtd;
    let rho = 1.0 / max(S.inv_rho, 1.0e-30);   // = ρ

    let af_sum = S.alpha_f * (C.fL + C.fR);
    let as_sum = S.alpha_s * (C.sL + C.sR);
    let af_dif = S.alpha_f * (C.fR - C.fL);
    let as_dif = S.alpha_s * (C.sR - C.sL);

    let qs_fdif = qs * (C.fL - C.fR);
    let qf_sdif = qf * (C.sR - C.sL);
    let aL_sum  = C.aL + C.aR;
    let aL_dif  = C.aR - C.aL;

    var W: PrimVec7;
    W.rho = rho * (af_sum + as_sum) + C.e;
    W.vn  = S.cf * af_dif + S.cs * as_dif;
    W.vt1 = S.bet1 * (qs_fdif + qf_sdif)
          + S.bet2 * (C.aR - C.aL);    // = bet2 * (vect(5) - vect(1)) in Athena
    W.vt2 = S.bet2 * (qs_fdif + qf_sdif)
          + S.bet1 * (C.aL - C.aR);    // = bet1 * (vect(1) - vect(5))
    W.p   = rho * S.asq * (af_sum + as_sum);
    W.bt1 = S.bet1 * (as_ * (C.fL + C.fR) - af * (C.sL + C.sR))
          - S.bet2 * S.sgn_bn * S.sqrtd * aL_sum;
    W.bt2 = S.bet2 * (as_ * (C.fL + C.fR) - af * (C.sL + C.sR))
          + S.bet1 * S.sgn_bn * S.sqrtd * aL_sum;
    // Athena writes v_2 = bet2*(qs·(v0−v6) + qf·(v4−v2)) + bet3·(v5−v1)
    // and v_3 = bet3*(…) + bet2*(v1−v5); the structure above matches with
    // (bet1,bet2)≡(bet2_Athena,bet3_Athena) — note our index ordering
    // (vt1,vt2)=(ivy,ivz) matches Athena's IBY/IBZ pairing.
    return W;
}

// PPM monotonicity limiter applied to a single characteristic component.
// Operates on the (dL, dR) deltas directly (dL = cell − left face,
// dR = right face − cell). Algebraically equivalent to the
// face-state-form limiter the previous primitive-PPM used:
//   q_L = q_c − dL,  q_R = q_c + dR
//   monotone:        dL · dR ≤ 0 → dL = dR = 0
//   parabola check:  ∂(parabola)/∂ξ has root inside [0,1] iff
//                    |dq|² < |q6|·|dq|, with dq=dL+dR, q6=3(dL−dR);
//                    case ① (q6>0 ⇔ parabola convex from above) clamp
//                    dL ≤ 2 dR (extremum near right face → push left
//                    face inward); case ② symmetric.
// The original code wrote the constraints in face-state form
// (qL := 3 q_c − 2 qR ⇒ dL := 2 dR); we keep that algebra here.
fn ppm_limit_delta(dL_in: f32, dR_in: f32) -> vec2<f32> {
    var dL = dL_in;
    var dR = dR_in;
    if (dL * dR <= 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let dq  = dL + dR;
    let q6  = 3.0 * (dL - dR);
    let dq2 = dq * dq;
    let test = dq * q6;
    if (test > dq2) {
        // Parabola extremum lies between cell center and right face
        // → fold the left face inward.
        dL = 2.0 * dR;
    } else if (test < -dq2) {
        // Parabola extremum lies between cell center and left face
        // → fold the right face inward.
        dR = 2.0 * dL;
    }
    return vec2<f32>(dL, dR);
}

// Per-component limiter on the 7 characteristic deltas.
struct CharLR {
    L: CharVec7,
    R: CharVec7,
};

fn ppm_limit_char(aL: CharVec7, aR: CharVec7) -> CharLR {
    let r0 = ppm_limit_delta(aL.fL, aR.fL);
    let r1 = ppm_limit_delta(aL.aL, aR.aL);
    let r2 = ppm_limit_delta(aL.sL, aR.sL);
    let r3 = ppm_limit_delta(aL.e,  aR.e );
    let r4 = ppm_limit_delta(aL.sR, aR.sR);
    let r5 = ppm_limit_delta(aL.aR, aR.aR);
    let r6 = ppm_limit_delta(aL.fR, aR.fR);
    var out: CharLR;
    out.L = CharVec7(r0.x, r1.x, r2.x, r3.x, r4.x, r5.x, r6.x);
    out.R = CharVec7(r0.y, r1.y, r2.y, r3.y, r4.y, r5.y, r6.y);
    return out;
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    // Dispatch over the extended interior range: [ghost-1, ghost+N+1).
    let extent = n_interior + 2u;
    let in_extent = (gid.x < extent) && (gid.y < extent);

    let axis = sweep.sweep_dir;
    let g    = U_uniforms.gamma;
    let pf   = U_uniforms.pressure_floor;

    let nt_max = i32(n_total) - 1;
    let gx = i32(gid.x) + i32(ghost) - 1;
    let gy = i32(gid.y) + i32(ghost) - 1;
    let lx = i32(lid.x);
    let ly = i32(lid.y);

    // ── Phase A: cooperative tile load ─────────────────────────────────
    let cx = u32(clamp(gx, 0, nt_max));
    let cy = u32(clamp(gy, 0, nt_max));
    tile[ly + 2][lx + 2] = cell_primitive_cache(cx, cy, n_total, g, pf);

    if (lid.x < 2u) {
        let sx = u32(clamp(gx - 2, 0, nt_max));
        let sy = u32(clamp(gy, 0, nt_max));
        tile[ly + 2][lx] = cell_primitive_cache(sx, sy, n_total, g, pf);
    }
    if (lid.x >= 6u) {
        let sx = u32(clamp(gx + 2, 0, nt_max));
        let sy = u32(clamp(gy, 0, nt_max));
        tile[ly + 2][lx + 4] = cell_primitive_cache(sx, sy, n_total, g, pf);
    }
    if (lid.y < 2u) {
        let sx = u32(clamp(gx, 0, nt_max));
        let sy = u32(clamp(gy - 2, 0, nt_max));
        tile[ly][lx + 2] = cell_primitive_cache(sx, sy, n_total, g, pf);
    }
    if (lid.y >= 6u) {
        let sx = u32(clamp(gx, 0, nt_max));
        let sy = u32(clamp(gy + 2, 0, nt_max));
        tile[ly + 4][lx + 2] = cell_primitive_cache(sx, sy, n_total, g, pf);
    }
    if (lid.x < 2u && lid.y < 2u) {
        let sx = u32(clamp(gx - 2, 0, nt_max));
        let sy = u32(clamp(gy - 2, 0, nt_max));
        tile[ly][lx] = cell_primitive_cache(sx, sy, n_total, g, pf);
    }
    if (lid.x >= 6u && lid.y < 2u) {
        let sx = u32(clamp(gx + 2, 0, nt_max));
        let sy = u32(clamp(gy - 2, 0, nt_max));
        tile[ly][lx + 4] = cell_primitive_cache(sx, sy, n_total, g, pf);
    }
    if (lid.x < 2u && lid.y >= 6u) {
        let sx = u32(clamp(gx - 2, 0, nt_max));
        let sy = u32(clamp(gy + 2, 0, nt_max));
        tile[ly + 4][lx] = cell_primitive_cache(sx, sy, n_total, g, pf);
    }
    if (lid.x >= 6u && lid.y >= 6u) {
        let sx = u32(clamp(gx + 2, 0, nt_max));
        let sy = u32(clamp(gy + 2, 0, nt_max));
        tile[ly + 4][lx + 4] = cell_primitive_cache(sx, sy, n_total, g, pf);
    }

    workgroupBarrier();

    // ── Phase B: PPM math (characteristic-variable limited) ────────────
    if (!in_extent) { return; }

    let ix = u32(gx);
    let iy = u32(gy);
    let idx = cell_idx_total(ix, iy, n_total);

    // Stencil-ok check (5-point sweep stencil fits in storage).
    var stencil_ok = true;
    if (axis == 0u) {
        stencil_ok = (ix >= 2u) && (ix + 2u < n_total);
    } else {
        stencil_ok = (iy >= 2u) && (iy + 2u < n_total);
    }

    let tc = tile[ly + 2][lx + 2];

    if (!stencil_ok) {
        // Piecewise-constant fallback: both edges equal cell value.
        // Use the existing pack_prim_pair shape for output consistency.
        let pcL = permute_prim(tc, axis);
        let pp  = pack_prim_pair_from_vec7(vec7_of(pcL), pcL.bn, axis);
        var l0 = pp.p0; var r0 = pp.p0;
        var l1 = pp.p1; var r1 = pp.p1;
        l0.x = max(l0.x, DENSITY_FLOOR);
        r0.x = max(r0.x, DENSITY_FLOOR);
        l1.x = max(l1.x, pf);
        r1.x = max(r1.x, pf);
        edge_l_0[idx] = l0;
        edge_l_1[idx] = l1;
        edge_r_0[idx] = r0;
        edge_r_1[idx] = r1;
        return;
    }

    // Read the 5-cell sweep-axis stencil from the cached tile.
    var tm2: MhdPrim; var tm1: MhdPrim; var tp1: MhdPrim; var tp2: MhdPrim;
    if (axis == 0u) {
        tm2 = tile[ly + 2][lx];
        tm1 = tile[ly + 2][lx + 1];
        tp1 = tile[ly + 2][lx + 3];
        tp2 = tile[ly + 2][lx + 4];
    } else {
        tm2 = tile[ly][lx + 2];
        tm1 = tile[ly + 1][lx + 2];
        tp1 = tile[ly + 3][lx + 2];
        tp2 = tile[ly + 4][lx + 2];
    }

    // Permute all 5 cells into sweep-aligned 7-vectors.
    let perm_c  = permute_prim(tc,  axis);
    let perm_m2 = permute_prim(tm2, axis);
    let perm_m1 = permute_prim(tm1, axis);
    let perm_p1 = permute_prim(tp1, axis);
    let perm_p2 = permute_prim(tp2, axis);

    let w_c  = vec7_of(perm_c);
    let w_m2 = vec7_of(perm_m2);
    let w_m1 = vec7_of(perm_m1);
    let w_p1 = vec7_of(perm_p1);
    let w_p2 = vec7_of(perm_p2);
    let bn_c = perm_c.bn;

    // 4th-order raw edge interpolants (CW 1984 in primitive variables).
    let c7 = 7.0 / 12.0;
    let c1 = 1.0 / 12.0;
    let qL_raw = prim_scaled_diff(w_m1, w_c,  c7, w_m2, w_p1, c1);
    let qR_raw = prim_scaled_diff(w_c,  w_p1, c7, w_m1, w_p2, c1);

    // Primitive deltas from cell center to face.
    let dL_prim = prim_sub(w_c, qL_raw);    // dL = w_c − qL_raw
    let dR_prim = prim_sub(qR_raw, w_c);    // dR = qR_raw − w_c

    // Build the MHD primitive eigensystem at the cell center.
    let eig = mhd_eigensystem(w_c, bn_c, g);

    // Project deltas to characteristic, limit per-wave, project back.
    let aL = project_to_char(dL_prim, eig);
    let aR = project_to_char(dR_prim, eig);
    let lim = ppm_limit_char(aL, aR);
    let dL_lim = project_from_char(lim.L, eig);
    let dR_lim = project_from_char(lim.R, eig);

    // Recover limited face states.
    let w_left  = prim_sub(w_c, dL_lim);
    let w_right = prim_add(w_c, dR_lim);

    // Pack and floor.
    let pp_L = pack_prim_pair_from_vec7(w_left,  bn_c, axis);
    let pp_R = pack_prim_pair_from_vec7(w_right, bn_c, axis);
    var l0 = pp_L.p0; var r0 = pp_R.p0;
    var l1 = pp_L.p1; var r1 = pp_R.p1;
    l0.x = max(l0.x, DENSITY_FLOOR);
    r0.x = max(r0.x, DENSITY_FLOOR);
    l1.x = max(l1.x, pf);
    r1.x = max(r1.x, pf);

    edge_l_0[idx] = l0;
    edge_l_1[idx] = l1;
    edge_r_0[idx] = r0;
    edge_r_1[idx] = r1;
}

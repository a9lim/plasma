// ─── riemann-hlld.wgsl ───────────────────────────────────────────────
// HLLD five-wave Riemann solver for ideal MHD (Miyoshi & Kusano 2005).
//
// Wave structure (left → right):
//   S_L  : fast-magnetosonic wave (left)
//   S_L* : Alfvén wave (left rotational discontinuity)
//   S_M  : contact / entropy
//   S_R* : Alfvén wave (right rotational discontinuity)
//   S_R  : fast-magnetosonic wave (right)
//
// Star-state intermediate states (M&K 2005 eqs 38-67), with B_n constant
// across all waves (continuous face-normal B).
//
// Degenerate-branch handling (per Phase 3a agent's pre-flight):
//   Branch A — Bx² < ε² · ρ   → Alfvén waves degenerate; fall back to HLLC
//                              (3-wave: SL, SM, SR; transverse B advected
//                              with average velocity, M&K eqs 51 reduced).
//   Branch B — wave-speed coincidence (rare; |S* - S| < tol)
//                              → fall back to HLL (no star states).
//   Branch C — star-state pressure < PRESSURE_FLOOR
//                              → pressure-floor + fall back to HLL.
//
// Reads PPM-reconstructed L/R face primitive states from edge_r_*[i-1]
// (for QL) and edge_l_*[i] (for QR). The face is owned by cell (i-1)
// on its right side; this kernel computes the flux through that face.
//
// Convention: kernel cell (i,j) computes flux at face (i+½, j) for
// x-sweep (axis=0), face (i, j+½) for y-sweep (axis=1) — same as
// Phase 3a's HLL.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0_in     (ro)
//   2 U1_in     (ro)
//   3 Bx_face   (ro)
//   4 By_face   (ro)
//   5 edge_l_0  (ro)
//   6 edge_l_1  (ro)
//   7 edge_r_0  (ro)
//   8 edge_r_1  (ro)
//   9 flux_0    (rw)
//  10 flux_1    (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:     array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:   array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:   array<f32>;
@group(0) @binding(5) var<storage, read>       edge_l_0:  array<vec4<f32>>;
@group(0) @binding(6) var<storage, read>       edge_l_1:  array<vec4<f32>>;
@group(0) @binding(7) var<storage, read>       edge_r_0:  array<vec4<f32>>;
@group(0) @binding(8) var<storage, read>       edge_r_1:  array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> flux_0:    array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> flux_1:   array<vec4<f32>>;

// Branch-A threshold: ε² used to declare Bx ≈ 0.
const HLLD_BX_EPS2: f32 = 1.0e-24;
// Branch-B tolerance: relative wave-speed coincidence trigger.
const HLLD_WS_TOL:  f32 = 1.0e-8;

// Unpack one cell's PPM-edge state pair (4-vec primitives) into an MhdPrim.
// `edge0` = (ρ, vx, vy, vz)
// `edge1` = (p, Bt1, Bz, _)   where Bt1 = By for x-sweep, Bx for y-sweep
// `b_normal` overwrites the face-normal B (continuous across the face).
fn unpack_edge_prim(edge0: vec4<f32>, edge1: vec4<f32>, b_normal: f32, axis: u32) -> MhdPrim {
    var Q: MhdPrim;
    Q.rho = max(edge0.x, DENSITY_FLOOR);
    Q.vx  = edge0.y;
    Q.vy  = edge0.z;
    Q.vz  = edge0.w;
    Q.p   = max(edge1.x, PRESSURE_FLOOR);
    Q.bz  = edge1.z;
    if (axis == 0u) {
        Q.bx = b_normal;
        Q.by = edge1.y;
    } else {
        Q.bx = edge1.y;
        Q.by = b_normal;
    }
    return Q;
}

// HLL flux helper (used as fallback by HLLD for degenerate branches).
struct HLLOut {
    f0:   vec4<f32>,
    f1:   vec4<f32>,
    fBt1: f32,
    fBt2: f32,
};

fn hll_flux_mhd(QL: MhdPrim, QR: MhdPrim, axis: u32, gamma: f32) -> HLLOut {
    let cfL = fast_mag_speed(QL, gamma, axis);
    let cfR = fast_mag_speed(QR, gamma, axis);
    let uL  = normal_velocity_mhd(QL, axis);
    let uR  = normal_velocity_mhd(QR, axis);
    let SL  = min(uL - cfL, uR - cfR);
    let SR  = max(uL + cfL, uR + cfR);

    let FL = mhd_flux(QL, gamma, axis);
    let FR = mhd_flux(QR, gamma, axis);

    let CL = prim_to_cons_pair(QL, gamma);
    let CR = prim_to_cons_pair(QR, gamma);

    var out: HLLOut;
    if (SL >= 0.0) {
        out.f0 = FL.f0; out.f1 = FL.f1;
        out.fBt1 = FL.f_bt1; out.fBt2 = FL.f_bt2;
    } else if (SR <= 0.0) {
        out.f0 = FR.f0; out.f1 = FR.f1;
        out.fBt1 = FR.f_bt1; out.fBt2 = FR.f_bt2;
    } else {
        let denom = max(SR - SL, 1.0e-12);
        out.f0 = (SR * FL.f0 - SL * FR.f0 + SL * SR * (CR.U0 - CL.U0)) / denom;
        out.f1 = (SR * FL.f1 - SL * FR.f1 + SL * SR * (CR.U1 - CL.U1)) / denom;
        let bt1L = select(QL.by, QL.bx, axis == 1u);
        let bt1R = select(QR.by, QR.bx, axis == 1u);
        out.fBt1 = (SR * FL.f_bt1 - SL * FR.f_bt1 + SL * SR * (bt1R - bt1L)) / denom;
        out.fBt2 = (SR * FL.f_bt2 - SL * FR.f_bt2 + SL * SR * (QR.bz - QL.bz)) / denom;
    }
    return out;
}

// Unified vector form of the MHD conservative state along the sweep axis.
// Returns (ρ, ρu_n, ρu_t1, ρu_t2, E, B_t1, B_t2) — 7 scalars in a vec4+vec4
// shape we can slice. We do this all in scalar form for HLLD's algebra.
struct AxisState {
    rho: f32,
    un:  f32,   // velocity normal to face
    ut1: f32,   // velocity transverse (1)
    ut2: f32,   // velocity transverse (2)
    bn:  f32,   // B normal (constant)
    bt1: f32,
    bt2: f32,
    p:   f32,
    pT:  f32,   // total pressure = p + ½B²
    E:   f32,   // total energy density
};

fn prim_to_axis_state(P: MhdPrim, axis: u32, gamma: f32) -> AxisState {
    var A: AxisState;
    A.rho = P.rho;
    A.p   = P.p;
    A.bn  = select(P.by, P.bx, axis == 0u);
    if (axis == 0u) {
        A.un = P.vx; A.ut1 = P.vy; A.ut2 = P.vz;
        A.bt1 = P.by; A.bt2 = P.bz;
    } else {
        A.un = P.vy; A.ut1 = P.vx; A.ut2 = P.vz;
        A.bt1 = P.bx; A.bt2 = P.bz;
    }
    let b2 = A.bn*A.bn + A.bt1*A.bt1 + A.bt2*A.bt2;
    A.pT  = A.p + 0.5 * b2;
    let v2 = A.un*A.un + A.ut1*A.ut1 + A.ut2*A.ut2;
    A.E   = A.p / (gamma - 1.0) + 0.5 * A.rho * v2 + 0.5 * b2;
    return A;
}

// Assemble axis-frame flux from state.
struct AxisFlux {
    f_rho: f32,
    f_mn:  f32,
    f_mt1: f32,
    f_mt2: f32,
    f_E:   f32,
    f_bt1: f32,   // = un·bt1 - ut1·bn
    f_bt2: f32,   // = un·bt2 - ut2·bn
};

fn axis_flux(A: AxisState) -> AxisFlux {
    var F: AxisFlux;
    F.f_rho = A.rho * A.un;
    F.f_mn  = A.rho * A.un * A.un + A.pT - A.bn * A.bn;
    F.f_mt1 = A.rho * A.un * A.ut1       - A.bn * A.bt1;
    F.f_mt2 = A.rho * A.un * A.ut2       - A.bn * A.bt2;
    let vdotb = A.un*A.bn + A.ut1*A.bt1 + A.ut2*A.bt2;
    F.f_E   = (A.E + A.pT) * A.un - A.bn * vdotb;
    F.f_bt1 = A.un * A.bt1 - A.ut1 * A.bn;
    F.f_bt2 = A.un * A.bt2 - A.ut2 * A.bn;
    return F;
}

// Pack an axis-frame conservative-state set into (U0, U1, bt1, bt2) for
// the downstream write. axis chooses whether the in-plane velocity pair
// (vx, vy) or (vy, vx) is reconstructed from (un, ut1).
struct PackedFlux {
    f0:   vec4<f32>,
    f1:   vec4<f32>,
    fBt1: f32,   // transverse-B flux for B_t1 (= By for x-sweep, Bx for y-sweep)
    fBt2: f32,   // = Bz flux's transverse-B contribution
};

fn pack_flux(F: AxisFlux, axis: u32) -> PackedFlux {
    var P: PackedFlux;
    if (axis == 0u) {
        // Normal = x → return (Fρ, Fρvx, Fρvy, Fρvz) and (FE, FBz, _, _).
        P.f0 = vec4<f32>(F.f_rho, F.f_mn, F.f_mt1, F.f_mt2);
        P.f1 = vec4<f32>(F.f_E,   F.f_bt2, 0.0, 0.0);   // .y = Bz flux
        P.fBt1 = F.f_bt1;
    } else {
        // Normal = y → vy is normal, vx is transverse-1.
        P.f0 = vec4<f32>(F.f_rho, F.f_mt1, F.f_mn, F.f_mt2);
        P.f1 = vec4<f32>(F.f_E,   F.f_bt2, 0.0, 0.0);
        P.fBt1 = F.f_bt1;
    }
    P.fBt2 = 0.0;  // unused; kept for the existing flux_1.w slot.
    return P;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i  = i32(n);
    let axis = U_uniforms.sweep_dir;
    let g    = U_uniforms.gamma;

    let i  = i32(gid.x);
    let j  = i32(gid.y);
    var ip = i;
    var jp = j;
    if (axis == 0u) { ip = i + 1; } else { jp = j + 1; }

    let idx_c = cell_index(gid.x, gid.y, n);
    let ip_u  = wrap_idx(ip, n_i);
    let jp_u  = wrap_idx(jp, n_i);
    let idx_p = cell_index(ip_u, jp_u, n);

    // Face-normal B: continuous staggered face value owned by cell (i,j).
    var b_normal: f32;
    if (axis == 0u) {
        b_normal = Bx_face[idx_c];
    } else {
        b_normal = By_face[idx_c];
    }

    // L state from right-edge of cell (i,j); R state from left-edge of (i+1,j).
    let QL = unpack_edge_prim(edge_r_0[idx_c], edge_r_1[idx_c], b_normal, axis);
    let QR = unpack_edge_prim(edge_l_0[idx_p], edge_l_1[idx_p], b_normal, axis);

    let AL = prim_to_axis_state(QL, axis, g);
    let AR = prim_to_axis_state(QR, axis, g);
    let FL = axis_flux(AL);
    let FR = axis_flux(AR);

    // Wave-speed estimates: fast magnetosonic on each side.
    let cfL = fast_mag_speed(QL, g, axis);
    let cfR = fast_mag_speed(QR, g, axis);
    let SL  = min(AL.un - cfL, AR.un - cfR);
    let SR  = max(AL.un + cfL, AR.un + cfR);

    // ── Early exits (M&K 2005, supersonic branches) ────────────────
    if (SL >= 0.0) {
        let pf = pack_flux(FL, axis);
        flux_0[idx_c] = pf.f0;
        flux_1[idx_c] = vec4<f32>(pf.f1.x, pf.f1.y, pf.fBt1, pf.fBt2);
        return;
    }
    if (SR <= 0.0) {
        let pf = pack_flux(FR, axis);
        flux_0[idx_c] = pf.f0;
        flux_1[idx_c] = vec4<f32>(pf.f1.x, pf.f1.y, pf.fBt1, pf.fBt2);
        return;
    }

    // ── Branch B: degenerate wave-speed coincidence → HLL fallback ──
    if (SR - SL < HLLD_WS_TOL * (abs(SR) + abs(SL) + 1.0e-12)) {
        let h = hll_flux_mhd(QL, QR, axis, g);
        flux_0[idx_c] = h.f0;
        flux_1[idx_c] = vec4<f32>(h.f1.x, h.f1.y, h.fBt1, h.fBt2);
        return;
    }

    // ── Compute contact speed S_M (M&K eq 38) ─────────────────────
    //   S_M = [(SR-uR)·ρR·uR - (SL-uL)·ρL·uL - p_T,R + p_T,L]
    //         / [(SR-uR)·ρR - (SL-uL)·ρL]
    let rcL = AL.rho * (SL - AL.un);
    let rcR = AR.rho * (SR - AR.un);
    let SM_num = rcR * AR.un - rcL * AL.un - AR.pT + AL.pT;
    let SM_den = rcR - rcL;
    let SM = SM_num / select(SM_den, sign(SM_den) * 1.0e-12, abs(SM_den) < 1.0e-30);

    // Star-state total pressure (M&K eq 41). Equal on both sides.
    //   p_T* = p_T,L + ρL·(SL - uL)·(SM - uL)
    //        = p_T,R + ρR·(SR - uR)·(SM - uR)
    let pT_star = AL.pT + AL.rho * (SL - AL.un) * (SM - AL.un);

    // ── Branch C: negative star pressure → HLL fallback ──
    if (pT_star <= PRESSURE_FLOOR) {
        let h = hll_flux_mhd(QL, QR, axis, g);
        flux_0[idx_c] = h.f0;
        flux_1[idx_c] = vec4<f32>(h.f1.x, h.f1.y, h.fBt1, h.fBt2);
        return;
    }

    // ── Branch A: Bx² < ε² · ρ → Alfvén waves degenerate, HLLC ─────
    // Use the average ρ on either side as the scale for the threshold.
    let bn2 = b_normal * b_normal;
    let rho_scale = max(0.5 * (AL.rho + AR.rho), DENSITY_FLOOR);
    let branchA = bn2 < HLLD_BX_EPS2 * rho_scale;

    if (branchA) {
        // HLLC for MHD with B_n ≈ 0: 3-wave structure (SL, SM, SR).
        // Star-region density (M&K eq 43 simplified with B_n = 0):
        //   ρ*_K = ρ_K · (S_K - u_K) / (S_K - S_M)
        //   transverse velocities & B advected at S_M (rotational discontinuity
        //   collapses onto the contact when B_n = 0).
        // SL - SM is strictly negative and SR - SM strictly positive when
        // we reach here (Branch B already filters the coincidence case);
        // small floor just guards against fp drift.
        let denom_L = min(SL - SM, -1.0e-20);
        let denom_R = max(SR - SM,  1.0e-20);
        let rhoLs = AL.rho * (SL - AL.un) / denom_L;
        let rhoRs = AR.rho * (SR - AR.un) / denom_R;

        // Conservative state vectors (axis frame): (ρ, ρu_n, ρu_t1, ρu_t2, E, B_t1, B_t2)
        // U*_K = ρ*_K · (1, S_M, u_t1_K, u_t2_K, E*_K/ρ*_K, B_t1_K/ρ*_K, B_t2_K/ρ*_K)
        // simplified — B_t1, B_t2 advect, energy from total-pressure jump:
        //   E*_K = ((S_K - u_K) E_K - p_T,K · u_K + p_T* · S_M) / (S_K - S_M)
        let E_Ls = ((SL - AL.un) * AL.E - AL.pT * AL.un + pT_star * SM) / (SL - SM);
        let E_Rs = ((SR - AR.un) * AR.E - AR.pT * AR.un + pT_star * SM) / (SR - SM);
        // Fluxes via HLLC formula: F* = F_K + S_K · (U*_K - U_K)
        var Fout: AxisFlux;
        if (SM >= 0.0) {
            // Left star.
            Fout.f_rho = FL.f_rho + SL * (rhoLs                - AL.rho);
            Fout.f_mn  = FL.f_mn  + SL * (rhoLs * SM           - AL.rho * AL.un);
            Fout.f_mt1 = FL.f_mt1 + SL * (rhoLs * AL.ut1       - AL.rho * AL.ut1);
            Fout.f_mt2 = FL.f_mt2 + SL * (rhoLs * AL.ut2       - AL.rho * AL.ut2);
            Fout.f_E   = FL.f_E   + SL * (E_Ls                 - AL.E);
            Fout.f_bt1 = FL.f_bt1 + SL * (AL.bt1 * (SL - AL.un)/(SL - SM) - AL.bt1);
            Fout.f_bt2 = FL.f_bt2 + SL * (AL.bt2 * (SL - AL.un)/(SL - SM) - AL.bt2);
        } else {
            Fout.f_rho = FR.f_rho + SR * (rhoRs                - AR.rho);
            Fout.f_mn  = FR.f_mn  + SR * (rhoRs * SM           - AR.rho * AR.un);
            Fout.f_mt1 = FR.f_mt1 + SR * (rhoRs * AR.ut1       - AR.rho * AR.ut1);
            Fout.f_mt2 = FR.f_mt2 + SR * (rhoRs * AR.ut2       - AR.rho * AR.ut2);
            Fout.f_E   = FR.f_E   + SR * (E_Rs                 - AR.E);
            Fout.f_bt1 = FR.f_bt1 + SR * (AR.bt1 * (SR - AR.un)/(SR - SM) - AR.bt1);
            Fout.f_bt2 = FR.f_bt2 + SR * (AR.bt2 * (SR - AR.un)/(SR - SM) - AR.bt2);
        }
        let pf = pack_flux(Fout, axis);
        flux_0[idx_c] = pf.f0;
        flux_1[idx_c] = vec4<f32>(pf.f1.x, pf.f1.y, pf.fBt1, pf.fBt2);
        return;
    }

    // ── Full HLLD 5-wave path ─────────────────────────────────────
    // Star-state densities (M&K eq 43):
    //   ρ*_K = ρ_K · (S_K - u_K) / (S_K - S_M)
    // SL-SM strictly negative, SR-SM strictly positive past Branch B.
    let dL = min(SL - SM, -1.0e-20);
    let dR = max(SR - SM,  1.0e-20);
    let rhoLs = AL.rho * (SL - AL.un) / dL;
    let rhoRs = AR.rho * (SR - AR.un) / dR;

    // Star-state transverse velocities & B (M&K eqs 44, 46 in axis frame):
    //   u_t* = u_t,K - B_n · B_t,K · (S_M - u_K) / [ρ_K (S_K - u_K)(S_K - S_M) - B_n²]
    //   B_t* = B_t,K · [ρ_K (S_K - u_K)² - B_n²] / [ρ_K (S_K - u_K)(S_K - S_M) - B_n²]
    // denomL/denomR vanish only at the rotational-discontinuity boundary
    // (Branch B coincidence, already filtered) or B_n=0 (Branch A). Guard
    // away from zero with a small floor preserving sign.
    let denomL_raw = AL.rho * (SL - AL.un) * (SL - SM) - bn2;
    let denomR_raw = AR.rho * (SR - AR.un) * (SR - SM) - bn2;
    // Physical denom is O(ρ·c_f²) ≫ 1e-12 when full HLLD is appropriate.
    // Branch A (B_n→0) and Branch B (wave coincidence) are filtered above;
    // remaining floor just kills fp drift singularities.
    let safeDL = select(denomL_raw, 1.0e-20, abs(denomL_raw) < 1.0e-20);
    let safeDR = select(denomR_raw, 1.0e-20, abs(denomR_raw) < 1.0e-20);

    let g_L = AL.rho * (SL - AL.un) * (SL - AL.un) - bn2;
    let g_R = AR.rho * (SR - AR.un) * (SR - AR.un) - bn2;

    let ut1_Ls = AL.ut1 - b_normal * AL.bt1 * (SM - AL.un) / safeDL;
    let ut2_Ls = AL.ut2 - b_normal * AL.bt2 * (SM - AL.un) / safeDL;
    let ut1_Rs = AR.ut1 - b_normal * AR.bt1 * (SM - AR.un) / safeDR;
    let ut2_Rs = AR.ut2 - b_normal * AR.bt2 * (SM - AR.un) / safeDR;

    let bt1_Ls = AL.bt1 * g_L / safeDL;
    let bt2_Ls = AL.bt2 * g_L / safeDL;
    let bt1_Rs = AR.bt1 * g_R / safeDR;
    let bt2_Rs = AR.bt2 * g_R / safeDR;

    // Energy in star states (M&K eq 48):
    //   E*_K = ((S_K - u_K) E_K - p_T,K u_K + p_T* S_M
    //           + B_n (v·B_K - v*·B*_K)) / (S_K - S_M)
    let vdotb_L  = AL.un  * b_normal + AL.ut1  * AL.bt1  + AL.ut2  * AL.bt2;
    let vdotb_R  = AR.un  * b_normal + AR.ut1  * AR.bt1  + AR.ut2  * AR.bt2;
    let vdotbLs  = SM     * b_normal + ut1_Ls  * bt1_Ls  + ut2_Ls  * bt2_Ls;
    let vdotbRs  = SM     * b_normal + ut1_Rs  * bt1_Rs  + ut2_Rs  * bt2_Rs;

    let E_Ls = ((SL - AL.un) * AL.E - AL.pT * AL.un + pT_star * SM
               + b_normal * (vdotb_L - vdotbLs)) / (SL - SM);
    let E_Rs = ((SR - AR.un) * AR.E - AR.pT * AR.un + pT_star * SM
               + b_normal * (vdotb_R - vdotbRs)) / (SR - SM);

    // Alfvén wave speeds (M&K eq 51):
    //   S_L* = S_M - |B_n| / sqrt(ρ*_L)
    //   S_R* = S_M + |B_n| / sqrt(ρ*_R)
    let absBn = abs(b_normal);
    let SLs = SM - absBn / sqrt(max(rhoLs, DENSITY_FLOOR));
    let SRs = SM + absBn / sqrt(max(rhoRs, DENSITY_FLOOR));

    // Double-star states (between the two Alfvén waves, M&K eqs 59-63):
    //   sqrt(ρ*_L)·u_t** = sqrt(ρ*_L) u_t*_L + sqrt(ρ*_R) u_t*_R + (B_t*_R - B_t*_L) sign(B_n)
    //                      all divided by (sqrt(ρ*_L) + sqrt(ρ*_R))
    //   B_t** = ...
    let srL = sqrt(max(rhoLs, DENSITY_FLOOR));
    let srR = sqrt(max(rhoRs, DENSITY_FLOOR));
    let srSum = srL + srR;
    let sgnBn = select(-1.0, 1.0, b_normal >= 0.0);

    let ut1_ss = (srL * ut1_Ls + srR * ut1_Rs + (bt1_Rs - bt1_Ls) * sgnBn) / srSum;
    let ut2_ss = (srL * ut2_Ls + srR * ut2_Rs + (bt2_Rs - bt2_Ls) * sgnBn) / srSum;
    let bt1_ss = (srL * bt1_Rs + srR * bt1_Ls + srL * srR * (ut1_Rs - ut1_Ls) * sgnBn) / srSum;
    let bt2_ss = (srL * bt2_Rs + srR * bt2_Ls + srL * srR * (ut2_Rs - ut2_Ls) * sgnBn) / srSum;

    // Energy in double-star (M&K eq 64). v**·B** is the double-star
    // velocity-B inner product (same on both sides of the contact):
    //   E**_L = E*_L - sqrt(ρ*_L) sign(B_n) (v*_L · B*_L - v** · B**)
    //   E**_R = E*_R + sqrt(ρ*_R) sign(B_n) (v*_R · B*_R - v** · B**)
    let vdotb_ss = SM * b_normal + ut1_ss * bt1_ss + ut2_ss * bt2_ss;
    let E_Lss = E_Ls - srL * sgnBn * (vdotbLs - vdotb_ss);
    let E_Rss = E_Rs + srR * sgnBn * (vdotbRs - vdotb_ss);

    // Choose the appropriate star state based on which wave region we're in.
    // SL < 0 < SR (already checked). Sub-regions:
    //   SL  < 0 ≤ SL*  → left star
    //   SL* < 0 ≤ SM   → left double-star
    //   SM  < 0 ≤ SR*  → right double-star
    //   SR* < 0 ≤ SR   → right star
    // Compute flux as F_K + S_K (U*_K - U_K) at left-star boundary, etc.
    var Fout: AxisFlux;
    if (SLs >= 0.0) {
        // Left star.
        Fout.f_rho = FL.f_rho + SL * (rhoLs                - AL.rho);
        Fout.f_mn  = FL.f_mn  + SL * (rhoLs * SM           - AL.rho * AL.un);
        Fout.f_mt1 = FL.f_mt1 + SL * (rhoLs * ut1_Ls       - AL.rho * AL.ut1);
        Fout.f_mt2 = FL.f_mt2 + SL * (rhoLs * ut2_Ls       - AL.rho * AL.ut2);
        Fout.f_E   = FL.f_E   + SL * (E_Ls                 - AL.E);
        Fout.f_bt1 = FL.f_bt1 + SL * (bt1_Ls               - AL.bt1);
        Fout.f_bt2 = FL.f_bt2 + SL * (bt2_Ls               - AL.bt2);
    } else if (SM >= 0.0) {
        // Left double-star.
        // F** = F_L + S_L* (U** - U*_L) + S_L (U*_L - U_L)
        //     = F_L + S_L (U*_L - U_L) + S_L* (U** - U*_L)
        Fout.f_rho = FL.f_rho + SL  * (rhoLs              - AL.rho)
                              + SLs * (rhoLs              - rhoLs);  // ρ unchanged across Alfvén
        Fout.f_mn  = FL.f_mn  + SL  * (rhoLs * SM         - AL.rho * AL.un)
                              + SLs * (rhoLs * SM         - rhoLs * SM);
        Fout.f_mt1 = FL.f_mt1 + SL  * (rhoLs * ut1_Ls     - AL.rho * AL.ut1)
                              + SLs * (rhoLs * ut1_ss     - rhoLs * ut1_Ls);
        Fout.f_mt2 = FL.f_mt2 + SL  * (rhoLs * ut2_Ls     - AL.rho * AL.ut2)
                              + SLs * (rhoLs * ut2_ss     - rhoLs * ut2_Ls);
        Fout.f_E   = FL.f_E   + SL  * (E_Ls               - AL.E)
                              + SLs * (E_Lss              - E_Ls);
        Fout.f_bt1 = FL.f_bt1 + SL  * (bt1_Ls             - AL.bt1)
                              + SLs * (bt1_ss             - bt1_Ls);
        Fout.f_bt2 = FL.f_bt2 + SL  * (bt2_Ls             - AL.bt2)
                              + SLs * (bt2_ss             - bt2_Ls);
    } else if (SRs >= 0.0) {
        // Right double-star.
        Fout.f_rho = FR.f_rho + SR  * (rhoRs              - AR.rho)
                              + SRs * (rhoRs              - rhoRs);
        Fout.f_mn  = FR.f_mn  + SR  * (rhoRs * SM         - AR.rho * AR.un)
                              + SRs * (rhoRs * SM         - rhoRs * SM);
        Fout.f_mt1 = FR.f_mt1 + SR  * (rhoRs * ut1_Rs     - AR.rho * AR.ut1)
                              + SRs * (rhoRs * ut1_ss     - rhoRs * ut1_Rs);
        Fout.f_mt2 = FR.f_mt2 + SR  * (rhoRs * ut2_Rs     - AR.rho * AR.ut2)
                              + SRs * (rhoRs * ut2_ss     - rhoRs * ut2_Rs);
        Fout.f_E   = FR.f_E   + SR  * (E_Rs               - AR.E)
                              + SRs * (E_Rss              - E_Rs);
        Fout.f_bt1 = FR.f_bt1 + SR  * (bt1_Rs             - AR.bt1)
                              + SRs * (bt1_ss             - bt1_Rs);
        Fout.f_bt2 = FR.f_bt2 + SR  * (bt2_Rs             - AR.bt2)
                              + SRs * (bt2_ss             - bt2_Rs);
    } else {
        // Right star.
        Fout.f_rho = FR.f_rho + SR * (rhoRs                - AR.rho);
        Fout.f_mn  = FR.f_mn  + SR * (rhoRs * SM           - AR.rho * AR.un);
        Fout.f_mt1 = FR.f_mt1 + SR * (rhoRs * ut1_Rs       - AR.rho * AR.ut1);
        Fout.f_mt2 = FR.f_mt2 + SR * (rhoRs * ut2_Rs       - AR.rho * AR.ut2);
        Fout.f_E   = FR.f_E   + SR * (E_Rs                 - AR.E);
        Fout.f_bt1 = FR.f_bt1 + SR * (bt1_Rs               - AR.bt1);
        Fout.f_bt2 = FR.f_bt2 + SR * (bt2_Rs               - AR.bt2);
    }

    let pf = pack_flux(Fout, axis);
    flux_0[idx_c] = pf.f0;
    flux_1[idx_c] = vec4<f32>(pf.f1.x, pf.f1.y, pf.fBt1, pf.fBt2);
}

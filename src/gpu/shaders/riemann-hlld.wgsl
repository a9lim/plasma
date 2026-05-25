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
// Phase 4: drops wrap_idx in favour of direct indexing into ghost-padded
// buffers. Each invocation writes the flux at the LEFT face of cell
// (ix, iy) — index `bx_face_idx(ix, iy)` for x-sweep, `by_face_idx(ix, iy)`
// for y-sweep — using QL from the right edge of cell (ix-1, iy) and QR
// from the left edge of cell (ix, iy). The flux buffer itself is sized
// (n_total × n_total) like the cell-centered arrays; we just don't write
// the last column / row for the relevant sweep.
//
// Dispatch range (sweep_dir == 0):
//   ix ∈ [ghost, ghost + n_interior + 1)   — N+1 x-faces per row
//   iy ∈ [ghost, ghost + n_interior)       — N rows
// Dispatch range (sweep_dir == 1):
//   ix ∈ [ghost, ghost + n_interior)       — N cols
//   iy ∈ [ghost, ghost + n_interior + 1)   — N+1 y-faces per col
//
// Degenerate-branch handling unchanged from Phase 3b.
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

// Threshold below which the normal-B² triggers HLLD's Alfvén-degenerate
// fallback to HLLC. Originally 1e-24 (essentially "exactly machine zero"),
// but HANDOFF flagged that as too conservative: at thin current sheets,
// |Bn| ~ 1e-5 is small enough that the full 5-wave HLLD path has tiny
// denominators (rho·(S-u)² - bn² and S-S*), and the 1e-20 denominator
// guards inflate bt_Ls = bt·g/safeDL into huge values that NaN-cascade.
// 1e-10 means fall back to HLLC whenever |Bn| < ~1e-5·√ρ — robust at
// near-degenerate sheets, no visible effect on bulk physics.
const HLLD_BX_EPS2: f32 = 1.0e-10;
const HLLD_WS_TOL:  f32 = 1.0e-8;

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

struct AxisState {
    rho: f32, un:  f32, ut1: f32, ut2: f32,
    bn:  f32, bt1: f32, bt2: f32,
    p:   f32, pT:  f32, E:   f32,
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

struct AxisFlux {
    f_rho: f32, f_mn: f32, f_mt1: f32, f_mt2: f32,
    f_E:   f32, f_bt1: f32, f_bt2: f32,
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

struct PackedFlux {
    f0:   vec4<f32>,
    f1:   vec4<f32>,
    fBt1: f32,
    fBt2: f32,
};

fn pack_flux(F: AxisFlux, axis: u32) -> PackedFlux {
    var P: PackedFlux;
    if (axis == 0u) {
        P.f0 = vec4<f32>(F.f_rho, F.f_mn, F.f_mt1, F.f_mt2);
        P.f1 = vec4<f32>(F.f_E,   F.f_bt2, 0.0, 0.0);
        P.fBt1 = F.f_bt1;
    } else {
        P.f0 = vec4<f32>(F.f_rho, F.f_mt1, F.f_mn, F.f_mt2);
        P.f1 = vec4<f32>(F.f_E,   F.f_bt2, 0.0, 0.0);
        P.fBt1 = F.f_bt1;
    }
    P.fBt2 = 0.0;
    return P;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let axis       = U_uniforms.sweep_dir;
    let g          = U_uniforms.gamma;

    // Dispatch shape: extended by one row/col on the transverse axis so
    // that compute-emf can read flux from cells one row/col outside the
    // interior on each side.
    //   x-sweep:  ix ∈ [ghost, ghost+N+1)         — N+1 x-faces
    //             iy ∈ [ghost-1, ghost+N+1)       — N+2 rows
    //   y-sweep:  ix ∈ [ghost-1, ghost+N+1)       — N+2 cols
    //             iy ∈ [ghost, ghost+N+1)         — N+1 y-faces
    var x_extent: u32;
    var y_extent: u32;
    if (axis == 0u) {
        x_extent = n_interior + 1u;
        y_extent = n_interior + 2u;
    } else {
        x_extent = n_interior + 2u;
        y_extent = n_interior + 1u;
    }
    if (gid.x >= x_extent || gid.y >= y_extent) { return; }

    // ix/iy is the cell on the HIGH side of the face. Offset depends on
    // sweep axis: for x-sweep, faces start at ix=ghost; rows extend by
    // one on each side.
    var ix: u32;
    var iy: u32;
    if (axis == 0u) {
        ix = ghost + gid.x;
        iy = ghost + gid.y - 1u;
    } else {
        ix = ghost + gid.x - 1u;
        iy = ghost + gid.y;
    }

    // Source PPM edges: QL = right edge of cell (i-1, j) for x-sweep,
    // QR = left edge of cell (i, j).
    var ix_l: u32 = ix;
    var iy_l: u32 = iy;
    if (axis == 0u) { ix_l = ix - 1u; } else { iy_l = iy - 1u; }
    let idx_l = cell_idx_total(ix_l, iy_l, n_total);
    let idx_r = cell_idx_total(ix,   iy,   n_total);

    // Face-normal B at this face.
    var b_normal: f32;
    if (axis == 0u) {
        b_normal = Bx_face[bx_face_idx(ix, iy, n_total)];
    } else {
        b_normal = By_face[by_face_idx(ix, iy, n_total)];
    }

    let QL = unpack_edge_prim(edge_r_0[idx_l], edge_r_1[idx_l], b_normal, axis);
    let QR = unpack_edge_prim(edge_l_0[idx_r], edge_l_1[idx_r], b_normal, axis);

    let AL = prim_to_axis_state(QL, axis, g);
    let AR = prim_to_axis_state(QR, axis, g);
    let FL = axis_flux(AL);
    let FR = axis_flux(AR);

    let cfL = fast_mag_speed(QL, g, axis);
    let cfR = fast_mag_speed(QR, g, axis);
    let SL  = min(AL.un - cfL, AR.un - cfR);
    let SR  = max(AL.un + cfL, AR.un + cfR);

    // Cell index of the FACE itself (the left face of cell idx_r). flux
    // arrays use the same cell-centered indexing scheme; we write at
    // idx_r (the cell on the "high" side of the face).
    let dst = idx_r;

    if (SL >= 0.0) {
        let pf = pack_flux(FL, axis);
        flux_0[dst] = pf.f0;
        flux_1[dst] = vec4<f32>(pf.f1.x, pf.f1.y, pf.fBt1, pf.fBt2);
        return;
    }
    if (SR <= 0.0) {
        let pf = pack_flux(FR, axis);
        flux_0[dst] = pf.f0;
        flux_1[dst] = vec4<f32>(pf.f1.x, pf.f1.y, pf.fBt1, pf.fBt2);
        return;
    }

    // Branch B: degenerate wave-speed coincidence → HLL fallback.
    if (SR - SL < HLLD_WS_TOL * (abs(SR) + abs(SL) + 1.0e-12)) {
        let h = hll_flux_mhd(QL, QR, axis, g);
        flux_0[dst] = h.f0;
        flux_1[dst] = vec4<f32>(h.f1.x, h.f1.y, h.fBt1, h.fBt2);
        return;
    }

    let rcL = AL.rho * (SL - AL.un);
    let rcR = AR.rho * (SR - AR.un);
    let SM_num = rcR * AR.un - rcL * AL.un - AR.pT + AL.pT;
    let SM_den = rcR - rcL;
    let SM = SM_num / select(SM_den, sign(SM_den) * 1.0e-12, abs(SM_den) < 1.0e-30);

    let pT_star = AL.pT + AL.rho * (SL - AL.un) * (SM - AL.un);

    // Branch C: negative star pressure → HLL fallback.
    if (pT_star <= PRESSURE_FLOOR) {
        let h = hll_flux_mhd(QL, QR, axis, g);
        flux_0[dst] = h.f0;
        flux_1[dst] = vec4<f32>(h.f1.x, h.f1.y, h.fBt1, h.fBt2);
        return;
    }

    // Branch A: Bx² < ε² · ρ → Alfvén waves degenerate, HLLC.
    let bn2 = b_normal * b_normal;
    let rho_scale = max(0.5 * (AL.rho + AR.rho), DENSITY_FLOOR);
    let branchA = bn2 < HLLD_BX_EPS2 * rho_scale;

    if (branchA) {
        let denom_L = min(SL - SM, -1.0e-20);
        let denom_R = max(SR - SM,  1.0e-20);
        let rhoLs = AL.rho * (SL - AL.un) / denom_L;
        let rhoRs = AR.rho * (SR - AR.un) / denom_R;

        let E_Ls = ((SL - AL.un) * AL.E - AL.pT * AL.un + pT_star * SM) / (SL - SM);
        let E_Rs = ((SR - AR.un) * AR.E - AR.pT * AR.un + pT_star * SM) / (SR - SM);
        var Fout: AxisFlux;
        if (SM >= 0.0) {
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
        flux_0[dst] = pf.f0;
        flux_1[dst] = vec4<f32>(pf.f1.x, pf.f1.y, pf.fBt1, pf.fBt2);
        return;
    }

    // Full HLLD 5-wave path.
    let dL = min(SL - SM, -1.0e-20);
    let dR = max(SR - SM,  1.0e-20);
    let rhoLs = AL.rho * (SL - AL.un) / dL;
    let rhoRs = AR.rho * (SR - AR.un) / dR;

    let denomL_raw = AL.rho * (SL - AL.un) * (SL - SM) - bn2;
    let denomR_raw = AR.rho * (SR - AR.un) * (SR - SM) - bn2;
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

    let vdotb_L  = AL.un  * b_normal + AL.ut1  * AL.bt1  + AL.ut2  * AL.bt2;
    let vdotb_R  = AR.un  * b_normal + AR.ut1  * AR.bt1  + AR.ut2  * AR.bt2;
    let vdotbLs  = SM     * b_normal + ut1_Ls  * bt1_Ls  + ut2_Ls  * bt2_Ls;
    let vdotbRs  = SM     * b_normal + ut1_Rs  * bt1_Rs  + ut2_Rs  * bt2_Rs;

    let E_Ls = ((SL - AL.un) * AL.E - AL.pT * AL.un + pT_star * SM
               + b_normal * (vdotb_L - vdotbLs)) / (SL - SM);
    let E_Rs = ((SR - AR.un) * AR.E - AR.pT * AR.un + pT_star * SM
               + b_normal * (vdotb_R - vdotbRs)) / (SR - SM);

    let absBn = abs(b_normal);
    let SLs = SM - absBn / sqrt(max(rhoLs, DENSITY_FLOOR));
    let SRs = SM + absBn / sqrt(max(rhoRs, DENSITY_FLOOR));

    let srL = sqrt(max(rhoLs, DENSITY_FLOOR));
    let srR = sqrt(max(rhoRs, DENSITY_FLOOR));
    let srSum = srL + srR;
    let sgnBn = select(-1.0, 1.0, b_normal >= 0.0);

    let ut1_ss = (srL * ut1_Ls + srR * ut1_Rs + (bt1_Rs - bt1_Ls) * sgnBn) / srSum;
    let ut2_ss = (srL * ut2_Ls + srR * ut2_Rs + (bt2_Rs - bt2_Ls) * sgnBn) / srSum;
    let bt1_ss = (srL * bt1_Rs + srR * bt1_Ls + srL * srR * (ut1_Rs - ut1_Ls) * sgnBn) / srSum;
    let bt2_ss = (srL * bt2_Rs + srR * bt2_Ls + srL * srR * (ut2_Rs - ut2_Ls) * sgnBn) / srSum;

    let vdotb_ss = SM * b_normal + ut1_ss * bt1_ss + ut2_ss * bt2_ss;
    let E_Lss = E_Ls - srL * sgnBn * (vdotbLs - vdotb_ss);
    let E_Rss = E_Rs + srR * sgnBn * (vdotbRs - vdotb_ss);

    var Fout: AxisFlux;
    if (SLs >= 0.0) {
        Fout.f_rho = FL.f_rho + SL * (rhoLs                - AL.rho);
        Fout.f_mn  = FL.f_mn  + SL * (rhoLs * SM           - AL.rho * AL.un);
        Fout.f_mt1 = FL.f_mt1 + SL * (rhoLs * ut1_Ls       - AL.rho * AL.ut1);
        Fout.f_mt2 = FL.f_mt2 + SL * (rhoLs * ut2_Ls       - AL.rho * AL.ut2);
        Fout.f_E   = FL.f_E   + SL * (E_Ls                 - AL.E);
        Fout.f_bt1 = FL.f_bt1 + SL * (bt1_Ls               - AL.bt1);
        Fout.f_bt2 = FL.f_bt2 + SL * (bt2_Ls               - AL.bt2);
    } else if (SM >= 0.0) {
        Fout.f_rho = FL.f_rho + SL  * (rhoLs              - AL.rho)
                              + SLs * (rhoLs              - rhoLs);
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
        Fout.f_rho = FR.f_rho + SR * (rhoRs                - AR.rho);
        Fout.f_mn  = FR.f_mn  + SR * (rhoRs * SM           - AR.rho * AR.un);
        Fout.f_mt1 = FR.f_mt1 + SR * (rhoRs * ut1_Rs       - AR.rho * AR.ut1);
        Fout.f_mt2 = FR.f_mt2 + SR * (rhoRs * ut2_Rs       - AR.rho * AR.ut2);
        Fout.f_E   = FR.f_E   + SR * (E_Rs                 - AR.E);
        Fout.f_bt1 = FR.f_bt1 + SR * (bt1_Rs               - AR.bt1);
        Fout.f_bt2 = FR.f_bt2 + SR * (bt2_Rs               - AR.bt2);
    }

    let pf = pack_flux(Fout, axis);
    flux_0[dst] = pf.f0;
    flux_1[dst] = vec4<f32>(pf.f1.x, pf.f1.y, pf.fBt1, pf.fBt2);
}

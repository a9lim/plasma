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
// Flux output layout (per face, two vec4<f32>):
//   flux_0 = (f_rho, f_mn, f_mt1, f_mt2)
//   flux_1 = (f_E,   f_bt2/f_by-or-bx, fBt1=±Ez, SM_face)
// The fourth component of flux_1 carries the HLLD contact-wave speed
// S_M (M&K 2005 eq 38) — the face-normal contact velocity. Consumed by
// compute-emf.wgsl as the upwind selector for the Gardiner-Stone 2005
// CT EMF (eqns 41-45). Computed once up-front from rcL/rcR so every
// branch (supersonic / HLL fallbacks / Branch A HLLC / full HLLD) can
// stamp it into flux_1.w with a consistent definition. Downstream
// update-conserved-weighted masks flux_1 with (1,1,0,0) so the SM
// channel costs zero in the conserved update — it only feeds CT.
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
//  11 sweep     (uniform SweepDir) — sweep_dir = 0 (x) or 1 (y)

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
@group(0) @binding(11) var<uniform>             sweep:    SweepDir;

// Dimensionless threshold below which the normal-B² triggers HLLD's
// Alfvén-degenerate fallback to HLLC. The test now reads
//   bn² < HLLD_BX_EPS2 · ρ_avg · ((SR - SL)/2)²
// where (SR-SL)/2 is the Davis-style fast-magnetosonic speed estimate
// from the wavespeed extrema. Both sides have units of (ρ · c²), so
// HLLD_BX_EPS2 is a pure dimensionless smallness parameter — the value
// 1e-10 means "fall back to HLLC whenever |Bn| is below ~1e-5 of the
// local Alfvén-equivalent magnetic field strength". HANDOFF Session 2
// #4 calibrated 1e-10 against the dimensionally-inconsistent form
// (bn² < ε² · ρ_avg), which absorbed an implicit c²-like factor — the
// trigger band may shift slightly under the corrected form, especially
// for Orszag-Tang and Harris-sheet runs. Smoke-test OT N=256/1024
// after this change.
const HLLD_BX_EPS2: f32 = 1.0e-10;
const HLLD_WS_TOL:  f32 = 1.0e-8;

fn finite_hlld(x: f32) -> bool {
    return (x == x) && (abs(x) < 1.0e30);
}

fn star_state_ok(rho: f32, E: f32) -> bool {
    return finite_hlld(rho) && finite_hlld(E) && rho > DENSITY_FLOOR && E > 0.0;
}

fn same_conserved_cell(a: u32, b: u32) -> bool {
    return all(U0_in[a] == U0_in[b]) && all(U1_in[a] == U1_in[b]);
}

fn unpack_edge_prim(edge0: vec4<f32>, edge1: vec4<f32>, b_normal: f32, axis: u32, p_floor: f32) -> MhdPrim {
    var Q: MhdPrim;
    Q.rho = max(edge0.x, DENSITY_FLOOR);
    Q.vx  = edge0.y;
    Q.vy  = edge0.z;
    Q.vz  = edge0.w;
    Q.p   = max(edge1.x, p_floor);
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

// Carries values the HLLD main path has already produced, so the HLL
// fallback can skip re-running fast_mag_speed / normal_velocity_mhd /
// mhd_flux. CL/CR (conservative pairs) and the tangential-B components
// used for the bt1/bt2 averaging are NOT pre-computed in AxisState form
// — hll_flux_mhd derives them from QL/QR/AL/AR inside, which is cheap
// compared to the cf / S / F recomputation we're skipping.
struct HllInputs {
    QL: MhdPrim,
    QR: MhdPrim,
    AL: AxisState,
    AR: AxisState,
    FL: AxisFlux,
    FR: AxisFlux,
    SL: f32,
    SR: f32,
};

fn hll_flux_mhd(in_: HllInputs, axis: u32, gamma: f32) -> HLLOut {
    let SL = in_.SL;
    let SR = in_.SR;
    let FL = in_.FL;
    let FR = in_.FR;

    var out: HLLOut;
    if (SL >= 0.0) {
        let pf = pack_flux(FL, axis);
        out.f0 = pf.f0; out.f1 = pf.f1;
        out.fBt1 = pf.fBt1; out.fBt2 = pf.fBt2;
        return out;
    }
    if (SR <= 0.0) {
        let pf = pack_flux(FR, axis);
        out.f0 = pf.f0; out.f1 = pf.f1;
        out.fBt1 = pf.fBt1; out.fBt2 = pf.fBt2;
        return out;
    }

    // Conservative pairs — not pre-computed by HLLD's AxisState path.
    let CL = prim_to_cons_pair(in_.QL, gamma, U_uniforms.pressure_floor);
    let CR = prim_to_cons_pair(in_.QR, gamma, U_uniforms.pressure_floor);

    // Build an AxisFlux for the HLL average, then route through
    // pack_flux to preserve the axis-dependent packing the call sites
    // expect. AL.bt1 / AL.bt2 already equal the select(QL.by, QL.bx,
    // axis==1u) / QL.bz components the old MhdFlux-based code used —
    // see prim_to_axis_state.
    let denom = max(SR - SL, 1.0e-12);
    let inv = 1.0 / denom;
    let dU0 = CR.U0 - CL.U0;
    let dU1 = CR.U1 - CL.U1;
    var Favg: AxisFlux;
    // CL.U0 = (rho, rho*vx, rho*vy, rho*vz). Map momentum components
    // back to axis-aligned (un, ut1, ut2):
    //   axis = 0 (x-sweep): mn = mx (dU0.y), mt1 = my (dU0.z), mt2 = mz (dU0.w)
    //   axis = 1 (y-sweep): mn = my (dU0.z), mt1 = mx (dU0.y), mt2 = mz (dU0.w)
    let dMn  = select(dU0.z, dU0.y, axis == 0u);
    let dMt1 = select(dU0.y, dU0.z, axis == 0u);
    let dMt2 = dU0.w;
    Favg.f_rho = (SR * FL.f_rho - SL * FR.f_rho + SL * SR * dU0.x) * inv;
    Favg.f_mn  = (SR * FL.f_mn  - SL * FR.f_mn  + SL * SR * dMn ) * inv;
    Favg.f_mt1 = (SR * FL.f_mt1 - SL * FR.f_mt1 + SL * SR * dMt1) * inv;
    Favg.f_mt2 = (SR * FL.f_mt2 - SL * FR.f_mt2 + SL * SR * dMt2) * inv;
    Favg.f_E   = (SR * FL.f_E   - SL * FR.f_E   + SL * SR * dU1.x) * inv;
    // bt1 in AxisState terms is AL.bt1 / AR.bt1 (= select(by, bx, axis==1u)).
    // bt2 is AL.bt2 / AR.bt2 (= bz on both axes).
    Favg.f_bt1 = (SR * FL.f_bt1 - SL * FR.f_bt1 + SL * SR * (in_.AR.bt1 - in_.AL.bt1)) * inv;
    Favg.f_bt2 = (SR * FL.f_bt2 - SL * FR.f_bt2 + SL * SR * (in_.AR.bt2 - in_.AL.bt2)) * inv;

    let pf = pack_flux(Favg, axis);
    out.f0 = pf.f0; out.f1 = pf.f1;
    out.fBt1 = pf.fBt1; out.fBt2 = pf.fBt2;
    return out;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let axis       = sweep.sweep_dir;
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

    let pf = U_uniforms.pressure_floor;
    // Periodic boundary faces should pair the two wrapped INTERIOR
    // parabolas, not the piecewise-constant reconstruction of the outer
    // ghost cell. `apply-bcs` copies periodic ghosts bit-for-bit from the
    // wrapped interior, so exact conserved-state equality is a cheap local
    // detector that avoids adding another storage binding for BcUniforms.
    var edge_l_src = idx_l;
    var edge_r_src = idx_r;
    if (axis == 0u) {
        if (ix == ghost) {
            let wrap_l = cell_idx_total(ghost + n_interior - 1u, iy, n_total);
            if (same_conserved_cell(idx_l, wrap_l)) { edge_l_src = wrap_l; }
        }
        if (ix == ghost + n_interior) {
            let wrap_r = cell_idx_total(ghost, iy, n_total);
            if (same_conserved_cell(idx_r, wrap_r)) { edge_r_src = wrap_r; }
        }
    } else {
        if (iy == ghost) {
            let wrap_l = cell_idx_total(ix, ghost + n_interior - 1u, n_total);
            if (same_conserved_cell(idx_l, wrap_l)) { edge_l_src = wrap_l; }
        }
        if (iy == ghost + n_interior) {
            let wrap_r = cell_idx_total(ix, ghost, n_total);
            if (same_conserved_cell(idx_r, wrap_r)) { edge_r_src = wrap_r; }
        }
    }

    let QL = unpack_edge_prim(edge_r_0[edge_l_src], edge_r_1[edge_l_src], b_normal, axis, pf);
    let QR = unpack_edge_prim(edge_l_0[edge_r_src], edge_l_1[edge_r_src], b_normal, axis, pf);

    let AL = prim_to_axis_state(QL, axis, g);
    let AR = prim_to_axis_state(QR, axis, g);
    let FL = axis_flux(AL);
    let FR = axis_flux(AR);

    let cfL = fast_mag_speed(QL, g, axis, pf);
    let cfR = fast_mag_speed(QR, g, axis, pf);
    let SL  = min(AL.un - cfL, AR.un - cfR);
    let SR  = max(AL.un + cfL, AR.un + cfR);

    // ── Contact-wave speed S_M (M&K 2005 eq 38) ──────────────────────
    // Computed up-front so every flux-write path can stash it into
    // flux_1.w. compute-emf.wgsl uses this as the upwind selector for
    // the Gardiner-Stone 2005 CT EMF — the sign of the contact velocity
    // at each face determines which adjacent cell's Ez_cell is upwind.
    //
    // Formula is identical to the HLL contact-velocity estimate, so
    // it's correct for HLL fallbacks (Branches B/C) as well as HLLD's
    // main path. For supersonic faces (SL>=0 or SR<=0) the entire
    // upstream is one side anyway; SM and the actual upstream velocity
    // coincide in the limit of vanishing jump and differ only by a
    // smooth fast-wave-speed correction otherwise — fine for the
    // upwind selector since only sign matters in practice.
    let rcL_pre = AL.rho * (SL - AL.un);
    let rcR_pre = AR.rho * (SR - AR.un);
    let SM_den_pre = rcR_pre - rcL_pre;
    let SM_face = (rcR_pre * AR.un - rcL_pre * AL.un - AR.pT + AL.pT)
                / select(SM_den_pre, sign(SM_den_pre) * 1.0e-12, abs(SM_den_pre) < 1.0e-30);

    // Cell index of the FACE itself (the left face of cell idx_r). flux
    // arrays use the same cell-centered indexing scheme; we write at
    // idx_r (the cell on the "high" side of the face).
    let dst = idx_r;

    if (SL >= 0.0) {
        let pfL = pack_flux(FL, axis);
        flux_0[dst] = pfL.f0;
        flux_1[dst] = vec4<f32>(pfL.f1.x, pfL.f1.y, pfL.fBt1, SM_face);
        return;
    }
    if (SR <= 0.0) {
        let pfR = pack_flux(FR, axis);
        flux_0[dst] = pfR.f0;
        flux_1[dst] = vec4<f32>(pfR.f1.x, pfR.f1.y, pfR.fBt1, SM_face);
        return;
    }

    // Branch B: degenerate wave-speed coincidence → HLL fallback.
    if (SR - SL < HLLD_WS_TOL * (abs(SR) + abs(SL) + 1.0e-12)) {
        var hin: HllInputs;
        hin.QL = QL; hin.QR = QR;
        hin.AL = AL; hin.AR = AR;
        hin.FL = FL; hin.FR = FR;
        hin.SL = SL; hin.SR = SR;
        let h = hll_flux_mhd(hin, axis, g);
        flux_0[dst] = h.f0;
        flux_1[dst] = vec4<f32>(h.f1.x, h.f1.y, h.fBt1, SM_face);
        return;
    }

    // Reuse the up-front SM_face (identical to M&K 2005 eq 38).
    let SM = SM_face;

    let pT_star = AL.pT + AL.rho * (SL - AL.un) * (SM - AL.un);

    // Branch C: negative star pressure → HLL fallback.
    if (pT_star <= pf) {
        var hin: HllInputs;
        hin.QL = QL; hin.QR = QR;
        hin.AL = AL; hin.AR = AR;
        hin.FL = FL; hin.FR = FR;
        hin.SL = SL; hin.SR = SR;
        let h = hll_flux_mhd(hin, axis, g);
        flux_0[dst] = h.f0;
        flux_1[dst] = vec4<f32>(h.f1.x, h.f1.y, h.fBt1, SM_face);
        return;
    }

    // Branch A: bn² < ε² · ρ_avg · ((SR-SL)/2)² → Alfvén waves degenerate, HLLC.
    // (SR-SL)/2 stands in for a representative fast-magnetosonic speed;
    // multiplying by ρ_avg gives the test the same units as bn² (energy
    // density), so HLLD_BX_EPS2 is dimensionless. See constant comment.
    let bn2 = b_normal * b_normal;
    let rho_scale = max(0.5 * (AL.rho + AR.rho), DENSITY_FLOOR);
    let half_dS = 0.5 * (SR - SL);
    let branchA = bn2 < HLLD_BX_EPS2 * rho_scale * half_dS * half_dS;

    if (branchA) {
        let denom_L = min(SL - SM, -1.0e-20);
        let denom_R = max(SR - SM,  1.0e-20);
        let rhoLs = AL.rho * (SL - AL.un) / denom_L;
        let rhoRs = AR.rho * (SR - AR.un) / denom_R;

        let E_Ls = ((SL - AL.un) * AL.E - AL.pT * AL.un + pT_star * SM) / (SL - SM);
        let E_Rs = ((SR - AR.un) * AR.E - AR.pT * AR.un + pT_star * SM) / (SR - SM);
        if (!(star_state_ok(rhoLs, E_Ls) && star_state_ok(rhoRs, E_Rs))) {
            var hin: HllInputs;
            hin.QL = QL; hin.QR = QR;
            hin.AL = AL; hin.AR = AR;
            hin.FL = FL; hin.FR = FR;
            hin.SL = SL; hin.SR = SR;
            let h = hll_flux_mhd(hin, axis, g);
            flux_0[dst] = h.f0;
            flux_1[dst] = vec4<f32>(h.f1.x, h.f1.y, h.fBt1, SM_face);
            return;
        }
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
        let pfA = pack_flux(Fout, axis);
        flux_0[dst] = pfA.f0;
        flux_1[dst] = vec4<f32>(pfA.f1.x, pfA.f1.y, pfA.fBt1, SM_face);
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

    if (!(finite_hlld(pT_star)
       && abs(denomL_raw) > 1.0e-20
       && abs(denomR_raw) > 1.0e-20
       && star_state_ok(rhoLs, E_Ls)
       && star_state_ok(rhoRs, E_Rs))) {
        var hin: HllInputs;
        hin.QL = QL; hin.QR = QR;
        hin.AL = AL; hin.AR = AR;
        hin.FL = FL; hin.FR = FR;
        hin.SL = SL; hin.SR = SR;
        let h = hll_flux_mhd(hin, axis, g);
        flux_0[dst] = h.f0;
        flux_1[dst] = vec4<f32>(h.f1.x, h.f1.y, h.fBt1, SM_face);
        return;
    }

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

    if (!(star_state_ok(rhoLs, E_Lss) && star_state_ok(rhoRs, E_Rss))) {
        var hin: HllInputs;
        hin.QL = QL; hin.QR = QR;
        hin.AL = AL; hin.AR = AR;
        hin.FL = FL; hin.FR = FR;
        hin.SL = SL; hin.SR = SR;
        let h = hll_flux_mhd(hin, axis, g);
        flux_0[dst] = h.f0;
        flux_1[dst] = vec4<f32>(h.f1.x, h.f1.y, h.fBt1, SM_face);
        return;
    }

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

    let pfH = pack_flux(Fout, axis);
    flux_0[dst] = pfH.f0;
    flux_1[dst] = vec4<f32>(pfH.f1.x, pfH.f1.y, pfH.fBt1, SM_face);
}

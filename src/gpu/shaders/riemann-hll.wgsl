// ─── riemann-hll.wgsl ────────────────────────────────────────────────
// Per-face HLL Riemann solver for 2.5D ideal MHD.
//
// Computes flux at the cell's high-side face for the active sweep
// direction (sweep_dir = 0 → face at (i+½,j); = 1 → face at (i,j+½)).
//
// Wave speed: fast magnetosonic for the face-normal direction
//   c_fast² = ½(c_s² + c_A²) + ½√((c_s² + c_A²)² − 4 c_s² c_An²)
// S_L = min(uL - cfL, uR - cfR),  S_R = max(uL + cfL, uR + cfR)
//
// HLL average state (Toro 10.21):
//   F = F_L                                       if 0 ≤ S_L
//     = (S_R F_L − S_L F_R + S_L S_R (U_R − U_L)) / (S_R − S_L)  otherwise
//     = F_R                                       if S_R ≤ 0
//
// L/R primitive face states come from PLM:
//   q_L = q_i     + 0.5·dx·σ_i
//   q_R = q_{i+1} − 0.5·dx·σ_{i+1}
//
// Normal-direction B at the face is the staggered face value directly
// (Bx_face for x-sweep, By_face for y-sweep) — guaranteed continuous,
// so q_L.B_normal = q_R.B_normal at the face.
//
// flux packing:
//   flux_0 = (Fρ, Fρvx, Fρvy, Fρvz)
//   flux_1 = (FE, FBz, fBt1, fBt2)
//     where fBt1 = vx·By - vy·Bx  on x-faces (= -Ez_face)
//                  vy·Bx - vx·By  on y-faces (= +Ez_face)
//     and fBt2 = transverse-B flux for Bz advection (carried for symmetry
//                / debug; the Bz update reads it from flux_*_1.y instead).
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0_in   (ro)
//   2 U1_in   (ro)
//   3 Bx_face (ro)
//   4 By_face (ro)
//   5 slopes_0 (ro)
//   6 slopes_1 (ro)
//   7 flux_0   (rw)
//   8 flux_1   (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:     array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:   array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:   array<f32>;
@group(0) @binding(5) var<storage, read>       slopes_0:  array<vec4<f32>>;
@group(0) @binding(6) var<storage, read>       slopes_1:  array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> flux_0:    array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> flux_1:    array<vec4<f32>>;

fn cell_prim_for_axis(ix: u32, iy: u32, n: u32, gamma: f32) -> MhdPrim {
    let idx = cell_index(ix, iy, n);
    let bx  = 0.5 * (Bx_face[bx_face_left_index(ix, iy, n)] + Bx_face[bx_face_right_index(ix, iy, n)]);
    let by  = 0.5 * (By_face[by_face_down_index(ix, iy, n)] + By_face[by_face_up_index(ix, iy, n)]);
    return cons_to_prim_mhd(U0_in[idx], U1_in[idx], bx, by, gamma);
}

// Apply the PLM slope reconstruction to one cell and assemble the
// primitive state at the face. `sign_dx` is +0.5*dx for the L state
// (extrapolate forward to the high face from the lower cell) or
// -0.5*dx for the R state (extrapolate backward to the low face from
// the upper cell). `axis` picks the transverse-B mapping in slopes_1.
fn reconstruct_face_prim(
    base: MhdPrim, s0: vec4<f32>, s1: vec4<f32>, sign_dx: f32, axis: u32,
) -> MhdPrim {
    var Q: MhdPrim;
    Q.rho = max(base.rho + sign_dx * s0.x, DENSITY_FLOOR);
    Q.vx  = base.vx  + sign_dx * s0.y;
    Q.vy  = base.vy  + sign_dx * s0.z;
    Q.vz  = base.vz  + sign_dx * s0.w;
    Q.p   = max(base.p + sign_dx * s1.x, PRESSURE_FLOOR);
    Q.bz  = base.bz  + sign_dx * s1.z;
    if (axis == 0u) {
        // x-sweep: slopes_1 holds (p, By, Bz, _). Bx is the face normal —
        // overridden by the caller with the face value.
        Q.bx  = base.bx;                       // unused; overwritten outside
        Q.by  = base.by + sign_dx * s1.y;
    } else {
        // y-sweep: slopes_1 holds (p, Bx, Bz, _).
        Q.bx  = base.bx + sign_dx * s1.y;
        Q.by  = base.by;
    }
    return Q;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i  = i32(n);
    let axis = U_uniforms.sweep_dir;
    let g    = U_uniforms.gamma;
    let dx   = U_uniforms.dx;
    let half = 0.5 * dx;

    let i  = i32(gid.x);
    let j  = i32(gid.y);
    var ip = i;
    var jp = j;
    if (axis == 0u) { ip = i + 1; } else { jp = j + 1; }

    let idx_c = cell_index(gid.x, gid.y, n);
    let ip_u  = wrap_idx(ip, n_i);
    let jp_u  = wrap_idx(jp, n_i);
    let idx_p = cell_index(ip_u, jp_u, n);

    let P_c = cell_prim_for_axis(gid.x, gid.y, n, g);
    let P_p = cell_prim_for_axis(ip_u,   jp_u,   n, g);
    let s0_c = slopes_0[idx_c]; let s1_c = slopes_1[idx_c];
    let s0_p = slopes_0[idx_p]; let s1_p = slopes_1[idx_p];

    var QL = reconstruct_face_prim(P_c, s0_c, s1_c,  half, axis);
    var QR = reconstruct_face_prim(P_p, s0_p, s1_p, -half, axis);

    // Override the normal-direction B at the face with the staggered face
    // value (continuous by construction; PLM extrapolation would break
    // ∇·B preservation if we used reconstructed values).
    var b_normal: f32;
    if (axis == 0u) {
        // face at (i+½, j) — owned by cell (i, j)
        b_normal = Bx_face[cell_index(gid.x, gid.y, n)];
        QL.bx = b_normal;
        QR.bx = b_normal;
    } else {
        // face at (i, j+½) — owned by cell (i, j)
        b_normal = By_face[cell_index(gid.x, gid.y, n)];
        QL.by = b_normal;
        QR.by = b_normal;
    }

    // Wave speeds — fast magnetosonic along the face normal.
    let cfL = fast_mag_speed(QL, g, axis);
    let cfR = fast_mag_speed(QR, g, axis);
    let uL  = normal_velocity_mhd(QL, axis);
    let uR  = normal_velocity_mhd(QR, axis);
    let SL  = min(uL - cfL, uR - cfR);
    let SR  = max(uL + cfL, uR + cfR);

    // L/R fluxes.
    let FL = mhd_flux(QL, g, axis);
    let FR = mhd_flux(QR, g, axis);

    // L/R conservative states for the HLL jump term. We only need U0 and
    // U1 — the face-normal B is shared (it's constant across the face),
    // so it doesn't enter (U_R − U_L). The transverse B in U is implicit
    // via the cell-centered Bz in U1; transverse-Bx/By are stored at
    // faces and updated separately by CT — they do not appear in U_cell.
    let CL = prim_to_cons_pair(QL, g);
    let CR = prim_to_cons_pair(QR, g);
    let UL_0 = CL.U0; let UL_1 = CL.U1;
    let UR_0 = CR.U0; let UR_1 = CR.U1;

    var F0: vec4<f32>;
    var F1: vec4<f32>;
    var fBt1: f32;
    var fBt2: f32;
    if (SL >= 0.0) {
        F0 = FL.f0; F1 = FL.f1;
        fBt1 = FL.f_bt1; fBt2 = FL.f_bt2;
    } else if (SR <= 0.0) {
        F0 = FR.f0; F1 = FR.f1;
        fBt1 = FR.f_bt1; fBt2 = FR.f_bt2;
    } else {
        let denom = max(SR - SL, 1.0e-12);
        F0 = (SR * FL.f0 - SL * FR.f0 + SL * SR * (UR_0 - UL_0)) / denom;
        F1 = (SR * FL.f1 - SL * FR.f1 + SL * SR * (UR_1 - UL_1)) / denom;
        // For transverse-B fluxes the HLL average uses the same wave
        // structure with the B "conserved" being the cell-centered
        // transverse value at L/R. Use the face-reconstructed values.
        let bt1L = select(QL.by, QL.bx, axis == 1u);
        let bt1R = select(QR.by, QR.bx, axis == 1u);
        let bt2L = QL.bz;
        let bt2R = QR.bz;
        fBt1 = (SR * FL.f_bt1 - SL * FR.f_bt1 + SL * SR * (bt1R - bt1L)) / denom;
        fBt2 = (SR * FL.f_bt2 - SL * FR.f_bt2 + SL * SR * (bt2R - bt2L)) / denom;
    }

    flux_0[idx_c] = F0;
    flux_1[idx_c] = vec4<f32>(F1.x, F1.y, fBt1, fBt2);
}

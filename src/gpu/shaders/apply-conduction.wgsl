// ─── apply-conduction.wgsl ────────────────────────────────────────────
// Anisotropic thermal conduction source term (Braginskii 1965 /
// Spitzer-Härm 1953 — restricted to the parallel-only piece for the
// breadth pass; cross-field κ_⊥ is exposed via conduction_iso_frac).
//
//   dE/dt |_cond  =  −∇ · q
//
// with the heat flux
//
//   q  =  κ_∥ · b̂ (b̂ · ∇T)  +  κ_⊥ · (∇T − b̂(b̂·∇T))
//
// = κ_∥ · ( (1−f) · b̂(b̂·∇T)  +  f · ∇T )   where f = κ_⊥/κ_∥.
//
// Saturated heat-flux limiter (Cowie & McKee 1977):
//
//   |q|  ≤  φ_sat · ρ c_s³           (φ_sat = conduction_sat_frac; 0 = no limit)
//
// applied as a smooth blend  q ← q · 1 / sqrt(1 + (|q|/q_sat)²).
//
// Integration: explicit forward Euler over the globally source-limited dt.
// The host compute-dt pass includes a conduction diffusion bound. This shader
// is split into compute_delta/apply_delta so all heat fluxes are evaluated
// from a frozen U/B state before U1.E is mutated.
//
// Discretization: cell-centered T from cons_to_prim_mhd, face-centered
// ∇T via central difference between the two neighbour cell values, then
// the divergence is computed as a centered difference of the face fluxes
// over the cell.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0       (ro) — ρ, momentum (for KE in cons→prim)
//   2 U1       (rw) — read for T in compute_delta, written in apply_delta
//   3 Bx_face  (ro)
//   4 By_face  (ro)
//   5 dt_buf   (uniform)
//   6 dE       (rw) — one scalar energy delta per cell

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:    array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:    array<f32>;
@group(0) @binding(5) var<uniform>             dt_buf:     DtUniform;
@group(0) @binding(6) var<storage, read_write> dE_cond:    array<f32>;

// Cell-centered temperature T = p / ρ in code units.
fn cell_T(ix: u32, iy: u32, n_total: u32, p_floor: f32, gamma: f32) -> f32 {
    let c = cell_idx_total(ix, iy, n_total);
    let u0 = U0[c];
    let u1 = U1[c];
    let rho = max(u0.x, DENSITY_FLOOR);
    let bx_c = 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                    + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
    let by_c = 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                    + By_face[by_face_idx(ix, iy + 1u, n_total)]);
    let ke = 0.5 * (u0.y*u0.y + u0.z*u0.z + u0.w*u0.w) / rho;
    let mb = 0.5 * (bx_c*bx_c + by_c*by_c + u1.y*u1.y);
    let p  = max((gamma - 1.0) * (u1.x - ke - mb), p_floor);
    return p / rho;
}

fn cell_bx(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
}
fn cell_by(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                + By_face[by_face_idx(ix, iy + 1u, n_total)]);
}

fn heat_flux_sat_factor(qx: f32, qy: f32, rho_face: f32, T_face: f32, gamma: f32) -> f32 {
    let phi_sat = U_uniforms.conduction_sat_frac;
    if (phi_sat <= 0.0) { return 1.0; }

    // Cowie-McKee saturated flux in code units. Since T = p/ρ, c_s² = γT.
    let cs = sqrt(max(gamma * T_face, 0.0));
    let q_sat = max(phi_sat * max(rho_face, DENSITY_FLOOR) * cs * cs * cs, 1.0e-30);
    let q_mag = sqrt(max(qx*qx + qy*qy, 0.0));
    return 1.0 / sqrt(1.0 + (q_mag / q_sat) * (q_mag / q_sat));
}

// Compute the x-component of q at the LEFT face of cell (ix, iy) —
// i.e., at the face shared with cell (ix-1, iy).
fn q_x_face(ix: u32, iy: u32, n_total: u32, p_floor: f32, gamma: f32) -> f32 {
    let kappa = U_uniforms.conduction_kappa;
    if (kappa <= 0.0) { return 0.0; }
    let f_iso = U_uniforms.conduction_iso_frac;
    let dx    = U_uniforms.dx;

    let T_l = cell_T(ix - 1u, iy, n_total, p_floor, gamma);
    let T_r = cell_T(ix,      iy, n_total, p_floor, gamma);
    // Face-centered ∇T using the two neighbours straddling the face.
    let dTdx = (T_r - T_l) / dx;
    // Transverse derivative ∂T/∂y at the face — average of the two
    // adjacent cell-centered values.
    let T_dl = cell_T(ix - 1u, iy - 1u, n_total, p_floor, gamma);
    let T_ul = cell_T(ix - 1u, iy + 1u, n_total, p_floor, gamma);
    let T_dr = cell_T(ix,      iy - 1u, n_total, p_floor, gamma);
    let T_ur = cell_T(ix,      iy + 1u, n_total, p_floor, gamma);
    let dTdy_l = (T_ul - T_dl) / (2.0 * dx);
    let dTdy_r = (T_ur - T_dr) / (2.0 * dx);
    let dTdy   = 0.5 * (dTdy_l + dTdy_r);

    // Face-centered B unit vector.
    let bx_face = Bx_face[bx_face_idx(ix, iy, n_total)];
    let by_l    = cell_by(ix - 1u, iy, n_total);
    let by_r    = cell_by(ix,      iy, n_total);
    let by_face = 0.5 * (by_l + by_r);
    let b_mag   = sqrt(bx_face*bx_face + by_face*by_face) + 1.0e-30;
    let bxh = bx_face / b_mag;
    let byh = by_face / b_mag;
    let b_dot_gT = bxh * dTdx + byh * dTdy;

    let qx_raw = -kappa * ((1.0 - f_iso) * bxh * b_dot_gT + f_iso * dTdx);
    let qy_raw = -kappa * ((1.0 - f_iso) * byh * b_dot_gT + f_iso * dTdy);
    let rho_l = max(U0[cell_idx_total(ix - 1u, iy, n_total)].x, DENSITY_FLOOR);
    let rho_r = max(U0[cell_idx_total(ix,      iy, n_total)].x, DENSITY_FLOOR);
    let rho_face = 0.5 * (rho_l + rho_r);
    let T_face = max(0.5 * (T_l + T_r), 0.0);
    return qx_raw * heat_flux_sat_factor(qx_raw, qy_raw, rho_face, T_face, gamma);
}

// Symmetric for the BOTTOM face — q_y.
fn q_y_face(ix: u32, iy: u32, n_total: u32, p_floor: f32, gamma: f32) -> f32 {
    let kappa = U_uniforms.conduction_kappa;
    if (kappa <= 0.0) { return 0.0; }
    let f_iso = U_uniforms.conduction_iso_frac;
    let dx    = U_uniforms.dx;

    let T_d = cell_T(ix, iy - 1u, n_total, p_floor, gamma);
    let T_u = cell_T(ix, iy,      n_total, p_floor, gamma);
    let dTdy = (T_u - T_d) / dx;
    // Transverse ∂T/∂x.
    let T_ld = cell_T(ix - 1u, iy - 1u, n_total, p_floor, gamma);
    let T_rd = cell_T(ix + 1u, iy - 1u, n_total, p_floor, gamma);
    let T_lu = cell_T(ix - 1u, iy,      n_total, p_floor, gamma);
    let T_ru = cell_T(ix + 1u, iy,      n_total, p_floor, gamma);
    let dTdx_d = (T_rd - T_ld) / (2.0 * dx);
    let dTdx_u = (T_ru - T_lu) / (2.0 * dx);
    let dTdx   = 0.5 * (dTdx_d + dTdx_u);

    let by_face = By_face[by_face_idx(ix, iy, n_total)];
    let bx_d    = cell_bx(ix, iy - 1u, n_total);
    let bx_u    = cell_bx(ix, iy,      n_total);
    let bx_face = 0.5 * (bx_d + bx_u);
    let b_mag   = sqrt(bx_face*bx_face + by_face*by_face) + 1.0e-30;
    let bxh = bx_face / b_mag;
    let byh = by_face / b_mag;
    let b_dot_gT = bxh * dTdx + byh * dTdy;

    let qx_raw = -kappa * ((1.0 - f_iso) * bxh * b_dot_gT + f_iso * dTdx);
    let qy_raw = -kappa * ((1.0 - f_iso) * byh * b_dot_gT + f_iso * dTdy);
    let rho_d = max(U0[cell_idx_total(ix, iy - 1u, n_total)].x, DENSITY_FLOOR);
    let rho_u = max(U0[cell_idx_total(ix, iy,      n_total)].x, DENSITY_FLOOR);
    let rho_face = 0.5 * (rho_d + rho_u);
    let T_face = max(0.5 * (T_d + T_u), 0.0);
    return qy_raw * heat_flux_sat_factor(qx_raw, qy_raw, rho_face, T_face, gamma);
}

@compute @workgroup_size(8, 8, 1)
fn compute_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_CONDUCTION)) { return; }
    if (U_uniforms.conduction_kappa <= 0.0)                  { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    let p_floor = U_uniforms.pressure_floor;
    let gamma   = U_uniforms.gamma;
    let dx      = U_uniforms.dx;
    let dt      = dt_buf.dt;

    // Heat-flux divergence: ∇·q = (q_x[i+1] − q_x[i])/dx + (q_y[j+1] − q_y[j])/dx
    let qxL = q_x_face(ix,      iy, n_total, p_floor, gamma);
    let qxR = q_x_face(ix + 1u, iy, n_total, p_floor, gamma);
    let qyD = q_y_face(ix, iy,      n_total, p_floor, gamma);
    let qyU = q_y_face(ix, iy + 1u, n_total, p_floor, gamma);
    let divq = (qxR - qxL + qyU - qyD) / dx;

    // dE/dt = -∇·q. (Heat flux is the energy flux; divergence subtracts from local energy.)
    let dE = -divq * dt;

    dE_cond[c] = dE;
}

@compute @workgroup_size(8, 8, 1)
fn apply_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_CONDUCTION)) { return; }
    if (U_uniforms.conduction_kappa <= 0.0)                  { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    let u1 = U1[c];
    U1[c] = vec4<f32>(u1.x + dE_cond[c], u1.y, u1.z, u1.w);
}

// ─── view-field.wgsl ─────────────────────────────────────────────────
// Extract a scalar field from MHD state into a flat f32 buffer for
// downstream colormapping. View mode set via Uniforms.view_mode.
//
// Phase 4: reads from ghost-padded buffers; dispatches over interior
// cells only. The field buffer is sized (N+4)² for indexing compat
// with cell-centered storage; only interior cells are written.
//
// Session 15 extends the view set with three extended-physics fields:
//   5 (VIEW_T)    — temperature T = p / ρ
//   6 (VIEW_QMAG) — anisotropic Spitzer heat-flux magnitude |q|
//   7 (VIEW_PHI)  — gravitational potential φ (Poisson solve output)
//   8 (VIEW_ENTROPY) — dual-energy entropy proxy K = p / ρ^γ
//
// |q| is computed at cell center from a local cell-centered ∇T and the
// local b̂. The conduction shader uses face-centered fluxes for the
// divergence; this view mode is intentionally simpler — it shows where
// heat flux is large, not where energy is being deposited.
//
// φ reads the most recently written Poisson buffer (`phi`). With the
// default gravity_poisson_iters = 30 (even), the final iterate lands
// in `phi`. Odd iterations would leave it one Jacobi step stale —
// acceptable for visualization.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0_in    (ro)
//   2 U1_in    (ro)
//   3 Bx_face  (ro)
//   4 By_face  (ro)
//   5 field    (rw)
//   6 phi      (ro) — Poisson potential for VIEW_PHI

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:    array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:  array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:  array<f32>;
@group(0) @binding(5) var<storage, read_write> field:    array<f32>;
@group(0) @binding(6) var<storage, read>       phi:      array<f32>;

fn bx_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_left_idx(ix, iy, n_total)]
                + Bx_face[bx_face_right_idx(ix, iy, n_total)]);
}
fn by_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_down_idx(ix, iy, n_total)]
                + By_face[by_face_up_idx(ix, iy, n_total)]);
}

// Cell-centered T = p / ρ at (ix, iy). Reuses the same KE / magnetic
// subtraction as primitive recovery but divides by ρ at the end. The
// dual-energy fallback in pressure_from_dual_energy is intentionally used
// here so visualized temperature matches the state the Riemann solver sees.
fn cell_temp(ix: u32, iy: u32, n_total: u32) -> f32 {
    let idx = cell_idx_total(ix, iy, n_total);
    let U0 = U0_in[idx];
    let U1 = U1_in[idx];
    let rho = max(U0.x, 1.0e-6);
    let bx_c = bx_at(ix, iy, n_total);
    let by_c = by_at(ix, iy, n_total);
    let p = pressure_from_dual_energy(U0, U1, bx_c, by_c,
                                      U_uniforms.gamma, U_uniforms.pressure_floor);
    return p / rho;
}

fn heat_flux_sat_factor_view(qx: f32, qy: f32, rho: f32, T: f32) -> f32 {
    let phi_sat = U_uniforms.conduction_sat_frac;
    if (phi_sat <= 0.0) { return 1.0; }
    let cs = sqrt(max(U_uniforms.gamma * T, 0.0));
    let q_sat = max(phi_sat * max(rho, DENSITY_FLOOR) * cs * cs * cs, 1.0e-30);
    let q_mag = sqrt(max(qx*qx + qy*qy, 0.0));
    return 1.0 / sqrt(1.0 + (q_mag / q_sat) * (q_mag / q_sat));
}

fn transport_scale_view(theta: f32) -> f32 {
    // Same default transport closure used by compute-dt. The render
    // diagnostic intentionally avoids another table binding.
    return pow(max(theta, 1.0e-30), 2.5);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;

    let idx_c = cell_idx_total(ix, iy, n_total);
    let bx_c  = bx_at(ix, iy, n_total);
    let by_c  = by_at(ix, iy, n_total);

    let U0 = U0_in[idx_c];
    let U1 = U1_in[idx_c];
    let rho = max(U0.x, 1.0e-6);

    let mode = U_uniforms.view_mode;
    var v: f32 = 0.0;
    if (mode == 0u) {
        v = U0.x;                                                      // ρ
    } else if (mode == 1u) {
        v = pressure_from_dual_energy(U0, U1, bx_c, by_c,
                                      U_uniforms.gamma, U_uniforms.pressure_floor); // p
    } else if (mode == 2u) {
        let vx = U0.y / rho;
        let vy = U0.z / rho;
        let vz = U0.w / rho;
        v = sqrt(vx*vx + vy*vy + vz*vz);                              // |v|
    } else if (mode == 3u) {
        v = sqrt(bx_c*bx_c + by_c*by_c + U1.y*U1.y);                  // |B|
    } else if (mode == 4u) {
        // Jz = ∂By/∂x - ∂Bx/∂y via central differences. Ghost cells
        // adjacent to interior are filled by apply-bcs, so these
        // neighbour reads are always in-range.
        let by_cR = by_at(ix + 1u, iy,      n_total);
        let by_cL = by_at(ix - 1u, iy,      n_total);
        let bx_cU = bx_at(ix,      iy + 1u, n_total);
        let bx_cD = bx_at(ix,      iy - 1u, n_total);
        let dby_dx = (by_cR - by_cL) / (2.0 * U_uniforms.dx);
        let dbx_dy = (bx_cU - bx_cD) / (2.0 * U_uniforms.dx);
        v = dby_dx - dbx_dy;                                           // Jz
    } else if (mode == 5u) {
        v = cell_temp(ix, iy, n_total);                                // T = p/ρ
    } else if (mode == 6u) {
        // |q| anisotropic Spitzer heat flux magnitude.
        //   q = κ_∥ b̂(b̂·∇T) + κ_⊥ (∇T − b̂(b̂·∇T))
        // ∇T at cell center via central diffs of neighbour cell T.
        let TR = cell_temp(ix + 1u, iy,      n_total);
        let TL = cell_temp(ix - 1u, iy,      n_total);
        let TU = cell_temp(ix,      iy + 1u, n_total);
        let TD = cell_temp(ix,      iy - 1u, n_total);
        let inv_2dx = 1.0 / (2.0 * U_uniforms.dx);
        let dTx = (TR - TL) * inv_2dx;
        let dTy = (TU - TD) * inv_2dx;
        let bz_c = U1.y;
        let b2 = bx_c*bx_c + by_c*by_c + bz_c*bz_c;
        let inv_bmag = 1.0 / sqrt(max(b2, 1.0e-30));
        let bhat_x = bx_c * inv_bmag;
        let bhat_y = by_c * inv_bmag;
        let b_dot_gradT = bhat_x * dTx + bhat_y * dTy;
        let theta_c = cell_temp(ix, iy, n_total) / max(U_uniforms.cooling_T_ref, 1.0e-30);
        let kappa_par = U_uniforms.conduction_kappa * transport_scale_view(theta_c);
        let kappa_per = kappa_par * U_uniforms.conduction_iso_frac;
        // q_par along b̂; q_perp the component of ∇T orthogonal to b̂.
        let qpar_x = kappa_par * bhat_x * b_dot_gradT;
        let qpar_y = kappa_par * bhat_y * b_dot_gradT;
        let qperp_x = kappa_per * (dTx - bhat_x * b_dot_gradT);
        let qperp_y = kappa_per * (dTy - bhat_y * b_dot_gradT);
        var qx = -(qpar_x + qperp_x);  // q = -κ ∇T form
        var qy = -(qpar_y + qperp_y);
        let sat = heat_flux_sat_factor_view(qx, qy, rho, cell_temp(ix, iy, n_total));
        qx = qx * sat;
        qy = qy * sat;
        v = sqrt(qx*qx + qy*qy);
    } else if (mode == 7u) {
        // φ — Poisson potential. Reads the canonical phi buffer; with
        // the default even iters count, this is the final Jacobi
        // iterate from the most recent _encodeExtendedPhysics call.
        v = phi[idx_c];
    } else {
        let p = pressure_from_dual_energy(U0, U1, bx_c, by_c,
                                          U_uniforms.gamma, U_uniforms.pressure_floor);
        v = entropy_proxy(rho, p, U_uniforms.gamma, U_uniforms.pressure_floor);
    }
    field[idx_c] = v;
}

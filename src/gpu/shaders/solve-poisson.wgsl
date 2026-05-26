// ─── solve-poisson.wgsl ───────────────────────────────────────────────
// Jacobi iterator for the gravitational potential ∇²φ = 4πG·(ρ − ρ̄)
// on the cell-centered ghost-padded grid. ρ̄ is the volume-averaged
// density — subtracting it makes the periodic Poisson problem
// well-posed (otherwise the constant mode is unconstrained / drifts
// to infinity).
//
// Two entry points share the same bind group:
//   `reset_avg`     writes ρ̄ into rho_mean[0] (single 1×1 dispatch).
//                   For the breadth pass we just take the corner sample
//                   ρ(ghost, ghost) as a stand-in for the spatial mean.
//                   A proper reduction is the obvious follow-up.
//   `iterate`       one Jacobi sweep:
//                       φ_next[i,j] = ¼(φ[i±1,j] + φ[i,j±1] − dx²·4πG·ρ')
//                   with ρ' = ρ − ρ̄. Caller runs this `gravity_poisson_iters`
//                   times, ping-ponging phi ↔ phi_next via the bind group.
//
// Boundary handling for the breadth pass: this kernel assumes periodic
// wrap via the existing ghost cells (apply-bcs.wgsl is expected to
// have filled the ghosts before the Poisson solve runs). For non-
// periodic BCs the right move is Dirichlet φ = 0 at the boundary
// (zero potential far from the mass); that's a follow-up.
//
// Bindings:
//   0 uniforms     (uniform)
//   1 U0           (ro) — for ρ in source
//   2 phi_in       (ro) — current iterate
//   3 phi_out      (rw) — next iterate
//   4 rho_mean     (rw) — single-f32 holding ρ̄ for the periodic compatibility
//                         condition

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       phi_in:     array<f32>;
@group(0) @binding(3) var<storage, read_write> phi_out:    array<f32>;
@group(0) @binding(4) var<storage, read_write> rho_mean:   array<f32>;

@compute @workgroup_size(1, 1, 1)
fn reset_avg(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Stand-in for a proper reduction: sample the center cell.
    let n_total = U_uniforms.grid_n_total;
    let ghost   = U_uniforms.ghost_w;
    let cx      = ghost + U_uniforms.grid_n / 2u;
    let cy      = ghost + U_uniforms.grid_n / 2u;
    rho_mean[0] = U0[cell_idx_total(cx, cy, n_total)].x;
}

@compute @workgroup_size(8, 8, 1)
fn iterate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_GRAVITY_SELF)) { return; }
    if (U_uniforms.gravity_G <= 0.0)                           { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    // Source: 4πG · (ρ − ρ̄)
    let rho     = U0[c].x;
    let rho_bar = rho_mean[0];
    let rhs     = 4.0 * 3.14159265358979 * U_uniforms.gravity_G * (rho - rho_bar);

    // Five-point Laplacian. Reads from phi_in across neighbour cells
    // (ghosts are assumed to carry the periodic image — apply-bcs ran).
    let phi_l = phi_in[cell_idx_total(ix - 1u, iy,      n_total)];
    let phi_r = phi_in[cell_idx_total(ix + 1u, iy,      n_total)];
    let phi_d = phi_in[cell_idx_total(ix,      iy - 1u, n_total)];
    let phi_u = phi_in[cell_idx_total(ix,      iy + 1u, n_total)];

    let dx = U_uniforms.dx;
    // (φ_l + φ_r − 2φ_c)/dx² + (φ_d + φ_u − 2φ_c)/dx² = rhs
    // ⇒ φ_c = ¼(φ_l + φ_r + φ_d + φ_u − dx² · rhs)
    phi_out[c] = 0.25 * (phi_l + phi_r + phi_d + phi_u - dx * dx * rhs);
}

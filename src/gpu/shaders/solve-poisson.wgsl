// ─── solve-poisson.wgsl ───────────────────────────────────────────────
// Periodic Jacobi iterator for the gravitational potential
//
//   ∇²φ = 4πG · (ρ − ρ̄)
//
// on the cell-centered grid. The mean density is reduced from the actual
// interior domain each step so the periodic Poisson compatibility condition
// is satisfied. φ itself is indexed periodically; it does not rely on ghost
// cells being filled for the potential.
//
// Entry points:
//   reduce_mean     one partial density sum per 8×8 tile.
//   finalize_mean   accumulates tile sums into rho_mean[0].
//   iterate         one periodic Jacobi sweep, ping-ponging phi ↔ phi_next.

@group(0) @binding(0) var<uniform>             U_uniforms:       Uniforms;
@group(0) @binding(1) var<storage, read>       U0:               array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       phi_in:           array<f32>;
@group(0) @binding(3) var<storage, read_write> phi_out:          array<f32>;
@group(0) @binding(4) var<storage, read_write> rho_mean:         array<f32>;
@group(0) @binding(5) var<storage, read_write> rho_mean_partials: array<f32>;

const POISSON_WG: u32 = 8u;
const POISSON_LANES: u32 = 64u;
const PI_POISSON: f32 = 3.141592653589793;

var<workgroup> rho_tile: array<f32, 64>;

fn periodic_cell_idx_from_gid(gx: u32, gy: u32, n_interior: u32, n_total: u32, ghost: u32) -> u32 {
    let ix = ghost + gx;
    let iy = ghost + gy;
    return cell_idx_total(ix, iy, n_total);
}

fn periodic_phi(ix: u32, iy: u32, n_interior: u32, n_total: u32, ghost: u32) -> f32 {
    let gx = (ix + n_interior - ghost) % n_interior;
    let gy = (iy + n_interior - ghost) % n_interior;
    return phi_in[periodic_cell_idx_from_gid(gx, gy, n_interior, n_total, ghost)];
}

@compute @workgroup_size(8, 8, 1)
fn reduce_mean(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    var rho: f32 = 0.0;
    if (gid.x < n_interior && gid.y < n_interior) {
        let c = periodic_cell_idx_from_gid(gid.x, gid.y, n_interior, n_total, ghost);
        rho = U0[c].x;
    }
    rho_tile[lid] = rho;
    workgroupBarrier();

    var stride: u32 = POISSON_LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            rho_tile[lid] = rho_tile[lid] + rho_tile[lid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid == 0u) {
        let tiles = (n_interior + POISSON_WG - 1u) / POISSON_WG;
        rho_mean_partials[wid.y * tiles + wid.x] = rho_tile[0];
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize_mean() {
    let n_interior = U_uniforms.grid_n;
    let tiles = (n_interior + POISSON_WG - 1u) / POISSON_WG;
    let tile_count = tiles * tiles;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < tile_count; i = i + 1u) {
        sum = sum + rho_mean_partials[i];
    }
    let cells = f32(n_interior * n_interior);
    rho_mean[0] = sum / max(cells, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn iterate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_GRAVITY_SELF)) { return; }
    if (U_uniforms.gravity_G <= 0.0)                            { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    let rho     = U0[c].x;
    let rho_bar = rho_mean[0];
    let rhs     = 4.0 * PI_POISSON * U_uniforms.gravity_G * (rho - rho_bar);

    let gx_l = (gid.x + n_interior - 1u) % n_interior;
    let gx_r = (gid.x + 1u) % n_interior;
    let gy_d = (gid.y + n_interior - 1u) % n_interior;
    let gy_u = (gid.y + 1u) % n_interior;
    let phi_l = phi_in[periodic_cell_idx_from_gid(gx_l, gid.y, n_interior, n_total, ghost)];
    let phi_r = phi_in[periodic_cell_idx_from_gid(gx_r, gid.y, n_interior, n_total, ghost)];
    let phi_d = phi_in[periodic_cell_idx_from_gid(gid.x, gy_d, n_interior, n_total, ghost)];
    let phi_u = phi_in[periodic_cell_idx_from_gid(gid.x, gy_u, n_interior, n_total, ghost)];

    let dx = U_uniforms.dx;
    let softened = U_uniforms.gravity_softening > 0.0;
    let soft2 = select(0.0,
                       (dx / max(U_uniforms.gravity_softening, 1.0e-30))
                     * (dx / max(U_uniforms.gravity_softening, 1.0e-30)),
                       softened);
    let jacobi = (phi_l + phi_r + phi_d + phi_u - dx * dx * rhs)
               / max(4.0 + soft2, 1.0e-6);
    let omega = clamp(U_uniforms.gravity_poisson_omega, 0.05, 1.95);
    phi_out[c] = mix(phi_in[c], jacobi, omega);
}

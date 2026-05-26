// ─── apply-hall.wgsl ──────────────────────────────────────────────────
// Hall MHD correction to the induction equation:
//
//   E_H = (d_i / ρ) · (J × B − ∇p_e)
//   ∂B/∂t |_hall = -∇ × E_H
//
// The implementation is deliberately split into ordered dispatches:
//
//   compute_emf    evaluates corner-centered E_H from a frozen U/B state and
//                  stores the pre-Hall cell magnetic energy.
//   apply_update   applies the CT curl to face B and the 2.5D Bz update.
//   repair_energy  adds Δ(½|B|²) to total energy so the Hall B update does
//                  not appear as spurious thermal heating/cooling.
//
// This is still an explicit Hall update, but the race-prone in-place EMF
// evaluation is gone; all stencils read immutable scratch within a dispatch.

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Bx_face:    array<f32>;
@group(0) @binding(3) var<storage, read_write> By_face:    array<f32>;
@group(0) @binding(4) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(5) var<uniform>             dt_buf:     DtUniform;
@group(0) @binding(6) var<storage, read_write> hall_E:     array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> hall_mb0:   array<f32>;

struct CornerJB {
    Jx: f32, Jy: f32, Jz: f32,
    Bx: f32, By: f32, Bz: f32,
    rho: f32,
};

fn cell_magnetic_energy(ix: u32, iy: u32, n_total: u32) -> f32 {
    let u1 = U1[cell_idx_total(ix, iy, n_total)];
    let bx_c = 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                    + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
    let by_c = 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                    + By_face[by_face_idx(ix, iy + 1u, n_total)]);
    return 0.5 * (bx_c * bx_c + by_c * by_c + u1.y * u1.y);
}

fn cell_pressure_hall(ix: u32, iy: u32, n_total: u32) -> f32 {
    let u0 = U0[cell_idx_total(ix, iy, n_total)];
    let u1 = U1[cell_idx_total(ix, iy, n_total)];
    let rho = max(u0.x, DENSITY_FLOOR);
    let ke = 0.5 * (u0.y*u0.y + u0.z*u0.z + u0.w*u0.w) / rho;
    let bx_c = 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                    + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
    let by_c = 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                    + By_face[by_face_idx(ix, iy + 1u, n_total)]);
    let mb = 0.5 * (bx_c*bx_c + by_c*by_c + u1.y*u1.y);
    return max((U_uniforms.gamma - 1.0) * (u1.x - ke - mb), U_uniforms.pressure_floor);
}

fn corner_jb(ix: u32, iy: u32, n_total: u32) -> CornerJB {
    let dx = U_uniforms.dx;
    var R: CornerJB;

    R.Bx = 0.5 * (Bx_face[bx_face_idx(ix, iy - 1u, n_total)]
                + Bx_face[bx_face_idx(ix, iy,      n_total)]);
    R.By = 0.5 * (By_face[by_face_idx(ix - 1u, iy, n_total)]
                + By_face[by_face_idx(ix,      iy, n_total)]);

    let bz_sw = U1[cell_idx_total(ix - 1u, iy - 1u, n_total)].y;
    let bz_se = U1[cell_idx_total(ix,      iy - 1u, n_total)].y;
    let bz_nw = U1[cell_idx_total(ix - 1u, iy,      n_total)].y;
    let bz_ne = U1[cell_idx_total(ix,      iy,      n_total)].y;
    R.Bz = 0.25 * (bz_sw + bz_se + bz_nw + bz_ne);

    let rho_sw = U0[cell_idx_total(ix - 1u, iy - 1u, n_total)].x;
    let rho_se = U0[cell_idx_total(ix,      iy - 1u, n_total)].x;
    let rho_nw = U0[cell_idx_total(ix - 1u, iy,      n_total)].x;
    let rho_ne = U0[cell_idx_total(ix,      iy,      n_total)].x;
    R.rho = max(0.25 * (rho_sw + rho_se + rho_nw + rho_ne), DENSITY_FLOOR);

    let by_l = By_face[by_face_idx(ix - 1u, iy, n_total)];
    let by_r = By_face[by_face_idx(ix,      iy, n_total)];
    let bx_d = Bx_face[bx_face_idx(ix, iy - 1u, n_total)];
    let bx_u = Bx_face[bx_face_idx(ix, iy,      n_total)];
    R.Jz = (by_r - by_l) / dx - (bx_u - bx_d) / dx;

    let bz_d_avg = 0.5 * (bz_sw + bz_se);
    let bz_u_avg = 0.5 * (bz_nw + bz_ne);
    let bz_l_avg = 0.5 * (bz_sw + bz_nw);
    let bz_r_avg = 0.5 * (bz_se + bz_ne);
    R.Jx =  (bz_u_avg - bz_d_avg) / dx;
    R.Jy = -(bz_r_avg - bz_l_avg) / dx;

    return R;
}

fn hall_e_corner(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    let s = corner_jb(ix, iy, n_total);
    let prefactor = U_uniforms.hall_di / s.rho;
    let pe_frac = clamp(U_uniforms.hall_electron_pressure_frac, 0.0, 1.0);
    let pe_sw = pe_frac * cell_pressure_hall(ix - 1u, iy - 1u, n_total);
    let pe_se = pe_frac * cell_pressure_hall(ix,      iy - 1u, n_total);
    let pe_nw = pe_frac * cell_pressure_hall(ix - 1u, iy,      n_total);
    let pe_ne = pe_frac * cell_pressure_hall(ix,      iy,      n_total);
    let grad_pe_x = 0.5 * ((pe_se + pe_ne) - (pe_sw + pe_nw)) / U_uniforms.dx;
    let grad_pe_y = 0.5 * ((pe_nw + pe_ne) - (pe_sw + pe_se)) / U_uniforms.dx;
    return vec3<f32>(
        prefactor * ((s.Jy * s.Bz - s.Jz * s.By) - grad_pe_x),
        prefactor * ((s.Jz * s.Bx - s.Jx * s.Bz) - grad_pe_y),
        prefactor * (s.Jx * s.By - s.Jy * s.Bx),
    );
}

fn load_E(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    let e = hall_E[ez_edge_idx(ix, iy, n_total)];
    return vec3<f32>(e.x, e.y, e.z);
}

@compute @workgroup_size(8, 8, 1)
fn compute_emf(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_HALL)) { return; }
    if (U_uniforms.hall_di <= 0.0)                      { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let extent     = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;
    let e = hall_e_corner(ix, iy, n_total);
    hall_E[ez_edge_idx(ix, iy, n_total)] = vec4<f32>(e.x, e.y, e.z, 0.0);

    if (gid.x < n_interior && gid.y < n_interior) {
        let c = cell_idx_total(ix, iy, n_total);
        hall_mb0[c] = cell_magnetic_energy(ix, iy, n_total);
    }
}

@compute @workgroup_size(8, 8, 1)
fn apply_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_HALL)) { return; }
    if (U_uniforms.hall_di <= 0.0)                      { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let extent     = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;
    let dt_dx = dt_buf.dt / U_uniforms.dx;

    if (gid.y < n_interior) {
        let e0 = load_E(ix, iy,      n_total);
        let e1 = load_E(ix, iy + 1u, n_total);
        let bxi = bx_face_idx(ix, iy, n_total);
        Bx_face[bxi] = Bx_face[bxi] - dt_dx * (e1.z - e0.z);
    }
    if (gid.x < n_interior) {
        let e0 = load_E(ix,      iy, n_total);
        let e1 = load_E(ix + 1u, iy, n_total);
        let byi = by_face_idx(ix, iy, n_total);
        By_face[byi] = By_face[byi] + dt_dx * (e1.z - e0.z);
    }
    if (gid.x < n_interior && gid.y < n_interior) {
        let e_sw = load_E(ix,      iy,      n_total);
        let e_se = load_E(ix + 1u, iy,      n_total);
        let e_nw = load_E(ix,      iy + 1u, n_total);
        let e_ne = load_E(ix + 1u, iy + 1u, n_total);
        let dEy_dx = 0.5 * ((e_se.y + e_ne.y) - (e_sw.y + e_nw.y)) / U_uniforms.dx;
        let dEx_dy = 0.5 * ((e_nw.x + e_ne.x) - (e_sw.x + e_se.x)) / U_uniforms.dx;
        let c = cell_idx_total(ix, iy, n_total);
        let u1 = U1[c];
        U1[c] = vec4<f32>(u1.x, u1.y + (-dEy_dx + dEx_dy) * dt_buf.dt, u1.z, u1.w);
    }
}

@compute @workgroup_size(8, 8, 1)
fn repair_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_HALL)) { return; }
    if (U_uniforms.hall_di <= 0.0)                      { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;
    let c = cell_idx_total(ix, iy, n_total);
    let u1 = U1[c];
    let dmb = cell_magnetic_energy(ix, iy, n_total) - hall_mb0[c];
    U1[c] = vec4<f32>(u1.x + dmb, u1.y, u1.z, u1.w);
}

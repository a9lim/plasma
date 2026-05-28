// ─── apply-ohm.wgsl ───────────────────────────────────────────────────
// Unified generalized-Ohm source layer for the single-fluid MHD engine.
//
// The ideal electric field is still produced by compute-emf.wgsl from the
// Riemann fluxes. Ohmic resistivity still lives in the RKL2 curl(ηJ) pass,
// because that parabolic operator needs super-time-stepping. This shader
// unifies the remaining generalized-Ohm terms that previously lived in
// separate Hall and partial-ionization kernels:
//
//   E_H  = (d_i / ρ) · (J×B − ∇p_e)       nondissipative
//   E_AD = η_A · J_perp                  dissipative ion-neutral drift
//   E_ei = −η₄ ∇²J                       kinetic-length current smoothing
//   ∂B_z/∂t|Biermann = C_B (∇ρ×∇p_e)_z / ρ²
//
// compute_emf evaluates all terms from one frozen state. apply_hall_update
// applies the Hall part alone, repair_hall_energy restores the total energy
// by Δ magnetic energy, then apply_dissipative_update applies ambipolar and
// Biermann terms at fixed total energy so magnetic losses/gains exchange
// with the gas internal energy budget. The electron-inertia path is a
// single-fluid hyper-resistive closure, not a full two-fluid momentum solve;
// it regularizes unresolved current sheets at a configurable kinetic scale
// d_e while preserving the CT curl form.

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Bx_face:    array<f32>;
@group(0) @binding(4) var<storage, read_write> By_face:    array<f32>;
@group(0) @binding(5) var<uniform>             dt_buf:     DtUniform;
@group(0) @binding(6) var<storage, read_write> ohm_E:      array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> hall_E:     array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> hall_mb0:   array<f32>;
@group(0) @binding(9) var<storage, read>       micro:      array<vec4<f32>>;

const MICRO_ION_START: u32 = 24u;
const MICRO_ION_COUNT: u32 = 24u;
const INV_LN10_OHM: f32 = 0.4342944819032518;

struct CornerJB {
    Jx: f32, Jy: f32, Jz: f32,
    Bx: f32, By: f32, Bz: f32,
    rho: f32,
};

fn micro_log_interp_ohm(start: u32, count: u32, theta: f32) -> f32 {
    let log_theta = log(max(theta, 1.0e-30)) * INV_LN10_OHM;
    var idx = start;
    for (var i: u32 = 0u; i < 23u; i = i + 1u) {
        if (i + 1u >= count) { break; }
        let next = micro[start + i + 1u];
        if (log_theta < next.x) {
            idx = start + i;
            break;
        }
        idx = start + i + 1u;
    }
    let row = micro[idx];
    return row.y + row.z * (log_theta - row.x);
}

fn cell_bx_ohm(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
}

fn cell_by_ohm(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                + By_face[by_face_idx(ix, iy + 1u, n_total)]);
}

fn cell_pressure_ohm(ix: u32, iy: u32, n_total: u32) -> f32 {
    let u0 = U0[cell_idx_total(ix, iy, n_total)];
    let u1 = U1[cell_idx_total(ix, iy, n_total)];
    return pressure_from_dual_energy(u0, u1,
                                     cell_bx_ohm(ix, iy, n_total),
                                     cell_by_ohm(ix, iy, n_total),
                                     U_uniforms.gamma,
                                     U_uniforms.pressure_floor);
}

fn cell_temperature_ohm(ix: u32, iy: u32, n_total: u32) -> f32 {
    let rho = max(U0[cell_idx_total(ix, iy, n_total)].x, DENSITY_FLOOR);
    return cell_pressure_ohm(ix, iy, n_total) / rho;
}

fn neutral_fraction_ohm(ix: u32, iy: u32, n_total: u32) -> f32 {
    let f0 = clamp(U_uniforms.neutral_frac, 0.0, 1.0);
    if (f0 <= 0.0) { return 0.0; }
    let theta = cell_temperature_ohm(ix, iy, n_total)
              / max(U_uniforms.ionization_T0, 1.0e-30);
    let log_f = micro_log_interp_ohm(MICRO_ION_START, MICRO_ION_COUNT, theta);
    return clamp(f0 * pow(10.0, log_f), 0.0, 1.0);
}

fn corner_jb_ohm(ix: u32, iy: u32, n_total: u32) -> CornerJB {
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

fn cell_magnetic_energy_ohm(ix: u32, iy: u32, n_total: u32) -> f32 {
    let u1 = U1[cell_idx_total(ix, iy, n_total)];
    let bx_c = cell_bx_ohm(ix, iy, n_total);
    let by_c = cell_by_ohm(ix, iy, n_total);
    return 0.5 * (bx_c * bx_c + by_c * by_c + u1.y * u1.y);
}

fn hall_e_corner_ohm(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    if (!flag_set(U_uniforms.physics_flags, FLAG_HALL) || U_uniforms.hall_di <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let s = corner_jb_ohm(ix, iy, n_total);
    let prefactor = U_uniforms.hall_di / s.rho;
    let pe_frac = clamp(U_uniforms.hall_electron_pressure_frac, 0.0, 1.0);
    let pe_sw = pe_frac * cell_pressure_ohm(ix - 1u, iy - 1u, n_total);
    let pe_se = pe_frac * cell_pressure_ohm(ix,      iy - 1u, n_total);
    let pe_nw = pe_frac * cell_pressure_ohm(ix - 1u, iy,      n_total);
    let pe_ne = pe_frac * cell_pressure_ohm(ix,      iy,      n_total);
    let grad_pe_x = 0.5 * ((pe_se + pe_ne) - (pe_sw + pe_nw)) / U_uniforms.dx;
    let grad_pe_y = 0.5 * ((pe_nw + pe_ne) - (pe_sw + pe_se)) / U_uniforms.dx;
    return vec3<f32>(
        prefactor * ((s.Jy * s.Bz - s.Jz * s.By) - grad_pe_x),
        prefactor * ((s.Jz * s.Bx - s.Jx * s.Bz) - grad_pe_y),
        prefactor * (s.Jx * s.By - s.Jy * s.Bx),
    );
}

fn ambipolar_e_corner_ohm(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    if (!flag_set(U_uniforms.physics_flags, FLAG_AMBIPOLAR) || U_uniforms.ambipolar_eta <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let s = corner_jb_ohm(ix, iy, n_total);
    let b2 = s.Bx*s.Bx + s.By*s.By + s.Bz*s.Bz;
    if (b2 <= 1.0e-20) { return vec3<f32>(0.0, 0.0, 0.0); }

    let jdotb = s.Jx*s.Bx + s.Jy*s.By + s.Jz*s.Bz;
    let jperp = vec3<f32>(s.Jx, s.Jy, s.Jz)
              - vec3<f32>(s.Bx, s.By, s.Bz) * (jdotb / b2);
    let f_sw = neutral_fraction_ohm(ix - 1u, iy - 1u, n_total);
    let f_se = neutral_fraction_ohm(ix,      iy - 1u, n_total);
    let f_nw = neutral_fraction_ohm(ix - 1u, iy,      n_total);
    let f_ne = neutral_fraction_ohm(ix,      iy,      n_total);
    let neutral = 0.25 * (f_sw + f_se + f_nw + f_ne);
    return max(U_uniforms.ambipolar_eta, 0.0) * neutral * jperp;
}

fn electron_inertia_e_corner_ohm(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    if (!flag_set(U_uniforms.physics_flags, FLAG_ELECTRON_INERTIA)
        || U_uniforms.electron_inertia_length <= 0.0
        || U_uniforms.electron_inertia_damping <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let dx = U_uniforms.dx;
    let j0 = corner_jb_ohm(ix,      iy,      n_total).Jz;
    let jl = corner_jb_ohm(ix - 1u, iy,      n_total).Jz;
    let jr = corner_jb_ohm(ix + 1u, iy,      n_total).Jz;
    let jd = corner_jb_ohm(ix,      iy - 1u, n_total).Jz;
    let ju = corner_jb_ohm(ix,      iy + 1u, n_total).Jz;
    let lap_jz = (jl + jr + jd + ju - 4.0 * j0) / max(dx * dx, 1.0e-30);
    let eta4 = U_uniforms.electron_inertia_damping
             * U_uniforms.electron_inertia_length
             * U_uniforms.electron_inertia_length;
    return vec3<f32>(0.0, 0.0, -eta4 * lap_jz);
}

fn biermann_cell_ohm(ix: u32, iy: u32, n_total: u32) -> f32 {
    if (!flag_set(U_uniforms.physics_flags, FLAG_BIERMANN)
        || U_uniforms.biermann_coeff == 0.0
        || U_uniforms.hall_electron_pressure_frac <= 0.0) {
        return 0.0;
    }
    let pe_frac = clamp(U_uniforms.hall_electron_pressure_frac, 0.0, 1.0);
    let pe_l = pe_frac * cell_pressure_ohm(ix - 1u, iy, n_total);
    let pe_r = pe_frac * cell_pressure_ohm(ix + 1u, iy, n_total);
    let pe_d = pe_frac * cell_pressure_ohm(ix, iy - 1u, n_total);
    let pe_u = pe_frac * cell_pressure_ohm(ix, iy + 1u, n_total);

    let rho_l = max(U0[cell_idx_total(ix - 1u, iy, n_total)].x, DENSITY_FLOOR);
    let rho_r = max(U0[cell_idx_total(ix + 1u, iy, n_total)].x, DENSITY_FLOOR);
    let rho_d = max(U0[cell_idx_total(ix, iy - 1u, n_total)].x, DENSITY_FLOOR);
    let rho_u = max(U0[cell_idx_total(ix, iy + 1u, n_total)].x, DENSITY_FLOOR);
    let rho_c = max(U0[cell_idx_total(ix, iy, n_total)].x, DENSITY_FLOOR);

    let inv_2dx = 0.5 / U_uniforms.dx;
    let grad_rho_x = (rho_r - rho_l) * inv_2dx;
    let grad_rho_y = (rho_u - rho_d) * inv_2dx;
    let grad_pe_x = (pe_r - pe_l) * inv_2dx;
    let grad_pe_y = (pe_u - pe_d) * inv_2dx;
    return U_uniforms.biermann_coeff
         * (grad_rho_x * grad_pe_y - grad_rho_y * grad_pe_x)
         / max(rho_c * rho_c, 1.0e-20);
}

fn load_total_E(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    let e = ohm_E[ez_edge_idx(ix, iy, n_total)];
    return vec3<f32>(e.x, e.y, e.z);
}

fn load_hall_E(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    let e = hall_E[ez_edge_idx(ix, iy, n_total)];
    return vec3<f32>(e.x, e.y, e.z);
}

fn load_ambipolar_E(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    let idx = ez_edge_idx(ix, iy, n_total);
    let total = ohm_E[idx];
    let hall = hall_E[idx];
    return vec3<f32>(total.x - hall.x, total.y - hall.y, total.z - hall.z);
}

@compute @workgroup_size(8, 8, 1)
fn compute_emf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let any_on = (flag_set(U_uniforms.physics_flags, FLAG_HALL) && U_uniforms.hall_di > 0.0)
              || (flag_set(U_uniforms.physics_flags, FLAG_AMBIPOLAR) && U_uniforms.ambipolar_eta > 0.0)
              || (flag_set(U_uniforms.physics_flags, FLAG_BIERMANN) && U_uniforms.biermann_coeff != 0.0)
              || (flag_set(U_uniforms.physics_flags, FLAG_ELECTRON_INERTIA)
                  && U_uniforms.electron_inertia_length > 0.0
                  && U_uniforms.electron_inertia_damping > 0.0);
    if (!any_on) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let extent     = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;
    let e_hall = hall_e_corner_ohm(ix, iy, n_total);
    let e_ambi = ambipolar_e_corner_ohm(ix, iy, n_total);
    let e_ei = electron_inertia_e_corner_ohm(ix, iy, n_total);
    var battery = 0.0;
    if (gid.x < n_interior && gid.y < n_interior) {
        battery = biermann_cell_ohm(ix, iy, n_total);
        hall_mb0[cell_idx_total(ix, iy, n_total)] = cell_magnetic_energy_ohm(ix, iy, n_total);
    }
    let e_total = e_hall + e_ambi + e_ei;
    ohm_E[ez_edge_idx(ix, iy, n_total)] = vec4<f32>(e_total.x, e_total.y, e_total.z, battery);
    hall_E[ez_edge_idx(ix, iy, n_total)] = vec4<f32>(e_hall.x, e_hall.y, e_hall.z, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn apply_hall_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_HALL) || U_uniforms.hall_di <= 0.0) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let extent     = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;
    let dt_dx = dt_buf.dt / U_uniforms.dx;

    if (gid.y < n_interior) {
        let e0 = load_hall_E(ix, iy,      n_total);
        let e1 = load_hall_E(ix, iy + 1u, n_total);
        let bxi = bx_face_idx(ix, iy, n_total);
        Bx_face[bxi] = Bx_face[bxi] - dt_dx * (e1.z - e0.z);
    }
    if (gid.x < n_interior) {
        let e0 = load_hall_E(ix,      iy, n_total);
        let e1 = load_hall_E(ix + 1u, iy, n_total);
        let byi = by_face_idx(ix, iy, n_total);
        By_face[byi] = By_face[byi] + dt_dx * (e1.z - e0.z);
    }
    if (gid.x < n_interior && gid.y < n_interior) {
        let e_sw = load_hall_E(ix,      iy,      n_total);
        let e_se = load_hall_E(ix + 1u, iy,      n_total);
        let e_nw = load_hall_E(ix,      iy + 1u, n_total);
        let e_ne = load_hall_E(ix + 1u, iy + 1u, n_total);
        let dEy_dx = 0.5 * ((e_se.y + e_ne.y) - (e_sw.y + e_nw.y)) / U_uniforms.dx;
        let dEx_dy = 0.5 * ((e_nw.x + e_ne.x) - (e_sw.x + e_se.x)) / U_uniforms.dx;
        let c = cell_idx_total(ix, iy, n_total);
        let u1 = U1[c];
        U1[c] = vec4<f32>(u1.x, u1.y + (-dEy_dx + dEx_dy) * dt_buf.dt, u1.z, u1.w);
    }
}

@compute @workgroup_size(8, 8, 1)
fn repair_hall_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_HALL) || U_uniforms.hall_di <= 0.0) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;
    let c = cell_idx_total(ix, iy, n_total);
    let u0 = U0[c];
    let u1 = U1[c];
    let dmb = cell_magnetic_energy_ohm(ix, iy, n_total) - hall_mb0[c];
    let E = u1.x + dmb;
    let rho = max(u0.x, DENSITY_FLOOR);
    let ke = 0.5 * (u0.y*u0.y + u0.z*u0.z + u0.w*u0.w) / rho;
    let mb = cell_magnetic_energy_ohm(ix, iy, n_total);
    let p = max((U_uniforms.gamma - 1.0) * (E - ke - mb), U_uniforms.pressure_floor);
    U1[c] = pack_u1_aux(E, u1.y, rho, p, U_uniforms.gamma, U_uniforms.pressure_floor);
}

@compute @workgroup_size(8, 8, 1)
fn apply_dissipative_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ambi_on = flag_set(U_uniforms.physics_flags, FLAG_AMBIPOLAR)
               && U_uniforms.ambipolar_eta > 0.0
               && U_uniforms.neutral_frac > 0.0;
    let electron_inertia_on = flag_set(U_uniforms.physics_flags, FLAG_ELECTRON_INERTIA)
                           && U_uniforms.electron_inertia_length > 0.0
                           && U_uniforms.electron_inertia_damping > 0.0;
    let nonhall_on = ambi_on || electron_inertia_on;
    let biermann_on = flag_set(U_uniforms.physics_flags, FLAG_BIERMANN)
                   && U_uniforms.biermann_coeff != 0.0
                   && U_uniforms.hall_electron_pressure_frac > 0.0;
    if (!nonhall_on && !biermann_on) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let extent     = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;
    let dt = dt_buf.dt;
    let dt_dx = dt / U_uniforms.dx;

    if (nonhall_on) {
        if (gid.y < n_interior) {
            let e0 = load_ambipolar_E(ix, iy,      n_total);
            let e1 = load_ambipolar_E(ix, iy + 1u, n_total);
            let bxi = bx_face_idx(ix, iy, n_total);
            Bx_face[bxi] = Bx_face[bxi] - dt_dx * (e1.z - e0.z);
        }
        if (gid.x < n_interior) {
            let e0 = load_ambipolar_E(ix,      iy, n_total);
            let e1 = load_ambipolar_E(ix + 1u, iy, n_total);
            let byi = by_face_idx(ix, iy, n_total);
            By_face[byi] = By_face[byi] + dt_dx * (e1.z - e0.z);
        }
    }

    if (gid.x < n_interior && gid.y < n_interior) {
        let c = cell_idx_total(ix, iy, n_total);
        var u1 = U1[c];
        if (nonhall_on) {
            let e_sw = load_ambipolar_E(ix,      iy,      n_total);
            let e_se = load_ambipolar_E(ix + 1u, iy,      n_total);
            let e_nw = load_ambipolar_E(ix,      iy + 1u, n_total);
            let e_ne = load_ambipolar_E(ix + 1u, iy + 1u, n_total);
            let dEy_dx = 0.5 * ((e_se.y + e_ne.y) - (e_sw.y + e_nw.y)) / U_uniforms.dx;
            let dEx_dy = 0.5 * ((e_nw.x + e_ne.x) - (e_sw.x + e_se.x)) / U_uniforms.dx;
            u1 = vec4<f32>(u1.x, u1.y + (-dEy_dx + dEx_dy) * dt, u1.z, u1.w);
        }
        if (biermann_on) {
            let b = ohm_E[ez_edge_idx(ix, iy, n_total)].w;
            u1 = vec4<f32>(u1.x, u1.y + b * dt, u1.z, u1.w);
        }
        let u0 = U0[c];
        let rho = max(u0.x, DENSITY_FLOOR);
        let ke = 0.5 * (u0.y*u0.y + u0.z*u0.z + u0.w*u0.w) / rho;
        let mb = cell_magnetic_energy_ohm(ix, iy, n_total);
        let p = max((U_uniforms.gamma - 1.0) * (u1.x - ke - mb),
                    U_uniforms.pressure_floor);
        U1[c] = pack_u1_aux(u1.x, u1.y, rho, p,
                             U_uniforms.gamma, U_uniforms.pressure_floor);
    }
}

// ─── apply-nonideal.wgsl ──────────────────────────────────────────────
// Partial-ionization / generalized-Ohm corrections beyond the ideal,
// resistive, and Hall paths:
//
//   • Ambipolar diffusion: E_AD = η_A · J_perp, where
//       J_perp = J − B (J·B)/|B|².
//     This damps currents perpendicular to the magnetic field and models
//     ion-neutral drift in weakly ionized plasma. Unlike Hall, this is
//     dissipative, so total energy is not repaired back to preserve thermal
//     energy; magnetic energy lost by the induction update remains available
//     as heat through the conserved total-E budget.
//
//   • Biermann battery: ∂B_z/∂t = C_B · (∇ρ × ∇p_e)_z / ρ².
//     This creates seed out-of-plane field in baroclinic flows where density
//     and electron-pressure gradients are not parallel.
//
// Split form mirrors Hall/conduction: compute all source EMFs from a frozen
// state into scratch, then apply the CT curl/update in a separate dispatch.

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Bx_face:    array<f32>;
@group(0) @binding(4) var<storage, read_write> By_face:    array<f32>;
@group(0) @binding(5) var<uniform>             dt_buf:     DtUniform;
@group(0) @binding(6) var<storage, read_write> nonideal_E: array<vec4<f32>>;

struct CornerJB {
    Jx: f32, Jy: f32, Jz: f32,
    Bx: f32, By: f32, Bz: f32,
    rho: f32,
};

fn cell_bx_ni(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
}

fn cell_by_ni(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                + By_face[by_face_idx(ix, iy + 1u, n_total)]);
}

fn cell_pressure_ni(ix: u32, iy: u32, n_total: u32) -> f32 {
    let c = cell_idx_total(ix, iy, n_total);
    let u0 = U0[c];
    let u1 = U1[c];
    let rho = max(u0.x, DENSITY_FLOOR);
    let ke = 0.5 * (u0.y*u0.y + u0.z*u0.z + u0.w*u0.w) / rho;
    let bx = cell_bx_ni(ix, iy, n_total);
    let by = cell_by_ni(ix, iy, n_total);
    let mb = 0.5 * (bx*bx + by*by + u1.y*u1.y);
    return max((U_uniforms.gamma - 1.0) * (u1.x - ke - mb),
               U_uniforms.pressure_floor);
}

fn cell_temperature_ni(ix: u32, iy: u32, n_total: u32) -> f32 {
    let rho = max(U0[cell_idx_total(ix, iy, n_total)].x, DENSITY_FLOOR);
    return cell_pressure_ni(ix, iy, n_total) / rho;
}

fn neutral_fraction_cell(ix: u32, iy: u32, n_total: u32) -> f32 {
    let f0 = clamp(U_uniforms.neutral_frac, 0.0, 1.0);
    let T0 = max(U_uniforms.ionization_T0, 1.0e-30);
    let T  = max(cell_temperature_ni(ix, iy, n_total), 0.0);
    return f0 / (1.0 + (T / T0) * (T / T0));
}

fn corner_jb_ni(ix: u32, iy: u32, n_total: u32) -> CornerJB {
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

fn ambipolar_e_corner(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    let s = corner_jb_ni(ix, iy, n_total);
    let b2 = s.Bx*s.Bx + s.By*s.By + s.Bz*s.Bz;
    if (b2 <= 1.0e-20) { return vec3<f32>(0.0, 0.0, 0.0); }

    let jdotb = s.Jx*s.Bx + s.Jy*s.By + s.Jz*s.Bz;
    let jperp = vec3<f32>(s.Jx, s.Jy, s.Jz)
              - vec3<f32>(s.Bx, s.By, s.Bz) * (jdotb / b2);
    let f_sw = neutral_fraction_cell(ix - 1u, iy - 1u, n_total);
    let f_se = neutral_fraction_cell(ix,      iy - 1u, n_total);
    let f_nw = neutral_fraction_cell(ix - 1u, iy,      n_total);
    let f_ne = neutral_fraction_cell(ix,      iy,      n_total);
    let neutral = 0.25 * (f_sw + f_se + f_nw + f_ne);
    let eta_a = max(U_uniforms.ambipolar_eta, 0.0) * neutral;
    return eta_a * jperp;
}

fn biermann_cell(ix: u32, iy: u32, n_total: u32) -> f32 {
    let pe_frac = clamp(U_uniforms.hall_electron_pressure_frac, 0.0, 1.0);
    let pe_l = pe_frac * cell_pressure_ni(ix - 1u, iy, n_total);
    let pe_r = pe_frac * cell_pressure_ni(ix + 1u, iy, n_total);
    let pe_d = pe_frac * cell_pressure_ni(ix, iy - 1u, n_total);
    let pe_u = pe_frac * cell_pressure_ni(ix, iy + 1u, n_total);

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

fn load_E(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    let e = nonideal_E[ez_edge_idx(ix, iy, n_total)];
    return vec3<f32>(e.x, e.y, e.z);
}

@compute @workgroup_size(8, 8, 1)
fn compute_emf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ambi_on = flag_set(U_uniforms.physics_flags, FLAG_AMBIPOLAR)
               && U_uniforms.ambipolar_eta > 0.0
               && U_uniforms.neutral_frac > 0.0;
    let biermann_on = flag_set(U_uniforms.physics_flags, FLAG_BIERMANN)
                   && U_uniforms.biermann_coeff != 0.0
                   && U_uniforms.hall_electron_pressure_frac > 0.0;
    if (!ambi_on && !biermann_on) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let extent     = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;

    var e = vec3<f32>(0.0, 0.0, 0.0);
    if (ambi_on) {
        e = ambipolar_e_corner(ix, iy, n_total);
    }
    var battery = 0.0;
    if (biermann_on && gid.x < n_interior && gid.y < n_interior) {
        battery = biermann_cell(ix, iy, n_total);
    }
    nonideal_E[ez_edge_idx(ix, iy, n_total)] = vec4<f32>(e.x, e.y, e.z, battery);
}

@compute @workgroup_size(8, 8, 1)
fn apply_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ambi_on = flag_set(U_uniforms.physics_flags, FLAG_AMBIPOLAR)
               && U_uniforms.ambipolar_eta > 0.0
               && U_uniforms.neutral_frac > 0.0;
    let biermann_on = flag_set(U_uniforms.physics_flags, FLAG_BIERMANN)
                   && U_uniforms.biermann_coeff != 0.0
                   && U_uniforms.hall_electron_pressure_frac > 0.0;
    if (!ambi_on && !biermann_on) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let extent     = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;
    let dt = dt_buf.dt;
    let dt_dx = dt / U_uniforms.dx;

    if (ambi_on) {
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
    }

    if (gid.x < n_interior && gid.y < n_interior) {
        let c = cell_idx_total(ix, iy, n_total);
        var u1 = U1[c];
        if (ambi_on) {
            let e_sw = load_E(ix,      iy,      n_total);
            let e_se = load_E(ix + 1u, iy,      n_total);
            let e_nw = load_E(ix,      iy + 1u, n_total);
            let e_ne = load_E(ix + 1u, iy + 1u, n_total);
            let dEy_dx = 0.5 * ((e_se.y + e_ne.y) - (e_sw.y + e_nw.y)) / U_uniforms.dx;
            let dEx_dy = 0.5 * ((e_nw.x + e_ne.x) - (e_sw.x + e_se.x)) / U_uniforms.dx;
            u1 = vec4<f32>(u1.x, u1.y + (-dEy_dx + dEx_dy) * dt, u1.z, u1.w);
        }
        if (biermann_on) {
            let b = nonideal_E[ez_edge_idx(ix, iy, n_total)].w;
            u1 = vec4<f32>(u1.x, u1.y + b * dt, u1.z, u1.w);
        }
        U1[c] = u1;
    }
}

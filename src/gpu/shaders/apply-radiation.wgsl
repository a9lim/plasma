// ─── apply-radiation.wgsl ─────────────────────────────────────────────
// Grey radiation-energy source layer.
//
// This opt-in source advances a separate cell-centered radiation energy
// density E_r with flux-limited diffusion and local thermal coupling:
//
//   ∂E_r/∂t = -∇·F_r + c κ_P ρ (a_r T^4 - E_r)
//   ∂E_g/∂t =          - c κ_P ρ (a_r T^4 - E_r)
//
//   F_r = -D ∇E_r,     D = c λ(R) / ((κ_P + κ_R) ρ)
//   λ(R) = (2 + R) / (6 + 3R + R²), R = |∇E_r| / ((κ_P + κ_R)ρE_r)
//
// The limiter recovers diffusion (λ≈1/3) in optically thick regions and
// caps the streaming speed in optically thin gradients. This is still a
// grey, nonrelativistic closure, but it adds a conserved radiation reservoir
// instead of treating all radiation as instant optically-thin loss.
//
// Bindings:
//   0 uniforms       (uniform)
//   1 U0            (ro)
//   2 U1            (rw)
//   3 Bx_face       (ro)
//   4 By_face       (ro)
//   5 rad_dt        (uniform)
//   6 radiation_E   (rw, f32 per cell)
//   7 radiation_dE  (rw, vec4 per cell: x=dE_r, y=dE_gas)
//   8 microphysics table (ro storage; absorption/scattering opacity modifiers)
//   9 bc_uniforms   (ro storage; scalar radiation boundary sampling)

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms:  Uniforms;
@group(0) @binding(1) var<storage, read>       U0:          array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:          array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:     array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:     array<f32>;
@group(0) @binding(5) var<uniform>             dt_buf:      DtUniform;
@group(0) @binding(6) var<storage, read_write> radiation_E: array<f32>;
@group(0) @binding(7) var<storage, read_write> radiation_dE: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read>       micro:       array<vec4<f32>>;
@group(0) @binding(9) var<storage, read>       bc:          BcUniforms;

const MICRO_RAD_ABS_START: u32 = 96u;
const MICRO_RAD_SCAT_START: u32 = 120u;
const MICRO_RAD_COUNT: u32 = 24u;
const INV_LN10_RAD: f32 = 0.4342944819032518;

fn radiation_on() -> bool {
    return flag_set(U_uniforms.physics_flags, FLAG_RADIATION)
        && U_uniforms.radiation_c > 0.0
        && (U_uniforms.radiation_kappa_abs > 0.0 || U_uniforms.radiation_kappa_scat > 0.0);
}

fn sample_axis(idx: u32, offset: i32, n_interior: u32, ghost: u32,
               lo_mode: u32, hi_mode: u32) -> u32 {
    let local = i32(idx) - i32(ghost) + offset;
    let n = i32(n_interior);
    if (local < 0) {
        if (lo_mode == BC_PERIODIC) { return ghost + n_interior - 1u; }
        return ghost;
    }
    if (local >= n) {
        if (hi_mode == BC_PERIODIC) { return ghost; }
        return ghost + n_interior - 1u;
    }
    return u32(local) + ghost;
}

fn sample_ix(ix: u32, offset: i32, n_interior: u32, ghost: u32) -> u32 {
    return sample_axis(ix, offset, n_interior, ghost, bc.mode_w, bc.mode_e);
}

fn sample_iy(iy: u32, offset: i32, n_interior: u32, ghost: u32) -> u32 {
    return sample_axis(iy, offset, n_interior, ghost, bc.mode_s, bc.mode_n);
}

fn rad_idx(ix: u32, iy: u32, n_total: u32) -> u32 {
    return cell_idx_total(ix, iy, n_total);
}

fn rad_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return max(radiation_E[rad_idx(ix, iy, n_total)], U_uniforms.radiation_floor);
}

fn micro_log_interp_rad(start: u32, count: u32, theta: f32) -> f32 {
    let log_theta = log(max(theta, 1.0e-30)) * INV_LN10_RAD;
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

fn bx_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_left_idx(ix, iy, n_total)]
                + Bx_face[bx_face_right_idx(ix, iy, n_total)]);
}

fn by_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_down_idx(ix, iy, n_total)]
                + By_face[by_face_up_idx(ix, iy, n_total)]);
}

fn cell_pressure(ix: u32, iy: u32, n_total: u32) -> f32 {
    let c = cell_idx_total(ix, iy, n_total);
    return pressure_from_dual_energy(U0[c], U1[c], bx_at(ix, iy, n_total), by_at(ix, iy, n_total),
                                     U_uniforms.gamma, U_uniforms.pressure_floor);
}

fn cell_temperature_rad(ix: u32, iy: u32, n_total: u32) -> f32 {
    let rho = max(U0[cell_idx_total(ix, iy, n_total)].x, DENSITY_FLOOR);
    return cell_pressure(ix, iy, n_total) / rho;
}

fn opacity_abs_scale(theta: f32) -> f32 {
    return clamp(pow(10.0, micro_log_interp_rad(MICRO_RAD_ABS_START, MICRO_RAD_COUNT, theta)),
                 0.01, 32.0);
}

fn opacity_scat_scale(theta: f32) -> f32 {
    return clamp(pow(10.0, micro_log_interp_rad(MICRO_RAD_SCAT_START, MICRO_RAD_COUNT, theta)),
                 0.01, 4.0);
}

fn radiation_theta(ix: u32, iy: u32, n_total: u32) -> f32 {
    return cell_temperature_rad(ix, iy, n_total)
         / max(U_uniforms.cooling_T_ref, 1.0e-30);
}

fn radiation_kappa_abs_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return max(U_uniforms.radiation_kappa_abs, 0.0)
         * opacity_abs_scale(radiation_theta(ix, iy, n_total));
}

fn radiation_kappa_scat_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return max(U_uniforms.radiation_kappa_scat, 0.0)
         * opacity_scat_scale(radiation_theta(ix, iy, n_total));
}

fn radiation_kappa_total_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return radiation_kappa_abs_at(ix, iy, n_total)
         + radiation_kappa_scat_at(ix, iy, n_total);
}

fn diffusion_coeff(ix: u32, iy: u32, n_interior: u32, n_total: u32, ghost: u32) -> f32 {
    let kappa = radiation_kappa_total_at(ix, iy, n_total);
    if (kappa <= 0.0 || U_uniforms.radiation_c <= 0.0) { return 0.0; }

    let ix_l = sample_ix(ix, -1, n_interior, ghost);
    let ix_r = sample_ix(ix,  1, n_interior, ghost);
    let iy_d = sample_iy(iy, -1, n_interior, ghost);
    let iy_u = sample_iy(iy,  1, n_interior, ghost);
    let er = rad_at(ix, iy, n_total);
    let dEx = (rad_at(ix_r, iy,   n_total) - rad_at(ix_l, iy,   n_total)) / (2.0 * U_uniforms.dx);
    let dEy = (rad_at(ix,   iy_u, n_total) - rad_at(ix,   iy_d, n_total)) / (2.0 * U_uniforms.dx);
    let rho = max(U0[cell_idx_total(ix, iy, n_total)].x, DENSITY_FLOOR);
    let denom = max(kappa * rho * er, 1.0e-30);
    let R = sqrt(max(dEx*dEx + dEy*dEy, 0.0)) / denom;
    let lambda = (2.0 + R) / max(6.0 + 3.0 * R + R * R, 1.0e-30);
    return U_uniforms.radiation_c * lambda / max(kappa * rho, 1.0e-30);
}

@compute @workgroup_size(8, 8, 1)
fn compute_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);
    if (!radiation_on()) {
        radiation_dE[c] = vec4<f32>(0.0);
        return;
    }

    let ix_l = sample_ix(ix, -1, n_interior, ghost);
    let ix_r = sample_ix(ix,  1, n_interior, ghost);
    let iy_d = sample_iy(iy, -1, n_interior, ghost);
    let iy_u = sample_iy(iy,  1, n_interior, ghost);

    let er_c = rad_at(ix,   iy,   n_total);
    let er_l = rad_at(ix_l, iy,   n_total);
    let er_r = rad_at(ix_r, iy,   n_total);
    let er_d = rad_at(ix,   iy_d, n_total);
    let er_u = rad_at(ix,   iy_u, n_total);

    let D_c = diffusion_coeff(ix,   iy,   n_interior, n_total, ghost);
    let D_l = 0.5 * (D_c + diffusion_coeff(ix_l, iy,   n_interior, n_total, ghost));
    let D_r = 0.5 * (D_c + diffusion_coeff(ix_r, iy,   n_interior, n_total, ghost));
    let D_d = 0.5 * (D_c + diffusion_coeff(ix,   iy_d, n_interior, n_total, ghost));
    let D_u = 0.5 * (D_c + diffusion_coeff(ix,   iy_u, n_interior, n_total, ghost));

    let inv_dx = 1.0 / U_uniforms.dx;
    let flux_l = -D_l * (er_c - er_l) * inv_dx;
    let flux_r = -D_r * (er_r - er_c) * inv_dx;
    let flux_d = -D_d * (er_c - er_d) * inv_dx;
    let flux_u = -D_u * (er_u - er_c) * inv_dx;
    let div_flux = (flux_r - flux_l + flux_u - flux_d) * inv_dx;

    let rho = max(U0[c].x, DENSITY_FLOOR);
    let T = cell_temperature_rad(ix, iy, n_total);
    let er_lte = max(U_uniforms.radiation_const, 0.0) * pow(max(T, 0.0), 4.0);
    let kappa_abs = radiation_kappa_abs_at(ix, iy, n_total);
    let exchange = U_uniforms.radiation_c * kappa_abs * rho * (er_lte - er_c);
    let dt = max(dt_buf.dt, 0.0);

    radiation_dE[c] = vec4<f32>((-div_flux + exchange) * dt,
                                -exchange * dt,
                                0.0,
                                0.0);
}

@compute @workgroup_size(8, 8, 1)
fn apply_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);
    if (!radiation_on()) { return; }

    let d = radiation_dE[c];
    radiation_E[c] = max(radiation_E[c] + d.x, U_uniforms.radiation_floor);

    let u0 = U0[c];
    let u1 = U1[c];
    let rho = max(u0.x, DENSITY_FLOOR);
    let mx = u0.y;
    let my = u0.z;
    let mz = u0.w;
    let bz = u1.y;
    let bx_c = bx_at(ix, iy, n_total);
    let by_c = by_at(ix, iy, n_total);
    let ke = 0.5 * (mx*mx + my*my + mz*mz) / rho;
    let mb = 0.5 * (bx_c*bx_c + by_c*by_c + bz*bz);
    let E_min = ke + mb + U_uniforms.pressure_floor / (U_uniforms.gamma - 1.0);
    let E_new = clamp(u1.x + d.y, E_min, 1.0e30);
    let p_new = max((U_uniforms.gamma - 1.0) * (E_new - ke - mb), U_uniforms.pressure_floor);
    U1[c] = pack_u1_aux(E_new, bz, rho, p_new, U_uniforms.gamma, U_uniforms.pressure_floor);
}

// ─── apply-viscosity.wgsl ─────────────────────────────────────────────
// Explicit viscous transport for the 2.5D MHD state.
//
// The source is split into compute_delta/apply_delta so all velocity
// gradients are evaluated from a frozen state. This is still a compact
// interactive-model approximation, but it gives the engine a real transport
// channel beyond thermal conduction:
//
//   d(ρv)/dt = ∇·τ
//   dE/dt    = v·∇·τ + τ:∇v
//
// with isotropic shear viscosity, bulk/compressional viscosity, an optional
// field-aligned projection, and a small shock-viscosity term gated on
// compression.

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:    array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:    array<f32>;
@group(0) @binding(5) var<uniform>             dt_buf:     DtUniform;
@group(0) @binding(6) var<storage, read_write> dU_visc:    array<vec4<f32>>;
@group(0) @binding(7) var<storage, read>       micro:      array<vec4<f32>>;

const MICRO_TRANSPORT_START_VISC: u32 = 48u;
const MICRO_TRANSPORT_COUNT_VISC: u32 = 16u;
const INV_LN10_VISC: f32 = 0.4342944819032518;

fn cell_bx_visc(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
}

fn cell_by_visc(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                + By_face[by_face_idx(ix, iy + 1u, n_total)]);
}

fn micro_log_interp_visc(start: u32, count: u32, theta: f32) -> f32 {
    let log_theta = log(max(theta, 1.0e-30)) * INV_LN10_VISC;
    var idx = start;
    for (var i: u32 = 0u; i < 15u; i = i + 1u) {
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

fn viscosity_transport_scale(ix: u32, iy: u32, n_total: u32) -> f32 {
    let u0 = U0[cell_idx_total(ix, iy, n_total)];
    let u1 = U1[cell_idx_total(ix, iy, n_total)];
    let rho = max(u0.x, DENSITY_FLOOR);
    let p = pressure_from_dual_energy(u0, u1, cell_bx_visc(ix, iy, n_total),
                                      cell_by_visc(ix, iy, n_total),
                                      U_uniforms.gamma, U_uniforms.pressure_floor);
    let theta = (p / rho) / max(U_uniforms.cooling_T_ref, 1.0e-30);
    return pow(10.0, micro_log_interp_visc(MICRO_TRANSPORT_START_VISC,
                                           MICRO_TRANSPORT_COUNT_VISC,
                                           theta));
}

fn velocity_at(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    let u0 = U0[cell_idx_total(ix, iy, n_total)];
    let rho = max(u0.x, DENSITY_FLOOR);
    return vec3<f32>(u0.y / rho, u0.z / rho, u0.w / rho);
}

fn rho_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return max(U0[cell_idx_total(ix, iy, n_total)].x, DENSITY_FLOOR);
}

fn div_v_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    let inv_2dx = 0.5 / U_uniforms.dx;
    let vx_r = velocity_at(ix + 1u, iy, n_total).x;
    let vx_l = velocity_at(ix - 1u, iy, n_total).x;
    let vy_u = velocity_at(ix, iy + 1u, n_total).y;
    let vy_d = velocity_at(ix, iy - 1u, n_total).y;
    return (vx_r - vx_l + vy_u - vy_d) * inv_2dx;
}

fn cell_bhat(ix: u32, iy: u32, n_total: u32) -> vec3<f32> {
    let bx = 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                  + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
    let by = 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                  + By_face[by_face_idx(ix, iy + 1u, n_total)]);
    let bz = U1[cell_idx_total(ix, iy, n_total)].y;
    let b2 = bx*bx + by*by + bz*bz;
    if (b2 <= 1.0e-20) { return vec3<f32>(0.0, 0.0, 0.0); }
    return vec3<f32>(bx, by, bz) / sqrt(b2);
}

@compute @workgroup_size(8, 8, 1)
fn compute_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_VISCOSITY)) { return; }
    let nu0 = max(U_uniforms.viscosity_nu, 0.0);
    let zeta0 = max(U_uniforms.viscosity_bulk, 0.0);
    let shock0 = max(U_uniforms.viscosity_shock, 0.0);
    if (nu0 <= 0.0 && zeta0 <= 0.0 && shock0 <= 0.0) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    let dx = U_uniforms.dx;
    let inv_dx = 1.0 / dx;
    let inv_2dx = 0.5 * inv_dx;
    let inv_dx2 = inv_dx * inv_dx;
    let dt = dt_buf.dt;

    let vc = velocity_at(ix, iy, n_total);
    let vl = velocity_at(ix - 1u, iy, n_total);
    let vr = velocity_at(ix + 1u, iy, n_total);
    let vd = velocity_at(ix, iy - 1u, n_total);
    let vu = velocity_at(ix, iy + 1u, n_total);

    let lap = (vl + vr + vd + vu - 4.0 * vc) * inv_dx2;

    let div_c = div_v_at(ix, iy, n_total);
    let div_l = div_v_at(ix - 1u, iy, n_total);
    let div_r = div_v_at(ix + 1u, iy, n_total);
    let div_d = div_v_at(ix, iy - 1u, n_total);
    let div_u = div_v_at(ix, iy + 1u, n_total);
    let grad_div = vec3<f32>(
        (div_r - div_l) * inv_2dx,
        (div_u - div_d) * inv_2dx,
        0.0,
    );

    let rho = rho_at(ix, iy, n_total);
    let shock_nu = shock0 * dx * dx * max(-div_c, 0.0);
    let tscale = viscosity_transport_scale(ix, iy, n_total);
    let nu = nu0 * tscale + shock_nu;
    let zeta = zeta0 * tscale + shock_nu;

    let b = cell_bhat(ix, iy, n_total);
    let aniso = clamp(U_uniforms.viscosity_aniso_frac, 0.0, 1.0);
    let lap_parallel = b * dot(lap, b);
    let lap_eff = mix(lap, lap_parallel, aniso);

    let force = rho * (nu * lap_eff + zeta * grad_div);
    let d_mom = force * dt;

    let dvdx = (vr - vl) * inv_2dx;
    let dvdy = (vu - vd) * inv_2dx;
    let div3 = div_c / 3.0;
    let sxx = dvdx.x - div3;
    let syy = dvdy.y - div3;
    let szz = -div3;
    let sxy = 0.5 * (dvdy.x + dvdx.y);
    let sxz = 0.5 * dvdx.z;
    let syz = 0.5 * dvdy.z;
    let shear_norm = sxx*sxx + syy*syy + szz*szz
                   + 2.0 * (sxy*sxy + sxz*sxz + syz*syz);
    let heat_rate = rho * (2.0 * nu * shear_norm + zeta * div_c * div_c);
    let dE = dot(vc, d_mom) + heat_rate * dt;

    dU_visc[c] = vec4<f32>(d_mom.x, d_mom.y, d_mom.z, dE);
}

@compute @workgroup_size(8, 8, 1)
fn apply_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_VISCOSITY)) { return; }
    let nu0 = max(U_uniforms.viscosity_nu, 0.0);
    let zeta0 = max(U_uniforms.viscosity_bulk, 0.0);
    let shock0 = max(U_uniforms.viscosity_shock, 0.0);
    if (nu0 <= 0.0 && zeta0 <= 0.0 && shock0 <= 0.0) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    let du = dU_visc[c];
    let u0 = U0[c];
    let u1 = U1[c];
    let u0_new = vec4<f32>(u0.x, u0.y + du.x, u0.z + du.y, u0.w + du.z);
    let E = u1.x + du.w;
    let rho = max(u0_new.x, DENSITY_FLOOR);
    let bx = cell_bx_visc(ix, iy, n_total);
    let by = cell_by_visc(ix, iy, n_total);
    let ke = 0.5 * (u0_new.y*u0_new.y + u0_new.z*u0_new.z + u0_new.w*u0_new.w) / rho;
    let mb = 0.5 * (bx*bx + by*by + u1.y*u1.y);
    let p = max((U_uniforms.gamma - 1.0) * (E - ke - mb), U_uniforms.pressure_floor);
    U0[c] = u0_new;
    U1[c] = pack_u1_aux(E, u1.y, rho, p, U_uniforms.gamma, U_uniforms.pressure_floor);
}

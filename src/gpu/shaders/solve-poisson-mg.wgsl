// ─── solve-poisson-mg.wgsl ────────────────────────────────────────────
// Geometric multigrid V-cycle helper for Cartesian self-gravity.
//
// The main solve-poisson.wgsl path remains the complete fallback for
// cylindrical geometry. This shader accelerates Cartesian periodic/isolated
// Poisson solves by working on compact no-ghost level buffers:
//
//   init_level0      full-resolution RHS + warm-start phi from main phi
//   smooth_level     weighted-Jacobi smoother on one level
//   restrict_residual full-weight residual restriction to the next level
//   prolongate_add   bilinear prolongation of coarse correction
//   copy_to_main     copy level-0 phi back to the ghost-padded main buffer

struct MgUniform {
    level_n: u32,
    stride:  u32,
    _pad0:   u32,
    _pad1:   u32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<uniform>             mg:         MgUniform;
@group(0) @binding(2) var<storage, read>       U0:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       rho_mean:   array<f32>;
@group(0) @binding(4) var<storage, read_write> phi_main:   array<f32>;
@group(0) @binding(5) var<storage, read>       phi_in:     array<f32>;
@group(0) @binding(6) var<storage, read_write> phi_out:    array<f32>;
@group(0) @binding(7) var<storage, read>       rhs_in:     array<f32>;
@group(0) @binding(8) var<storage, read_write> rhs_out:    array<f32>;
@group(0) @binding(9) var<storage, read>       phi_coarse: array<f32>;

const PI_MG: f32 = 3.141592653589793;

fn mg_idx(ix: u32, iy: u32, n: u32) -> u32 {
    return iy * n + ix;
}

fn poisson_isolated_mg() -> bool {
    return U_uniforms.gravity_boundary_mode == 1u;
}

fn sample_phi(gx_in: i32, gy_in: i32, n: u32) -> f32 {
    let ni = i32(n);
    if (poisson_isolated_mg()) {
        if (gx_in < 0 || gx_in >= ni || gy_in < 0 || gy_in >= ni) {
            return 0.0;
        }
        return phi_in[mg_idx(u32(gx_in), u32(gy_in), n)];
    }
    let gx = ((gx_in % ni) + ni) % ni;
    let gy = ((gy_in % ni) + ni) % ni;
    return phi_in[mg_idx(u32(gx), u32(gy), n)];
}

fn sample_rhs(gx_in: i32, gy_in: i32, n: u32) -> f32 {
    let ni = i32(n);
    if (poisson_isolated_mg()) {
        if (gx_in < 0 || gx_in >= ni || gy_in < 0 || gy_in >= ni) {
            return 0.0;
        }
        return rhs_in[mg_idx(u32(gx_in), u32(gy_in), n)];
    }
    let gx = ((gx_in % ni) + ni) % ni;
    let gy = ((gy_in % ni) + ni) % ni;
    return rhs_in[mg_idx(u32(gx), u32(gy), n)];
}

fn sample_coarse(gx_in: i32, gy_in: i32, n: u32) -> f32 {
    let ni = i32(n);
    if (poisson_isolated_mg()) {
        if (gx_in < 0 || gx_in >= ni || gy_in < 0 || gy_in >= ni) {
            return 0.0;
        }
        return phi_coarse[mg_idx(u32(gx_in), u32(gy_in), n)];
    }
    let gx = ((gx_in % ni) + ni) % ni;
    let gy = ((gy_in % ni) + ni) % ni;
    return phi_coarse[mg_idx(u32(gx), u32(gy), n)];
}

fn dx_level(stride: u32) -> f32 {
    return U_uniforms.dx * f32(max(stride, 1u));
}

fn soft_inv2() -> f32 {
    if (U_uniforms.gravity_softening <= 0.0) { return 0.0; }
    let s = max(U_uniforms.gravity_softening, 1.0e-30);
    return 1.0 / (s * s);
}

fn residual_at(gx: i32, gy: i32, n: u32, dx: f32) -> f32 {
    let c = sample_phi(gx, gy, n);
    let l = sample_phi(gx - 1, gy, n);
    let r = sample_phi(gx + 1, gy, n);
    let d = sample_phi(gx, gy - 1, n);
    let u = sample_phi(gx, gy + 1, n);
    let lap = (l + r + d + u - 4.0 * c) / max(dx * dx, 1.0e-30)
            - c * soft_inv2();
    let rhs = sample_rhs(gx, gy, n);
    return rhs - lap;
}

@compute @workgroup_size(8, 8, 1)
fn init_level0(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = mg.level_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let ghost = U_uniforms.ghost_w;
    let n_total = U_uniforms.grid_n_total;
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c_main = cell_idx_total(ix, iy, n_total);
    let c = mg_idx(gid.x, gid.y, n);

    let rho = U0[c_main].x;
    let rho_bar = select(rho_mean[0], 0.0, poisson_isolated_mg());
    rhs_out[c] = 4.0 * PI_MG * U_uniforms.gravity_G * (rho - rho_bar);
    phi_out[c] = phi_main[c_main];
}

@compute @workgroup_size(8, 8, 1)
fn smooth_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = mg.level_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let gx = i32(gid.x);
    let gy = i32(gid.y);
    let c = mg_idx(gid.x, gid.y, n);
    let l = sample_phi(gx - 1, gy, n);
    let r = sample_phi(gx + 1, gy, n);
    let d = sample_phi(gx, gy - 1, n);
    let u = sample_phi(gx, gy + 1, n);
    let dx = dx_level(mg.stride);
    let soft2 = select(0.0,
                       (dx / max(U_uniforms.gravity_softening, 1.0e-30))
                     * (dx / max(U_uniforms.gravity_softening, 1.0e-30)),
                       U_uniforms.gravity_softening > 0.0);
    let jacobi = (l + r + d + u - dx * dx * rhs_in[c])
               / max(4.0 + soft2, 1.0e-6);
    let omega = clamp(U_uniforms.gravity_poisson_omega, 0.05, 1.95);
    phi_out[c] = mix(phi_in[c], jacobi, omega);
}

@compute @workgroup_size(8, 8, 1)
fn restrict_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_coarse = mg.level_n;
    if (gid.x >= n_coarse || gid.y >= n_coarse) { return; }

    let n_fine = n_coarse * 2u;
    let stride_fine = max(mg.stride / 2u, 1u);
    let dx_fine = dx_level(stride_fine);
    let fx = i32(gid.x * 2u);
    let fy = i32(gid.y * 2u);

    let r_c  = residual_at(fx,     fy,     n_fine, dx_fine);
    let r_l  = residual_at(fx - 1, fy,     n_fine, dx_fine);
    let r_r  = residual_at(fx + 1, fy,     n_fine, dx_fine);
    let r_d  = residual_at(fx,     fy - 1, n_fine, dx_fine);
    let r_u  = residual_at(fx,     fy + 1, n_fine, dx_fine);
    let r_ld = residual_at(fx - 1, fy - 1, n_fine, dx_fine);
    let r_lu = residual_at(fx - 1, fy + 1, n_fine, dx_fine);
    let r_rd = residual_at(fx + 1, fy - 1, n_fine, dx_fine);
    let r_ru = residual_at(fx + 1, fy + 1, n_fine, dx_fine);

    let c = mg_idx(gid.x, gid.y, n_coarse);
    rhs_out[c] = (4.0 * r_c + 2.0 * (r_l + r_r + r_d + r_u)
               + (r_ld + r_lu + r_rd + r_ru)) / 16.0;
    phi_out[c] = 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn prolongate_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_fine = mg.level_n;
    if (gid.x >= n_fine || gid.y >= n_fine) { return; }

    let n_coarse = max(n_fine / 2u, 1u);
    let ux = (f32(gid.x) + 0.5) * 0.5 - 0.5;
    let uy = (f32(gid.y) + 0.5) * 0.5 - 0.5;
    let x0 = i32(floor(ux));
    let y0 = i32(floor(uy));
    let tx = clamp(ux - floor(ux), 0.0, 1.0);
    let ty = clamp(uy - floor(uy), 0.0, 1.0);

    let c00 = sample_coarse(x0,     y0,     n_coarse);
    let c10 = sample_coarse(x0 + 1, y0,     n_coarse);
    let c01 = sample_coarse(x0,     y0 + 1, n_coarse);
    let c11 = sample_coarse(x0 + 1, y0 + 1, n_coarse);
    let e0 = mix(c00, c10, tx);
    let e1 = mix(c01, c11, tx);
    let corr = mix(e0, e1, ty);

    let c = mg_idx(gid.x, gid.y, n_fine);
    phi_out[c] = phi_in[c] + corr;
}

@compute @workgroup_size(8, 8, 1)
fn copy_to_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = mg.level_n;
    if (gid.x >= n || gid.y >= n) { return; }
    let ghost = U_uniforms.ghost_w;
    let n_total = U_uniforms.grid_n_total;
    let c_main = cell_idx_total(gid.x + ghost, gid.y + ghost, n_total);
    phi_main[c_main] = phi_in[mg_idx(gid.x, gid.y, n)];
}

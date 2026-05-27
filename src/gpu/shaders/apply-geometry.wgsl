// ─── apply-geometry.wgsl ──────────────────────────────────────────────
// Geometry and boundary-environment source terms.
//
//   geometry_mode = 0: Cartesian no-op.
//   geometry_mode = 1: axisymmetric cylindrical correction with x as radius
//                      r and y as axial coordinate z. The stored components
//                      map as vx=v_r, vy=v_z, vz=v_phi, and Bz=B_phi.
//
// The cylindrical piece is intentionally a source-layer approximation rather
// than a full finite-volume r-weighted rewrite: enough to capture toroidal
// curvature/tension and expansion effects in interactive experiments without
// disturbing the existing CT/HLLD Cartesian core.
//
// The sponge damps momentum and B_phi near all edges while preserving the
// local thermal pressure, acting as a crude absorbing layer for open-boundary
// experiments.

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Bx_face:    array<f32>;
@group(0) @binding(4) var<storage, read_write> By_face:    array<f32>;
@group(0) @binding(5) var<uniform>             dt_buf:     DtUniform;

fn cell_bx_geom(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
}

fn cell_by_geom(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                + By_face[by_face_idx(ix, iy + 1u, n_total)]);
}

fn pressure_from_state(u0: vec4<f32>, u1: vec4<f32>, bx: f32, by: f32) -> f32 {
    return pressure_from_dual_energy(
        u0,
        u1,
        bx,
        by,
        U_uniforms.gamma,
        U_uniforms.pressure_floor,
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let geom_on = flag_set(U_uniforms.physics_flags, FLAG_GEOMETRY)
               && U_uniforms.geometry_mode == 1u;
    let sponge_on = flag_set(U_uniforms.physics_flags, FLAG_SPONGE)
                 && U_uniforms.sponge_width > 0.0
                 && U_uniforms.sponge_strength > 0.0;
    if (!geom_on && !sponge_on) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    let dt = dt_buf.dt;
    let dx = U_uniforms.dx;
    let bx = cell_bx_geom(ix, iy, n_total);
    let by = cell_by_geom(ix, iy, n_total);
    var u0 = U0[c];
    var u1 = U1[c];
    var rho = max(u0.x, DENSITY_FLOOR);
    var vr = u0.y / rho;
    var vz = u0.z / rho;
    var vphi = u0.w / rho;
    var bphi = u1.y;
    var p = pressure_from_state(u0, u1, bx, by);

    if (geom_on) {
        let r = max(U_uniforms.geometry_r_min + (f32(gid.x) + 0.5) * dx,
                    0.5 * dx);
        let inv_r = 1.0 / r;
        let br = bx;

        // Axisymmetric mass/momentum/toroidal-field source terms. The
        // conservative r-weighted flux part is approximated as a local
        // source, which is acceptable for exploratory runs but documented
        // as a geometry layer rather than a full cylindrical solver.
        let drho = -rho * vr * inv_r * dt;
        let radial_force = (rho * vphi * vphi - bphi * bphi) * inv_r;
        let toroidal_mom = -(rho * vr * vphi - br * bphi) * inv_r;
        let toroidal_B   = -(vr * bphi - vphi * br) * inv_r;

        rho = max(rho + drho, DENSITY_FLOOR);
        let dmx = radial_force * dt;
        let dmz = toroidal_mom * dt;
        let dBphi = toroidal_B * dt;

        u0 = vec4<f32>(
            rho,
            u0.y + dmx,
            u0.z,
            u0.w + dmz,
        );
        u1 = vec4<f32>(
            u1.x + radial_force * vr * dt,
            bphi + dBphi,
            u1.z,
            u1.w,
        );
        rho = max(u0.x, DENSITY_FLOOR);
        vr = u0.y / rho;
        vz = u0.z / rho;
        vphi = u0.w / rho;
        bphi = u1.y;
        p = pressure_from_state(u0, u1, bx, by);
    }

    if (sponge_on) {
        let nx = n_interior - 1u;
        let d0 = min(gid.x, gid.y);
        let d1 = min(nx - gid.x, nx - gid.y);
        let dist = f32(min(d0, d1));
        let width = max(U_uniforms.sponge_width, 1.0e-6);
        if (dist < width) {
            let x = clamp(1.0 - dist / width, 0.0, 1.0);
            let damp = exp(-U_uniforms.sponge_strength * x * x * dt);
            let rho_s = max(u0.x, DENSITY_FLOOR);
            let mx = u0.y * damp;
            let my = u0.z * damp;
            let mz = u0.w * damp;
            let bz = u1.y * damp;
            let ke = 0.5 * (mx*mx + my*my + mz*mz) / rho_s;
            let mb = 0.5 * (bx*bx + by*by + bz*bz);
            u0 = vec4<f32>(rho_s, mx, my, mz);
            u1 = vec4<f32>(ke + mb + p / (U_uniforms.gamma - 1.0),
                            bz, u1.z, u1.w);
        }
    }

    let p_final = pressure_from_state(u0, u1, bx, by);
    U0[c] = u0;
    U1[c] = pack_u1_aux(u1.x, u1.y, max(u0.x, DENSITY_FLOOR), p_final,
                         U_uniforms.gamma, U_uniforms.pressure_floor);
}

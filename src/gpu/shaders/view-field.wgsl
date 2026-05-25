// ─── view-field.wgsl ─────────────────────────────────────────────────
// Extract a scalar field from MHD state into a flat f32 buffer for
// downstream colormapping. View mode set via Uniforms.view_mode:
//   0 = ρ           (default)
//   1 = p           (gas pressure)
//   2 = |v|         (kinetic speed magnitude)
//   3 = |B|         (magnetic field magnitude — uses face averages)
//   4 = Jz          (out-of-plane current density at cell center)
//
// Jz uses central differences on cell-centered B components (each itself
// the average of the two owning faces). This is well-defined under
// periodic wrap and lines up with the |B| mode's averaging.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0_in    (ro)
//   2 U1_in    (ro)
//   3 Bx_face  (ro)
//   4 By_face  (ro)
//   5 field    (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:    array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:  array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:  array<f32>;
@group(0) @binding(5) var<storage, read_write> field:    array<f32>;

fn bx_at(ix: u32, iy: u32, n: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_left_index(ix, iy, n)] + Bx_face[bx_face_right_index(ix, iy, n)]);
}
fn by_at(ix: u32, iy: u32, n: u32) -> f32 {
    return 0.5 * (By_face[by_face_down_index(ix, iy, n)] + By_face[by_face_up_index(ix, iy, n)]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i = i32(n);
    let i   = i32(gid.x);
    let j   = i32(gid.y);

    let idx_c = cell_index(gid.x, gid.y, n);
    let bx_c  = bx_at(gid.x, gid.y, n);
    let by_c  = by_at(gid.x, gid.y, n);

    let U0 = U0_in[idx_c];
    let U1 = U1_in[idx_c];
    let rho = max(U0.x, 1.0e-6);

    let mode = U_uniforms.view_mode;
    var v: f32 = 0.0;
    if (mode == 0u) {
        v = U0.x;                                                      // ρ
    } else if (mode == 1u) {
        let vx = U0.y / rho;
        let vy = U0.z / rho;
        let vz = U0.w / rho;
        let ke = 0.5 * rho * (vx*vx + vy*vy + vz*vz);
        let mb = 0.5 * (bx_c*bx_c + by_c*by_c + U1.y*U1.y);
        v = max((U_uniforms.gamma - 1.0) * (U1.x - ke - mb), 1.0e-6); // p
    } else if (mode == 2u) {
        let vx = U0.y / rho;
        let vy = U0.z / rho;
        let vz = U0.w / rho;
        v = sqrt(vx*vx + vy*vy + vz*vz);                              // |v|
    } else if (mode == 3u) {
        v = sqrt(bx_c*bx_c + by_c*by_c + U1.y*U1.y);                  // |B|
    } else {
        let ix_r = wrap_idx(i + 1, n_i);
        let ix_l = wrap_idx(i - 1, n_i);
        let iy_u = wrap_idx(j + 1, n_i);
        let iy_d = wrap_idx(j - 1, n_i);
        let by_cR = by_at(ix_r, gid.y, n);
        let by_cL = by_at(ix_l, gid.y, n);
        let bx_cU = bx_at(gid.x, iy_u, n);
        let bx_cD = bx_at(gid.x, iy_d, n);
        let dby_dx = (by_cR - by_cL) / (2.0 * U_uniforms.dx);
        let dbx_dy = (bx_cU - bx_cD) / (2.0 * U_uniforms.dx);
        v = dby_dx - dbx_dy;                                           // Jz
    }
    field[idx_c] = v;
}

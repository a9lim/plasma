// ─── view-field.wgsl ─────────────────────────────────────────────────
// Extract a scalar field from MHD state into a flat f32 buffer for
// downstream colormapping. View mode set via Uniforms.view_mode.
//
// Phase 4: reads from ghost-padded buffers; dispatches over interior
// cells only. The field buffer is sized (N+4)² for indexing compat
// with cell-centered storage; only interior cells are written.
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

fn bx_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_left_idx(ix, iy, n_total)]
                + Bx_face[bx_face_right_idx(ix, iy, n_total)]);
}
fn by_at(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_down_idx(ix, iy, n_total)]
                + By_face[by_face_up_idx(ix, iy, n_total)]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;

    let idx_c = cell_idx_total(ix, iy, n_total);
    let bx_c  = bx_at(ix, iy, n_total);
    let by_c  = by_at(ix, iy, n_total);

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
        // Jz = ∂By/∂x - ∂Bx/∂y via central differences. Ghost cells
        // adjacent to interior are filled by apply-bcs, so these
        // neighbour reads are always in-range.
        let by_cR = by_at(ix + 1u, iy,      n_total);
        let by_cL = by_at(ix - 1u, iy,      n_total);
        let bx_cU = bx_at(ix,      iy + 1u, n_total);
        let bx_cD = bx_at(ix,      iy - 1u, n_total);
        let dby_dx = (by_cR - by_cL) / (2.0 * U_uniforms.dx);
        let dbx_dy = (bx_cU - bx_cD) / (2.0 * U_uniforms.dx);
        v = dby_dx - dbx_dy;                                           // Jz
    }
    field[idx_c] = v;
}

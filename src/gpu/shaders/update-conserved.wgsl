// ─── update-conserved.wgsl ───────────────────────────────────────────
// Forward-Euler unsplit update of the 6-component cell-centered
// conserved state from x- and y-direction face fluxes:
//
//   U^{n+1}[i,j] = U^n[i,j]
//                - (dt/dx) · (Fx[i,j] - Fx[i-1,j])
//                - (dt/dx) · (Fy[i,j] - Fy[i,j-1])
//
// (dx = dy.)
//
// Cell-centered state pack:
//   U0 = (ρ, ρvx, ρvy, ρvz)   ← updated from flux_*_0
//   U1 = (E,  Bz,    _pad, _pad) ← E from flux_*_1.x, Bz from flux_*_1.y
//
// Bx and By are face-centered and updated by update-b.wgsl from -∇×E,
// not by this kernel. The transverse-B flux lanes (flux_*_1.z) feed
// compute-emf, also not used here.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0_in    (ro)
//   2 U1_in    (ro)
//   3 flux_x_0 (ro)
//   4 flux_x_1 (ro)
//   5 flux_y_0 (ro)
//   6 flux_y_1 (ro)
//   7 dt_buf   (ro)
//   8 U0_out   (rw)
//   9 U1_out   (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:    array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       flux_x_0: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       flux_x_1: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       flux_y_0: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read>       flux_y_1: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read>       dt_buf:   array<f32, 1>;
@group(0) @binding(8) var<storage, read_write> U0_out:   array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> U1_out:   array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i = i32(n);
    let i   = i32(gid.x);
    let j   = i32(gid.y);

    let idx_c   = cell_index(gid.x, gid.y, n);
    let idx_xlo = cell_index_wrapped(i - 1, j, n_i);   // cell to the left owns x-face at i-½
    let idx_ylo = cell_index_wrapped(i, j - 1, n_i);   // cell below owns y-face at j-½

    let dt   = dt_buf[0];
    let coef = dt / U_uniforms.dx;

    let dFx_0 = flux_x_0[idx_c] - flux_x_0[idx_xlo];
    let dFy_0 = flux_y_0[idx_c] - flux_y_0[idx_ylo];
    let dFx_1 = flux_x_1[idx_c] - flux_x_1[idx_xlo];
    let dFy_1 = flux_y_1[idx_c] - flux_y_1[idx_ylo];

    // Mask out the transverse-B lanes (.z) and the spare (.w) before
    // writing — only U1.x (E) and U1.y (Bz) are real conserved cell-
    // centered quantities. The other two components stay at zero.
    let mask = vec4<f32>(1.0, 1.0, 0.0, 0.0);

    U0_out[idx_c] = U0_in[idx_c] - coef * (dFx_0 + dFy_0);
    U1_out[idx_c] = U1_in[idx_c] - coef * mask * (dFx_1 + dFy_1);
}

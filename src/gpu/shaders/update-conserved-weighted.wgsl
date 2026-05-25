// ─── update-conserved-weighted.wgsl ──────────────────────────────────
// Weighted RK3 SSP stage update for the cell-centered conserved state.
//
//   U_out[i,j] = a0 · U_n[i,j] + a1 · U_other[i,j] + dt_w · dt · L[i,j]
//
// where L is the spatial RHS (-∇·F) and a0/a1/dt_w are SSP weights.
//
// Phase 4: direct indexing into ghost-padded buffers; the flux stencil
// uses flux[i] (LEFT face of cell i) and flux[i+1] (LEFT face of cell
// i+1, which is the RIGHT face of cell i):
//
//   -∂F/∂x ≈ -(flux_x[i+1, j] - flux_x[i, j]) / dx
//   -∂F/∂y ≈ -(flux_y[i, j+1] - flux_y[i, j]) / dx
//
// Dispatch covers interior cells: [ghost, ghost+N) × [ghost, ghost+N).
//
// Bindings:
//   0 uniforms       (uniform)
//   1 stage_params   (uniform) — (a0, a1, dt_w, _)
//   2 U0_n           (ro)
//   3 U1_n           (ro)
//   4 U0_other       (ro)
//   5 U1_other       (ro)
//   6 flux_x_0       (ro)
//   7 flux_x_1       (ro)
//   8 flux_y_0       (ro)
//   9 flux_y_1       (ro)
//  10 dt_buf         (ro)
//  11 U0_out         (rw)
//  12 U1_out         (rw)

struct StageParams {
    a0:    f32,
    a1:    f32,
    dt_w:  f32,
    _pad:  f32,
};

@group(0) @binding(0)  var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1)  var<uniform> stage_params: StageParams;
@group(0) @binding(2)  var<storage, read>       U0_n:     array<vec4<f32>>;
@group(0) @binding(3)  var<storage, read>       U1_n:     array<vec4<f32>>;
@group(0) @binding(4)  var<storage, read>       U0_other: array<vec4<f32>>;
@group(0) @binding(5)  var<storage, read>       U1_other: array<vec4<f32>>;
@group(0) @binding(6)  var<storage, read>       flux_x_0: array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read>       flux_x_1: array<vec4<f32>>;
@group(0) @binding(8)  var<storage, read>       flux_y_0: array<vec4<f32>>;
@group(0) @binding(9)  var<storage, read>       flux_y_1: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read>       dt_buf:   array<f32, 1>;
@group(0) @binding(11) var<storage, read_write> U0_out:   array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> U1_out:   array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;

    let idx_c    = cell_idx_total(ix,      iy,      n_total);
    let idx_xhi  = cell_idx_total(ix + 1u, iy,      n_total);  // right face: flux_x at (i+1, j)
    let idx_yhi  = cell_idx_total(ix,      iy + 1u, n_total);  // top   face: flux_y at (i, j+1)

    let dt   = dt_buf[0];
    let dx   = U_uniforms.dx;
    let scale = stage_params.dt_w * dt / dx;

    let dFx_0 = flux_x_0[idx_xhi] - flux_x_0[idx_c];
    let dFy_0 = flux_y_0[idx_yhi] - flux_y_0[idx_c];
    let dFx_1 = flux_x_1[idx_xhi] - flux_x_1[idx_c];
    let dFy_1 = flux_y_1[idx_yhi] - flux_y_1[idx_c];

    let mask = vec4<f32>(1.0, 1.0, 0.0, 0.0);

    let L0 = -(dFx_0 + dFy_0) / dx;
    let L1 = -mask * (dFx_1 + dFy_1) / dx;

    U0_out[idx_c] =
        stage_params.a0 * U0_n[idx_c]
      + stage_params.a1 * U0_other[idx_c]
      + stage_params.dt_w * dt * L0;
    U1_out[idx_c] =
        stage_params.a0 * U1_n[idx_c]
      + stage_params.a1 * U1_other[idx_c]
      + stage_params.dt_w * dt * L1;
}

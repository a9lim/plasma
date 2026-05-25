// ─── update-b-weighted.wgsl ───────────────────────────────────────────
// Weighted RK3 SSP stage update for the face-centered transverse B field
// via constrained transport.
//
//   Bx_out[i,j] = a0·Bx_n[i,j] + a1·Bx_other[i,j]
//               - dt_w·(dt/dy)·(Ez[i,j] - Ez[i,j-1])
//   By_out[i,j] = a0·By_n[i,j] + a1·By_other[i,j]
//               + dt_w·(dt/dx)·(Ez[i,j] - Ez[i-1,j])
//
// Ez is the edge-EMF computed from the fluxes of U_other (matching the
// L(U_other) operator that drives U_cell). Weights identical to
// update-conserved-weighted.
//
// Bindings:
//   0 uniforms       (uniform)
//   1 stage_params   (uniform) — (a0, a1, dt_w, _)
//   2 Bx_n           (ro)
//   3 By_n           (ro)
//   4 Bx_other       (ro)
//   5 By_other       (ro)
//   6 Ez_edge        (ro)
//   7 dt_buf         (ro)
//   8 Bx_out         (rw)
//   9 By_out         (rw)

struct StageParamsB {
    a0:    f32,
    a1:    f32,
    dt_w:  f32,
    _pad:  f32,
};

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<uniform> stage_params: StageParamsB;
@group(0) @binding(2) var<storage, read>       Bx_n:     array<f32>;
@group(0) @binding(3) var<storage, read>       By_n:     array<f32>;
@group(0) @binding(4) var<storage, read>       Bx_other: array<f32>;
@group(0) @binding(5) var<storage, read>       By_other: array<f32>;
@group(0) @binding(6) var<storage, read>       Ez_edge:  array<f32>;
@group(0) @binding(7) var<storage, read>       dt_buf:   array<f32, 1>;
@group(0) @binding(8) var<storage, read_write> Bx_out:   array<f32>;
@group(0) @binding(9) var<storage, read_write> By_out:   array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i = i32(n);
    let i   = i32(gid.x);
    let j   = i32(gid.y);
    let dx  = U_uniforms.dx;
    let dt  = dt_buf[0];

    let idx_c    = cell_index(gid.x, gid.y, n);
    let idx_lftj = cell_index_wrapped(i - 1, j,     n_i);
    let idx_dwni = cell_index_wrapped(i,     j - 1, n_i);

    let ez_here = Ez_edge[idx_c];
    let ez_left = Ez_edge[idx_lftj];
    let ez_down = Ez_edge[idx_dwni];

    let coef = stage_params.dt_w * dt / dx;
    Bx_out[idx_c] =
        stage_params.a0 * Bx_n[idx_c]
      + stage_params.a1 * Bx_other[idx_c]
      - coef * (ez_here - ez_down);
    By_out[idx_c] =
        stage_params.a0 * By_n[idx_c]
      + stage_params.a1 * By_other[idx_c]
      + coef * (ez_here - ez_left);
}

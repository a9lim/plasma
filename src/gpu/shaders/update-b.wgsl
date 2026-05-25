// ─── update-b.wgsl ───────────────────────────────────────────────────
// Constrained-transport update of the face-centered transverse B field
// from the edge-centered Ez. Forward Euler.
//
//   ∂Bx/∂t = -∂Ez/∂y
//   ∂By/∂t = +∂Ez/∂x
//
// Discrete (Bx at face (i+½, j), edges at (i+½, j±½) = Ez_edge[i, j±½]):
//   Bx_face[i,j]^{n+1} = Bx_face[i,j]^n
//                       - (dt/dy) · (Ez_edge[i, j  ] - Ez_edge[i, j-1])
//   By_face[i,j]^{n+1} = By_face[i,j]^n
//                       + (dt/dx) · (Ez_edge[i, j  ] - Ez_edge[i-1, j])
//
// dx = dy on our square grid.
//
// Why this preserves ∇·B exactly:
//   discrete ∇·B per cell sums the four face contributions; the
//   d/dt update writes -∇×E to faces. The cell-summed divergence of
//   a discrete curl telescopes to zero — each corner Ez appears in
//   the divergence with equal positive and negative contributions
//   that cancel exactly. With Bx/By updated jointly from one Ez,
//   ∇·B preservation is bit-exact (no floating-point drift beyond
//   the ε already in the initial condition).
//
// Bindings:
//   0 uniforms (uniform)
//   1 Bx_in    (ro)
//   2 By_in    (ro)
//   3 Ez_edge  (ro)
//   4 dt_buf   (ro)
//   5 Bx_out   (rw)
//   6 By_out   (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       Bx_in:   array<f32>;
@group(0) @binding(2) var<storage, read>       By_in:   array<f32>;
@group(0) @binding(3) var<storage, read>       Ez_edge: array<f32>;
@group(0) @binding(4) var<storage, read>       dt_buf:  array<f32, 1>;
@group(0) @binding(5) var<storage, read_write> Bx_out:  array<f32>;
@group(0) @binding(6) var<storage, read_write> By_out:  array<f32>;

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

    let ez_here  = Ez_edge[idx_c];
    let ez_left  = Ez_edge[idx_lftj];  // (i-½, j+½) — the edge to the left of this cell's upper-right
    let ez_down  = Ez_edge[idx_dwni];  // (i+½, j-½) — the edge below this cell's upper-right

    let coef = dt / dx;
    Bx_out[idx_c] = Bx_in[idx_c] - coef * (ez_here - ez_down);
    By_out[idx_c] = By_in[idx_c] + coef * (ez_here - ez_left);
}

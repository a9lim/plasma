// ─── compute-emf.wgsl ────────────────────────────────────────────────
// Edge-centered Ez at corner (i+½, j+½) — owned by cell (i,j) — from
// the four neighboring face fluxes. Balsara & Spicer 1999 arithmetic-
// mean CT: no upwinding, just average.
//
// Sign convention from the MHD flux:
//   x-face flux of By:  vx·By - vy·Bx  = -Ez   (at face (i+½, j))
//   y-face flux of Bx:  vy·Bx - vx·By  = +Ez   (at face (i, j+½))
// (Both are stored in flux_*_1.z by riemann-hll.wgsl.)
//
// Four neighbors around edge (i+½, j+½):
//   x-face at (i+½, j)   → -flux_x_1[i,j].z
//   x-face at (i+½, j+1) → -flux_x_1[i,j+1].z
//   y-face at (i,   j+½) → +flux_y_1[i,j].z
//   y-face at (i+1, j+½) → +flux_y_1[i+1,j].z
//
// Ez_edge[i,j] = ¼ · sum.
//
// Upwind CT (Gardiner & Stone 2005) is the Phase-3b upgrade; this simple
// mean is robust but slightly more diffusive on the transverse B
// components.
//
// Bindings:
//   0 uniforms (uniform)
//   1 flux_x_1 (ro)
//   2 flux_y_1 (ro)
//   3 Ez_edge  (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       flux_x_1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       flux_y_1: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Ez_edge:  array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i = i32(n);
    let i   = i32(gid.x);
    let j   = i32(gid.y);

    let idx_c   = cell_index(gid.x, gid.y, n);
    let idx_up  = cell_index_wrapped(i,     j + 1, n_i);
    let idx_rgt = cell_index_wrapped(i + 1, j,     n_i);

    let ez_x_lo = -flux_x_1[idx_c].z;    // x-face at (i+½, j)
    let ez_x_hi = -flux_x_1[idx_up].z;   // x-face at (i+½, j+1)
    let ez_y_lo =  flux_y_1[idx_c].z;    // y-face at (i,   j+½)
    let ez_y_hi =  flux_y_1[idx_rgt].z;  // y-face at (i+1, j+½)

    Ez_edge[idx_c] = 0.25 * (ez_x_lo + ez_x_hi + ez_y_lo + ez_y_hi);
}

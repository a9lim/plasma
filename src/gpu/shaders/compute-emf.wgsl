// ─── compute-emf.wgsl ────────────────────────────────────────────────
// Edge-centered Ez at cell corners from the four neighbouring face
// fluxes. Balsara & Spicer 1999 arithmetic-mean CT.
//
// Phase 4 (LEFT/DOWN face owner convention):
//   Ez_edge[i, j] sits at the corner (x = (i-ghost)·dx, y = (j-ghost)·dx)
//   = BOTTOM-LEFT corner of cell (i, j).
//
// Four neighbouring face fluxes meet at this corner:
//   x-face flux at top    of corner: flux_x_1[i, j-1].z  (left face of cell (i, j-1))
//   x-face flux at bottom of corner: flux_x_1[i, j  ].z  (left face of cell (i, j))
//   y-face flux at right  of corner: flux_y_1[i,   j].z  (bottom face of cell (i, j))
//   y-face flux at left   of corner: flux_y_1[i-1, j].z  (bottom face of cell (i-1, j))
//
// Sign convention from the MHD flux:
//   x-face flux of By:  vx·By - vy·Bx  = -Ez   (stored in flux_x_1.z)
//   y-face flux of Bx:  vy·Bx - vx·By  = +Ez   (stored in flux_y_1.z)
//
// Ez_edge[i, j] = ¼ · ( -flux_x_1[i, j-1].z
//                        -flux_x_1[i, j  ].z
//                        +flux_y_1[i-1, j].z
//                        +flux_y_1[i,   j].z )
//
// Dispatch over corners that bound interior cells:
//   ix ∈ [ghost, ghost + n_interior + 1)
//   iy ∈ [ghost, ghost + n_interior + 1)
// All four neighbour reads are in-range because flux_x covers x-faces
// over rows [ghost-1, ghost+N+1) (1-extended) and flux_y covers y-faces
// over cols [ghost-1, ghost+N+1) (1-extended).
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
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    let extent = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;

    let ez_x_top = -flux_x_1[cell_idx_total(ix, iy - 1u, n_total)].z;
    let ez_x_bot = -flux_x_1[cell_idx_total(ix, iy,       n_total)].z;
    let ez_y_lft =  flux_y_1[cell_idx_total(ix - 1u, iy, n_total)].z;
    let ez_y_rgt =  flux_y_1[cell_idx_total(ix,      iy, n_total)].z;

    Ez_edge[ez_edge_idx(ix, iy, n_total)] = 0.25 * (ez_x_top + ez_x_bot + ez_y_lft + ez_y_rgt);
}

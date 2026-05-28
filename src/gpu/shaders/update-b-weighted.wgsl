// ─── update-b-weighted.wgsl ───────────────────────────────────────────
// Weighted RK3 SSP stage update for the face-centered transverse B field
// via constrained transport.
//
//   Bx_out[i,j] = a0·Bx_n[i,j] + a1·Bx_other[i,j]
//               - dt_w·(dt/dy)·(Ez[i, j+1] - Ez[i, j])
//   By_out[i,j] = a0·By_n[i,j] + a1·By_other[i,j]
//               + dt_w·(dt/dx)·(Ez[i+1, j] - Ez[i, j])
//
// In cylindrical mode, x is radius r and y is axial z. The Br update is
// still -∂Eφ/∂z, but the axial field update must be finite-volume weighted:
//
//   Bz_out += dt · (r_{i+1/2}Eφ_{i+1/2} - r_{i-1/2}Eφ_{i-1/2}) / (r_i Δr)
//
// so the discrete cylindrical divergence
//   (r_{i+1/2}Br_{i+1/2} - r_{i-1/2}Br_{i-1/2})/(r_i Δr) + ∂Bz/∂z
// telescopes with the same corner EMFs.
//
// Ez is the edge-EMF at corners (LEFT/DOWN face owner convention):
//   Ez_edge[i, j] sits at the BOTTOM-LEFT corner of cell (i, j).
//   So x-face Bx_face[i, j] (left face of cell (i, j)) has its bottom
//   corner at Ez_edge[i, j] and top corner at Ez_edge[i, j+1].
//   Similarly y-face By_face[i, j] has left corner at Ez_edge[i, j] and
//   right corner at Ez_edge[i+1, j].
//
// Dispatch ranges (per axis):
//   Bx_face interior faces:  ix ∈ [ghost, ghost+N+1), iy ∈ [ghost, ghost+N)
//                            — N+1 cols × N rows
//   By_face interior faces:  ix ∈ [ghost, ghost+N),  iy ∈ [ghost, ghost+N+1)
//                            — N cols × N+1 rows
//
// The two dispatches use different extents per axis. We compress into
// one kernel by dispatching over (N+1)×(N+1) and branching on which face
// axis the index is valid for; this matches the apply-bcs.wgsl pattern.
//
// Bindings:
//   0 uniforms       (uniform)
//   1 stage_params   (uniform)
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
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let dx         = U_uniforms.dx;
    let dt         = dt_buf[0];
    let coef       = stage_params.dt_w * dt / dx;
    let geom_cyl   = flag_set(U_uniforms.physics_flags, FLAG_GEOMETRY)
                  && U_uniforms.geometry_mode == 1u;

    let ix = gid.x;
    let iy = gid.y;

    // ── Bx_face interior x-faces: ix ∈ [ghost, ghost+N+1), iy ∈ [ghost, ghost+N) ──
    if (ix < n_interior + 1u && iy < n_interior) {
        let bix = ix + ghost;
        let biy = iy + ghost;
        let dst = bx_face_idx(bix, biy, n_total);
        let ez_top = Ez_edge[ez_edge_idx(bix, biy + 1u, n_total)];
        let ez_bot = Ez_edge[ez_edge_idx(bix, biy,      n_total)];
        Bx_out[dst] =
            stage_params.a0 * Bx_n[dst]
          + stage_params.a1 * Bx_other[dst]
          - coef * (ez_top - ez_bot);
    }

    // ── By_face interior y-faces: ix ∈ [ghost, ghost+N), iy ∈ [ghost, ghost+N+1) ──
    if (ix < n_interior && iy < n_interior + 1u) {
        let bix = ix + ghost;
        let biy = iy + ghost;
        let dst = by_face_idx(bix, biy, n_total);
        let ez_rgt = Ez_edge[ez_edge_idx(bix + 1u, biy, n_total)];
        let ez_lft = Ez_edge[ez_edge_idx(bix,      biy, n_total)];
        var curl_e = ez_rgt - ez_lft;
        if (geom_cyl) {
            let r_l = max(U_uniforms.geometry_r_min + f32(ix) * dx, 0.0);
            let r_r = max(U_uniforms.geometry_r_min + (f32(ix) + 1.0) * dx, 0.0);
            let r_c = max(U_uniforms.geometry_r_min + (f32(ix) + 0.5) * dx, 0.5 * dx);
            curl_e = (r_r * ez_rgt - r_l * ez_lft) / r_c;
        }
        By_out[dst] =
            stage_params.a0 * By_n[dst]
          + stage_params.a1 * By_other[dst]
          + coef * curl_e;
    }
}

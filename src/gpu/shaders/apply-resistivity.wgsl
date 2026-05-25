// ─── apply-resistivity.wgsl ───────────────────────────────────────────
// Explicit resistive diffusion. Runs once per RK3 stage AFTER the CT
// face-B update. Adds η ∇²B to:
//   • Bx_face (per x-face, central differences along x and y)
//   • By_face (per y-face, central differences along x and y)
//   • Bz (cell-centered, lives in U1.y)
//
// All reads use direct indexing into the ghost-padded buffers — ghost
// cells were filled by apply-bcs.wgsl at the start of the stage. The
// preceding update-b-weighted writes Bx_face / By_face only at interior
// faces, so ghost faces still hold valid BC-derived values for the
// Laplacian stencil. Similarly, U1's Bz component sits in interior
// cells; ghost cells were filled by apply-bcs.
//
// Update form (per stage, per component B_k):
//   B_k_out[i,j] = B_k_in[i,j] + dt_w · dt · η · ∇²B_k[i,j]
// where ∇² is the standard 5-point central-difference Laplacian. This is
// SSP-compatible by linearity: the resistive operator is linear in B, so
// it composes correctly with the RK3 SSP weighted combinations.
//
// dt_w is the per-stage SSP weight (1, 1/4, 2/3); dt comes from compute-
// dt and includes the parabolic CFL via dt_res = 0.5 · dx² / η (taken
// as min with the hyperbolic CFL inside compute-dt.wgsl).
//
// Concurrency note: within a single compute dispatch WebGPU does not
// synchronize cross-invocation writes, so neighbor reads through a
// `read_write` binding return the dispatch's PRE-CALL value. We rely on
// this: each invocation reads 5 cells (self + 4 neighbors) and writes
// only its own cell. No two invocations write the same destination; the
// stencil pattern is well-defined under this contract.
//
// Bindings:
//   0 uniforms       (uniform)
//   1 stage_params   (uniform)        — (a0, a1, dt_w, _) — only dt_w used
//   2 Bx_face        (rw)             — in-place
//   3 By_face        (rw)             — in-place
//   4 U1_out         (rw)             — Bz lives in U1.y
//   5 dt_buf         (ro)

struct StageParamsResis {
    a0:    f32,
    a1:    f32,
    dt_w:  f32,
    _pad:  f32,
};

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<uniform> stage_params: StageParamsResis;
@group(0) @binding(2) var<storage, read_write> Bx_face:  array<f32>;
@group(0) @binding(3) var<storage, read_write> By_face:  array<f32>;
@group(0) @binding(4) var<storage, read_write> U1_out:   array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       dt_buf:   array<f32, 1>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_total    = U_uniforms.grid_n_total;
    let n_interior = U_uniforms.grid_n;
    let ghost      = U_uniforms.ghost_w;
    let eta        = U_uniforms.eta;
    let dx         = U_uniforms.dx;
    let dt         = dt_buf[0];
    let coef       = stage_params.dt_w * dt * eta / (dx * dx);

    // Early-out: if η is exactly zero, nothing to do.
    if (eta == 0.0) { return; }

    let ix = gid.x;
    let iy = gid.y;

    // ── Bz diffusion (cell-centered, U1.y) ──────────────────────────
    if (ix >= ghost && ix < ghost + n_interior &&
        iy >= ghost && iy < ghost + n_interior) {
        let c    = cell_idx_total(ix,      iy,      n_total);
        let xl   = cell_idx_total(ix - 1u, iy,      n_total);
        let xr   = cell_idx_total(ix + 1u, iy,      n_total);
        let yd   = cell_idx_total(ix,      iy - 1u, n_total);
        let yu   = cell_idx_total(ix,      iy + 1u, n_total);
        let bz_c = U1_out[c].y;
        let lap  = U1_out[xr].y + U1_out[xl].y + U1_out[yu].y + U1_out[yd].y - 4.0 * bz_c;
        var u1   = U1_out[c];
        u1.y = bz_c + coef * lap;
        U1_out[c] = u1;
    }

    // ── Bx_face diffusion ────────────────────────────────────────────
    // Truly INTERIOR x-faces (between two interior cells) are at
    //   i ∈ [ghost+1, ghost + n_interior),  j ∈ [ghost, ghost + n_interior).
    // The boundary x-faces (i = ghost on the W wall, i = ghost + n_interior
    // on the E wall) are OWNED by the BC shader and not diffused —
    // diffusing them would clobber reflecting (Bx=0) or driven values.
    if (ix > ghost && ix < ghost + n_interior &&
        iy >= ghost && iy < ghost + n_interior) {
        let c  = bx_face_idx(ix,      iy,      n_total);
        let xl = bx_face_idx(ix - 1u, iy,      n_total);
        let xr = bx_face_idx(ix + 1u, iy,      n_total);
        let yd = bx_face_idx(ix,      iy - 1u, n_total);
        let yu = bx_face_idx(ix,      iy + 1u, n_total);
        let v   = Bx_face[c];
        let lap = Bx_face[xr] + Bx_face[xl] + Bx_face[yu] + Bx_face[yd] - 4.0 * v;
        Bx_face[c] = v + coef * lap;
    }

    // ── By_face diffusion ────────────────────────────────────────────
    // Truly INTERIOR y-faces:
    //   i ∈ [ghost, ghost + n_interior),    j ∈ [ghost+1, ghost + n_interior).
    if (ix >= ghost && ix < ghost + n_interior &&
        iy > ghost && iy < ghost + n_interior) {
        let c  = by_face_idx(ix,      iy,      n_total);
        let xl = by_face_idx(ix - 1u, iy,      n_total);
        let xr = by_face_idx(ix + 1u, iy,      n_total);
        let yd = by_face_idx(ix,      iy - 1u, n_total);
        let yu = by_face_idx(ix,      iy + 1u, n_total);
        let v   = By_face[c];
        let lap = By_face[xr] + By_face[xl] + By_face[yu] + By_face[yd] - 4.0 * v;
        By_face[c] = v + coef * lap;
    }
}

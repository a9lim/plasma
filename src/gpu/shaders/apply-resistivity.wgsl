// ─── apply-resistivity.wgsl ───────────────────────────────────────────
// Explicit resistive diffusion. Two entry points run back-to-back per
// stage:
//
//   1. `snapshot` — race-free per-cell copy of (Bx_face, By_face, U1)
//      → (Bx_snap, By_snap, U1_snap). Each invocation touches only its
//      own cell, so there are no neighbor reads and no race possible.
//
//   2. `main` — adds η ∇²B to (Bx_face, By_face, U1.y), reading the
//      Laplacian's 5-point stencil from the SNAPSHOTS. Because the
//      snapshots are populated by a prior dispatch (which WebGPU
//      sequences-after), the reads are guaranteed race-free.
//
// History: this shader used to do an in-place RMW on `read_write`
// storage, relying on the (incorrect) assumption that neighbor reads
// inside a dispatch return pre-call values. They don't — WebGPU has no
// cross-invocation memory ordering inside a dispatch, so neighbor reads
// at workgroup-tile boundaries pick up post-write values from already-
// executed tiles. At low η the noise was below the floor; at high η
// it manifested as regular-spacing bright "blebs" at ~workgroup tile
// stride. The snapshot pass eliminates the race entirely.
//
// Update form (per stage, per component B_k):
//   B_k_out[i,j] = B_k_snap[i,j] + dt_w · dt · η · ∇²B_k_snap[i,j]
// where ∇² is the standard 5-point central-difference Laplacian.
// SSP-compatible by linearity.
//
// dt_w is the per-stage SSP weight (1, 1/4, 2/3); dt comes from compute-
// dt and includes the parabolic CFL via dt_res = 0.5 · dx² / η (taken
// as min with the hyperbolic CFL inside compute-dt.wgsl).
//
// Bindings:
//   0 uniforms       (uniform)
//   1 stage_params   (uniform)        — (a0, a1, dt_w, _) — only dt_w used
//   2 Bx_face        (rw)             — destination (main writes)
//   3 By_face        (rw)             — destination (main writes)
//   4 U1_out         (rw)             — destination; Bz lives in U1.y
//   5 dt_buf         (ro)
//   6 Bx_snap        (rw)             — snapshot writes here; main reads
//   7 By_snap        (rw)             — snapshot writes here; main reads
//   8 U1_snap        (rw)             — snapshot writes here; main reads

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
@group(0) @binding(6) var<storage, read_write> Bx_snap:  array<f32>;
@group(0) @binding(7) var<storage, read_write> By_snap:  array<f32>;
@group(0) @binding(8) var<storage, read_write> U1_snap:  array<vec4<f32>>;

// Per-cell copy dst → snap. Race-free: each invocation touches only its
// own buffer cell, no neighbor reads. Dispatched over (N+3)² so that
// the snapshotted region covers exactly the Laplacian's read footprint:
// interior cells [ghost, ghost+N) plus one ghost-cell margin on every
// side (the 5-point stencil reads (i±1, j) and (i, j±1)). Bx_face and
// By_face have one extra index along their normal axis, so the N+3 wide
// dispatch covers them too. Indices are shifted by (ghost − 1) inside
// the shader.
@compute @workgroup_size(8, 8, 1)
fn snapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_total = U_uniforms.grid_n_total;
    let ghost   = U_uniforms.ghost_w;
    let eta     = U_uniforms.eta;
    // Skip the copy entirely when ideal MHD is active — main will also
    // early-out, so the snapshots are never consumed.
    if (eta == 0.0) { return; }

    // Shift dispatch index into [ghost-1, ghost+N+2). Covers interior +
    // 1-cell ghost margin (Laplacian footprint) + the extra face index
    // along the normal axis (high-end bound below picks this up).
    let ix = gid.x + ghost - 1u;
    let iy = gid.y + ghost - 1u;

    // U1: N_total × N_total
    if (ix < n_total && iy < n_total) {
        let c = cell_idx_total(ix, iy, n_total);
        U1_snap[c] = U1_out[c];
    }
    // Bx_face: (N_total+1) × N_total
    if (ix <= n_total && iy < n_total) {
        let cx = bx_face_idx(ix, iy, n_total);
        Bx_snap[cx] = Bx_face[cx];
    }
    // By_face: N_total × (N_total+1)
    if (ix < n_total && iy <= n_total) {
        let cy = by_face_idx(ix, iy, n_total);
        By_snap[cy] = By_face[cy];
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_total    = U_uniforms.grid_n_total;
    let n_interior = U_uniforms.grid_n;
    let ghost      = U_uniforms.ghost_w;
    let eta        = U_uniforms.eta;
    let dx         = U_uniforms.dx;
    let dt         = dt_buf[0];
    let coef       = stage_params.dt_w * dt * eta / (dx * dx);

    // Early-out: if η is exactly zero, nothing to do (and the snapshot
    // dispatch above also no-op'd, so reads from snap would be stale).
    if (eta == 0.0) { return; }

    // Same dispatch shape as `snapshot` — (N+3)² with a (ghost-1) shift.
    // The main pass only writes interior cells/faces; the dispatch is
    // sized to match the snapshot's so a single workgroup count suffices
    // for both. The interior-bounds checks below filter the extras out.
    let ix = gid.x + ghost - 1u;
    let iy = gid.y + ghost - 1u;

    // ── Bz diffusion (cell-centered, U1.y) ──────────────────────────
    if (ix >= ghost && ix < ghost + n_interior &&
        iy >= ghost && iy < ghost + n_interior) {
        let c    = cell_idx_total(ix,      iy,      n_total);
        let xl   = cell_idx_total(ix - 1u, iy,      n_total);
        let xr   = cell_idx_total(ix + 1u, iy,      n_total);
        let yd   = cell_idx_total(ix,      iy - 1u, n_total);
        let yu   = cell_idx_total(ix,      iy + 1u, n_total);
        // Read center + neighbors from SNAPSHOT (race-free).
        let bz_c = U1_snap[c].y;
        let lap  = U1_snap[xr].y + U1_snap[xl].y + U1_snap[yu].y + U1_snap[yd].y - 4.0 * bz_c;
        // Write back to dst U1, preserving the other components (which
        // weren't touched by this stage's CT update).
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
        let v   = Bx_snap[c];
        let lap = Bx_snap[xr] + Bx_snap[xl] + Bx_snap[yu] + Bx_snap[yd] - 4.0 * v;
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
        let v   = By_snap[c];
        let lap = By_snap[xr] + By_snap[xl] + By_snap[yu] + By_snap[yd] - 4.0 * v;
        By_face[c] = v + coef * lap;
    }
}

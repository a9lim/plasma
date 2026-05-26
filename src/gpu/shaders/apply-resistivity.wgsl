// ─── apply-resistivity.wgsl ───────────────────────────────────────────
// RKL2 super-time-stepping for resistive diffusion (Meyer, Diehl & Kupka
// 2014 — Algorithm 1 / Eq 13). One super-step covers the hyperbolic Δt
// with s first-order Laplacian substeps, each within the standard
// forward-Euler stability bound  Δt_sub ≤ ½ · dx² / η_max.
//
// Stability boundary (RKL2):  Δt_super ≤ ((s² + s − 2) / 2) · dt_parabolic
// So for moderate-to-high η where dt_parabolic ≪ dt_hyperbolic,
// `s = ceil(0.5 · (√(1 + 8 · dt_super/dt_parabolic) − 1))` cleanly
// outruns forward-Euler (s grows as √(ratio) but does ~s²-fast work).
//
// Recurrence (MDK14 eq 13):
//   Y_0 = U^n
//   Y_1 = U^n + μ̃_1 · Δt · L(U^n)                                     (j=1)
//   Y_j = μ_j · Y_{j-1} + ν_j · Y_{j-2} + (1 − μ_j − ν_j) · U^n
//         + μ̃_j · Δt · L(Y_{j-1}) + γ̃_j · Δt · L(U^n)                 (j ≥ 2)
//   U^{n+1} = Y_s
//
// L(B) = η ∇²B applied component-wise. η is allowed to be SPATIALLY-
// VARYING (anomalous resistivity, Birn 2001 GEM closure):
//   η_local(i, j) = η_0 + α · max(0, |J_z(i,j)| / J_crit − 1)²
// J_z is computed from face B via central differences. η is FROZEN at
// the start of the super-step (sampled from U^n) — RKL2's stability
// proof assumes a linear time-invariant operator, and re-evaluating η
// every substep would re-enter nonlinear territory. With α = 0 (anomalous
// off) the local-η evaluation skips the J_z compute entirely and returns
// the uniform η_0.
//
// Operator splitting: Lie split (resistivity AFTER the hyperbolic step).
// 1st-order. Strang doubles the cost for negligible gain given the rest
// of the scheme is also 1st-order in time.
//
// ── Pass split / bind-group rationale ─────────────────────────────────
// The recurrence reads three field snapshots (Y_init = U^n, Y_pprev,
// Y_prev) and writes one (Y_curr). Combined into ONE kernel that would
// be 4 buffer sets × 3 components = 12 storage bindings, over the
// per-stage 10 cap. We split into two single-bind-group kernels (in
// separate files) plus a shared snapshot kernel:
//
//   `snapshot`              dst = src (per-cell, race-free).
//   `apply-resistivity-init` Y_tmp = (1−μ−ν)·U^n + ν·Y_{j-2} + γ̃·Δt·L(U^n).
//                            Reads init + pprev; writes tmp. 9 storage.
//   `apply-resistivity-prev` Y_tmp += μ·Y_{j-1} + μ̃·Δt·L(Y_{j-1}).
//                            Reads init (η_local only) + prev; rw tmp. 8 storage.
//
// Host orchestration (sim.js `_encodeResistivitySuperStep`):
//
//   1.  `snapshot` ×3        — copy dst → init, dst → pprev, dst → prev.
//                              (super-step boot; init stays frozen after.)
//   2.  For j = 1..s:
//         a. `apply-resistivity-init`  (writes tmp using ν_j, γ̃_j)
//         b. `apply-resistivity-prev`  (adds μ_j Y_{j-1} + μ̃_j L(Y_{j-1}))
//         c. Rotate roles by rebinding (no GPU copies):
//                 new_prev  ← tmp           (was Y_j)
//                 new_pprev ← old prev      (was Y_{j-1})
//                 new_tmp   ← old pprev     (was Y_{j-2}, free to overwrite)
//   3.  Copy final Y_s (in `prev` after substep s's rotation) → dst.
//
// Total buffer-set footprint: dst (1) + init (1) + 3 rotating sets
// (pprev, prev, tmp) = 5 × 3 buffers = 15 GPU buffers. At N=256:
// 15 × (260²) × 4 B ≈ 4 MB. Negligible vs the U/B main buffers.
//
// ── This file: `snapshot` only ────────────────────────────────────────
// The init / prev kernels live in apply-resistivity-init.wgsl and
// apply-resistivity-prev.wgsl respectively. Separating shader files
// keeps each pipeline at one bind-group with under 10 storage entries.
//
// Bindings (snapshot):
//   0  uniforms
//   1  Bx_src   (ro)
//   2  By_src   (ro)
//   3  U1_src   (ro)
//   4  Bx_dst   (rw)
//   5  By_dst   (rw)
//   6  U1_dst   (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       Bx_src: array<f32>;
@group(0) @binding(2) var<storage, read>       By_src: array<f32>;
@group(0) @binding(3) var<storage, read>       U1_src: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Bx_dst: array<f32>;
@group(0) @binding(5) var<storage, read_write> By_dst: array<f32>;
@group(0) @binding(6) var<storage, read_write> U1_dst: array<vec4<f32>>;

// Race-free per-cell copy. (N+3)² dispatch with (ghost-1) shift —
// covers interior + 1-cell ghost margin (the RKL2 5-point Laplacian
// footprint) + the extra face-index along the normal axis.
@compute @workgroup_size(8, 8, 1)
fn snapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_total = U_uniforms.grid_n_total;
    let ghost   = U_uniforms.ghost_w;

    let ix = gid.x + ghost - 1u;
    let iy = gid.y + ghost - 1u;

    if (ix < n_total && iy < n_total) {
        let c = cell_idx_total(ix, iy, n_total);
        U1_dst[c] = U1_src[c];
    }
    if (ix <= n_total && iy < n_total) {
        let cx = bx_face_idx(ix, iy, n_total);
        Bx_dst[cx] = Bx_src[cx];
    }
    if (ix < n_total && iy <= n_total) {
        let cy = by_face_idx(ix, iy, n_total);
        By_dst[cy] = By_src[cy];
    }
}

// ─── lic-normalize.wgsl ──────────────────────────────────────────────
// Min/max contrast stretch on the per-cell LIC luminance buffer.
//
// Reads the global (min, max) computed by lic-reduce and rewrites
// lic_out[i] := (lic_out[i] - min) / max(max - min, EPS), clamped to
// [0, 1]. In a region with strong field variation min and max are
// well-separated and the colour pass-through is unchanged; in a flat,
// field-free region (where lic-advect just returned average noise),
// the stretch pulls whatever residual variation exists out into the
// full [0, 1] luminance range, so the LIC texture continues to read.
//
// Choice of min/max over mean/std: this is the canonical "show me the
// variation" choice — research-grade vis on LIC traces typically uses
// it. mean/std is gentler but biases against shock-front regions where
// the LIC trace concentrates the texture into a small band of high
// values; min/max preserves the relative dominance of those bands.
//
// Bindings:
//   0 uniforms        (uniform)  — reads grid_n, grid_n_total, ghost_w
//   1 lic_minmax      (ro)       — [min_bits, max_bits] u32 storage
//   2 lic_out         (rw)       — read current cell, write normalized
//
// ── Transpiler audit ───────────────────────────────────────────────
//   • No workgroup-shared memory, no barriers, no atomics on this
//     entry point — purely per-invocation.
//   • bitcast<f32>(u32) confined to this entry (mirrors compute-dt
//     finalize()'s pattern).
//   • No textures, samplers, dynamic offsets, or push constants.

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       lic_minmax: array<u32, 2>;
@group(0) @binding(2) var<storage, read_write> lic_out:    array<f32>;

const NORM_EPS: f32 = 1.0e-4;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    let n_total = U_uniforms.grid_n_total;
    let ghost   = U_uniforms.ghost_w;

    let lo = bitcast<f32>(lic_minmax[0]);
    let hi = bitcast<f32>(lic_minmax[1]);
    let denom = max(hi - lo, NORM_EPS);

    let ix  = gid.x + ghost;
    let iy  = gid.y + ghost;
    let idx = cell_idx_total(ix, iy, n_total);

    let raw  = lic_out[idx];
    let norm = clamp((raw - lo) / denom, 0.0, 1.0);
    lic_out[idx] = norm;
}

// ─── view-field.wgsl ─────────────────────────────────────────────────
// Extract a scalar field from the conservative state into a flat f32
// buffer for downstream colormapping.
//
// Phase 2 default: density (U.x). Other view modes ride a uniform
// switch — pressure (via cons→prim) and |v| are wired so Phase 5's
// UI work doesn't need to touch the shader. View mode encoded in
// uniforms struct... but Phase 2 only ships density. Hold the other
// branches as no-ops to avoid binding-layout churn later.

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U_in:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> field:   array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }
    let idx = cell_index(gid.x, gid.y, n);
    // Phase 2: density only.
    field[idx] = U_in[idx].x;
}

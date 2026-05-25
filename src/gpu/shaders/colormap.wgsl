// ─── colormap.wgsl ───────────────────────────────────────────────────
// Sample the LUT for each interior cell of the scalar field and write a
// vec4<f32> (RGB in [0,1], alpha=1) into the colored buffer.
//
// Phase 4: field buffer is sized (N_total)² (cell-centered storage with
// ghosts). We dispatch over the INTERIOR N×N and write the result into
// the same storage at the SAME ghost-padded index — the composite
// fragment shader samples interior-relative UVs and maps them back to
// ghost-padded indices via the grid_n_total / ghost_w uniforms.
//
// Normalization: u = clamp((v - view_min) / (view_max - view_min), 0, 1)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       field:    array<f32>;
@group(0) @binding(2) var<storage, read>       lut:      array<vec4<f32>, 256>;
@group(0) @binding(3) var<storage, read_write> colored:  array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let idx = cell_idx_total(ix, iy, n_total);
    let v   = field[idx];

    let lo = U_uniforms.view_min;
    let hi = U_uniforms.view_max;
    let span = max(hi - lo, 1.0e-12);
    let u = clamp((v - lo) / span, 0.0, 1.0);

    let t   = u * 255.0;
    let i0  = u32(floor(t));
    let i1  = min(i0 + 1u, 255u);
    let frac = t - f32(i0);

    let c0 = lut[i0];
    let c1 = lut[i1];
    let c  = mix(c0, c1, frac);

    colored[idx] = vec4<f32>(c.rgb, 1.0);
}

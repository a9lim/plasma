// ─── composite.wgsl ──────────────────────────────────────────────────
// Render the colored buffer to the canvas, modulated by the LIC
// luminance. Fullscreen triangle, no vertex buffers. Each pixel samples
// the interior cell whose normalized UV contains it; ghost cells are
// NOT shown.
//
// Phase 4: colored is sized (N_total)², with valid data only at
// interior indices [ghost, ghost+N) per axis.
//
// Phase 6: lic_out is also (N_total)²-sized for indexing parity. The
// LIC pass writes interior cells only; we read at the same ghost-padded
// index as `colored`. Blend:
//   final.rgb = colored.rgb * mix(1.0, L, intensity * 0.5 + 0.5)
// At intensity=0 the colormap passes through unchanged. At intensity=1
// the LIC modulates between 0.5× and 1.5× of the colormap value (since
// the mix factor reaches 1 and L is in [0,1]) — visible but not washing
// out the underlying field colormap.
//
// Bindings (group 0):
//   0: Uniforms       (vertex + fragment)
//   1: colored        (fragment, read storage)
//   2: lic_out        (fragment, read storage)

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> colored: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lic_out: array<f32>;

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vid & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vid >> 1u)) * 4.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(0.5 * (x + 1.0), 0.5 * (y + 1.0));
    return out;
}

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let intensity  = U_uniforms.lic_intensity;
    let nf = f32(n_interior);
    let ix_int = u32(clamp(floor(in.uv.x * nf), 0.0, nf - 1.0));
    let iy_int = u32(clamp(floor(in.uv.y * nf), 0.0, nf - 1.0));
    let idx = cell_idx_total(ix_int + ghost, iy_int + ghost, n_total);

    let base = colored[idx];
    // LIC luminance in [0, 1]; clamp defensively. Remap into a
    // symmetric multiplicative modulation around 1.0 so the average-noise
    // case leaves the colormap untouched and only the L extremes
    // brighten or darken it.
    let L = clamp(lic_out[idx], 0.0, 1.0);
    let alpha = clamp(intensity, 0.0, 2.0);
    // mod_factor sweeps from 1.0 (no modulation when alpha=0) to
    // 0.5 + L (full modulation when alpha=1), which gives a ±50%
    // luminance swing at full intensity. alpha > 1 deepens the swing
    // for users who want a stronger effect (slider exposes up to 2×).
    let mod_factor = mix(1.0, 0.5 + L, alpha);
    return vec4<f32>(base.rgb * mod_factor, 1.0);
}

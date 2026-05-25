// ─── composite.wgsl ──────────────────────────────────────────────────
// Render the colored buffer to the canvas. Fullscreen triangle, no
// vertex buffers. Each pixel samples the cell whose normalized UV
// contains it (nearest-neighbor at low resolution preserves the grid
// texture). LIC composition lives in Phase 6.
//
// Canvas format is non-sRGB (preferred format from the WebGPU spec is
// rgba8unorm / bgra8unorm), so we already-gamma-encoded sRGB values
// pass through unchanged.

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> colored: array<vec4<f32>>;

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vid & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vid >> 1u)) * 4.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    // UV in [0,1]; the Y-flip aligns "grid row 0 at bottom of screen"
    // with the conventional simulation orientation (origin lower-left).
    out.uv = vec2<f32>(0.5 * (x + 1.0), 0.5 * (y + 1.0));
    return out;
}

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = U_uniforms.grid_n;
    let nf = f32(n);
    let ix = u32(clamp(floor(in.uv.x * nf), 0.0, nf - 1.0));
    let iy = u32(clamp(floor(in.uv.y * nf), 0.0, nf - 1.0));
    let idx = cell_index(ix, iy, n);
    return colored[idx];
}

// ─── Phase 1: minimal fullscreen-quad clear shader ───
// Single fullscreen triangle covers the viewport; fragment outputs the
// brand red #e11107. Replaced in Phase 6 by a real composite pass.
//
// Canvas format from navigator.gpu.getPreferredCanvasFormat() is
// non-sRGB (rgba8unorm / bgra8unorm), so we emit gamma-encoded sRGB
// values directly without a linear→sRGB conversion.

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
};

// Fullscreen triangle: 3 vertices at (-1,-1), (3,-1), (-1,3).
// Covers the entire NDC quad with no vertex buffer.
@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vid & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vid >> 1u)) * 4.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// #e11107 = rgb(225, 17, 7) — brand red, gamma-encoded sRGB.
@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.88235, 0.06667, 0.02745, 1.0);
}

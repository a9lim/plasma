// ─── composite.wgsl ──────────────────────────────────────────────────
// Render the colored buffer to the canvas. Fullscreen triangle, no
// vertex buffers. Each pixel samples the interior cell whose normalized
// UV contains it; ghost cells are NOT shown.
//
// Phase 4: colored is sized (N_total)², with valid data only at
// interior indices [ghost, ghost+N) per axis. Fragment shader maps UV
// ∈ [0, 1] to interior cells via (ghost + floor(uv·N_interior)).

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
    out.uv = vec2<f32>(0.5 * (x + 1.0), 0.5 * (y + 1.0));
    return out;
}

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let nf = f32(n_interior);
    let ix_int = u32(clamp(floor(in.uv.x * nf), 0.0, nf - 1.0));
    let iy_int = u32(clamp(floor(in.uv.y * nf), 0.0, nf - 1.0));
    let idx = cell_idx_total(ix_int + ghost, iy_int + ghost, n_total);
    return colored[idx];
}

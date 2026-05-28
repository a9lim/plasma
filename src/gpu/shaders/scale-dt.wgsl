// ─── scale-dt.wgsl ───────────────────────────────────────────────────
// Small utility pass for source-term operator splitting.
//
// compute-dt writes the macro-step dt into dt_in[0] on the GPU. For
// Strang-style source bracketing, source shaders need a half-step dt without
// stalling the queue for CPU readback. This pass writes dt_out[0] = 0.5 dt_in[0]
// and mirrors diagnostic slots so the same DtUniform/storage layout can be
// rebound by source kernels.

@group(0) @binding(0) var<storage, read>       dt_in:  array<f32, 8>;
@group(0) @binding(1) var<storage, read_write> dt_out: array<f32, 8>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    dt_out[0] = 0.5 * dt_in[0];
    dt_out[1] = dt_in[1];
    dt_out[2] = dt_in[2];
    dt_out[3] = dt_in[3];
    dt_out[4] = dt_in[4];
    dt_out[5] = 0.0;
    dt_out[6] = 0.0;
    dt_out[7] = 0.0;
}

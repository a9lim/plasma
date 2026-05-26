// ─── conservation-finalize.wgsl ──────────────────────────────────────
// Second pass of the conservation reduction (see conservation-reduce.wgsl
// for the per-tile partial-sum stage). Single workgroup (64 threads)
// strides over all per-tile partials, accumulates per-thread sums,
// barrier-reduces through workgroup-shared memory, and one thread
// commits the seven final scalars to the output buffer.
//
// At the largest shipped grid (1024², 128×128 = 16384 tiles), one
// thread handles ⌈16384 / 64⌉ = 256 partials × 7 quantities = ~1.8k
// ALU per thread — negligible compared to the per-step RK3 cost.
//
// Bindings:
//   0 uniforms        (uniform)
//   1 tile_partials   (ro f32, max 16384 × 7)
//   2 cons_out        (rw f32, 8 slots — 7 live + 1 pad)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       tile_partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> cons_out:      array<f32>;

const QUANTITY_COUNT: u32 = 7u;
const FINAL_THREADS:  u32 = 64u;

var<workgroup> final_scratch: array<f32, 448>;  // 64 threads × 7 quantities

@compute @workgroup_size(64, 1, 1)
fn finalize(
    @builtin(local_invocation_index) lid: u32,
) {
    let n_interior     = U_uniforms.grid_n;
    let tiles_per_axis = (n_interior + 7u) / 8u;       // ceil(N / WG)
    let num_tiles      = tiles_per_axis * tiles_per_axis;

    var s0: f32 = 0.0;
    var s1: f32 = 0.0;
    var s2: f32 = 0.0;
    var s3: f32 = 0.0;
    var s4: f32 = 0.0;
    var s5: f32 = 0.0;
    var s6: f32 = 0.0;
    for (var t: u32 = lid; t < num_tiles; t = t + FINAL_THREADS) {
        let off = t * QUANTITY_COUNT;
        s0 = s0 + tile_partials[off + 0u];
        s1 = s1 + tile_partials[off + 1u];
        s2 = s2 + tile_partials[off + 2u];
        s3 = s3 + tile_partials[off + 3u];
        s4 = s4 + tile_partials[off + 4u];
        s5 = s5 + tile_partials[off + 5u];
        s6 = s6 + tile_partials[off + 6u];
    }
    let base = lid * QUANTITY_COUNT;
    final_scratch[base + 0u] = s0;
    final_scratch[base + 1u] = s1;
    final_scratch[base + 2u] = s2;
    final_scratch[base + 3u] = s3;
    final_scratch[base + 4u] = s4;
    final_scratch[base + 5u] = s5;
    final_scratch[base + 6u] = s6;

    workgroupBarrier();

    if (lid == 0u) {
        var t0: f32 = 0.0;
        var t1: f32 = 0.0;
        var t2: f32 = 0.0;
        var t3: f32 = 0.0;
        var t4: f32 = 0.0;
        var t5: f32 = 0.0;
        var t6: f32 = 0.0;
        for (var k: u32 = 0u; k < FINAL_THREADS; k = k + 1u) {
            let off = k * QUANTITY_COUNT;
            t0 = t0 + final_scratch[off + 0u];
            t1 = t1 + final_scratch[off + 1u];
            t2 = t2 + final_scratch[off + 2u];
            t3 = t3 + final_scratch[off + 3u];
            t4 = t4 + final_scratch[off + 4u];
            t5 = t5 + final_scratch[off + 5u];
            t6 = t6 + final_scratch[off + 6u];
        }
        cons_out[0] = t0;
        cons_out[1] = t1;
        cons_out[2] = t2;
        cons_out[3] = t3;
        cons_out[4] = t4;
        cons_out[5] = t5;
        cons_out[6] = t6;
        cons_out[7] = 0.0;  // pad for 32 B alignment
    }
}

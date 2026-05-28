// ─── conservation-finalize.wgsl ──────────────────────────────────────
// Final pass for conservation-reduce.wgsl. One workgroup strides over all
// per-tile partials, combines sum/min/max quantities, and writes the scalar
// diagnostics consumed by stats-display.js.

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       tile_partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> cons_out:      array<f32>;

const QUANTITY_COUNT: u32 = 24u;
const FINAL_THREADS:  u32 = 64u;

var<workgroup> final_scratch: array<f32, 1536>;  // 64 threads × 24 quantities

@compute @workgroup_size(64, 1, 1)
fn finalize(
    @builtin(local_invocation_index) lid: u32,
) {
    let n_interior     = U_uniforms.grid_n;
    let tiles_per_axis = (n_interior + 7u) / 8u;
    let num_tiles      = tiles_per_axis * tiles_per_axis;

    var s0:  f32 = 0.0;
    var s1:  f32 = 0.0;
    var s2:  f32 = 0.0;
    var s3:  f32 = 0.0;
    var s4:  f32 = 0.0;
    var s5:  f32 = 0.0;
    var s6:  f32 = 0.0;
    var s7:  f32 = 0.0;
    var s8:  f32 = 0.0;
    var s9:  f32 = 0.0;
    var s10: f32 = 1.0e30;
    var s11: f32 = 0.0;
    var s12: f32 = 0.0;
    var s13: f32 = 0.0;
    var s14: f32 = 0.0;
    var s15: f32 = 0.0;
    var s16: f32 = 0.0;
    var s17: f32 = 0.0;
    var s18: f32 = 0.0;
    var s19: f32 = 0.0;
    var s20: f32 = 0.0;

    for (var t: u32 = lid; t < num_tiles; t = t + FINAL_THREADS) {
        let off = t * QUANTITY_COUNT;
        s0  = s0  + tile_partials[off + 0u];
        s1  = s1  + tile_partials[off + 1u];
        s2  = s2  + tile_partials[off + 2u];
        s3  = s3  + tile_partials[off + 3u];
        s4  = s4  + tile_partials[off + 4u];
        s5  = s5  + tile_partials[off + 5u];
        s6  = s6  + tile_partials[off + 6u];
        s7  = s7  + tile_partials[off + 7u];
        s8  = s8  + tile_partials[off + 8u];
        s9  = s9  + tile_partials[off + 9u];
        s10 = min(s10, tile_partials[off + 10u]);
        s11 = max(s11, tile_partials[off + 11u]);
        s12 = max(s12, tile_partials[off + 12u]);
        s13 = max(s13, tile_partials[off + 13u]);
        s14 = max(s14, tile_partials[off + 14u]);
        s15 = s15 + tile_partials[off + 15u];
        s16 = s16 + tile_partials[off + 16u];
        s17 = s17 + tile_partials[off + 17u];
        s18 = s18 + tile_partials[off + 18u];
        s19 = s19 + tile_partials[off + 19u];
        s20 = s20 + tile_partials[off + 20u];
    }

    let base = lid * QUANTITY_COUNT;
    final_scratch[base + 0u]  = s0;
    final_scratch[base + 1u]  = s1;
    final_scratch[base + 2u]  = s2;
    final_scratch[base + 3u]  = s3;
    final_scratch[base + 4u]  = s4;
    final_scratch[base + 5u]  = s5;
    final_scratch[base + 6u]  = s6;
    final_scratch[base + 7u]  = s7;
    final_scratch[base + 8u]  = s8;
    final_scratch[base + 9u]  = s9;
    final_scratch[base + 10u] = s10;
    final_scratch[base + 11u] = s11;
    final_scratch[base + 12u] = s12;
    final_scratch[base + 13u] = s13;
    final_scratch[base + 14u] = s14;
    final_scratch[base + 15u] = s15;
    final_scratch[base + 16u] = s16;
    final_scratch[base + 17u] = s17;
    final_scratch[base + 18u] = s18;
    final_scratch[base + 19u] = s19;
    final_scratch[base + 20u] = s20;
    final_scratch[base + 21u] = 0.0;
    final_scratch[base + 22u] = 0.0;
    final_scratch[base + 23u] = 0.0;

    workgroupBarrier();

    if (lid == 0u) {
        var t0:  f32 = 0.0;
        var t1:  f32 = 0.0;
        var t2:  f32 = 0.0;
        var t3:  f32 = 0.0;
        var t4:  f32 = 0.0;
        var t5:  f32 = 0.0;
        var t6:  f32 = 0.0;
        var t7:  f32 = 0.0;
        var t8:  f32 = 0.0;
        var t9:  f32 = 0.0;
        var t10: f32 = 1.0e30;
        var t11: f32 = 0.0;
        var t12: f32 = 0.0;
        var t13: f32 = 0.0;
        var t14: f32 = 0.0;
        var t15: f32 = 0.0;
        var t16: f32 = 0.0;
        var t17: f32 = 0.0;
        var t18: f32 = 0.0;
        var t19: f32 = 0.0;
        var t20: f32 = 0.0;
        for (var k: u32 = 0u; k < FINAL_THREADS; k = k + 1u) {
            let off = k * QUANTITY_COUNT;
            t0  = t0  + final_scratch[off + 0u];
            t1  = t1  + final_scratch[off + 1u];
            t2  = t2  + final_scratch[off + 2u];
            t3  = t3  + final_scratch[off + 3u];
            t4  = t4  + final_scratch[off + 4u];
            t5  = t5  + final_scratch[off + 5u];
            t6  = t6  + final_scratch[off + 6u];
            t7  = t7  + final_scratch[off + 7u];
            t8  = t8  + final_scratch[off + 8u];
            t9  = t9  + final_scratch[off + 9u];
            t10 = min(t10, final_scratch[off + 10u]);
            t11 = max(t11, final_scratch[off + 11u]);
            t12 = max(t12, final_scratch[off + 12u]);
            t13 = max(t13, final_scratch[off + 13u]);
            t14 = max(t14, final_scratch[off + 14u]);
            t15 = t15 + final_scratch[off + 15u];
            t16 = t16 + final_scratch[off + 16u];
            t17 = t17 + final_scratch[off + 17u];
            t18 = t18 + final_scratch[off + 18u];
            t19 = t19 + final_scratch[off + 19u];
            t20 = t20 + final_scratch[off + 20u];
        }

        cons_out[0]  = t0;
        cons_out[1]  = t1;
        cons_out[2]  = t2;
        cons_out[3]  = t3;
        cons_out[4]  = t4;
        cons_out[5]  = t5;
        cons_out[6]  = t6;
        cons_out[7]  = t7;
        cons_out[8]  = t8;
        cons_out[9]  = t9;
        cons_out[10] = t10;
        cons_out[11] = t11;
        cons_out[12] = t12;
        cons_out[13] = t13;
        cons_out[14] = t14;
        cons_out[15] = t15;
        cons_out[16] = t16;
        cons_out[17] = t17;
        cons_out[18] = t18;
        cons_out[19] = t19;
        cons_out[20] = t20;
        cons_out[21] = 0.0;
        cons_out[22] = 0.0;
        cons_out[23] = 0.0;
    }
}

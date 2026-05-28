// ─── conservation-reduce.wgsl ────────────────────────────────────────
// Per-tile diagnostic reduction for the Stats tab and conservation panel.
//
// Output slots are straight sums over interior cells unless noted:
//   0  ρ                  1  ρv_x              2  ρv_y
//   3  ρv_z               4  E                 5  ½|B|²
//   6  |∇·B|              7  kinetic energy    8  internal energy
//   9  plasma beta sum    10 beta min (min)    11 beta max (max)
//   12 |B|max (max)       13 |v|max (max)      14 |J_z|max (max)
//   15 (∇·B)²             16 nonfinite count   17 ρ-floor count
//   18 p-floor count      19 Harris ψ column   20 valid-cell count
//   21..23 pad/reserved
//
// The final pass scales by dx² on the CPU where the UI wants integrals.
// Keeping these reductions on-GPU avoids per-cadence full-field readback.

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:       array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:       array<f32>;
@group(0) @binding(5) var<storage, read_write> tile_partials: array<f32>;

const QUANTITY_COUNT: u32 = 24u;
const TILE_THREADS:   u32 = 64u;

var<workgroup> tile_scratch: array<f32, 1536>;  // 64 threads × 24 quantities

fn finiteish(v: f32) -> bool {
    return v == v && abs(v) < 1.0e30;
}

@compute @workgroup_size(8, 8, 1)
fn tile(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let dx_inv     = 1.0 / U_uniforms.dx;
    let p_floor    = U_uniforms.pressure_floor;
    let gamma_m1   = max(U_uniforms.gamma - 1.0, 1.0e-6);

    var c0:  f32 = 0.0;
    var c1:  f32 = 0.0;
    var c2:  f32 = 0.0;
    var c3:  f32 = 0.0;
    var c4:  f32 = 0.0;
    var c5:  f32 = 0.0;
    var c6:  f32 = 0.0;
    var c7:  f32 = 0.0;
    var c8:  f32 = 0.0;
    var c9:  f32 = 0.0;
    var c10: f32 = 1.0e30;
    var c11: f32 = 0.0;
    var c12: f32 = 0.0;
    var c13: f32 = 0.0;
    var c14: f32 = 0.0;
    var c15: f32 = 0.0;
    var c16: f32 = 0.0;
    var c17: f32 = 0.0;
    var c18: f32 = 0.0;
    var c19: f32 = 0.0;
    var c20: f32 = 0.0;

    if (gid.x < n_interior && gid.y < n_interior) {
        let ix = gid.x + ghost;
        let iy = gid.y + ghost;
        let idx = cell_idx_total(ix, iy, n_total);

        let u0 = U0_in[idx];
        let u1 = U1_in[idx];

        let bxL = Bx_face[bx_face_left_idx(ix, iy, n_total)];
        let bxR = Bx_face[bx_face_right_idx(ix, iy, n_total)];
        let byD = By_face[by_face_down_idx(ix, iy, n_total)];
        let byU = By_face[by_face_up_idx(ix, iy, n_total)];
        let bx_c = 0.5 * (bxL + bxR);
        let by_c = 0.5 * (byD + byU);
        let bz_c = u1.y;

        let good = finiteish(u0.x) && finiteish(u0.y) && finiteish(u0.z) && finiteish(u0.w)
                && finiteish(u1.x) && finiteish(u1.y) && finiteish(u1.z) && finiteish(u1.w)
                && finiteish(bxL) && finiteish(bxR) && finiteish(byD) && finiteish(byU);

        if (good) {
            let rho = max(u0.x, DENSITY_FLOOR);
            let vx = u0.y / rho;
            let vy = u0.z / rho;
            let vz = u0.w / rho;
            let ke = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
            let mb = 0.5 * (bx_c * bx_c + by_c * by_c + bz_c * bz_c);
            let eth_floor = p_floor / gamma_m1;
            let eth_total = u1.x - ke - mb;
            let total_ok = finiteish(eth_total)
                        && eth_total > max(eth_floor, DUAL_ENERGY_FRACTION * max(abs(u1.x), eth_floor));
            let eth = select(max(u1.z, eth_floor), eth_total, total_ok);
            let p = max(gamma_m1 * eth, p_floor);
            let beta = (2.0 * p) / max(2.0 * mb, 1.0e-12);
            let bmag = sqrt(max(bx_c * bx_c + by_c * by_c + bz_c * bz_c, 0.0));
            let vmag = sqrt(max(vx * vx + vy * vy + vz * vz, 0.0));
            let divB = (bxR - bxL) * dx_inv + (byU - byD) * dx_inv;

            c0 = u0.x;
            c1 = u0.y;
            c2 = u0.z;
            c3 = u0.w;
            c4 = u1.x;
            c5 = mb;
            c6 = abs(divB);
            c7 = ke;
            c8 = p / gamma_m1;
            c9 = beta;
            c10 = beta;
            c11 = beta;
            c12 = bmag;
            c13 = vmag;
            c15 = divB * divB;
            c20 = 1.0;

            if (u0.x <= 1.001 * DENSITY_FLOOR) { c17 = 1.0; }
            if (p <= 1.001 * p_floor) { c18 = 1.0; }

            if (gid.x == (n_interior >> 1u) && gid.y >= (n_interior >> 1u)) {
                c19 = Bx_face[bx_face_idx(ix, iy, n_total)];
            }

            if (gid.x > 0u && gid.x < n_interior - 1u && gid.y > 0u && gid.y < n_interior - 1u) {
                let byR = 0.5 * (By_face[by_face_idx(ix + 1u, iy, n_total)]
                               + By_face[by_face_idx(ix + 1u, iy + 1u, n_total)]);
                let byL = 0.5 * (By_face[by_face_idx(ix - 1u, iy, n_total)]
                               + By_face[by_face_idx(ix - 1u, iy + 1u, n_total)]);
                let bxU = 0.5 * (Bx_face[bx_face_idx(ix, iy + 1u, n_total)]
                               + Bx_face[bx_face_idx(ix + 1u, iy + 1u, n_total)]);
                let bxD = 0.5 * (Bx_face[bx_face_idx(ix, iy - 1u, n_total)]
                               + Bx_face[bx_face_idx(ix + 1u, iy - 1u, n_total)]);
                c14 = abs((byR - byL) * 0.5 * dx_inv - (bxU - bxD) * 0.5 * dx_inv);
            }
        } else {
            c16 = 1.0;
        }
    }

    let base = lid * QUANTITY_COUNT;
    tile_scratch[base + 0u]  = c0;
    tile_scratch[base + 1u]  = c1;
    tile_scratch[base + 2u]  = c2;
    tile_scratch[base + 3u]  = c3;
    tile_scratch[base + 4u]  = c4;
    tile_scratch[base + 5u]  = c5;
    tile_scratch[base + 6u]  = c6;
    tile_scratch[base + 7u]  = c7;
    tile_scratch[base + 8u]  = c8;
    tile_scratch[base + 9u]  = c9;
    tile_scratch[base + 10u] = c10;
    tile_scratch[base + 11u] = c11;
    tile_scratch[base + 12u] = c12;
    tile_scratch[base + 13u] = c13;
    tile_scratch[base + 14u] = c14;
    tile_scratch[base + 15u] = c15;
    tile_scratch[base + 16u] = c16;
    tile_scratch[base + 17u] = c17;
    tile_scratch[base + 18u] = c18;
    tile_scratch[base + 19u] = c19;
    tile_scratch[base + 20u] = c20;
    tile_scratch[base + 21u] = 0.0;
    tile_scratch[base + 22u] = 0.0;
    tile_scratch[base + 23u] = 0.0;

    workgroupBarrier();

    if (lid == 0u) {
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
        for (var k: u32 = 0u; k < TILE_THREADS; k = k + 1u) {
            let off = k * QUANTITY_COUNT;
            s0  = s0  + tile_scratch[off + 0u];
            s1  = s1  + tile_scratch[off + 1u];
            s2  = s2  + tile_scratch[off + 2u];
            s3  = s3  + tile_scratch[off + 3u];
            s4  = s4  + tile_scratch[off + 4u];
            s5  = s5  + tile_scratch[off + 5u];
            s6  = s6  + tile_scratch[off + 6u];
            s7  = s7  + tile_scratch[off + 7u];
            s8  = s8  + tile_scratch[off + 8u];
            s9  = s9  + tile_scratch[off + 9u];
            s10 = min(s10, tile_scratch[off + 10u]);
            s11 = max(s11, tile_scratch[off + 11u]);
            s12 = max(s12, tile_scratch[off + 12u]);
            s13 = max(s13, tile_scratch[off + 13u]);
            s14 = max(s14, tile_scratch[off + 14u]);
            s15 = s15 + tile_scratch[off + 15u];
            s16 = s16 + tile_scratch[off + 16u];
            s17 = s17 + tile_scratch[off + 17u];
            s18 = s18 + tile_scratch[off + 18u];
            s19 = s19 + tile_scratch[off + 19u];
            s20 = s20 + tile_scratch[off + 20u];
        }
        let tile_idx = wid.y * nwg.x + wid.x;
        let out = tile_idx * QUANTITY_COUNT;
        tile_partials[out + 0u]  = s0;
        tile_partials[out + 1u]  = s1;
        tile_partials[out + 2u]  = s2;
        tile_partials[out + 3u]  = s3;
        tile_partials[out + 4u]  = s4;
        tile_partials[out + 5u]  = s5;
        tile_partials[out + 6u]  = s6;
        tile_partials[out + 7u]  = s7;
        tile_partials[out + 8u]  = s8;
        tile_partials[out + 9u]  = s9;
        tile_partials[out + 10u] = s10;
        tile_partials[out + 11u] = s11;
        tile_partials[out + 12u] = s12;
        tile_partials[out + 13u] = s13;
        tile_partials[out + 14u] = s14;
        tile_partials[out + 15u] = s15;
        tile_partials[out + 16u] = s16;
        tile_partials[out + 17u] = s17;
        tile_partials[out + 18u] = s18;
        tile_partials[out + 19u] = s19;
        tile_partials[out + 20u] = s20;
        tile_partials[out + 21u] = 0.0;
        tile_partials[out + 22u] = 0.0;
        tile_partials[out + 23u] = 0.0;
    }
}

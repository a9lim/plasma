// ─── conservation-reduce.wgsl ────────────────────────────────────────
// Two-pass conservation-diagnostic reduction. Sums seven scalar
// integrals over the interior grid each step:
//
//   slot 0: ∫ρ                  (total mass)
//   slot 1: ∫ρv_x               (total x-momentum)
//   slot 2: ∫ρv_y               (total y-momentum)
//   slot 3: ∫ρv_z               (total z-momentum)
//   slot 4: ∫E                  (total energy density)
//   slot 5: ∫½|B|²              (total magnetic energy)
//   slot 6: ∫|∇·B|              (L1 of solenoidality residual)
//
// Output is NOT multiplied by cell area dx² — the CPU side scales as
// needed for drift % vs. baseline. Mass / energy values therefore read
// as straight summations across interior cells.
//
// ── Why two passes ───────────────────────────────────────────────────
// Per-quantity sums need negative-value support (momentum components).
// The f32-as-u32-via-bitcast atomic-min/max trick that compute-dt and
// lic-reduce use only preserves ordering for non-negative floats — it's
// wrong for atomicAdd in any case (the bit pattern arithmetic of float
// addition doesn't agree with integer addition). The standard GPU
// reduction pattern is:
//
//   Pass 1 (`tile`):  per-workgroup reduction in shared memory; one
//                     thread writes seven f32 partial sums into a
//                     per-tile output slab. Top-level barrier between
//                     per-thread fill and per-tile commit.
//   Pass 2 (`finalize`): single workgroup iterates over all tile
//                        partials, sums them in workgroup-local
//                        arrays, one thread writes the seven final
//                        scalars to the output.
//
// Tile partials buffer is sized for the largest grid we ship (1024² →
// 128×128 = 16384 tiles → 16384 × 7 × 4 B = ~460 KB). Per-tile shared
// memory cost: 7 × 64 × 4 B = 1.75 KB per workgroup — well under WebGPU's
// 16 KB workgroup-storage floor.
//
// The two entry points use SEPARATE bind-group layouts (see
// pipelines.js — `conservationTileBGL` vs `conservationFinalizeBGL`).
// Each pipeline binds the layout it needs; the underlying tile-partials
// storage buffer is the same on the host side.
//
// ── Transpiler contract ──────────────────────────────────────────────
// Mirrors the verified `testTwoDSharedTileHaloBarrier` /
// `testBarrierReduction` patterns in tests/wgsl-transpile/smoke.js:
// one top-level workgroupBarrier(), workgroup-shared arrays of plain
// f32, no atomics (per-tile reduction is serialized through a single
// thread post-barrier), one bind group per pipeline.

// ── Tile pass bindings ───────────────────────────────────────────────
@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:       array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:       array<f32>;
@group(0) @binding(5) var<storage, read_write> tile_partials: array<f32>;

const QUANTITY_COUNT: u32 = 7u;
const TILE_THREADS:   u32 = 64u;

// Per-thread scratch slab. Layout: thread-major then quantity-minor
// (lid * 7 + q). One thread sums after the barrier — sequential
// CPU-style reduction the transpiler maps to a plain loop.
var<workgroup> tile_scratch: array<f32, 448>;  // 64 threads × 7 quantities

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

    // Per-thread contributions. Threads outside the interior just write
    // zero so the per-tile sum is well-defined.
    var c0: f32 = 0.0;  // ρ
    var c1: f32 = 0.0;  // ρ vx
    var c2: f32 = 0.0;  // ρ vy
    var c3: f32 = 0.0;  // ρ vz
    var c4: f32 = 0.0;  // E
    var c5: f32 = 0.0;  // ½|B|²
    var c6: f32 = 0.0;  // |∇·B|

    if (gid.x < n_interior && gid.y < n_interior) {
        let ix = gid.x + ghost;
        let iy = gid.y + ghost;
        let idx = cell_idx_total(ix, iy, n_total);

        let u0 = U0_in[idx];
        let u1 = U1_in[idx];

        // Face-B averaging for cell-centered B_x / B_y.
        let bxL = Bx_face[bx_face_left_idx(ix, iy, n_total)];
        let bxR = Bx_face[bx_face_right_idx(ix, iy, n_total)];
        let byD = By_face[by_face_down_idx(ix, iy, n_total)];
        let byU = By_face[by_face_up_idx(ix, iy, n_total)];
        let bx_c = 0.5 * (bxL + bxR);
        let by_c = 0.5 * (byD + byU);
        let bz_c = u1.y;

        c0 = u0.x;
        c1 = u0.y;
        c2 = u0.z;
        c3 = u0.w;
        c4 = u1.x;
        c5 = 0.5 * (bx_c * bx_c + by_c * by_c + bz_c * bz_c);

        // Discrete ∇·B at cell — staggered-grid formula; CT preserves
        // this at machine precision regardless of EMF recipe.
        let divB = (bxR - bxL) * dx_inv + (byU - byD) * dx_inv;
        c6 = abs(divB);
    }

    let base = lid * QUANTITY_COUNT;
    tile_scratch[base + 0u] = c0;
    tile_scratch[base + 1u] = c1;
    tile_scratch[base + 2u] = c2;
    tile_scratch[base + 3u] = c3;
    tile_scratch[base + 4u] = c4;
    tile_scratch[base + 5u] = c5;
    tile_scratch[base + 6u] = c6;

    workgroupBarrier();

    // Thread 0 of each workgroup commits the tile sum to global storage.
    // Sequential sum over 64 entries × 7 quantities — fits in a few
    // microseconds; mirrors the per-tile commit phase the smoke tests
    // verify.
    if (lid == 0u) {
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        var s2: f32 = 0.0;
        var s3: f32 = 0.0;
        var s4: f32 = 0.0;
        var s5: f32 = 0.0;
        var s6: f32 = 0.0;
        for (var k: u32 = 0u; k < TILE_THREADS; k = k + 1u) {
            let off = k * QUANTITY_COUNT;
            s0 = s0 + tile_scratch[off + 0u];
            s1 = s1 + tile_scratch[off + 1u];
            s2 = s2 + tile_scratch[off + 2u];
            s3 = s3 + tile_scratch[off + 3u];
            s4 = s4 + tile_scratch[off + 4u];
            s5 = s5 + tile_scratch[off + 5u];
            s6 = s6 + tile_scratch[off + 6u];
        }
        let tile_idx = wid.y * nwg.x + wid.x;
        let out_base = tile_idx * QUANTITY_COUNT;
        tile_partials[out_base + 0u] = s0;
        tile_partials[out_base + 1u] = s1;
        tile_partials[out_base + 2u] = s2;
        tile_partials[out_base + 3u] = s3;
        tile_partials[out_base + 4u] = s4;
        tile_partials[out_base + 5u] = s5;
        tile_partials[out_base + 6u] = s6;
    }
}

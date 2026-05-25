// ─── lic-reduce.wgsl ─────────────────────────────────────────────────
// Reduce the interior of `lic_out` to a global (min, max) pair for the
// contrast-normalization pass. Mirrors `compute-dt.wgsl`'s reduction
// shape exactly: per-tile atomicMin/Max into workgroup-shared u32 atoms,
// a single top-level workgroupBarrier(), then thread 0 atomicMin/Max-es
// the tile result into the global `lic_minmax` storage.
//
// Two entry points (transpiler-friendly: each has at most one top-level
// barrier, neither nests barriers inside flow control):
//   reset()  — 1×1 workgroup, initialises lic_minmax to (+inf-ish, 0.0).
//              Encoded once per render frame before `main`.
//   main()   — 8×8 workgroup over interior cells. Per-tile reduce
//              into workgroup-shared tile_min/tile_max, then commit
//              into lic_minmax.
//
// Bitcast trick (same as compute-dt): LIC luminance is bounded in
// [0, 1] by lic-advect's averaging contract, so the u32 bit-pattern
// preserves ordering and atomicMin/Max-on-bitcast just works. We
// still defensively gate with `select(0.0, L, L >= 0 && L == L)` so
// any future change to LIC's range (or a transient NaN) doesn't
// poison the reduction.
//
// Bindings:
//   0 uniforms        (uniform)  — reads grid_n, grid_n_total, ghost_w
//   1 lic_out         (ro)
//   2 lic_minmax      (rw)       — array<atomic<u32>, 2> [min_bits, max_bits]
//
// ── Transpiler audit ───────────────────────────────────────────────
//   • Mirrors compute-dt's barrier-at-top-level pattern.
//   • One workgroup-shared atomic<u32> per reduction direction (min, max).
//   • bitcast<u32>(f32) confined to the entry function — same as compute-dt.
//   • No textures, samplers, dynamic offsets, or push constants.

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       lic_out:    array<f32>;
@group(0) @binding(2) var<storage, read_write> lic_minmax: array<atomic<u32>, 2>;

// Bit pattern for f32 +1.0 — chosen as the initial "min" because LIC
// luminance is in [0, 1]; anything we actually observe is ≤ 1.0 so the
// atomicMin will lower the value monotonically.
const F32_ONE_BITS: u32 = 0x3F800000u;
// Initial "max" — f32 0.0 (LIC luminance is ≥ 0).
const F32_ZERO_BITS: u32 = 0u;

@compute @workgroup_size(1, 1, 1)
fn reset() {
    atomicStore(&lic_minmax[0], F32_ONE_BITS);   // min seed
    atomicStore(&lic_minmax[1], F32_ZERO_BITS);  // max seed
}

var<workgroup> tile_min: atomic<u32>;
var<workgroup> tile_max: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if (lid == 0u) {
        atomicStore(&tile_min, F32_ONE_BITS);
        atomicStore(&tile_max, F32_ZERO_BITS);
    }
    workgroupBarrier();

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x < n_interior && gid.y < n_interior) {
        let ix  = gid.x + ghost;
        let iy  = gid.y + ghost;
        let raw = lic_out[cell_idx_total(ix, iy, n_total)];
        // Defensive sanitization — clamp to [0, 1]. lic-advect's
        // averaging contract already keeps it there but a transient
        // NaN or a future relaxation of the range would otherwise
        // bitcast to a u32 outside the bit-preserving-order window.
        let L_safe = select(0.0, raw, raw >= 0.0 && raw == raw);
        let L      = clamp(L_safe, 0.0, 1.0);
        let bits   = bitcast<u32>(L);
        atomicMin(&tile_min, bits);
        atomicMax(&tile_max, bits);
    }

    workgroupBarrier();
    if (lid == 0u) {
        let lo = atomicLoad(&tile_min);
        let hi = atomicLoad(&tile_max);
        atomicMin(&lic_minmax[0], lo);
        atomicMax(&lic_minmax[1], hi);
    }
}

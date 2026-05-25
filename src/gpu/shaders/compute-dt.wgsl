// ─── compute-dt.wgsl ─────────────────────────────────────────────────
// Two entry points:
//
//   reset:      Single-thread dispatch. Zeroes the atomic wave-speed
//               buffer so reduce can start fresh.
//
//   reduce:     One workgroup per tile; each invocation computes
//               (|vx| + c_s) + (|vy| + c_s) at its cell and combines
//               into a workgroup-local atomic, then to a global atomic.
//               Uses bitcast<u32>(f32) — valid for non-negative floats
//               because IEEE-754 magnitude is monotonic on positive
//               values.
//
//   finalize:   Single-thread dispatch. Reads the bitcasted u32 max
//               wave speed, computes dt = CFL · dx / max_speed, clamps
//               to [DT_MIN, DT_MAX], writes to dt_buf[0].
//
// 2D sweep-split CFL: the per-cell signal speed across the dimensional
// split is max over directions, so a single combined |v|+c_s estimate
// per axis works.  We take the max of the x- and y-direction estimates.

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>           U_in:        array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write>     wavespeed:   atomic<u32>;
@group(0) @binding(3) var<storage, read_write>     dt_buf:      array<f32, 1>;

const CFL_NUMBER: f32 = 0.4;
const DT_MIN: f32 = 1.0e-8;
const DT_MAX: f32 = 1.0e-2;

@compute @workgroup_size(1, 1, 1)
fn reset() {
    atomicStore(&wavespeed, 0u);
}

var<workgroup> tile_max: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if (lid == 0u) { atomicStore(&tile_max, 0u); }
    workgroupBarrier();

    let n = U_uniforms.grid_n;
    if (gid.x < n && gid.y < n) {
        let prim = cons_to_prim(U_in[cell_index(gid.x, gid.y, n)], U_uniforms.gamma);
        let cs   = sound_speed(prim, U_uniforms.gamma);
        // Max over directions — what dimensional splitting actually
        // bounds per substep.
        let sx = abs(prim.y) + cs;
        let sy = abs(prim.z) + cs;
        let s  = max(sx, sy);
        atomicMax(&tile_max, bitcast<u32>(s));
    }

    workgroupBarrier();
    if (lid == 0u) {
        let m = atomicLoad(&tile_max);
        atomicMax(&wavespeed, m);
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize() {
    let s_bits = atomicLoad(&wavespeed);
    let s = max(bitcast<f32>(s_bits), 1.0e-12);
    var dt = CFL_NUMBER * U_uniforms.dx / s;
    dt = clamp(dt, DT_MIN, DT_MAX);
    dt_buf[0] = dt;
}

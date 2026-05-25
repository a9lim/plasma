// ─── compute-dt.wgsl ─────────────────────────────────────────────────
// Three entry points (reset / reduce / finalize), unchanged in shape
// from Phase 2.  The per-cell signal speed estimate now uses the fast
// magnetosonic speed instead of pure sound speed:
//
//   per direction: |v_dir| + c_fast_dir
//   reduce to max over both directions across the grid
//
// c_fast for the x-direction uses Bx² in the Alfvén term; for the y-
// direction it uses By². We take the larger of the two.
//
// Bindings:
//   0 uniforms   (uniform)
//   1 U0_in      (ro)
//   2 U1_in      (ro)
//   3 Bx_face    (ro)
//   4 By_face    (ro)
//   5 wavespeed  (atomic<u32>)
//   6 dt_buf     (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>           U0_in:       array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>           U1_in:       array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>           Bx_face:     array<f32>;
@group(0) @binding(4) var<storage, read>           By_face:     array<f32>;
@group(0) @binding(5) var<storage, read_write>     wavespeed:   atomic<u32>;
@group(0) @binding(6) var<storage, read_write>     dt_buf:      array<f32, 1>;

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
        let bx = 0.5 * (Bx_face[bx_face_left_index(gid.x, gid.y, n)] + Bx_face[bx_face_right_index(gid.x, gid.y, n)]);
        let by = 0.5 * (By_face[by_face_down_index(gid.x, gid.y, n)] + By_face[by_face_up_index(gid.x, gid.y, n)]);
        let idx = cell_index(gid.x, gid.y, n);
        let P  = cons_to_prim_mhd(U0_in[idx], U1_in[idx], bx, by, U_uniforms.gamma);
        let cfx = fast_mag_speed(P, U_uniforms.gamma, 0u);
        let cfy = fast_mag_speed(P, U_uniforms.gamma, 1u);
        let sx = abs(P.vx) + cfx;
        let sy = abs(P.vy) + cfy;
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

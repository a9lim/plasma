// ─── reconstruct-plm.wgsl ────────────────────────────────────────────
// Per-cell PLM slope computation on MHD primitive variables, with
// minmod limiter. Stencil reads neighbor cell-centered primitives along
// the sweep axis.
//
// Primitive pack — vectors a 2× vec4 mirror of the cell-state layout:
//   slopes_x_0[idx] = (ρ', vx', vy', vz')
//   slopes_x_1[idx] = (p', By', Bz', _)   ← x-sweep: transverse Bs are By, Bz
//   slopes_y_0[idx] = (ρ', vx', vy', vz')
//   slopes_y_1[idx] = (p', Bx', Bz', _)   ← y-sweep: transverse Bs are Bx, Bz
//
// The normal-direction B (Bx for x-sweep, By for y-sweep) is NOT
// reconstructed at cell centers — it comes from the face values directly
// (continuous across the face by the staggered CT discretization).
//
// One pipeline writes both _x_ and _y_ slope sets; sweep_dir picks the
// transverse-B mapping. We still emit BOTH directions in this Phase-3a
// implementation by running the pipeline twice (once per direction)
// rather than splitting into separate pipelines — see sim.js.
//
// Bindings (slope-bind-group, used for both _x_ and _y_ via dispatch-time
// bind-group swap):
//   0 uniforms (uniform)
//   1 U0_in   (ro)
//   2 U1_in   (ro)
//   3 Bx_face (ro)
//   4 By_face (ro)
//   5 slopes_0 (rw)  — target half determined by sweep_dir via bind group
//   6 slopes_1 (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_in:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       U1_in:     array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:   array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:   array<f32>;
@group(0) @binding(5) var<storage, read_write> slopes_0:  array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> slopes_1:  array<vec4<f32>>;

fn minmod_scalar(a: f32, b: f32) -> f32 {
    if (a * b <= 0.0) { return 0.0; }
    let s = sign(a);
    return s * min(abs(a), abs(b));
}

fn minmod_vec(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        minmod_scalar(a.x, b.x),
        minmod_scalar(a.y, b.y),
        minmod_scalar(a.z, b.z),
        minmod_scalar(a.w, b.w),
    );
}

// Pack a cell's primitive state into the two vec4s used for slopes.
// For x-sweep:  (ρ, vx, vy, vz),  (p, By, Bz, 0)
// For y-sweep:  (ρ, vx, vy, vz),  (p, Bx, Bz, 0)
fn cell_primitive_pair(ix: u32, iy: u32, n: u32, gamma: f32, axis: u32) -> PrimPair {
    let idx = cell_index(ix, iy, n);
    let bx  = 0.5 * (Bx_face[bx_face_left_index(ix, iy, n)] + Bx_face[bx_face_right_index(ix, iy, n)]);
    let by  = 0.5 * (By_face[by_face_down_index(ix, iy, n)] + By_face[by_face_up_index(ix, iy, n)]);
    let P   = cons_to_prim_mhd(U0_in[idx], U1_in[idx], bx, by, gamma);
    var R: PrimPair;
    R.p0    = vec4<f32>(P.rho, P.vx, P.vy, P.vz);
    if (axis == 0u) {
        R.p1 = vec4<f32>(P.p, P.by, P.bz, 0.0);
    } else {
        R.p1 = vec4<f32>(P.p, P.bx, P.bz, 0.0);
    }
    return R;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i  = i32(n);
    let ix   = i32(gid.x);
    let iy   = i32(gid.y);
    let axis = U_uniforms.sweep_dir;
    let g    = U_uniforms.gamma;

    let p_c = cell_primitive_pair(gid.x, gid.y, n, g, axis);
    var ix_m: u32; var iy_m: u32; var ix_p: u32; var iy_p: u32;
    if (axis == 0u) {
        ix_m = wrap_idx(ix - 1, n_i); iy_m = gid.y;
        ix_p = wrap_idx(ix + 1, n_i); iy_p = gid.y;
    } else {
        ix_m = gid.x; iy_m = wrap_idx(iy - 1, n_i);
        ix_p = gid.x; iy_p = wrap_idx(iy + 1, n_i);
    }
    let p_m = cell_primitive_pair(ix_m, iy_m, n, g, axis);
    let p_p = cell_primitive_pair(ix_p, iy_p, n, g, axis);

    let dx = U_uniforms.dx;
    let sl0 = (p_c.p0 - p_m.p0) / dx;
    let sr0 = (p_p.p0 - p_c.p0) / dx;
    let sl1 = (p_c.p1 - p_m.p1) / dx;
    let sr1 = (p_p.p1 - p_c.p1) / dx;

    let idx = cell_index(gid.x, gid.y, n);
    slopes_0[idx] = minmod_vec(sl0, sr0);
    slopes_1[idx] = minmod_vec(sl1, sr1);
}

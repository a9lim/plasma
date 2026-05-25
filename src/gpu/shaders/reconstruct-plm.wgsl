// ─── reconstruct-plm.wgsl ────────────────────────────────────────────
// Per-cell PLM slope computation on primitive variables (ρ, vx, vy, p)
// with minmod limiter. Output is a vec4 of per-component slopes σ_i
// stored in the same N×N indexing as U. The slope is in (units of q)/dx
// — i.e. already divided by dx. The Riemann pass reconstructs face
// states as q_i ± 0.5·dx·σ_i.
//
// Stencil: cell (i,j) reads (i±1, j) for x-sweep, (i, j±1) for y-sweep.
// Periodic BCs via cell_index_wrapped(); no ghost-cell pass needed.

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U_in:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> slopes:  array<vec4<f32>>;

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i = i32(n);
    let ix  = i32(gid.x);
    let iy  = i32(gid.y);

    // Center cell — convert to primitive once.
    let q_c = cons_to_prim(U_in[cell_index(gid.x, gid.y, n)], U_uniforms.gamma);

    // Neighbor offsets driven by sweep direction.
    var qm: vec4<f32>;
    var qp: vec4<f32>;
    if (U_uniforms.sweep_dir == 0u) {
        qm = cons_to_prim(U_in[cell_index_wrapped(ix - 1, iy, n_i)], U_uniforms.gamma);
        qp = cons_to_prim(U_in[cell_index_wrapped(ix + 1, iy, n_i)], U_uniforms.gamma);
    } else {
        qm = cons_to_prim(U_in[cell_index_wrapped(ix, iy - 1, n_i)], U_uniforms.gamma);
        qp = cons_to_prim(U_in[cell_index_wrapped(ix, iy + 1, n_i)], U_uniforms.gamma);
    }

    let dx = U_uniforms.dx;
    let sigma_l = (q_c - qm) / dx;
    let sigma_r = (qp - q_c) / dx;

    slopes[cell_index(gid.x, gid.y, n)] = minmod_vec(sigma_l, sigma_r);
}

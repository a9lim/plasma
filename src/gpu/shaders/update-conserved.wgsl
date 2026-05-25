// ─── update-conserved.wgsl ───────────────────────────────────────────
// Apply -∇·F to U via forward-Euler:
//
//   U_new[i] = U[i] - (dt/dx) · (F[i] - F[i-1])
//
// where F[i] is the flux at face (i+1/2), i.e. the high face of cell i,
// and F[i-1] is the flux at face (i-1/2), i.e. the high face of cell
// i-1 (its "low" face is our "low" face). Periodic BC via wrap.
//
// dt is supplied as a single-element storage buffer (written by
// compute-dt's finalize entry); this lets the JS side avoid a host-side
// queue.writeBuffer roundtrip every step.

// Bind-group layout (shared with reconstruct-plm / riemann-hll):
//   0 uniforms · 1 U_in (ro) · 2 slopes (rw) · 3 flux (rw) ·
//   4 U_out (rw) · 5 dt_buf (ro)
// Slot 2 (slopes) is unused here — WebGPU permits a shader to use a
// subset of BGL slots, so we omit it from the declarations below.

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U_in:    array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> flux:    array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> U_out:   array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       dt_buf:  array<f32, 1>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i = i32(n);
    let i   = i32(gid.x);
    let j   = i32(gid.y);

    var im = i;
    var jm = j;
    if (U_uniforms.sweep_dir == 0u) { im = i - 1; } else { jm = j - 1; }

    let idx_c  = cell_index(gid.x, gid.y, n);
    let idx_lo = cell_index_wrapped(im, jm, n_i);

    let F_hi = flux[idx_c];   // flux at face (i+1/2)
    let F_lo = flux[idx_lo];  // flux at face (i-1/2) — owned by neighbor

    let dt = dt_buf[0];
    let coef = dt / U_uniforms.dx;
    U_out[idx_c] = U_in[idx_c] - coef * (F_hi - F_lo);
}

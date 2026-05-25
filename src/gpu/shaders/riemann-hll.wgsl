// ─── riemann-hll.wgsl ────────────────────────────────────────────────
// Per-face HLL Riemann solver. "Per face" = per cell, computing the
// flux at the cell's high-side face: i+1/2 for x-sweep, j+1/2 for
// y-sweep. The cell at index (i,j) owns face F[i,j] = F_{i+1/2,j}
// (or F_{i,j+1/2}).
//
// Toro 1999, eq. 10.21:
//   F_HLL = F_L                                    if 0 ≤ S_L
//         = (S_R F_L − S_L F_R + S_L S_R (U_R − U_L)) / (S_R − S_L)
//                                                  if S_L < 0 < S_R
//         = F_R                                    if S_R ≤ 0
//
// Wave-speed estimates (Davis 1988):
//   S_L = min(uL − cL, uR − cR)
//   S_R = max(uL + cL, uR + cR)
//
// L/R primitive states at the face come from PLM slopes:
//   q_L = q_i     + 0.5·dx·σ_i
//   q_R = q_{i+1} − 0.5·dx·σ_{i+1}

// Bind-group layout (shared with reconstruct-plm / update-conserved):
//   0 uniforms · 1 U_in (ro) · 2 slopes (rw) · 3 flux (rw) ·
//   4 U_out (rw) · 5 dt_buf (ro)
// HLL writes flux (3) and reads slopes (2) + U_in (1). U_out and
// dt_buf are unused here and intentionally omitted from the shader
// bindings — WebGPU permits a shader to use a subset of BGL slots.

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U_in:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> slopes:  array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> flux:    array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    if (gid.x >= n || gid.y >= n) { return; }

    let n_i  = i32(n);
    let axis = U_uniforms.sweep_dir;
    let g    = U_uniforms.gamma;
    let dx   = U_uniforms.dx;

    // Center cell and its high-side neighbor along the sweep axis.
    let i  = i32(gid.x);
    let j  = i32(gid.y);
    var ip = i;
    var jp = j;
    if (axis == 0u) { ip = i + 1; } else { jp = j + 1; }

    let idx_c = cell_index(gid.x, gid.y, n);
    let idx_p = cell_index_wrapped(ip, jp, n_i);

    let prim_c = cons_to_prim(U_in[idx_c], g);
    let prim_p = cons_to_prim(U_in[idx_p], g);

    let sig_c = slopes[idx_c];
    let sig_p = slopes[idx_p];

    let half_dx = 0.5 * dx;
    let qL = prim_c + half_dx * sig_c;  // extrapolated forward from cell i
    let qR = prim_p - half_dx * sig_p;  // extrapolated backward from cell i+1

    // Convert L/R back to conservative states for the flux jump term.
    let UL = prim_to_cons(qL, g);
    let UR = prim_to_cons(qR, g);

    let cL = sound_speed(qL, g);
    let cR = sound_speed(qR, g);
    let uL = normal_velocity(qL, axis);
    let uR = normal_velocity(qR, axis);

    let SL = min(uL - cL, uR - cR);
    let SR = max(uL + cL, uR + cR);

    let FL = euler_flux(qL, g, axis);
    let FR = euler_flux(qR, g, axis);

    var F: vec4<f32>;
    if (SL >= 0.0) {
        F = FL;
    } else if (SR <= 0.0) {
        F = FR;
    } else {
        // Guard against SR == SL exactly (degenerate sonic point).
        let denom = max(SR - SL, 1.0e-12);
        F = (SR * FL - SL * FR + SL * SR * (UR - UL)) / denom;
    }

    flux[idx_c] = F;
}

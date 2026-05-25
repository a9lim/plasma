// ─── energy-floor.wgsl ───────────────────────────────────────────────
// Magnetic-pressure-aware energy floor. Runs between update-conserved-
// weighted (step 7) and update-b-weighted (step 8), one invocation per
// interior cell. Sanitizes E in U1.x using
//
//   E_min = KE + ½|B|² + p_floor/(γ-1)
//
// where KE = ½|m|²/ρ uses the just-updated cell-centered conserved
// state, and |B|² uses the SOURCE-stage face B (averaged to cell center)
// plus Bz from U1.y. The face B at this point is whatever the stage
// reads as input — update-b-weighted has NOT yet written this stage's
// destination B (that's step 8), so we get a magnetic floor that's
// consistent with the conserved-state update we're protecting.
//
// History: update-conserved-weighted.wgsl already does a defensive
// floor using `E_min = KE + p_floor/(γ-1)` — omitting the magnetic-
// pressure term because adding Bx_face/By_face bindings there would
// exceed the 10-storage-binding cap. This kernel does the magnetic
// correction in a separate pass with its own (light) binding layout.
// Together with update-conserved's sanitization, the floor is tight at
// thin current sheets where ½|B|² dominates the total energy.
//
// Bindings (5 storage + 1 uniform — well under cap):
//   0 uniforms (uniform)
//   1 U0_out   (ro)              — read ρ, momentum
//   2 U1_out   (rw)              — read Bz, read+write E
//   3 Bx_face  (ro)              — stage input face B
//   4 By_face  (ro)              — stage input face B
//
// Dispatch: interior cells (N × N), workgroup 8×8. No barriers, no
// shared memory, no atomics. Per-invocation only.

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0_out:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1_out:  array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face: array<f32>;
@group(0) @binding(4) var<storage, read>       By_face: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    let u0  = U0_out[c];
    let u1  = U1_out[c];
    let rho = max(u0.x, DENSITY_FLOOR);
    let mx  = u0.y;
    let my  = u0.z;
    let mz  = u0.w;
    let bz  = u1.y;

    // Cell-centered Bx,By from the (input) face arrays.
    let bx_c = 0.5 * (Bx_face[bx_face_left_idx(ix, iy, n_total)]
                    + Bx_face[bx_face_right_idx(ix, iy, n_total)]);
    let by_c = 0.5 * (By_face[by_face_down_idx(ix, iy, n_total)]
                    + By_face[by_face_up_idx(ix, iy, n_total)]);

    let mb = 0.5 * (bx_c*bx_c + by_c*by_c + bz*bz);
    let ke = 0.5 * (mx*mx + my*my + mz*mz) / rho;
    let E_min = ke + mb + U_uniforms.pressure_floor / (U_uniforms.gamma - 1.0);
    let E = clamp(u1.x, E_min, 1.0e30);

    U1_out[c] = vec4<f32>(E, bz, 0.0, 0.0);
}

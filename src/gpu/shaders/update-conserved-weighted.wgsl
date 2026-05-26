// ─── update-conserved-weighted.wgsl ──────────────────────────────────
// Weighted RK3 SSP stage update for the cell-centered conserved state.
//
//   U_out[i,j] = a0 · U_n[i,j] + a1 · U_other[i,j] + dt_w · dt · L[i,j]
//
// where L is the spatial RHS (-∇·F) and a0/a1/dt_w are SSP weights.
//
// Phase 4: direct indexing into ghost-padded buffers; the flux stencil
// uses flux[i] (LEFT face of cell i) and flux[i+1] (LEFT face of cell
// i+1, which is the RIGHT face of cell i):
//
//   -∂F/∂x ≈ -(flux_x[i+1, j] - flux_x[i, j]) / dx
//   -∂F/∂y ≈ -(flux_y[i, j+1] - flux_y[i, j]) / dx
//
// Dispatch covers interior cells: [ghost, ghost+N) × [ghost, ghost+N).
//
// Bindings:
//   0 uniforms       (uniform)
//   1 stage_params   (uniform) — (a0, a1, dt_w, _)
//   2 U0_n           (ro)
//   3 U1_n           (ro)
//   4 U0_other       (ro)
//   5 U1_other       (ro)
//   6 flux_x_0       (ro)
//   7 flux_x_1       (ro)
//   8 flux_y_0       (ro)
//   9 flux_y_1       (ro)
//  10 dt_buf         (uniform — keeps storage-buffer count at 10, the per-stage cap)
//  11 U0_out         (rw)
//  12 U1_out         (rw)

struct StageParams {
    a0:    f32,
    a1:    f32,
    dt_w:  f32,
    _pad:  f32,
};

// Uniform wrapper around the dt scalar. compute-dt writes the underlying
// buffer as storage<read_write>; we read it here as uniform to keep this
// stage's storage-buffer binding count at exactly 10 (the per-stage cap
// most desktop adapters expose). Padded to 16 B to match buffers.js.
struct DtUniform {
    dt:    f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0)  var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1)  var<uniform> stage_params: StageParams;
@group(0) @binding(2)  var<storage, read>       U0_n:     array<vec4<f32>>;
@group(0) @binding(3)  var<storage, read>       U1_n:     array<vec4<f32>>;
@group(0) @binding(4)  var<storage, read>       U0_other: array<vec4<f32>>;
@group(0) @binding(5)  var<storage, read>       U1_other: array<vec4<f32>>;
@group(0) @binding(6)  var<storage, read>       flux_x_0: array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read>       flux_x_1: array<vec4<f32>>;
@group(0) @binding(8)  var<storage, read>       flux_y_0: array<vec4<f32>>;
@group(0) @binding(9)  var<storage, read>       flux_y_1: array<vec4<f32>>;
@group(0) @binding(10) var<uniform>             dt_buf:   DtUniform;
@group(0) @binding(11) var<storage, read_write> U0_out:   array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> U1_out:   array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;

    let idx_c    = cell_idx_total(ix,      iy,      n_total);
    let idx_xhi  = cell_idx_total(ix + 1u, iy,      n_total);  // right face: flux_x at (i+1, j)
    let idx_yhi  = cell_idx_total(ix,      iy + 1u, n_total);  // top   face: flux_y at (i, j+1)

    let dt   = dt_buf.dt;
    let dx   = U_uniforms.dx;
    let scale = stage_params.dt_w * dt / dx;

    let dFx_0 = flux_x_0[idx_xhi] - flux_x_0[idx_c];
    let dFy_0 = flux_y_0[idx_yhi] - flux_y_0[idx_c];
    let dFx_1 = flux_x_1[idx_xhi] - flux_x_1[idx_c];
    let dFy_1 = flux_y_1[idx_yhi] - flux_y_1[idx_c];

    let mask = vec4<f32>(1.0, 1.0, 0.0, 0.0);

    let L0 = -(dFx_0 + dFy_0) / dx;
    let L1 = -mask * (dFx_1 + dFy_1) / dx;

    let u0_raw =
        stage_params.a0 * U0_n[idx_c]
      + stage_params.a1 * U0_other[idx_c]
      + stage_params.dt_w * dt * L0;
    let u1_raw =
        stage_params.a0 * U1_n[idx_c]
      + stage_params.a1 * U1_other[idx_c]
      + stage_params.dt_w * dt * L1;

    // ── Defensive sanitization ──────────────────────────────────────
    // Catches NaN/Inf/sub-floor cells before they cascade through the
    // next RK3 stage's HLLD + wavespeed reduction (a single NaN cell
    // poisons the entire dt computation, and within ~5 steps the whole
    // field is non-finite). The downstream cons_to_prim floor catches
    // residual sub-physical states but only AFTER NaN has already
    // propagated; intervening at the conserved-state write is the only
    // place to break the cycle.
    //
    // NaN handling exploits IEEE-754 maxNum semantics: max(NaN, x) = x
    // and min(NaN, x) = x. So clamp(NaN, low, high) = min(max(NaN, low),
    // high) = min(low, high) = low. Density and energy thus snap to
    // their floors automatically. For momentum and Bz (which can be
    // any sign), we use `x == x` (false for NaN) to gate the value.

    // Momentum: zero if non-finite (interpretation: "no motion").
    var mx = select(0.0, u0_raw.y, u0_raw.y == u0_raw.y);
    var my = select(0.0, u0_raw.z, u0_raw.z == u0_raw.z);
    var mz = select(0.0, u0_raw.w, u0_raw.w == u0_raw.w);

    // Density: clamp to [floor, large]. NaN → floor.
    let rho = clamp(u0_raw.x, DENSITY_FLOOR, 1.0e30);

    // ── Momentum sanitization for the floored-density case ─────────────
    // Session 12 retrospective (fifth Harris bug): when ρ_raw falls below
    // DENSITY_FLOOR at a cell, the bare floor write keeps momentum at
    // whatever HLLD produced. Next stage / next step reads v = m / ρ_floor,
    // which can be huge — KE = m²/(2ρ_floor) = huge, vMax → 1000+, CFL
    // forces dt → DT_MIN, sim explodes. This is what kills tight-loop
    // Harris at step ~150 even with the curl(η J) + corner BC fix landed
    // in Session 11. Mechanism is downstream of (not caused by) sheet
    // thinning: the sheet IS supposed to thin via reconnection, but the
    // outflow region from the X-point should never see ρ → floor under
    // any physical solver — it's a numerical artifact from HLLD over-
    // depleting density at the X-point. The proper fix would be HLLD
    // positivity-preservation (Janhunen 2000-style) but that's a deep
    // re-write. Stop-gap: scale momentum to keep v ≤ V_MAX_SANE when
    // ρ is floored. V_MAX_SANE = 10 ≈ 3×c_f (Harris background Alfvén
    // speed ~2.2, fast speed ~3.5). At ρ_floor = 1e-6 this caps KE per
    // cell at ½·1e-6·100 = 5e-5 — negligible vs background pressure
    // contribution, so the sanitization doesn't bias physics in cells
    // that didn't need it.
    let V_MAX_SANE: f32 = 10.0;
    let v_inv_rho = 1.0 / rho;
    let vx_raw = mx * v_inv_rho;
    let vy_raw = my * v_inv_rho;
    let vz_raw = mz * v_inv_rho;
    let v_mag = sqrt(vx_raw*vx_raw + vy_raw*vy_raw + vz_raw*vz_raw);
    if (v_mag > V_MAX_SANE) {
        let scale = V_MAX_SANE / v_mag;
        mx = mx * scale;
        my = my * scale;
        mz = mz * scale;
    }

    // KE from sanitized momentum and density. Guaranteed finite, bounded.
    let ke = 0.5 * (mx*mx + my*my + mz*mz) / rho;

    // Energy: must be at least KE + p_floor/(γ−1). We can't add the
    // magnetic contribution here (Bx/By live on faces, not in U), but
    // the downstream cons_to_prim floor catches the residual slop.
    let E_min = ke + U_uniforms.pressure_floor / (U_uniforms.gamma - 1.0);
    let E = clamp(u1_raw.x, E_min, 1.0e30);

    // Bz: zero if non-finite.
    let bz = select(0.0, u1_raw.y, u1_raw.y == u1_raw.y);

    U0_out[idx_c] = vec4<f32>(rho, mx, my, mz);
    U1_out[idx_c] = vec4<f32>(E, bz, 0.0, 0.0);
}

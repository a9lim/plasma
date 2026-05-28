// ─── perturb.wgsl ─────────────────────────────────────────────────────
// Pointer-driven user perturbation. Three entry points sharing a single
// bind group:
//
//   apply_drag             — left-click drag: deposit a Gaussian-weighted
//                            momentum impulse along (dvec_x, dvec_y), then
//                            update E for the KE change. Touches only own
//                            cell — race-free.
//
//   apply_excite_b         — right-click drag: divergence-preserving B
//                            perturbation. Uses an analytic vector
//                            potential Az(x,y) whose curl at the click
//                            center equals amp·(dvec_x, dvec_y):
//
//                              Az(x,y) = amp · (dvec_x·dy − dvec_y·dx)
//                                              · exp(-r²/(2σ²))
//
//                            where (dx, dy) = (x − c_x, y − c_y). The
//                            discrete curl on the Yee grid is identically
//                            ∇·B = 0 by the same telescoping argument as
//                            CT — corner contributions cancel pairwise
//                            around every cell.
//
//   apply_excite_energy    — re-syncs cell-centered E with the new |B|²
//                            after apply_excite_b. Reads the freshly-
//                            updated face buffers (guaranteed by inter-
//                            dispatch ordering within a single compute
//                            pass), computes ΔE_mag = ½(|B|²_new −
//                            |B|²_old), adds it to E.
//
// PerturbUniforms layout (32 B):
//   cx, cy        f32  domain-coord center (interior frame: x = (ix - ghost)·dx)
//   dvec_x, dvec_y f32 perturbation direction × magnitude
//   sigma         f32  Gaussian sigma in domain units
//   amplitude     f32  overall scale (the curl at the center will be
//                       amplitude · (dvec_x, dvec_y); for drag the
//                       deposited momentum is amplitude · (dvec_x, dvec_y))
//   _pad0, _pad1  u32
//
// All three entries dispatch over interior cells [0, n)² with workgroup
// 8×8. apply_excite_b also dispatches the rightmost / topmost row+col so
// every interior face has an owning invocation — see the bound check.
//
// Bind group (one BGL shared across all three entries; each entry uses a
// subset of bindings, which WebGPU permits):
//   0  Uniforms                (uniform)
//   1  U0          read_write  — apply_drag writes ρvx/ρvy
//   2  U1          read_write  — apply_drag writes E; apply_excite_energy writes E
//   3  Bx_face     read_write  — apply_excite_b writes
//   4  By_face     read_write  — apply_excite_b writes
//   5  PerturbUniforms         (uniform)

struct PerturbUniforms {
    cx:        f32,
    cy:        f32,
    dvec_x:    f32,
    dvec_y:    f32,
    sigma:     f32,
    amplitude: f32,
    _pad0:     u32,
    _pad1:     u32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Bx_face:    array<f32>;
@group(0) @binding(4) var<storage, read_write> By_face:    array<f32>;
@group(0) @binding(5) var<uniform>             p:          PerturbUniforms;

// Cull threshold for the Gaussian envelope. Beyond ~4σ the contribution
// drops below 1e-3 of peak — skip the work entirely.
const PERTURB_GAUSS_CULL: f32 = 1.0e-4;

// Analytic vector potential Az for the excite perturbation. (x, y) are in
// interior-frame domain units (x = (ix - ghost) · dx).
fn excite_az(x: f32, y: f32) -> f32 {
    let rx = x - p.cx;
    let ry = y - p.cy;
    let r2 = rx * rx + ry * ry;
    let sigma2 = max(p.sigma * p.sigma, 1.0e-12);
    let g = exp(-0.5 * r2 / sigma2);
    return p.amplitude * (p.dvec_x * ry - p.dvec_y * rx) * g;
}

// Gaussian envelope (scalar). Same exp(-r²/2σ²) used by the drag deposit.
fn perturb_gauss(x: f32, y: f32) -> f32 {
    let rx = x - p.cx;
    let ry = y - p.cy;
    let r2 = rx * rx + ry * ry;
    let sigma2 = max(p.sigma * p.sigma, 1.0e-12);
    return exp(-0.5 * r2 / sigma2);
}

// ── Drag: Gaussian momentum deposit ─────────────────────────────────
@compute @workgroup_size(8, 8)
fn apply_drag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= n || iy >= n) { return; }

    let ghost = U_uniforms.ghost_w;
    let nT = U_uniforms.grid_n_total;
    let i = ix + ghost;
    let j = iy + ghost;
    let idx = cell_index(i, j, nT);

    let dxv = U_uniforms.dx;
    // Cell-center in interior-frame domain coords. Matches the apply-bcs
    // / apply-gravity / cooling convention.
    let x = (f32(ix) + 0.5) * dxv;
    let y = (f32(iy) + 0.5) * dxv;

    let w = perturb_gauss(x, y);
    if (w < PERTURB_GAUSS_CULL) { return; }

    let p_floor = U_uniforms.pressure_floor;
    let u0_old = U0[idx];
    let rho = max(u0_old.x, 1.0e-12);

    let dmx = p.amplitude * w * p.dvec_x * rho;
    let dmy = p.amplitude * w * p.dvec_y * rho;

    let mvx_old = u0_old.y;
    let mvy_old = u0_old.z;
    // ΔKE = ((m + δm)² - m²) / (2ρ) = (m·δm)/ρ + |δm|²/(2ρ)
    let dKE = (mvx_old * dmx + mvy_old * dmy) / rho
            + 0.5 * (dmx * dmx + dmy * dmy) / rho;

    var u0 = u0_old;
    u0.y = mvx_old + dmx;
    u0.z = mvy_old + dmy;
    U0[idx] = u0;

    var u1 = U1[idx];
    u1.x = u1.x + dKE;
    // Internal energy unchanged — kinetic injection is reversible work,
    // not heating. K (entropy proxy) untouched for the same reason.
    U1[idx] = u1;
}

// ── Excite (B-field): curl-of-Az face update ─────────────────────────
@compute @workgroup_size(8, 8)
fn apply_excite_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Dispatch shape: (n+1) × (n+1). Each invocation MAY write Bx_face
    // and By_face for the LEFT/BOTTOM face of cell (ix, iy); the extra
    // row+col ensures the rightmost/topmost interior boundary faces also
    // get covered (Bx_face[ghost+n, *] is the right face of cell (n-1, *)
    // and By_face[*, ghost+n] is the top face of cell (*, n-1)).
    let n = U_uniforms.grid_n;
    let ix = gid.x;
    let iy = gid.y;
    if (ix > n || iy > n) { return; }

    let ghost = U_uniforms.ghost_w;
    let nT = U_uniforms.grid_n_total;
    let dxv = U_uniforms.dx;

    // Corner positions in interior-frame domain coords.
    // Corner (ix, iy) sits at the bottom-left of cell (ix, iy), i.e. at
    // (ix · dx, iy · dx).
    let x0 = f32(ix) * dxv;
    let y0 = f32(iy) * dxv;
    let x1 = x0 + dxv;
    let y1 = y0 + dxv;

    let az_00 = excite_az(x0, y0);
    let az_10 = excite_az(x1, y0);
    let az_01 = excite_az(x0, y1);

    // Bx_face[i, j] is the LEFT face of cell (i, j). δBx = ∂Az/∂y.
    // Only write if the face actually belongs to an interior cell — i.e.,
    // ix in [0, n] (one extra to catch the rightmost boundary face) and
    // iy in [0, n).
    if (ix <= n && iy < n) {
        let i = ix + ghost;
        let j = iy + ghost;
        let dBx = (az_01 - az_00) / dxv;
        let bxi = bx_face_idx(i, j, nT);
        Bx_face[bxi] = Bx_face[bxi] + dBx;
    }

    // By_face[i, j] is the BOTTOM face of cell (i, j). δBy = -∂Az/∂x.
    if (ix < n && iy <= n) {
        let i = ix + ghost;
        let j = iy + ghost;
        let dBy = -(az_10 - az_00) / dxv;
        let byi = by_face_idx(i, j, nT);
        By_face[byi] = By_face[byi] + dBy;
    }
}

// ── Excite (E sync): re-pack magnetic energy ─────────────────────────
@compute @workgroup_size(8, 8)
fn apply_excite_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = U_uniforms.grid_n;
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= n || iy >= n) { return; }

    let ghost = U_uniforms.ghost_w;
    let nT = U_uniforms.grid_n_total;
    let i = ix + ghost;
    let j = iy + ghost;
    let idx = cell_index(i, j, nT);
    let dxv = U_uniforms.dx;

    // Compute the four cell-corner δAz values analytically, derive
    // δBx_left/right and δBy_bot/top, then cell-center δBx, δBy.
    let x0 = f32(ix) * dxv;
    let y0 = f32(iy) * dxv;
    let x1 = x0 + dxv;
    let y1 = y0 + dxv;
    let az_00 = excite_az(x0, y0);
    let az_10 = excite_az(x1, y0);
    let az_01 = excite_az(x0, y1);
    let az_11 = excite_az(x1, y1);

    let dBx_left  = (az_01 - az_00) / dxv;
    let dBx_right = (az_11 - az_10) / dxv;
    let dBy_bot   = -(az_10 - az_00) / dxv;
    let dBy_top   = -(az_11 - az_01) / dxv;
    let dBx_cell  = 0.5 * (dBx_left + dBx_right);
    let dBy_cell  = 0.5 * (dBy_bot + dBy_top);

    // Skip the cell if both deltas are negligible — saves the energy work
    // in the vast majority of cells far from the click.
    if (abs(dBx_cell) + abs(dBy_cell) < PERTURB_GAUSS_CULL * max(p.amplitude, 1.0)) { return; }

    let bx_new = 0.5 * (Bx_face[bx_face_left_idx(i, j, nT)]
                      + Bx_face[bx_face_right_idx(i, j, nT)]);
    let by_new = 0.5 * (By_face[by_face_down_idx(i, j, nT)]
                      + By_face[by_face_up_idx(i, j, nT)]);
    let bx_old = bx_new - dBx_cell;
    let by_old = by_new - dBy_cell;

    let dEmag = 0.5 * (bx_new * bx_new + by_new * by_new)
              - 0.5 * (bx_old * bx_old + by_old * by_old);

    var u1 = U1[idx];
    u1.x = u1.x + dEmag;
    U1[idx] = u1;
}

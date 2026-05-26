// ─── compute-emf.wgsl ────────────────────────────────────────────────
// Edge-centered Ez at cell corners from face Riemann fluxes + cell-
// centered Ez, using the Gardiner & Stone 2005 upwind CT EMF formula
// (eqns 41-45). Replaces the Balsara-Spicer arithmetic-mean Ez used
// through v1 — the upwind version is the research-code-standard CT
// formula (Athena++ default; Stone+ 2008 §3.5), avoids grid-aligned
// dissipation, and better preserves plane-parallel flows.
//
// ── Geometry ─────────────────────────────────────────────────────────
// Phase 4 (LEFT/DOWN face owner convention):
//   Ez_edge[ix, iy] sits at the BOTTOM-LEFT corner of cell (ix, iy),
//   physically at (x, y) = ((ix-ghost-½)·dx, (iy-ghost-½)·dx).
//
// In Gardiner-Stone notation, that corner is (i+½, j+½) with
//   i = ix - 1,   j = iy - 1
// so the four CELLS around the corner map to shader indices as:
//        G&S          ↦  shader cell
//   (i,   j  )     ↦  (ix-1, iy-1)   SW
//   (i+1, j  )     ↦  (ix,   iy-1)   SE
//   (i,   j+1)     ↦  (ix-1, iy  )   NW
//   (i+1, j+1)     ↦  (ix,   iy  )   NE
// and the four FACES adjacent to the corner map as:
//   x-face (i+½, j  ) lo  ↦  flux_x_1[ix,   iy-1]   (below corner)
//   x-face (i+½, j+1) hi  ↦  flux_x_1[ix,   iy  ]   (above corner)
//   y-face (i,   j+½) le  ↦  flux_y_1[ix-1, iy  ]   (left  of corner)
//   y-face (i+1, j+½) ri  ↦  flux_y_1[ix,   iy  ]   (right of corner)
//
// ── Sign convention ──────────────────────────────────────────────────
// The MHD flux of magnetic field carries Ez with these signs:
//   x-face flux of By = vx·By - vy·Bx = -Ez  →  Ez_face = -flux_x_1.z
//   y-face flux of Bx = vy·Bx - vx·By = +Ez  →  Ez_face = +flux_y_1.z
// Cell-centered Ez uses the same convention: Ez_cell = vy·Bx - vx·By.
//
// ── Contact-velocity channel ────────────────────────────────────────
// riemann-hlld.wgsl now stashes the contact-wave speed S_M (M&K 2005
// eq 38; coincides with the HLL contact estimate so it's defined in
// every branch) into flux_*_1.w. compute-emf reads this as the upwind
// selector: at an x-face, S_M is the face-normal velocity v_x*; at a
// y-face, S_M is v_y*. The sign of v_x* picks the upwind cell for the
// y-derivative at an x-face (i.e. for the y-half of the formula);
// the sign of v_y* picks the upwind cell for the x-derivative at a
// y-face (the x-half of the formula).
//
// ── Gardiner-Stone 2005 eqns (41)-(45), worked out ──────────────────
// Eq 45: Ez_{i+½,j+½} =
//     ¼ (Ez^x_lo + Ez^x_hi + Ez^y_le + Ez^y_ri)
//   + (dy/8) [(∂Ez/∂y)_{i+½, j+¼} - (∂Ez/∂y)_{i+½, j+¾}]
//   + (dx/8) [(∂Ez/∂x)_{i+¼, j+½} - (∂Ez/∂x)_{i+¾, j+½}]
//
// Eqns 41-44 give the upwind one-sided quarter-derivatives. Each is a
// half-cell-distance finite difference from the upwind cell-center to
// the face, picking the cell upwind of the face. At (i+½, j+¼):
//     if v_x^*(i+½, j) > 0:  ∂y = (Ez^x_lo - Ez^c_{i,   j}) / (dy/2)
//     if v_x^*(i+½, j) < 0:  ∂y = (Ez^x_lo - Ez^c_{i+1, j}) / (dy/2)
//     if v_x^*(i+½, j) = 0:  ∂y = ½ · (sum of both)
// and at (i+½, j+¾) (above the corner, sign flips because we walk from
// the corner upward TO the cell-center):
//     if v_x^*(i+½, j+1) > 0:  ∂y = (Ez^c_{i,   j+1} - Ez^x_hi) / (dy/2)
//     if v_x^*(i+½, j+1) < 0:  ∂y = (Ez^c_{i+1, j+1} - Ez^x_hi) / (dy/2)
//     if v_x^*(i+½, j+1) = 0:  ∂y = ½ · (sum of both)
// Symmetrically for the x-derivatives at y-faces, with v_y^* selecting
// the upwind cell between (i, ·) and (i+1, ·).
//
// Plugging into eq 45 and absorbing the (dy/8)·(2/dy) = ¼ prefactor
// (and the symmetric x-side ¼), the formula collapses to (Stone+ 2008
// §3.5, eq 23 form):
//
// Ez_corner =
//      ¼ (Ez^x_lo + Ez^x_hi + Ez^y_le + Ez^y_ri)                       — BS base
//    + ¼ (Ez^x_lo - Ez^c_{up_lo, j}) + ¼ (Ez^x_hi - Ez^c_{up_hi, j+1})
//    + ¼ (Ez^y_le - Ez^c_{i,   up_le}) + ¼ (Ez^y_ri - Ez^c_{i+1, up_ri})
//
// where up_lo/up_hi are chosen by sign(v_x^*) at the lo/hi x-faces and
// up_le/up_ri by sign(v_y^*) at the le/ri y-faces. This is the form
// implemented below, with `select(...)` for the upwind branches and a
// half-and-half average at exactly zero contact velocity (the v=0
// case from G&S 2005 eqn 42-44 "otherwise" clause).
//
// ── Smooth-field limit ──────────────────────────────────────────────
// If face Ez values exactly equal the cell-centered Ez at the upwind
// neighbor (uniform Ez field), each correction term vanishes and the
// formula reduces to the BS arithmetic mean. So a constant background
// Ez is reproduced exactly.
//
// ── Degenerate (Bn ≈ 0) limit ───────────────────────────────────────
// SM_face is well-defined whenever rcR ≠ rcL; under near-zero face B
// (Branch A in HLLD), HLLC fires and SM is still computed from the
// same M&K 2005 formula. Both lo/hi (or le/ri) faces independently
// select their upwind side — no special case needed here, and the
// formula falls smoothly back toward an upwind-of-cell-Ez average.
//
// ── Bindings ────────────────────────────────────────────────────────
//   0 uniforms (uniform)
//   1 flux_x_1 (ro)       — (-,-,-Ez_x, SM=v_x*)
//   2 flux_y_1 (ro)       — (-,-,+Ez_y, SM=v_y*)
//   3 Ez_edge  (rw)
//   4 U0       (ro)       — (ρ, ρvx, ρvy, ρvz)  for cell-Ez velocity
//   5 Bx_face  (ro)       — for cell-Ez Bx average
//   6 By_face  (ro)       — for cell-Ez By average
//
// 6 storage bindings, well under the 10-per-pipeline cap (HANDOFF
// Session 2 §1).

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       flux_x_1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       flux_y_1: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Ez_edge:  array<f32>;
@group(0) @binding(4) var<storage, read>       U0:       array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       Bx_face:  array<f32>;
@group(0) @binding(6) var<storage, read>       By_face:  array<f32>;

// Cell-centered Ez = vy·Bx - vx·By. Reads ρ + momenta from U0 and
// averages the two adjacent face B values for cell-center B.
fn ez_cell(ix: u32, iy: u32, n_total: u32) -> f32 {
    let u0 = U0[cell_idx_total(ix, iy, n_total)];
    let rho = max(u0.x, DENSITY_FLOOR);
    let vx = u0.y / rho;
    let vy = u0.z / rho;
    let bx = 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                  + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
    let by = 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                  + By_face[by_face_idx(ix, iy + 1u, n_total)]);
    return vy * bx - vx * by;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    let extent = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }

    let ix = ghost + gid.x;
    let iy = ghost + gid.y;

    // ── Face Ez values (Riemann-solver outputs) ─────────────────────
    // Sign convention: x-face flux stores -Ez, y-face flux stores +Ez.
    let fxl = flux_x_1[cell_idx_total(ix, iy - 1u, n_total)]; // x-face below corner
    let fxh = flux_x_1[cell_idx_total(ix, iy,       n_total)]; // x-face above corner
    let fyl = flux_y_1[cell_idx_total(ix - 1u, iy, n_total)]; // y-face left of corner
    let fyr = flux_y_1[cell_idx_total(ix,      iy, n_total)]; // y-face right of corner

    let ez_x_lo = -fxl.z;
    let ez_x_hi = -fxh.z;
    let ez_y_le =  fyl.z;
    let ez_y_ri =  fyr.z;

    // ── Contact velocities at the four faces (HLLD S_M / HLL contact) ─
    // .w slot, stashed by riemann-hlld.wgsl in every branch. At an
    // x-face this is v_x*; at a y-face this is v_y*. Used as the
    // upwind selector for the G&S 2005 quarter-derivatives.
    let vx_lo = fxl.w;
    let vx_hi = fxh.w;
    let vy_le = fyl.w;
    let vy_ri = fyr.w;

    // ── Cell-centered Ez at the four neighbours ─────────────────────
    //   (ix-1, iy-1) SW    (ix, iy-1) SE
    //   (ix-1, iy  ) NW    (ix, iy  ) NE
    let ez_sw = ez_cell(ix - 1u, iy - 1u, n_total);
    let ez_se = ez_cell(ix,      iy - 1u, n_total);
    let ez_nw = ez_cell(ix - 1u, iy,      n_total);
    let ez_ne = ez_cell(ix,      iy,      n_total);

    // ── Upwind cell selection at each face (G&S 2005 eqns 41-44) ────
    // At lo x-face (below corner, j-row): v_x>0 → SW (i,j); v_x<0 → SE (i+1,j).
    //   v_x = 0 ⇒ "otherwise" branch: average of both.
    let TOL: f32 = 1.0e-12;
    let up_lo = select(
                  select(0.5 * (ez_sw + ez_se), ez_se, vx_lo < -TOL),
                  ez_sw, vx_lo >  TOL);

    // At hi x-face (above corner, (j+1)-row): v_x>0 → NW (i,j+1); v_x<0 → NE (i+1,j+1).
    let up_hi = select(
                  select(0.5 * (ez_nw + ez_ne), ez_ne, vx_hi < -TOL),
                  ez_nw, vx_hi >  TOL);

    // At le y-face (left of corner, i-col): v_y>0 → SW (i,j); v_y<0 → NW (i,j+1).
    let up_le = select(
                  select(0.5 * (ez_sw + ez_nw), ez_nw, vy_le < -TOL),
                  ez_sw, vy_le >  TOL);

    // At ri y-face (right of corner, (i+1)-col): v_y>0 → SE (i+1,j); v_y<0 → NE (i+1,j+1).
    let up_ri = select(
                  select(0.5 * (ez_se + ez_ne), ez_ne, vy_ri < -TOL),
                  ez_se, vy_ri >  TOL);

    // ── Balsara-Spicer 1999 arithmetic-mean corner EMF (temporary) ──
    // The Gardiner-Stone 2005 upwind formulation above-the-line in
    // the file header decomposes to:
    //   Ez_corner = ½·(four face Ez) − ¼·(four upwind cell Ez)
    // which is consistent in the smooth-flow limit but empirically
    // produces a runaway CT B-update on Orszag-Tang at N=256, η=0.
    // The OT cascade onsets at step ~250, exactly when the
    // Riemann-solver face Ez at the four current sheets reaches
    // ~2× the local cell Ez = vy·Bx − vx·By and the formula
    // extrapolates outside the input range. The bisect that
    // located this is documented in HANDOFF Session 9 — the upwind
    // formula matches both the in-file derivation AND Stone+ 2008
    // §3.5 eq 23, so the bug is subtle and pending re-investigation
    // (likely flux-sign convention mismatch with HLLD's output, or
    // a missing damping coefficient that Athena++ ships).
    //
    // Until the upwind formula is repaired, fall back to the
    // Balsara-Spicer 1999 arithmetic mean. This costs us Session 4's
    // claimed sharpness improvement on plane-parallel flows but
    // restores OT stability. The cell-Ez upwind machinery above
    // (lines 162-188) and the U0/Bx_face/By_face bindings (4-6)
    // are left in place so re-introducing a corrected upwind term
    // doesn't require pipeline-layout changes.
    Ez_edge[ez_edge_idx(ix, iy, n_total)] = 0.25 * (ez_x_lo + ez_x_hi + ez_y_le + ez_y_ri);
}

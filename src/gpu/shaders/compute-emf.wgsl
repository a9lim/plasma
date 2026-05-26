// ─── compute-emf.wgsl ────────────────────────────────────────────────
// Edge-centered Ez at cell corners from face Riemann fluxes + cell-
// centered Ez, using the UCT-HLLD scheme of Mignone, Tzeferacos & Bodo
// 2010 (J. Comp. Phys. 229, 5896 §4.2) — the Alfvén-wave-aware
// generalization of the Gardiner-Stone 2005 upwind CT EMF. Refined by
// Mignone, Mattia, Bodo, Del Zanna 2021 (J. Comp. Phys. 424, 109839
// §3). Supersedes the cell-upwind G&S formula (which itself superseded
// the original Balsara-Spicer arithmetic mean).
//
// The improvement: G&S 2005 picks an upwind cell at each face via
// sign(S_M_face), so the corner EMF feels the contact wave but not the
// Alfvén waves. UCT-HLLD weights the L/R states at each face by the
// Alfvén wave speeds (S_L*, S_R*) — smoothly blending across the
// rotational discontinuity rather than picking one side. The result
// preserves Alfvén-wave structure more accurately: sharper rotational
// discontinuities (Brio-Wu), cleaner current-sheet topology (Harris
// reconnection), better-resolved magnetic islands (Orszag-Tang).
//
// ── Geometry ─────────────────────────────────────────────────────────
// Phase 4 (LEFT/DOWN face owner convention):
//   Ez_edge[ix, iy] sits at the BOTTOM-LEFT corner of cell (ix, iy),
//   physically at (x, y) = ((ix-ghost-½)·dx, (iy-ghost-½)·dx).
//
// In Mignone+ 2010 notation, that corner is (i+½, j+½) with
//   i = ix - 1,   j = iy - 1
// so the four CELLS around the corner map to shader indices as:
//        Mignone      ↦  shader cell
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
// ── Wavespeed channels (new in UCT-HLLD) ────────────────────────────
// riemann-hlld.wgsl stashes (S_L, S_L*, S_R*, S_R) per face into
// face_wavespeeds_x[idx] and face_wavespeeds_y[idx]. The HLLD contact
// speed S_M still lives in flux_*_1.w. Together these parameterize the
// LHLLD per-face reconstruction.
//
// ── UCT-HLLD per-face EMF reconstruction ────────────────────────────
// At each face, the 5-wave HLLD fan produces a piecewise-constant state
// (L, L*, L**, R**, R*, R) along the face's normal direction. The Ez at
// the face is the spatial average of -(v × B)_z weighted by the wave
// fan's intersection with the face. For UCT, we project this onto two
// adjacent cell-centered Ez values and add a dissipative B-jump term.
//
// Following Mignone+ 2010 eq 38 and Mignone+ 2021 eqs 17-20, the
// per-face UCT-HLLD reconstruction is:
//
//   Ez_face_lhlld = α^L · Ez_L + α^R · Ez_R - ν · (B_t,R - B_t,L)
//
// where Ez_L / Ez_R are the cell-centered Ez values immediately on
// either side of the face, B_t is the tangential B at the face (the
// jump comes from the L/R reconstructed primitives), and the weights
// are determined by the inner wave fan (S_L*, S_R*):
//
//   α^R = max(0, S_R*)  / (max(0, S_R*) - min(0, S_L*))
//   α^L = -min(0, S_L*) / (max(0, S_R*) - min(0, S_L*))
//   ν   = max(0, S_R*) · (-min(0, S_L*)) / (max(0, S_R*) - min(0, S_L*))
//
// These satisfy α^L + α^R = 1 by construction. Wave-fan-limit checks:
//   - S_L* ≥ 0  (supersonic-fast-left at the face): α^L = 1, ν = 0.
//   - S_R* ≤ 0  (supersonic-fast-right):            α^R = 1, ν = 0.
//   - S_L* < 0 < S_R*: subsonic-Alfvénic; smooth L↔R blend with
//     dissipation proportional to the tangential-B jump.
//
// ── 2D corner EMF: average over four adjacent faces ─────────────────
// Each of the four faces (lo, hi x-faces and le, ri y-faces) around a
// corner contributes its own UCT-HLLD reconstruction:
//
//   Ez_corner = ¼ · (Ez_lo_lhlld + Ez_hi_lhlld + Ez_le_lhlld + Ez_ri_lhlld)
//
// This is the corner-symmetric (orthogonal-sum) form of Mignone+ 2010
// eq 38 — picks the same "fan" decomposition that the 1D Riemann
// solver picked, but applies it at the 2D corner via averaging. The
// dissipation now adapts per direction.
//
// ── Reduction to G&S 2005 in the smooth-flow / B_n = 0 limit ────────
// When |B_n| → 0 (e.g. pure-hydro Sod with B = 0 everywhere), riemann-
// hlld's Branch A (HLLC fallback) stashes (S_L*, S_R*) = (S_M, S_M).
// Then min(0, S_L*) = min(0, S_M), max(0, S_R*) = max(0, S_M), so:
//   - S_M > 0: α^L = 1, α^R = 0, ν = 0  → Ez_face = Ez_L
//   - S_M < 0: α^L = 0, α^R = 1, ν = 0  → Ez_face = Ez_R
//   - S_M = 0: α^L = α^R = ½, ν = 0    → Ez_face = ½(Ez_L + Ez_R)
// This is EXACTLY the G&S 2005 upwind-cell-Ez selector (eqns 41-44)
// for the case where the "face Ez correction" equals the upwind cell
// Ez — which it does in any smooth-Ez region (face Ez ≈ cell Ez). So
// in the hydrodynamic / smooth-flow limit, UCT-HLLD reduces to the G&S
// upwind formula. At B = 0 specifically, Ez_cell = vy·Bx - vx·By = 0
// everywhere, so the corner EMF is identically zero — same as G&S.
//
// In the smooth-MHD limit where the wave fan is well-resolved (no
// degeneracy), UCT-HLLD differs from G&S by using the FULL Alfvén
// structure rather than just sign(S_M). Both schemes preserve ∇·B by
// construction (the corner-EMF curl property depends only on Ez_corner
// being SHARED across the four cells whose edges touch the corner, not
// on its recipe). Both schemes reduce to BS arithmetic mean of face Ez
// in the uniform-Ez / vanishing-jump limit.
//
// ── Tangential-B jump for the dissipation term ───────────────────────
// The B_t,R - B_t,L jump is taken between the cell-centered B
// components on either side of the face. At an x-face between cells
// (ix-1, *) and (ix, *), the tangential B is B_y; we use the cell-
// averaged B_y at each side. Cell-averaged B comes from the same
// face-B averaging used in ez_cell (sum of two adjacent face B values,
// /2). This is consistent with the cell-centered Ez we're combining
// against, and avoids re-reading the PPM edge buffers.
//
// ── Bindings ────────────────────────────────────────────────────────
//   0 uniforms (uniform)
//   1 flux_x_1 (ro)       — (-,-,-Ez_x, SM=v_x*)
//   2 flux_y_1 (ro)       — (-,-,+Ez_y, SM=v_y*)
//   3 Ez_edge  (rw)
//   4 U0       (ro)       — (ρ, ρvx, ρvy, ρvz)  for cell-Ez velocity
//   5 Bx_face  (ro)       — for cell-Ez Bx average + face-B jump
//   6 By_face  (ro)       — for cell-Ez By average + face-B jump
//   7 face_wavespeeds_x (ro) — (S_L, S_L*, S_R*, S_R) per x-face
//   8 face_wavespeeds_y (ro) — (S_L, S_L*, S_R*, S_R) per y-face
//
// 8 storage bindings, under the 10-per-pipeline cap (HANDOFF Session 2
// §1).

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       flux_x_1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       flux_y_1: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Ez_edge:  array<f32>;
@group(0) @binding(4) var<storage, read>       U0:       array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       Bx_face:  array<f32>;
@group(0) @binding(6) var<storage, read>       By_face:  array<f32>;
@group(0) @binding(7) var<storage, read>       face_wavespeeds_x: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read>       face_wavespeeds_y: array<vec4<f32>>;

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

// Cell-averaged tangential B. For the x-face dissipation term we need
// B_y on either side; for the y-face dissipation term, B_x. These are
// the same averages ez_cell uses.
fn cell_by(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                + By_face[by_face_idx(ix, iy + 1u, n_total)]);
}
fn cell_bx(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
}

// UCT-HLLD per-face EMF reconstruction (Mignone+ 2010 eq 38 / 2021 eqs
// 17-20). Given the face's inner Alfvén wave speeds (SLs, SRs), the
// L/R cell-centered Ez immediately adjacent, and the tangential-B jump
// across the face, returns the per-face EMF.
//
// Bt_jump = B_t,R - B_t,L (cell-centered tangential B on either side).
// Sign convention: the dissipation term opposes the jump, so a
// positive (B_t,R - B_t,L) at a sub-Alfvénic face produces a negative
// EMF contribution (correct dispersive behavior).
fn lhlld_face_ez(SLs: f32, SRs: f32, Ez_L: f32, Ez_R: f32, Bt_jump: f32) -> f32 {
    // Clamp wave speeds for the weight calculation. SR+ = max(0, SRs);
    // SL- = -min(0, SLs) = max(0, -SLs). Both ≥ 0 by construction.
    let SR_p = max(0.0, SRs);
    let SL_n = max(0.0, -SLs);
    let denom = SR_p + SL_n;

    // When the wave fan is entirely on one side (SR_p = 0 → α^R = 0;
    // SL_n = 0 → α^L = 0), the other weight goes to 1 and dissipation
    // vanishes. The 1e-30 denom floor handles the degenerate
    // SLs = SRs = 0 case (both supersonic-towards-stationary, vanishing
    // probability in practice) by collapsing to ½(Ez_L + Ez_R) with
    // zero dissipation.
    let safe_denom = max(denom, 1.0e-30);
    let alpha_R = SR_p / safe_denom;
    let alpha_L = SL_n / safe_denom;
    let nu      = SR_p * SL_n / safe_denom;

    // When denom is effectively zero (both supersonic-into-corner),
    // bias to the simple ½/½ average — symmetric, no-bias choice.
    let degenerate = denom < 1.0e-20;
    let aL = select(alpha_L, 0.5, degenerate);
    let aR = select(alpha_R, 0.5, degenerate);
    let nu_eff = select(nu, 0.0, degenerate);

    return aL * Ez_L + aR * Ez_R - nu_eff * Bt_jump;
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

    // ── Face indices ────────────────────────────────────────────────
    // x-face below corner: cell (ix, iy-1)  (the LEFT face of that cell)
    // x-face above corner: cell (ix, iy  )
    // y-face left  of corner: cell (ix-1, iy)
    // y-face right of corner: cell (ix,   iy)
    let idx_x_lo = cell_idx_total(ix,      iy - 1u, n_total);
    let idx_x_hi = cell_idx_total(ix,      iy,      n_total);
    let idx_y_le = cell_idx_total(ix - 1u, iy,      n_total);
    let idx_y_ri = cell_idx_total(ix,      iy,      n_total);

    // ── Cell-centered Ez at the four neighbours ─────────────────────
    //   (ix-1, iy-1) SW    (ix, iy-1) SE
    //   (ix-1, iy  ) NW    (ix, iy  ) NE
    let ez_sw = ez_cell(ix - 1u, iy - 1u, n_total);
    let ez_se = ez_cell(ix,      iy - 1u, n_total);
    let ez_nw = ez_cell(ix - 1u, iy,      n_total);
    let ez_ne = ez_cell(ix,      iy,      n_total);

    // ── Tangential-B at the four corner-adjacent cells ──────────────
    // For x-face dissipation we need B_y on either side of the face.
    // For y-face dissipation we need B_x. Use cell-averaged values for
    // consistency with the cell-Ez computation.
    let by_sw = cell_by(ix - 1u, iy - 1u, n_total);
    let by_se = cell_by(ix,      iy - 1u, n_total);
    let by_nw = cell_by(ix - 1u, iy,      n_total);
    let by_ne = cell_by(ix,      iy,      n_total);

    let bx_sw = cell_bx(ix - 1u, iy - 1u, n_total);
    let bx_se = cell_bx(ix,      iy - 1u, n_total);
    let bx_nw = cell_bx(ix - 1u, iy,      n_total);
    let bx_ne = cell_bx(ix,      iy,      n_total);

    // ── Face wave speeds (S_L, S_L*, S_R*, S_R) ─────────────────────
    let ws_x_lo = face_wavespeeds_x[idx_x_lo];
    let ws_x_hi = face_wavespeeds_x[idx_x_hi];
    let ws_y_le = face_wavespeeds_y[idx_y_le];
    let ws_y_ri = face_wavespeeds_y[idx_y_ri];

    // ── Per-face UCT-HLLD EMF reconstructions ───────────────────────
    // x-face below corner: L = SW (ix-1, iy-1), R = SE (ix, iy-1)
    //   Tangential B is B_y; jump is (by_se - by_sw).
    let ez_lo_lhlld = lhlld_face_ez(
        ws_x_lo.y, ws_x_lo.z,   // SLs, SRs
        ez_sw, ez_se,
        by_se - by_sw,
    );

    // x-face above corner: L = NW (ix-1, iy), R = NE (ix, iy)
    let ez_hi_lhlld = lhlld_face_ez(
        ws_x_hi.y, ws_x_hi.z,
        ez_nw, ez_ne,
        by_ne - by_nw,
    );

    // y-face left of corner: L = SW (ix-1, iy-1), R = NW (ix-1, iy)
    //   Tangential B is B_x; jump is (bx_nw - bx_sw).
    //   Note sign: at a y-face, Ez_face = +flux_y.z (positive sign,
    //   per the file-header convention). The dissipation term in the
    //   UCT formula is "-ν · jump"; the jump (B_t,R - B_t,L) for a
    //   y-face takes R = upper cell, L = lower cell.
    let ez_le_lhlld = lhlld_face_ez(
        ws_y_le.y, ws_y_le.z,
        ez_sw, ez_nw,
        bx_nw - bx_sw,
    );

    // y-face right of corner: L = SE (ix, iy-1), R = NE (ix, iy)
    let ez_ri_lhlld = lhlld_face_ez(
        ws_y_ri.y, ws_y_ri.z,
        ez_se, ez_ne,
        bx_ne - bx_se,
    );

    // ── 2D corner EMF: arithmetic mean of four per-face LHLLD EMFs ──
    // Mignone+ 2010 eq 38, corner-symmetric form. Reduces to BS
    // arithmetic-mean of face Ez in the uniform-Ez / vanishing-Bt-jump
    // limit (Ez_L = Ez_R = Ez_face_hlld, Bt_jump = 0 → per-face
    // reconstruction returns Ez_face_hlld exactly).
    Ez_edge[ez_edge_idx(ix, iy, n_total)] = 0.25 * (
        ez_lo_lhlld + ez_hi_lhlld + ez_le_lhlld + ez_ri_lhlld
    );
}

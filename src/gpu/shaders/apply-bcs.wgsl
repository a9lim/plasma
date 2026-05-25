// ─── apply-bcs.wgsl ──────────────────────────────────────────────────
// Fill the 2-layer ghost-cell band around the interior region per the
// four per-edge BC modes (periodic / outflow / reflecting / driven).
// Dispatched once at the start of each RK3 stage, BEFORE reconstruct-ppm
// reads its 5-point stencil. After this shader runs, every read of
// U0/U1/Bx_face/By_face inside the interior dispatch lands on a valid
// in-bounds index — no wrapping required.
//
// Coverage:
//   * Cell-centered (U0, U1):
//       - West strip:  i ∈ [0, ghost),         j ∈ [0, N_total)
//       - East strip:  i ∈ [ghost+N, N_total), j ∈ [0, N_total)
//       - South strip: i ∈ [0, N_total),       j ∈ [0, ghost)
//       - North strip: i ∈ [0, N_total),       j ∈ [ghost+N, N_total)
//     Corners belong to two edges. Rule: prefer the NON-periodic edge if
//     mixed; if both equal, just use one (the choice doesn't matter when
//     the modes match). Implemented as a priority: among the two
//     adjacent edges, pick the one whose mode is not BC_PERIODIC; if
//     both periodic, pick the horizontal (E/W) wrap, which behaves
//     identically to the vertical (N/S) wrap on its own ghost cells
//     since both copies preserve the interior data.
//
//   * Face-centered Bx_face:
//       - The interior x-face indices are [ghost+1, ghost+N+1] (i.e.,
//         left face of cell ghost, …, right face of cell ghost+N-1).
//       - West ghost x-faces: i ∈ [0, ghost+1) — under-extension
//         covering the LEFT face of each ghost cell and the BOUNDARY
//         face itself (i = ghost+1 is the leftmost interior face;
//         i = ghost is the left face of the rightmost W-ghost cell).
//         For reflecting BCs: the BOUNDARY face (i = ghost) must hold
//         Bx = 0 (no normal field through a perfectly conducting wall).
//       - East ghost x-faces: i ∈ [ghost+N+1, N_total+1).
//
//   * Face-centered By_face: symmetric with x roles swapped.
//
// Reflecting BC sign-flips:
//   - West/East wall (normal = x):  v_x and B_x flip; v_y, v_z, B_y,
//     B_z, ρ, p preserved.
//   - South/North wall (normal = y): v_y and B_y flip; v_x, v_z, B_x,
//     B_z, ρ, p preserved.
// The mirrored interior cell index for a ghost at distance d from the
// boundary is the interior cell at distance d on the other side of the
// boundary. With ghost = 2:
//   Left ghost (0, j) mirrors interior (3, j)
//   Left ghost (1, j) mirrors interior (2, j)
//   Right ghost (N+2, j) mirrors interior (N+1, j)
//   Right ghost (N+3, j) mirrors interior (N, j)
// (Here all indices are in the ghost-padded storage frame, so
// ghost = 2 and interior i ∈ [2, N+2).)
//
// Bindings:
//   0 uniforms       (uniform)
//   1 bc_uniforms    (ro storage) — mode_n, mode_s, mode_e, mode_w + driven state
//   2 U0             (rw)
//   3 U1             (rw)
//   4 Bx_face        (rw)
//   5 By_face        (rw)

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       bc:        BcUniforms;
@group(0) @binding(2) var<storage, read_write> U0:        array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> U1:        array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Bx_face:   array<f32>;
@group(0) @binding(5) var<storage, read_write> By_face:   array<f32>;

// Mirror an interior cell index across a west/east wall and apply the
// appropriate x-normal sign flip (vx, Bx negated). Other components
// unchanged.
fn reflect_x(U0v: vec4<f32>, U1v: vec4<f32>) -> array<vec4<f32>, 2> {
    var out_U0 = U0v;
    out_U0.y = -U0v.y;  // ρ·v_x sign flip
    return array<vec4<f32>, 2>(out_U0, U1v);
}
fn reflect_y(U0v: vec4<f32>, U1v: vec4<f32>) -> array<vec4<f32>, 2> {
    var out_U0 = U0v;
    out_U0.z = -U0v.z;  // ρ·v_y sign flip
    return array<vec4<f32>, 2>(out_U0, U1v);
}

// Convert driven primitive state to conservative pair (U0, U1).
fn driven_cons() -> array<vec4<f32>, 2> {
    var P: MhdPrim;
    P.rho = max(bc.driven_rho, DENSITY_FLOOR);
    P.vx  = bc.driven_vx;
    P.vy  = bc.driven_vy;
    P.vz  = bc.driven_vz;
    P.p   = max(bc.driven_p, PRESSURE_FLOOR);
    P.bx  = bc.driven_bx;
    P.by  = bc.driven_by;
    P.bz  = bc.driven_bz;
    let cp = prim_to_cons_pair(P, U_uniforms.gamma);
    return array<vec4<f32>, 2>(cp.U0, cp.U1);
}

// Choose the BC mode that "owns" a ghost cell at (ix, iy). For non-corner
// strips this is unambiguous. For corners, prefer the first non-periodic
// of the two adjacent edges.
fn pick_corner_mode(mode_h: u32, mode_v: u32) -> u32 {
    // If horizontal edge is non-periodic, use it. Otherwise fall back
    // to vertical (which may itself be periodic, fine).
    if (mode_h != BC_PERIODIC) { return mode_h; }
    return mode_v;
}

// Pick adjacent vertical (S/N) edge for a row.
fn vert_mode_for_row(iy: u32, ghost: u32, n_interior: u32) -> u32 {
    if (iy < ghost) { return bc.mode_s; }
    if (iy >= ghost + n_interior) { return bc.mode_n; }
    return BC_PERIODIC;
}

fn horiz_mode_for_col(ix: u32, ghost: u32, n_interior: u32) -> u32 {
    if (ix < ghost) { return bc.mode_w; }
    if (ix >= ghost + n_interior) { return bc.mode_e; }
    return BC_PERIODIC;
}

// Fill ONE cell-centered ghost (i, j) based on the appropriate edge mode.
fn fill_cell_ghost(ix: u32, iy: u32, ghost: u32, n_interior: u32, n_total: u32) {
    let h_mode = horiz_mode_for_col(ix, ghost, n_interior);
    let v_mode = vert_mode_for_row(iy, ghost, n_interior);
    let in_h_ghost = (ix < ghost) || (ix >= ghost + n_interior);
    let in_v_ghost = (iy < ghost) || (iy >= ghost + n_interior);
    if (!in_h_ghost && !in_v_ghost) { return; }   // interior — never touch.

    var mode: u32;
    if (in_h_ghost && in_v_ghost) {
        // Corner. Prefer non-periodic among horizontal vs vertical.
        mode = pick_corner_mode(h_mode, v_mode);
    } else if (in_h_ghost) {
        mode = h_mode;
    } else {
        mode = v_mode;
    }

    let dst = cell_idx_total(ix, iy, n_total);

    if (mode == BC_PERIODIC) {
        // Copy from the wrapped interior cell on the opposite side.
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) { src_i = ix + n_interior; }
        else if (ix >= ghost + n_interior) { src_i = ix - n_interior; }
        if (iy < ghost) { src_j = iy + n_interior; }
        else if (iy >= ghost + n_interior) { src_j = iy - n_interior; }
        let src = cell_idx_total(src_i, src_j, n_total);
        U0[dst] = U0[src];
        U1[dst] = U1[src];
        return;
    }

    if (mode == BC_OUTFLOW) {
        // Zero-gradient: copy from the nearest interior cell.
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) { src_i = ghost; }
        else if (ix >= ghost + n_interior) { src_i = ghost + n_interior - 1u; }
        if (iy < ghost) { src_j = ghost; }
        else if (iy >= ghost + n_interior) { src_j = ghost + n_interior - 1u; }
        let src = cell_idx_total(src_i, src_j, n_total);
        U0[dst] = U0[src];
        U1[dst] = U1[src];
        return;
    }

    if (mode == BC_REFLECTING) {
        // Mirror across the boundary. The wall sits between the
        // outermost ghost cell (i = ghost-1) and the first interior
        // cell (i = ghost). Index formulas:
        //     src_i = 2*ghost - 1 - i      for W ghost
        //     src_i = 2*(ghost+n) - 1 - i  for E ghost
        // Same shape vertically.
        //
        // Corner rule: the corner mode is owned by the first NON-periodic
        // adjacent edge (see pick_corner_mode). If the OTHER axis is
        // periodic, we still need to map its index into the interior
        // before sampling, otherwise the mirror source lands on stale
        // ghost data. We compose: reflect on the owning axis, periodic-
        // wrap on the other axis if it's periodic.
        var src_i = ix;
        var src_j = iy;
        var flip_x = false;
        var flip_y = false;
        let h_is_reflect = in_h_ghost && (h_mode == BC_REFLECTING);
        let v_is_reflect = in_v_ghost && (v_mode == BC_REFLECTING);
        if (h_is_reflect) {
            if (ix < ghost) { src_i = 2u * ghost - 1u - ix; }
            else            { src_i = 2u * (ghost + n_interior) - 1u - ix; }
            flip_x = true;
        } else if (in_h_ghost && h_mode == BC_PERIODIC) {
            if (ix < ghost) { src_i = ix + n_interior; }
            else            { src_i = ix - n_interior; }
        }
        if (v_is_reflect) {
            if (iy < ghost) { src_j = 2u * ghost - 1u - iy; }
            else            { src_j = 2u * (ghost + n_interior) - 1u - iy; }
            flip_y = true;
        } else if (in_v_ghost && v_mode == BC_PERIODIC) {
            if (iy < ghost) { src_j = iy + n_interior; }
            else            { src_j = iy - n_interior; }
        }
        let src = cell_idx_total(src_i, src_j, n_total);
        var u0 = U0[src];
        var u1 = U1[src];
        if (flip_x) { u0.y = -u0.y; }    // flip ρ·v_x
        if (flip_y) { u0.z = -u0.z; }    // flip ρ·v_y
        U0[dst] = u0;
        U1[dst] = u1;
        return;
    }

    // BC_DRIVEN
    let cons = driven_cons();
    U0[dst] = cons[0];
    U1[dst] = cons[1];
}

// Fill one Bx_face entry (index space (N_total+1) × N_total). The
// shape of the strips is different from cell-centered: x-faces have an
// extra column at i = N_total.
//   x-face index i corresponds to position i - ghost - 0.5 within the
//   interior. Interior x-faces (between interior cells, plus the two
//   boundary faces) are i ∈ [ghost, ghost+n_interior]. Ghost x-faces:
//     i ∈ [0, ghost)            → west ghost
//     i ∈ (ghost+n_interior, N_total]   → east ghost
fn fill_bx_face(ix: u32, iy: u32, ghost: u32, n_interior: u32, n_total: u32) {
    // Determine if this is a ghost x-face. The boundary faces (i = ghost
    // and i = ghost + n_interior) are TOUCHED ONLY for reflecting and
    // driven BCs — for periodic and outflow they're part of the interior
    // physics dispatch's writes (or just left alone).
    let in_h_ghost = (ix < ghost) || (ix > ghost + n_interior);
    let on_w_wall  = (ix == ghost);
    let on_e_wall  = (ix == ghost + n_interior);
    let in_v_ghost = (iy < ghost) || (iy >= ghost + n_interior);
    if (!in_h_ghost && !on_w_wall && !on_e_wall && !in_v_ghost) { return; }

    let h_mode = horiz_mode_for_col(ix, ghost, n_interior);
    let v_mode = vert_mode_for_row(iy, ghost, n_interior);

    // Choose mode. If on a boundary face (W or E wall), the horizontal
    // mode owns it unconditionally. Otherwise, corner logic.
    var mode: u32;
    if (on_w_wall) { mode = bc.mode_w; }
    else if (on_e_wall) { mode = bc.mode_e; }
    else if (in_h_ghost && in_v_ghost) { mode = pick_corner_mode(h_mode, v_mode); }
    else if (in_h_ghost) { mode = h_mode; }
    else                 { mode = v_mode; }

    let dst = bx_face_idx(ix, iy, n_total);

    if (mode == BC_PERIODIC) {
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) { src_i = ix + n_interior; }
        else if (ix > ghost + n_interior) { src_i = ix - n_interior; }
        if (iy < ghost) { src_j = iy + n_interior; }
        else if (iy >= ghost + n_interior) { src_j = iy - n_interior; }
        // Boundary faces: under periodic wrap, the W and E boundary
        // faces are the SAME face. We canonicalize the W wall as
        // authoritative: ALL boundary periodic invocations read from
        // the W boundary face (ix = ghost). The E wall invocation then
        // writes the same value back, ensuring both are in sync.
        if (on_w_wall || on_e_wall) { src_i = ghost; }
        Bx_face[dst] = Bx_face[bx_face_idx(src_i, src_j, n_total)];
        return;
    }

    if (mode == BC_OUTFLOW) {
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) { src_i = ghost; }
        else if (ix > ghost + n_interior) { src_i = ghost + n_interior; }
        // boundary face stays put (it IS the interior boundary).
        if (iy < ghost) { src_j = ghost; }
        else if (iy >= ghost + n_interior) { src_j = ghost + n_interior - 1u; }
        Bx_face[dst] = Bx_face[bx_face_idx(src_i, src_j, n_total)];
        return;
    }

    if (mode == BC_REFLECTING) {
        // Perfectly conducting wall: B normal to the wall is zero.
        // On a W/E wall, normal is x → Bx = 0 on that boundary face.
        // For x-ghost faces (away from the boundary), mirror across the
        // wall: face at distance d outside ↔ face at distance d inside,
        // with Bx negated.
        if (on_w_wall || on_e_wall) {
            Bx_face[dst] = 0.0;
            return;
        }
        var src_i = ix;
        var src_j = iy;
        var flip = false;
        if (ix < ghost) {
            // Mirror about the W boundary face at i = ghost.
            // i ∈ {0, 1, …, ghost-1} → src ∈ {2*ghost - i, …}
            src_i = 2u * ghost - ix;
            flip = true;
        } else if (ix > ghost + n_interior) {
            // Mirror about the E boundary face at i = ghost + n_interior.
            src_i = 2u * (ghost + n_interior) - ix;
            flip = true;
        }
        if (iy < ghost) {
            // S-wall reflection for v-ghost rows doesn't flip Bx (normal
            // is y). Just mirror the index.
            src_j = 2u * ghost - 1u - iy;
        } else if (iy >= ghost + n_interior) {
            src_j = 2u * (ghost + n_interior) - 1u - iy;
        }
        var v = Bx_face[bx_face_idx(src_i, src_j, n_total)];
        if (flip) { v = -v; }
        Bx_face[dst] = v;
        return;
    }

    // BC_DRIVEN — set Bx to the driven inflow Bx everywhere on this strip.
    Bx_face[dst] = bc.driven_bx;
}

fn fill_by_face(ix: u32, iy: u32, ghost: u32, n_interior: u32, n_total: u32) {
    let in_v_ghost = (iy < ghost) || (iy > ghost + n_interior);
    let on_s_wall  = (iy == ghost);
    let on_n_wall  = (iy == ghost + n_interior);
    let in_h_ghost = (ix < ghost) || (ix >= ghost + n_interior);
    if (!in_v_ghost && !on_s_wall && !on_n_wall && !in_h_ghost) { return; }

    let h_mode = horiz_mode_for_col(ix, ghost, n_interior);
    let v_mode = vert_mode_for_row(iy, ghost, n_interior);

    var mode: u32;
    if (on_s_wall) { mode = bc.mode_s; }
    else if (on_n_wall) { mode = bc.mode_n; }
    else if (in_h_ghost && in_v_ghost) { mode = pick_corner_mode(h_mode, v_mode); }
    else if (in_v_ghost) { mode = v_mode; }
    else                 { mode = h_mode; }

    let dst = by_face_idx(ix, iy, n_total);

    if (mode == BC_PERIODIC) {
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) { src_i = ix + n_interior; }
        else if (ix >= ghost + n_interior) { src_i = ix - n_interior; }
        if (iy < ghost) { src_j = iy + n_interior; }
        else if (iy > ghost + n_interior) { src_j = iy - n_interior; }
        // Canonicalize boundary y-faces: S wall is authoritative.
        if (on_s_wall || on_n_wall) { src_j = ghost; }
        By_face[dst] = By_face[by_face_idx(src_i, src_j, n_total)];
        return;
    }

    if (mode == BC_OUTFLOW) {
        var src_i = ix;
        var src_j = iy;
        if (ix < ghost) { src_i = ghost; }
        else if (ix >= ghost + n_interior) { src_i = ghost + n_interior - 1u; }
        if (iy < ghost) { src_j = ghost; }
        else if (iy > ghost + n_interior) { src_j = ghost + n_interior; }
        By_face[dst] = By_face[by_face_idx(src_i, src_j, n_total)];
        return;
    }

    if (mode == BC_REFLECTING) {
        if (on_s_wall || on_n_wall) {
            By_face[dst] = 0.0;
            return;
        }
        var src_i = ix;
        var src_j = iy;
        var flip = false;
        if (iy < ghost) {
            src_j = 2u * ghost - iy;
            flip = true;
        } else if (iy > ghost + n_interior) {
            src_j = 2u * (ghost + n_interior) - iy;
            flip = true;
        }
        if (ix < ghost) {
            src_i = 2u * ghost - 1u - ix;
        } else if (ix >= ghost + n_interior) {
            src_i = 2u * (ghost + n_interior) - 1u - ix;
        }
        var v = By_face[by_face_idx(src_i, src_j, n_total)];
        if (flip) { v = -v; }
        By_face[dst] = v;
        return;
    }

    // BC_DRIVEN
    By_face[dst] = bc.driven_by;
}

// Single-pass kernel. We dispatch over the FULL (N_total+1, N_total+1)
// box and let each invocation handle whichever buffers its index is valid
// for: cell-centered for (ix < N_total, iy < N_total), Bx_face for
// (ix < N_total+1, iy < N_total), By_face for (ix < N_total, iy < N_total+1).
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_total    = U_uniforms.grid_n_total;
    let n_interior = U_uniforms.grid_n;
    let ghost      = U_uniforms.ghost_w;
    let ix         = gid.x;
    let iy         = gid.y;

    // Cell-centered (U0, U1)
    if (ix < n_total && iy < n_total) {
        fill_cell_ghost(ix, iy, ghost, n_interior, n_total);
    }

    // Bx_face: (n_total+1) × n_total
    if (ix < n_total + 1u && iy < n_total) {
        fill_bx_face(ix, iy, ghost, n_interior, n_total);
    }

    // By_face: n_total × (n_total+1)
    if (ix < n_total && iy < n_total + 1u) {
        fill_by_face(ix, iy, ghost, n_interior, n_total);
    }
}

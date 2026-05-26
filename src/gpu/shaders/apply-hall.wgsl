// ─── apply-hall.wgsl ──────────────────────────────────────────────────
// Hall MHD correction to the induction equation. Augments Ohm's law:
//
//   E_total  =  -v × B  +  η J  +  (d_i / ρ) · (J × B)
//                                  └──────────────┘
//                                  Hall term — this shader
//
// where d_i = c/ω_pi is the ion inertial length (here a code-units
// scalar, uniforms.hall_di). With B in code units the prefactor is
// (1/ne) = d_i/ρ_e ≈ d_i/ρ in an electron-quasineutrality assumption.
//
// Δ-form face-B update from the Hall EMF:
//
//   ∂B/∂t |_hall  =  -∇ × ((d_i/ρ) · J × B)
//
// On the Yee grid this fits cleanly into the constrained-transport
// machinery — compute a CORNER-centered Hall electric field E_H^z,
// then apply the standard CT-curl to the face Bs:
//
//   Bx_face[i,j] += -(dt/dx) · (E_H^z[i, j+1] − E_H^z[i, j])
//   By_face[i,j] += +(dt/dx) · (E_H^z[i+1,j] − E_H^z[i, j])
//
// E_H^z at the corner (i+½, j+½) needs:
//   J_x, J_y, J_z, B_x, B_y, B_z   evaluated AT the corner.
//
// J = ∇ × B sampled at the corner:
//   J_z = (∂B_y/∂x − ∂B_x/∂y)                 — central diffs of face B
//   J_x =  ∂B_z/∂y                              — diff of cell-centered Bz
//   J_y = -∂B_z/∂x
//
// B at the corner = average of the four touching face / cell values.
//
// E_H = (d_i/ρ_corner) · (J × B). We keep only the z-component for the
// face-B CT-curl (the x/y components would update Bz, which is fine —
// see Bz handling below).
//
//   E_H^z  =  (d_i/ρ) · (J_x B_y − J_y B_x)
//
// Bz (cell-centered) also evolves under the Hall term:
//
//   ∂Bz/∂t  =  -(∂E_H^x/∂y − ∂E_H^y/∂x)         (only out-of-plane component)
//
// For the breadth pass we apply the Bz contribution via cell-centered
// central differences of the Hall E field, evaluated at corners as
// above. Cheaper alternative: ignore Bz for now (true 2.5D Hall is
// rare in the literature anyway). We include it for completeness.
//
// Integration: explicit forward Euler over dt_hyp. Hall is DISPERSIVE
// (whistler waves), so the formal stability bound is
//   dt_hall  ≤  dx² / (v_A · d_i)
// which becomes restrictive when d_i is resolved. The breadth pass
// uses dt_hyp directly and accepts whatever instability shows up —
// a proper sub-cycle or HDS scheme is the next layer of work.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0       (ro) — ρ at cells for ρ_corner
//   2 Bx_face  (rw) — updated in place
//   3 By_face  (rw) — updated in place
//   4 U1       (rw) — Bz lives in U1.y
//   5 dt_buf   (uniform)

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Bx_face:    array<f32>;
@group(0) @binding(3) var<storage, read_write> By_face:    array<f32>;
@group(0) @binding(4) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(5) var<uniform>             dt_buf:     DtUniform;

// ── Corner-centered J = ∇×B at corner (ix, iy) (Yee convention: this
//    corner sits at the bottom-left of cell (ix, iy)). ─────────────────
struct CornerJB {
    Jx: f32, Jy: f32, Jz: f32,
    Bx: f32, By: f32, Bz: f32,
    rho: f32,
};

fn cell_bx_avg(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
}
fn cell_by_avg(ix: u32, iy: u32, n_total: u32) -> f32 {
    return 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                + By_face[by_face_idx(ix, iy + 1u, n_total)]);
}

fn corner_jb(ix: u32, iy: u32, n_total: u32) -> CornerJB {
    let dx = U_uniforms.dx;
    var R: CornerJB;

    // Bx at corner = average of the two face Bxs touching the corner
    // along the y-axis:
    //   Bx_face[ix, iy-1] and Bx_face[ix, iy].
    R.Bx = 0.5 * (Bx_face[bx_face_idx(ix, iy - 1u, n_total)]
                + Bx_face[bx_face_idx(ix, iy,      n_total)]);
    // By at corner = average of the two face Bys touching along x.
    R.By = 0.5 * (By_face[by_face_idx(ix - 1u, iy, n_total)]
                + By_face[by_face_idx(ix,      iy, n_total)]);
    // Bz at corner = average of the four touching cells.
    let bz_sw = U1[cell_idx_total(ix - 1u, iy - 1u, n_total)].y;
    let bz_se = U1[cell_idx_total(ix,      iy - 1u, n_total)].y;
    let bz_nw = U1[cell_idx_total(ix - 1u, iy,      n_total)].y;
    let bz_ne = U1[cell_idx_total(ix,      iy,      n_total)].y;
    R.Bz = 0.25 * (bz_sw + bz_se + bz_nw + bz_ne);

    // ρ at corner = average of the four touching cells.
    let rho_sw = U0[cell_idx_total(ix - 1u, iy - 1u, n_total)].x;
    let rho_se = U0[cell_idx_total(ix,      iy - 1u, n_total)].x;
    let rho_nw = U0[cell_idx_total(ix - 1u, iy,      n_total)].x;
    let rho_ne = U0[cell_idx_total(ix,      iy,      n_total)].x;
    R.rho = max(0.25 * (rho_sw + rho_se + rho_nw + rho_ne), DENSITY_FLOOR);

    // J = ∇×B.
    // J_z = ∂By/∂x − ∂Bx/∂y, evaluated at the corner with face stencils.
    let by_l = By_face[by_face_idx(ix - 1u, iy, n_total)];
    let by_r = By_face[by_face_idx(ix,      iy, n_total)];
    let bx_d = Bx_face[bx_face_idx(ix, iy - 1u, n_total)];
    let bx_u = Bx_face[bx_face_idx(ix, iy,      n_total)];
    R.Jz = (by_r - by_l) / dx - (bx_u - bx_d) / dx;

    // J_x =  ∂Bz/∂y. Use the 2-cell stencil straddling the corner along y.
    // J_y = -∂Bz/∂x. Mirror.
    let bz_d_avg = 0.5 * (bz_sw + bz_se);
    let bz_u_avg = 0.5 * (bz_nw + bz_ne);
    let bz_l_avg = 0.5 * (bz_sw + bz_nw);
    let bz_r_avg = 0.5 * (bz_se + bz_ne);
    R.Jx =  (bz_u_avg - bz_d_avg) / dx;
    R.Jy = -(bz_r_avg - bz_l_avg) / dx;

    return R;
}

// Hall E at the corner = (d_i / ρ) · J × B.
struct HallE {
    Ex: f32, Ey: f32, Ez: f32,
};

fn hall_e_corner(ix: u32, iy: u32, n_total: u32) -> HallE {
    let s = corner_jb(ix, iy, n_total);
    let prefactor = U_uniforms.hall_di / s.rho;
    var E: HallE;
    E.Ex = prefactor * (s.Jy * s.Bz - s.Jz * s.By);
    E.Ey = prefactor * (s.Jz * s.Bx - s.Jx * s.Bz);
    E.Ez = prefactor * (s.Jx * s.By - s.Jy * s.Bx);
    return E;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_HALL)) { return; }
    if (U_uniforms.hall_di <= 0.0)                      { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    let extent = n_interior + 1u;
    if (gid.x >= extent || gid.y >= extent) { return; }
    let ix = ghost + gid.x;
    let iy = ghost + gid.y;

    let dt = dt_buf.dt;
    let dx = U_uniforms.dx;

    // ── Face B update from the corner Hall E^z ─────────────────────
    // Bx_face at (ix, iy) gets contributions from corners (ix, iy) and (ix, iy+1).
    //   ΔBx = -(dt/dx) · (E_H^z[ix, iy+1] − E_H^z[ix, iy])
    // By_face at (ix, iy) gets contributions from corners (ix, iy) and (ix+1, iy).
    //   ΔBy = +(dt/dx) · (E_H^z[ix+1, iy] − E_H^z[ix, iy])
    //
    // We dispatch over corners but each invocation writes to one face
    // of each axis where it's the "low" endpoint of the curl
    // difference; the corner at (ix+1, iy+1) handles the other side.
    // To avoid double-writing the same face from two corners we use
    // atomic-free additive updates by precomputing the field
    // difference here at THIS corner only and applying it as
    // bx_face[ix, iy-1] -= dt/dx · E^z[ix, iy]
    // bx_face[ix, iy  ] += dt/dx · E^z[ix, iy]
    // by_face[ix-1, iy] += dt/dx · E^z[ix, iy]
    // by_face[ix,   iy] -= dt/dx · E^z[ix, iy]
    //
    // *However* these are concurrent writes across workgroups — for the
    // breadth pass we sidestep that by using THIS corner only as the
    // "low-y" endpoint for Bx and "low-x" endpoint for By, paired with
    // a SECOND read at the opposite endpoint. That is, each invocation
    // touches exactly one Bx_face and one By_face cell, both indexed
    // by (ix, iy). Cleaner; cost is one extra corner-Ez evaluation per
    // cell, which is fine for the breadth pass.

    let E_here = hall_e_corner(ix, iy, n_total);
    if (gid.y < extent - 1u) {
        // Bx_face[ix, iy] update uses ΔE^z across the y-edge of cell.
        let E_up   = hall_e_corner(ix, iy + 1u, n_total);
        let bxi    = bx_face_idx(ix, iy, n_total);
        Bx_face[bxi] = Bx_face[bxi] - (dt / dx) * (E_up.Ez - E_here.Ez);
    }
    if (gid.x < extent - 1u) {
        // By_face[ix, iy] update uses ΔE^z across the x-edge of cell.
        let E_ri   = hall_e_corner(ix + 1u, iy, n_total);
        let byi    = by_face_idx(ix, iy, n_total);
        By_face[byi] = By_face[byi] + (dt / dx) * (E_ri.Ez - E_here.Ez);
    }

    // ── Bz update at cell centers from -∂Ey/∂x + ∂Ex/∂y ────────────
    // Only the interior cell with (ix, iy) as its SW corner gets the
    // contribution centered at this corner. Simpler: a separate pass
    // would be cleaner, but to keep one shader we attribute the Bz
    // update to the cell at (ix, iy) when it's interior, using
    // central diffs of Hall E across the cell's four corners.
    if (gid.x < n_interior && gid.y < n_interior
        && gid.x > 0u && gid.y > 0u) {
        let cell_ix = ix;
        let cell_iy = iy;
        let E_sw = E_here;
        let E_se = hall_e_corner(cell_ix + 1u, cell_iy,      n_total);
        let E_nw = hall_e_corner(cell_ix,      cell_iy + 1u, n_total);
        let E_ne = hall_e_corner(cell_ix + 1u, cell_iy + 1u, n_total);
        let dEy_dx = 0.5 * ((E_se.Ey + E_ne.Ey) - (E_sw.Ey + E_nw.Ey)) / dx;
        let dEx_dy = 0.5 * ((E_nw.Ex + E_ne.Ex) - (E_sw.Ex + E_se.Ex)) / dx;
        let dBz = (-dEy_dx + dEx_dy) * dt;
        let c   = cell_idx_total(cell_ix, cell_iy, n_total);
        let u1  = U1[c];
        U1[c] = vec4<f32>(u1.x, u1.y + dBz, u1.z, u1.w);
    }
}

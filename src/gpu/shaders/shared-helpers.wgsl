// ─── shared-helpers.wgsl ─────────────────────────────────────────────
// Phase 3a: 2.5D ideal MHD on a Yee-style staggered grid.
//
// ── State layout ────────────────────────────────────────────────────
// Cell-centered conserved state, packed as TWO vec4<f32> arrays per
// ping-pong slot:
//   U0[idx] = (ρ, ρ·vx, ρ·vy, ρ·vz)
//   U1[idx] = (E,  Bz,    _pad,  _pad)
// where E is total energy density:
//   E = p/(γ-1) + ½·ρ·|v|² + ½·|B|²
//
// Cell-centered primitive state (7 active components, stored as TWO vec4):
//   prim0 = (ρ, vx, vy, vz)
//   prim1 = (p, Bt1, Bt2, _) — packed for the active sweep direction:
//     x-sweep: Bt1 = By, Bt2 = Bz   (transverse to the x-face normal)
//     y-sweep: Bt1 = Bx, Bt2 = Bz   (transverse to the y-face normal)
//   The normal-direction B is read directly from Bx_face / By_face
//   (continuous across the face by the CT discretization of ∇·B=0).
//
// ── Face / edge convention (Yee staggered) ──────────────────────────
// Cell (i,j) owns:
//   Bx_face[i,j]  = Bx at face (i+½, j)         (x-face to its right)
//   By_face[i,j]  = By at face (i, j+½)         (y-face above it)
//   Ez_edge[i,j]  = Ez at corner (i+½, j+½)     (upper-right corner)
//   flux_x[i,j]   = MHD flux through face (i+½, j)
//   flux_y[i,j]   = MHD flux through face (i, j+½)
// Faces wrap mod N under periodic BCs.
//
// Discrete CT update (forward Euler, dt from compute-dt):
//   Bx_face[i,j] += -(dt/dy) · (Ez_edge[i,j] - Ez_edge[i,j-1])
//   By_face[i,j] += +(dt/dx) · (Ez_edge[i,j] - Ez_edge[i-1,j])
// With Ez_edge defined as Balsara-Spicer arithmetic mean of the four
// neighboring face fluxes (see compute-emf.wgsl), discrete ∇·B is
// preserved exactly to machine precision: the corner contributions
// cancel in pairs around every cell.
//
// ── γ & floors ──────────────────────────────────────────────────────
// γ comes from the Uniforms; pressure & density floors match config.js.

struct Uniforms {
    dx:           f32,
    gamma:        f32,
    view_min:     f32,
    view_max:     f32,
    grid_n:       u32,
    sweep_dir:    u32,  // 0 = x-sweep / x-face shaders, 1 = y-sweep / y-face shaders
    step_parity:  u32,  // 0 = even, 1 = odd — informational (drove Strang ordering in Phase 2)
    view_mode:    u32,  // 0=ρ, 1=p, 2=|v|, 3=|B|, 4=Jz
};

const PRESSURE_FLOOR: f32 = 1.0e-6;
const DENSITY_FLOOR:  f32 = 1.0e-6;

// ── Indexing helpers ────────────────────────────────────────────────
fn wrap_idx(i: i32, n: i32) -> u32 {
    return u32(((i % n) + n) % n);
}

fn cell_index(ix: u32, iy: u32, n: u32) -> u32 {
    return iy * n + ix;
}

fn cell_index_wrapped(ix: i32, iy: i32, n: i32) -> u32 {
    let wx = wrap_idx(ix, n);
    let wy = wrap_idx(iy, n);
    return wy * u32(n) + wx;
}

// ── Cons / prim conversion ──────────────────────────────────────────
// Bx, By must be supplied at the cell center (caller averages neighboring
// faces). Bz comes from U1.y.
struct MhdPrim {
    rho: f32,
    vx:  f32,
    vy:  f32,
    vz:  f32,
    p:   f32,
    bx:  f32,
    by:  f32,
    bz:  f32,
};

struct MhdCons {
    rho:  f32,
    mx:   f32,
    my:   f32,
    mz:   f32,
    E:    f32,
    bx:   f32,
    by:   f32,
    bz:   f32,
};

fn cons_to_prim_mhd(U0: vec4<f32>, U1: vec4<f32>, bx_c: f32, by_c: f32, gamma: f32) -> MhdPrim {
    var P: MhdPrim;
    P.rho = max(U0.x, DENSITY_FLOOR);
    P.vx  = U0.y / P.rho;
    P.vy  = U0.z / P.rho;
    P.vz  = U0.w / P.rho;
    P.bx  = bx_c;
    P.by  = by_c;
    P.bz  = U1.y;
    let ke = 0.5 * P.rho * (P.vx*P.vx + P.vy*P.vy + P.vz*P.vz);
    let mb = 0.5 * (P.bx*P.bx + P.by*P.by + P.bz*P.bz);
    P.p   = max((gamma - 1.0) * (U1.x - ke - mb), PRESSURE_FLOOR);
    return P;
}

// Compose cell-center Bx, By by averaging the two owning faces under
// periodic wrap. Cell (i,j) sits between Bx_face[i-1,j] (its left face)
// and Bx_face[i,j] (its right face); same for By. Callers inline the
// face reads — we expose only the index pair so we don't need to pass
// storage-buffer pointers through function calls (which adds WGSL
// boilerplate without much win).
fn bx_face_left_index(ix: u32, iy: u32, n: u32) -> u32 {
    return cell_index_wrapped(i32(ix) - 1, i32(iy), i32(n));
}
fn bx_face_right_index(ix: u32, iy: u32, n: u32) -> u32 {
    return cell_index(ix, iy, n);
}
fn by_face_down_index(ix: u32, iy: u32, n: u32) -> u32 {
    return cell_index_wrapped(i32(ix), i32(iy) - 1, i32(n));
}
fn by_face_up_index(ix: u32, iy: u32, n: u32) -> u32 {
    return cell_index(ix, iy, n);
}

fn fast_mag_speed(P: MhdPrim, gamma: f32, axis: u32) -> f32 {
    // c_fast² = ½(c_s² + c_A²) + ½√((c_s² + c_A²)² − 4 c_s² c_An²)
    // where c_An² is the squared Alfvén speed along the face normal.
    let rho = max(P.rho, DENSITY_FLOOR);
    let p   = max(P.p,   PRESSURE_FLOOR);
    let cs2 = gamma * p / rho;
    let b2  = P.bx*P.bx + P.by*P.by + P.bz*P.bz;
    let ca2 = b2 / rho;
    var can2: f32;
    if (axis == 0u) { can2 = P.bx * P.bx / rho; }
    else            { can2 = P.by * P.by / rho; }
    let sum = cs2 + ca2;
    // discriminant ≥ 0 by AM-GM on (c_s², c_An²); guard against fp drift.
    let disc = max(sum * sum - 4.0 * cs2 * can2, 0.0);
    let cf2  = 0.5 * (sum + sqrt(disc));
    return sqrt(max(cf2, 0.0));
}

// 1D MHD flux along the sweep axis given full primitive state.
// Returns (F_U0, F_U1) where F_U0 = (Fρ, Fρvx, Fρvy, Fρvz) and
// F_U1 = (FE, FBz, _, _).
// Stone+ 2008 eq. 13 (with B² = bx²+by²+bz², and Btot·v shorthand).
// We do NOT include the flux for the normal-direction B (it's continuous
// across the face by construction); the transverse-B fluxes feed the
// CT EMF stage. The flux components returned in F.y/F.z carry the
// transverse-B advection contributions used by the EMF kernel.
struct MhdFlux {
    f0: vec4<f32>,
    f1: vec4<f32>,
    // The transverse B fluxes (signed): for x-sweep these are
    //   fBy = vx·By - vy·Bx  (= -Ez at the x-face)
    //   fBz = vx·Bz - vz·Bx
    // For y-sweep:
    //   fBx = vy·Bx - vx·By  (= +Ez at the y-face)
    //   fBz = vy·Bz - vz·By
    f_bt1: f32,
    f_bt2: f32,
};

fn mhd_flux(P: MhdPrim, gamma: f32, axis: u32) -> MhdFlux {
    let rho = max(P.rho, DENSITY_FLOOR);
    let p   = max(P.p,   PRESSURE_FLOOR);
    let ke  = 0.5 * rho * (P.vx*P.vx + P.vy*P.vy + P.vz*P.vz);
    let mb  = 0.5 * (P.bx*P.bx + P.by*P.by + P.bz*P.bz);
    let E   = p / (gamma - 1.0) + ke + mb;
    let p_t = p + mb;                       // total (gas + magnetic) pressure
    let vdotb = P.vx*P.bx + P.vy*P.by + P.vz*P.bz;

    var F: MhdFlux;
    if (axis == 0u) {
        // x-sweep: normal is x, transverse-1 is By, transverse-2 is Bz
        F.f0 = vec4<f32>(
            rho * P.vx,
            rho * P.vx * P.vx + p_t - P.bx * P.bx,
            rho * P.vx * P.vy       - P.bx * P.by,
            rho * P.vx * P.vz       - P.bx * P.bz,
        );
        F.f1 = vec4<f32>(
            (E + p_t) * P.vx - P.bx * vdotb,
            P.vx * P.bz - P.vz * P.bx,
            0.0, 0.0,
        );
        F.f_bt1 = P.vx * P.by - P.vy * P.bx;  // flux of By in x-direction = -Ez_face
        F.f_bt2 = P.vx * P.bz - P.vz * P.bx;
    } else {
        // y-sweep: normal is y, transverse-1 is Bx, transverse-2 is Bz
        F.f0 = vec4<f32>(
            rho * P.vy,
            rho * P.vy * P.vx       - P.by * P.bx,
            rho * P.vy * P.vy + p_t - P.by * P.by,
            rho * P.vy * P.vz       - P.by * P.bz,
        );
        F.f1 = vec4<f32>(
            (E + p_t) * P.vy - P.by * vdotb,
            P.vy * P.bz - P.vz * P.by,
            0.0, 0.0,
        );
        F.f_bt1 = P.vy * P.bx - P.vx * P.by;  // flux of Bx in y-direction = +Ez_face
        F.f_bt2 = P.vy * P.bz - P.vz * P.by;
    }
    return F;
}

// Build the conservative state from primitive plus the three B components.
struct ConsPair {
    U0: vec4<f32>,
    U1: vec4<f32>,
};

fn prim_to_cons_pair(P: MhdPrim, gamma: f32) -> ConsPair {
    let rho = max(P.rho, DENSITY_FLOOR);
    let p   = max(P.p,   PRESSURE_FLOOR);
    let ke  = 0.5 * rho * (P.vx*P.vx + P.vy*P.vy + P.vz*P.vz);
    let mb  = 0.5 * (P.bx*P.bx + P.by*P.by + P.bz*P.bz);
    let E   = p / (gamma - 1.0) + ke + mb;
    var R: ConsPair;
    R.U0 = vec4<f32>(rho, rho*P.vx, rho*P.vy, rho*P.vz);
    R.U1 = vec4<f32>(E, P.bz, 0.0, 0.0);
    return R;
}

struct PrimPair {
    p0: vec4<f32>,
    p1: vec4<f32>,
};

fn normal_velocity_mhd(P: MhdPrim, axis: u32) -> f32 {
    if (axis == 0u) { return P.vx; } else { return P.vy; }
}

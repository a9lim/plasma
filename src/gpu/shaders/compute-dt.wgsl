// ─── compute-dt.wgsl ─────────────────────────────────────────────────
// Three entry points (reset / reduce / finalize). Per-cell signal speed
// uses the fast magnetosonic speed (hyperbolic CFL). Session 8 also
// reduces a per-cell anomalous-η_max (when α > 0; when α = 0 it reduces
// to the uniform η_0) so the encoder can compute the RKL2 substep count
// `s` from the parabolic CFL diagnostic:
//
//   dt_hyp        = CFL · dx / max(|v|+c_fast)    (CFL from U_uniforms.cfl)
//   eta_max       = max over interior of η_local(|J|)
//   dt_parabolic  = 0.5 · dx² / eta_max           (RKL2 forward-Euler bound)
//   dt            = clamp(dt_hyp, [DT_MIN, DT_MAX])    ← hyperbolic only
//                   dt_parabolic is reported as a diagnostic in dt_buf[1]
//                   for the host-side RKL2 substep-count calculation.
//
// Pre-RKL2 (Sessions 1–7), dt was the min of hyperbolic and parabolic.
// With RKL2 the parabolic limit is replaced by `s` substeps that cover
// the hyperbolic Δt within the cell-wise stability bound. We therefore
// report dt = dt_hyp here; the host reads dt_parabolic from dt_buf[1]
// to set up RKL2.
//
// Bindings:
//   0 uniforms   (uniform)
//   1 U0_in      (ro)
//   2 U1_in      (ro)
//   3 Bx_face    (ro)
//   4 By_face    (ro)
//   5 wavespeed  (atomic<u32>)
//   6 dt_buf     (rw)   — slot 0: dt_hyp;  slot 1: dt_parabolic;  slots 2-3: pad
//   7 eta_max_buf(atomic<u32>) — global η_max bit-pattern, reduce target

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>           U0_in:       array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>           U1_in:       array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>           Bx_face:     array<f32>;
@group(0) @binding(4) var<storage, read>           By_face:     array<f32>;
@group(0) @binding(5) var<storage, read_write>     wavespeed:   atomic<u32>;
@group(0) @binding(6) var<storage, read_write>     dt_buf:      array<f32, 4>;
@group(0) @binding(7) var<storage, read_write>     eta_max_buf: atomic<u32>;

const DT_MIN: f32 = 1.0e-8;
const DT_MAX: f32 = 1.0e-2;

@compute @workgroup_size(1, 1, 1)
fn reset() {
    atomicStore(&wavespeed,   0u);
    atomicStore(&eta_max_buf, 0u);
}

var<workgroup> tile_max_wave: atomic<u32>;
var<workgroup> tile_max_eta:  atomic<u32>;

// |J_z(ix, iy)| from face B — same recipe as view-field and the
// resistivity passes. Reads ix±1 / iy±1; caller must guarantee range.
fn jz_mag_at(ix: u32, iy: u32, n_total: u32, dx_inv: f32) -> f32 {
    let by_R = 0.5 * (By_face[by_face_down_idx(ix + 1u, iy, n_total)]
                    + By_face[by_face_up_idx  (ix + 1u, iy, n_total)]);
    let by_L = 0.5 * (By_face[by_face_down_idx(ix - 1u, iy, n_total)]
                    + By_face[by_face_up_idx  (ix - 1u, iy, n_total)]);
    let bx_U = 0.5 * (Bx_face[bx_face_left_idx (ix, iy + 1u, n_total)]
                    + Bx_face[bx_face_right_idx(ix, iy + 1u, n_total)]);
    let bx_D = 0.5 * (Bx_face[bx_face_left_idx (ix, iy - 1u, n_total)]
                    + Bx_face[bx_face_right_idx(ix, iy - 1u, n_total)]);
    let dby_dx = (by_R - by_L) * 0.5 * dx_inv;
    let dbx_dy = (bx_U - bx_D) * 0.5 * dx_inv;
    return abs(dby_dx - dbx_dy);
}

@compute @workgroup_size(8, 8, 1)
fn reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if (lid == 0u) {
        atomicStore(&tile_max_wave, 0u);
        atomicStore(&tile_max_eta,  0u);
    }
    workgroupBarrier();

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let dx_inv     = 1.0 / U_uniforms.dx;

    if (gid.x < n_interior && gid.y < n_interior) {
        let ix = gid.x + ghost;
        let iy = gid.y + ghost;
        let bx = 0.5 * (Bx_face[bx_face_left_idx(ix, iy, n_total)]
                      + Bx_face[bx_face_right_idx(ix, iy, n_total)]);
        let by = 0.5 * (By_face[by_face_down_idx(ix, iy, n_total)]
                      + By_face[by_face_up_idx(ix, iy, n_total)]);
        let idx = cell_idx_total(ix, iy, n_total);
        let pf  = U_uniforms.pressure_floor;
        let P  = cons_to_prim_mhd(U0_in[idx], U1_in[idx], bx, by, U_uniforms.gamma, pf);
        let cfx = fast_mag_speed(P, U_uniforms.gamma, 0u, pf);
        let cfy = fast_mag_speed(P, U_uniforms.gamma, 1u, pf);
        let sx = abs(P.vx) + cfx;
        let sy = abs(P.vy) + cfy;
        let s  = max(sx, sy);
        let s_safe = select(0.0, s, s >= 0.0 && s == s);
        atomicMax(&tile_max_wave, bitcast<u32>(s_safe));

        // η_max reduction. With anomalous off (α = 0) this collapses to
        // U_uniforms.eta uniformly — still cheap to leave the path here
        // since the eta_local helper short-circuits. With α > 0, we
        // sample J_z and apply the anomalous-η formula. The defensive
        // `select`/`clamp` matches Session 3's wavespeed pattern: NaN
        // or negative bits could latch as huge u32s otherwise.
        let alpha = U_uniforms.eta_anom_alpha;
        var eta_l: f32;
        if (alpha <= 0.0) {
            eta_l = U_uniforms.eta;
        } else {
            let jmag = jz_mag_at(ix, iy, n_total, dx_inv);
            eta_l = anomalous_eta(jmag, U_uniforms.eta, alpha, U_uniforms.eta_anom_jcrit);
        }
        let eta_safe = select(0.0, eta_l, eta_l >= 0.0 && eta_l == eta_l);
        atomicMax(&tile_max_eta, bitcast<u32>(eta_safe));
    }

    workgroupBarrier();
    if (lid == 0u) {
        let mw = atomicLoad(&tile_max_wave);
        let me = atomicLoad(&tile_max_eta);
        atomicMax(&wavespeed,   mw);
        atomicMax(&eta_max_buf, me);
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize() {
    let s_bits = atomicLoad(&wavespeed);
    let s = max(bitcast<f32>(s_bits), 1.0e-12);
    let dx = U_uniforms.dx;
    let cfl_safe = max(U_uniforms.cfl, 1.0e-6);
    let dt_hyp = clamp(cfl_safe * dx / s, DT_MIN, DT_MAX);
    dt_buf[0] = dt_hyp;

    // Parabolic diagnostic — host reads this to pick RKL2 substep count.
    // RKL2's per-substep forward-Euler bound is dt_sub ≤ 0.5 · dx²/η_max
    // (factor 0.5, not 0.25; the previous code used the explicit-Euler
    // half of that to be conservative). When η_max is effectively zero
    // we report a huge value so the host's `s = ceil(...)` returns 1.
    let e_bits = atomicLoad(&eta_max_buf);
    let eta_max = max(bitcast<f32>(e_bits), 0.0);
    var dt_par: f32;
    if (eta_max > 1.0e-30) {
        dt_par = 0.5 * dx * dx / eta_max;
    } else {
        dt_par = 1.0e30;
    }
    dt_buf[1] = dt_par;
    dt_buf[2] = eta_max;     // diagnostic for the host / stats panel
    dt_buf[3] = 0.0;
}

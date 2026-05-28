// ─── compute-dt.wgsl ─────────────────────────────────────────────────
// Three entry points (reset / reduce / finalize). Per-cell signal speed
// uses the fast magnetosonic speed plus explicit source-term timestep
// constraints for opt-in extended physics. Session 8 also reduces a per-cell
// anomalous-η_max (when α > 0; when α = 0 it reduces to the uniform η_0) so
// the encoder can compute the RKL2 substep count `s` from the parabolic CFL
// diagnostic:
//
//   dt            = CFL · dx / max_signal
//   max_signal    = MHD wave speed plus gravity and source-cap safety
//                   constraints recast as equivalent signal speeds
//   eta_max       = max over interior of η_local(|J|)
//   dt_parabolic  = 0.25 · dx² / eta_max          (RKL2 forward-Euler bound)
//                   dt_parabolic is reported as a diagnostic in dt_buf[1]
//                   for the host-side RKL2 substep-count calculation.
//
// Pre-RKL2 (Sessions 1–7), dt was the min of hyperbolic and parabolic.
// With RKL2 the resistive parabolic limit is replaced by `s` substeps
// that cover the hyperbolic Δt within the cell-wise stability bound. Hall
// and conduction are still source-subcycled, but the macro step now includes
// a cap-aware safety limiter: if the configured maximum substep count is too
// low for the current source stiffness, dt_hyp is reduced enough that the
// source half-step remains stable instead of silently exceeding the explicit
// bound.
//
// Bindings:
//   0 uniforms   (uniform)
//   1 U0_in      (ro)
//   2 U1_in      (ro)
//   3 Bx_face    (ro)
//   4 By_face    (ro)
//   5 wavespeed  (atomic<u32>)  — MHD signal-speed reduction
//   6 dt_buf     (rw)   — slot 0: dt_hyp;  slot 1: dt_parabolic;
//                          slot 2: eta_max;  slot 3: hall_rate_max;
//                          slot 4: cond_rate_max
//   7 eta_max_buf(atomic<u32>) — global η_max bit-pattern, reduce target
//   8 hall_speed_buf (atomic<u32>) — historical host name; stores the
//                                    whistler inverse-time rate
//                                    max(v_A·d_i/dx²). Host reads dt_buf[3]
//                                    to size N_hall.
//   9 cond_speed_buf (atomic<u32>) — historical host name; stores the
//                                    explicit heat-diffusion inverse-time
//                                    rate max(4·χ/dx²). Host reads dt_buf[4]
//                                    to size N_cond.

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>           U0_in:          array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>           U1_in:          array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>           Bx_face:        array<f32>;
@group(0) @binding(4) var<storage, read>           By_face:        array<f32>;
@group(0) @binding(5) var<storage, read_write>     wavespeed:      atomic<u32>;
@group(0) @binding(6) var<storage, read_write>     dt_buf:         array<f32, 8>;
@group(0) @binding(7) var<storage, read_write>     eta_max_buf:    atomic<u32>;
@group(0) @binding(8) var<storage, read_write>     hall_speed_buf: atomic<u32>;
@group(0) @binding(9) var<storage, read_write>     cond_speed_buf: atomic<u32>;

const DT_MIN: f32 = 1.0e-8;
const DT_MAX: f32 = 1.0e-2;

@compute @workgroup_size(1, 1, 1)
fn reset() {
    atomicStore(&wavespeed,      0u);
    atomicStore(&eta_max_buf,    0u);
    atomicStore(&hall_speed_buf, 0u);
    atomicStore(&cond_speed_buf, 0u);
}

var<workgroup> tile_max_wave: atomic<u32>;
var<workgroup> tile_max_eta:  atomic<u32>;
var<workgroup> tile_max_hall: atomic<u32>;
var<workgroup> tile_max_cond: atomic<u32>;

const TRANSPORT_SCALE_MAX_DT: f32 = 1.0e5;

fn transport_scale_dt(theta: f32) -> f32 {
    // Mirrors the default uploaded microphysics transport family
    // (Spitzer/Braginskii T^(5/2)). compute-dt is already at the storage
    // binding ceiling, so the stability estimate uses the analytic default
    // rather than binding the table directly.
    return clamp(pow(max(theta, 1.0e-30), 2.5), 0.0, TRANSPORT_SCALE_MAX_DT);
}

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
        atomicStore(&tile_max_hall, 0u);
        atomicStore(&tile_max_cond, 0u);
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
        let rho = max(P.rho, DENSITY_FLOOR);
        let cfx = fast_mag_speed(P, U_uniforms.gamma, 0u, pf);
        let cfy = fast_mag_speed(P, U_uniforms.gamma, 1u, pf);
        let sx = abs(P.vx) + cfx;
        let sy = abs(P.vy) + cfy;
        // Unsplit 2D finite-volume CFL. The previous max(sx, sy) estimate
        // is a 1D bound; diagonal waves and genuinely 2D MHD flows need the
        // sum so the x and y flux updates cannot jointly overrun a cell.
        var s  = sx + sy;

        // ── Extended-physics timestep limits ───────────────────────
        // Reduce source-operator inverse-time rates so the host can pick
        // integer Strang-half-step subcycle counts without a readback stall.
        // These terms do not restrict the macro hyperbolic CFL directly.
        let dx = U_uniforms.dx;
        let cfl_safe = max(U_uniforms.cfl, 1.0e-6);
        let flags = U_uniforms.physics_flags;
        let b2 = P.bx*P.bx + P.by*P.by + P.bz*P.bz;

        // Hall whistler inverse-time rate — reduced separately so the host can size
        // sub-cycles. We also add a source-cap limiter to `s`: for a Strang
        // half-step, dt_macro <= 2 * N_hall * safety / rate. Recast through
        // dt = CFL * dx / s, that is s >= CFL * dx * rate / (2 N safety).
        if (flag_set(flags, FLAG_HALL) && U_uniforms.hall_di > 0.0) {
            let vA = sqrt(max(b2 / rho, 0.0));
            let r_hall = vA * U_uniforms.hall_di / max(dx * dx, 1.0e-30);
            let r_hall_safe = select(0.0, r_hall, r_hall >= 0.0 && r_hall == r_hall);
            atomicMax(&tile_max_hall, bitcast<u32>(r_hall_safe));
            let n_hall = max(f32(max(U_uniforms.hall_substeps_max, 1u)), 1.0);
            let s_hall_cap = cfl_safe * dx * r_hall_safe / (2.0 * n_hall * 0.5);
            s = s + select(0.0, s_hall_cap, s_hall_cap >= 0.0 && s_hall_cap == s_hall_cap);
        }

        // Conduction parabolic inverse-time rate — reduced separately so the host can
        // size sub-cycles. As for Hall, add only the source-cap limiter,
        // not the raw FE parabolic bound. Typical cases still run at the
        // hyperbolic dt; very stiff cases shrink the macro step rather than
        // overrunning the configured source_substeps_max.
        // 2D explicit heat diffusion bound: dt ≤ dx² / (4χ),
        // χ = (γ-1)·κ(T)/ρ. As an inverse-time rate: 4χ/dx².
        if (flag_set(flags, FLAG_CONDUCTION) && U_uniforms.conduction_kappa > 0.0) {
            let theta = (P.p / rho) / max(U_uniforms.cooling_T_ref, 1.0e-30);
            let kappa_T = U_uniforms.conduction_kappa * transport_scale_dt(theta);
            let chi = max(U_uniforms.gamma - 1.0, 1.0e-6)
                    * kappa_T / rho;
            let r_cond = 4.0 * chi / max(dx * dx, 1.0e-30);
            let r_cond_safe = select(0.0, r_cond, r_cond >= 0.0 && r_cond == r_cond);
            atomicMax(&tile_max_cond, bitcast<u32>(r_cond_safe));
            let n_cond = max(f32(max(U_uniforms.source_substeps_max, 1u)), 1.0);
            let s_cond_cap = cfl_safe * dx * r_cond_safe / (2.0 * n_cond * 0.5);
            s = s + select(0.0, s_cond_cap, s_cond_cap >= 0.0 && s_cond_cap == s_cond_cap);
        }

        let n_src = max(f32(max(U_uniforms.source_substeps_max, 1u)), 1.0);

        // Host-sized diffusion/source closures cannot cheaply reduce their
        // true cell-wise rate here without more bindings. Use the same
        // conservative uniform upper bounds as the host sizing path so the
        // soft sourceSubstepsMax cap still feeds back into the macro dt.
        if (flag_set(flags, FLAG_VISCOSITY)) {
            let nu_eff = max(max(U_uniforms.viscosity_nu * TRANSPORT_SCALE_MAX_DT,
                                 U_uniforms.viscosity_bulk * TRANSPORT_SCALE_MAX_DT),
                             U_uniforms.viscosity_shock);
            let r_visc = 4.0 * max(nu_eff, 0.0) / max(dx * dx, 1.0e-30);
            let s_visc_cap = cfl_safe * dx * r_visc / (2.0 * n_src * 0.45);
            s = s + select(0.0, s_visc_cap, s_visc_cap >= 0.0 && s_visc_cap == s_visc_cap);
        }

        var r_nonideal = 0.0;
        if (flag_set(flags, FLAG_AMBIPOLAR) && U_uniforms.ambipolar_eta > 0.0 && U_uniforms.neutral_frac > 0.0) {
            r_nonideal = r_nonideal
                + 4.0 * U_uniforms.ambipolar_eta * max(U_uniforms.neutral_frac, 0.0)
                / max(dx * dx, 1.0e-30);
        }
        if (flag_set(flags, FLAG_BIERMANN) && U_uniforms.biermann_coeff != 0.0
            && U_uniforms.hall_electron_pressure_frac > 0.0) {
            r_nonideal = r_nonideal + abs(U_uniforms.biermann_coeff) / max(dx, 1.0e-30);
        }
        if (flag_set(flags, FLAG_ELECTRON_INERTIA)
            && U_uniforms.electron_inertia_length > 0.0
            && U_uniforms.electron_inertia_damping > 0.0) {
            let eta4 = U_uniforms.electron_inertia_damping
                * U_uniforms.electron_inertia_length
                * U_uniforms.electron_inertia_length;
            r_nonideal = r_nonideal + 16.0 * eta4 / max(dx * dx * dx * dx, 1.0e-30);
        }
        let s_nonideal_cap = cfl_safe * dx * r_nonideal / (2.0 * n_src * 0.45);
        s = s + select(0.0, s_nonideal_cap, s_nonideal_cap >= 0.0 && s_nonideal_cap == s_nonideal_cap);

        if (flag_set(flags, FLAG_RADIATION) && U_uniforms.radiation_c > 0.0
            && (U_uniforms.radiation_kappa_abs > 0.0 || U_uniforms.radiation_kappa_scat > 0.0)) {
            let kappa = max(U_uniforms.radiation_kappa_abs, 0.0)
                      + max(U_uniforms.radiation_kappa_scat, 0.0);
            let opacity_min = 0.01;
            let opacity_max_abs = 32.0;
            let r_rad_diff = select(0.0,
                                    4.0 * U_uniforms.radiation_c
                                      / max(kappa * opacity_min * dx * dx, 1.0e-30),
                                    kappa > 0.0);
            let r_rad_exchange = U_uniforms.radiation_c
                * max(U_uniforms.radiation_kappa_abs, 0.0)
                * opacity_max_abs;
            let r_rad = r_rad_diff + r_rad_exchange;
            let s_rad_cap = cfl_safe * dx * r_rad / (2.0 * n_src * 0.35);
            s = s + select(0.0, s_rad_cap, s_rad_cap >= 0.0 && s_rad_cap == s_rad_cap);
        }

        // Cooling no longer contributes a Δt bound — Session 15's
        // `apply-cooling.wgsl` uses an exact analytic integrator for the
        // single-power-law Λ(T) ∝ √(T−T_floor) shape (Townsend 2009 spirit)
        // that is unconditionally stable for any Δt. See the apply-cooling
        // header for the derivation.

        if (flag_set(flags, FLAG_GRAVITY_EXT)) {
            let g_ext = sqrt(U_uniforms.gravity_gx * U_uniforms.gravity_gx
                           + U_uniforms.gravity_gy * U_uniforms.gravity_gy);
            if (g_ext > 0.0) {
                // Acceleration should not move material more than O(dx) in a
                // source kick: dt <= 0.25 sqrt(dx/|g|).
                let s_g = 4.0 * cfl_safe * sqrt(max(dx * g_ext, 0.0));
                s = s + select(0.0, s_g, s_g >= 0.0 && s_g == s_g);
            }
        }

        if (flag_set(flags, FLAG_GRAVITY_SELF) && U_uniforms.gravity_G > 0.0) {
            // Local freefall/plasma-frequency-like bound for explicit
            // self-gravity coupling: dt <= 0.25 / sqrt(4πGρ).
            let omega_g = sqrt(max(4.0 * 3.141592653589793 * U_uniforms.gravity_G * rho, 0.0));
            let s_self_g = 4.0 * cfl_safe * dx * omega_g;
            s = s + select(0.0, s_self_g, s_self_g >= 0.0 && s_self_g == s_self_g);
        }

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
        let mh = atomicLoad(&tile_max_hall);
        let mc = atomicLoad(&tile_max_cond);
        atomicMax(&wavespeed,      mw);
        atomicMax(&eta_max_buf,    me);
        atomicMax(&hall_speed_buf, mh);
        atomicMax(&cond_speed_buf, mc);
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
    // This is the 2D forward-Euler stability bound for the 5-point
    // Laplacian:  Δt_FE ≤ dx² / (4 · η_max)  =  0.25 · dx² / η_max.
    // Derivation: the 5-point Laplacian has spectral radius 8/dx² in 2D
    // (sum of two 1D operators with spectral radius 4/dx² each), so FE
    // stability requires |1 − η·Δt·λ_max| ≤ 1 → Δt ≤ dx² / (4·η).
    //
    // Session 13 retrospective: this was 0.5·dx²/η for the prior several
    // sessions. The comment justified it as "factor 0.5, not 0.25; the
    // previous code used the explicit-Euler half of that to be
    // conservative" — but that's wrong: 0.25·dx²/η isn't "conservative",
    // it's the 2D bound; 0.5·dx²/η is the 1D bound applied incorrectly
    // in 2D. The 2× error made the RKL2 substep count (computed CPU-side
    // from dt_super/dt_parabolic) consistently too small. At high η on
    // Orszag-Tang N=1024 (ratio ~13) RKL2 ran with s=5 when s=7 was
    // required, leaving the highest-k modes unstable; they grew until
    // resistivity caught up, then the cycle repeated — visible as blobs
    // fading in and out in the J_z view. Fixed alongside the missing
    // `-2` stability margin in `_computeRKL2Coeffs` (see sim.js).
    //
    // When η_max is effectively zero we report a huge value so the
    // host's `s = ceil(...)` returns 1.
    let e_bits = atomicLoad(&eta_max_buf);
    let eta_max = max(bitcast<f32>(e_bits), 0.0);
    var dt_par: f32;
    if (eta_max > 1.0e-30) {
        dt_par = 0.25 * dx * dx / eta_max;
    } else {
        dt_par = 1.0e30;
    }
    dt_buf[1] = dt_par;
    dt_buf[2] = eta_max;     // diagnostic for the host / stats panel

    // hall_rate_max — reduces v_A·d_i/dx² over interior cells; host
    // multiplies by the Strang half-step to size the Hall sub-cycle.
    let h_bits = atomicLoad(&hall_speed_buf);
    dt_buf[3] = max(bitcast<f32>(h_bits), 0.0);

    // cond_rate_max — reduces 4·χ/dx² over interior cells; host
    // multiplies by the Strang half-step to size the conduction sub-cycle.
    let c_bits = atomicLoad(&cond_speed_buf);
    dt_buf[4] = max(bitcast<f32>(c_bits), 0.0);
}

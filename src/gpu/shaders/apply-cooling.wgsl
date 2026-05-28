// ─── apply-cooling.wgsl ───────────────────────────────────────────────
// Optically-thin radiative cooling source term applied to total energy
// E in the cell-centered conserved state.
//
//   dE/dt |_cool  =  −n² · Λ(T)
//
// where n = ρ/(μ m_p) is the number density and Λ(T) is the cooling
// function. In code units we take μ m_p = 1 so n ≡ ρ, and pick the
// simplest physically-motivated cooling curve — thermal bremsstrahlung:
//
//   Λ(T)  =  Λ_0 · sqrt(max(T − T_floor, 0) / T_ref)
//
// where Λ_0 is the rate normalization (uniforms.cooling_lambda0),
// T_floor is the temperature below which radiative cooling switches
// off, and T_ref is the reference temperature for the normalization.
// Bremsstrahlung has Λ ∝ √T at high T — the dominant cooling channel
// for fully-ionized H/He at T ≳ 10⁷ K.
//
// Integration modes:
//   cooling_curve_mode = 0: Session-15 Townsend-style exact integration for
//                          the single √T bremsstrahlung shape.
//   cooling_curve_mode = 1: Session-16 piecewise power-law table. This keeps
//                          the exact-per-segment Townsend update while adding
//                          a line-cooling peak and high-T brems tail.
//   cooling_curve_mode = 2: broader CIE-inspired solar-metallicity power-law
//                          table, with metallicity scaling and optional
//                          volumetric heating for thermal-balance tests.
//   cooling_curve_mode = 3: uploaded microphysics table. The table lives in
//                          a storage buffer so future presets can replace
//                          the compact built-in shape without WGSL edits.
//
// Single-shape derivation:
//
//   dT/dt = -(γ-1) ρ Λ_0 · √((T − T_floor) / T_ref)
//
// Substitute s = √((T − T_floor) / T_ref) ⇒ T − T_floor = T_ref · s².
// Differentiating both sides w.r.t. t:
//
//   dT/dt = 2 · T_ref · s · ds/dt
//   -(γ-1) ρ Λ_0 · s = 2 · T_ref · s · ds/dt
//   ds/dt = -(γ-1) ρ Λ_0 / (2 · T_ref)             ← constant in t
//
// So s evolves linearly in time. Exact solution over an arbitrary Δt:
//
//   s(t)  = max(s(0) − C · t, 0)            C = (γ-1) ρ Λ_0 / (2 T_ref)
//   T(t)  = T_floor + T_ref · s(t)²
//
// When s(t) hits 0 the cell pins at T_floor — no further cooling. This
// is unconditionally stable for any Δt (no FE timestep bound), and
// recovers the explicit-FE answer in the limit Δt → 0. Townsend 2009
// generalizes this construction to a piecewise-power-law Λ(T) on a
// tabulated grid; for our single √T shape the analytic form lives
// inline.
//
// Bindings:
//   0 uniforms (uniform)
//   1 U0       (ro) — for ρ → n
//   2 U1       (rw) — E gets the cooling source
//   3 Bx_face  (ro) — for cell-centered Bx → magnetic-energy floor
//   4 By_face  (ro) — for cell-centered By → magnetic-energy floor
//   5 dt_buf   (uniform, dt_hyp)
//   6 microphysics table (ro storage)

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

struct CoolingSeg {
    theta_lo:  f32,
    lambda_lo: f32,
    alpha:     f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:    array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:    array<f32>;
@group(0) @binding(5) var<uniform>             dt_buf:     DtUniform;
@group(0) @binding(6) var<storage, read>       micro:      array<vec4<f32>>;

const COOLING_CURVE_TABULATED: u32 = 3u;
const MICRO_COOL_START: u32 = 0u;
const MICRO_COOL_COUNT: u32 = 24u;
const INV_LN10: f32 = 0.4342944819032518;

fn micro_segment(start: u32, count: u32, theta: f32) -> CoolingSeg {
    let log_theta = log(max(theta, 1.0e-30)) * INV_LN10;
    var idx = start;
    for (var i: u32 = 0u; i < 23u; i = i + 1u) {
        if (i + 1u >= count) { break; }
        let next = micro[start + i + 1u];
        if (log_theta < next.x) {
            idx = start + i;
            break;
        }
        idx = start + i + 1u;
    }
    let r = micro[idx];
    return CoolingSeg(pow(10.0, r.x), pow(10.0, r.y), r.z);
}

fn cooling_table_segment(theta: f32) -> CoolingSeg {
    // Dimensionless table in θ = T/T_ref. λ values are relative to Λ_0.
    // The shape is intentionally compact: a steep low-T rise, a line-cooling
    // peak near θ≈0.03, a trough around θ≈3, and a √θ brems tail.
    var s: CoolingSeg;
    if (theta <= 3.0e-4) {
        s = CoolingSeg(1.0e-4, 0.02, 1.261860);
    } else if (theta <= 1.0e-3) {
        s = CoolingSeg(3.0e-4, 0.08, 1.336773);
    } else if (theta <= 3.0e-3) {
        s = CoolingSeg(1.0e-3, 0.40, 1.000000);
    } else if (theta <= 1.0e-2) {
        s = CoolingSeg(3.0e-3, 1.20, 0.642199);
    } else if (theta <= 3.0e-2) {
        s = CoolingSeg(1.0e-2, 2.60, 0.392116);
    } else if (theta <= 1.0e-1) {
        s = CoolingSeg(3.0e-2, 4.00, -0.238944);
    } else if (theta <= 3.0e-1) {
        s = CoolingSeg(1.0e-1, 3.00, -0.572184);
    } else if (theta <= 1.0) {
        s = CoolingSeg(3.0e-1, 1.60, -0.390377);
    } else if (theta <= 3.0) {
        s = CoolingSeg(1.0, 1.00, -0.261860);
    } else if (theta <= 10.0) {
        s = CoolingSeg(3.0, 0.75, 0.238944);
    } else if (theta <= 30.0) {
        s = CoolingSeg(10.0, 1.00, 0.535026);
    } else if (theta <= 100.0) {
        s = CoolingSeg(30.0, 1.80, 0.503446);
    } else {
        s = CoolingSeg(100.0, 3.30, 0.5);
    }
    return s;
}

fn cooling_cie_segment(theta: f32) -> CoolingSeg {
    // Dimensionless CIE-inspired optically thin cooling shape in θ = T/T_ref.
    // The knot values follow the qualitative solar-metallicity structure used
    // by Sutherland-Dopita / CHIANTI-style curves: low-T cutoff, strong metal
    // line peak near 10^5-10^6 K, trough around virial/coronal temperatures,
    // and a high-T bremsstrahlung tail. Λ_0 sets the absolute code-unit scale.
    var s: CoolingSeg;
    if (theta <= 3.0e-4) {
        s = CoolingSeg(1.0e-4, 0.005, 0.40);
    } else if (theta <= 1.0e-3) {
        s = CoolingSeg(3.0e-4, 0.008, 0.82);
    } else if (theta <= 3.0e-3) {
        s = CoolingSeg(1.0e-3, 0.020, 1.47);
    } else if (theta <= 1.0e-2) {
        s = CoolingSeg(3.0e-3, 0.10, 1.34);
    } else if (theta <= 3.0e-2) {
        s = CoolingSeg(1.0e-2, 0.50, 1.47);
    } else if (theta <= 1.0e-1) {
        s = CoolingSeg(3.0e-2, 2.50, 0.58);
    } else if (theta <= 3.0e-1) {
        s = CoolingSeg(1.0e-1, 5.00, -0.63);
    } else if (theta <= 1.0) {
        s = CoolingSeg(3.0e-1, 2.50, -0.47);
    } else if (theta <= 3.0) {
        s = CoolingSeg(1.0, 1.40, -0.31);
    } else if (theta <= 10.0) {
        s = CoolingSeg(3.0, 1.00, 0.19);
    } else if (theta <= 30.0) {
        s = CoolingSeg(10.0, 1.25, 0.33);
    } else if (theta <= 100.0) {
        s = CoolingSeg(30.0, 1.80, 0.43);
    } else if (theta <= 300.0) {
        s = CoolingSeg(100.0, 3.00, 0.50);
    } else {
        s = CoolingSeg(300.0, 5.20, 0.50);
    }
    return s;
}

fn cooling_segment(theta: f32) -> CoolingSeg {
    if (U_uniforms.cooling_curve_mode == COOLING_CURVE_TABULATED) {
        return micro_segment(MICRO_COOL_START, MICRO_COOL_COUNT, theta);
    }
    if (U_uniforms.cooling_curve_mode == 2u) {
        return cooling_cie_segment(theta);
    }
    return cooling_table_segment(theta);
}

fn cooling_metal_scale(theta: f32) -> f32 {
    // CIE line cooling is metal-sensitive around the line peak; high-T
    // free-free emission remains even for metal-poor gas. This compact scale
    // keeps Z=0 finite while making Z≈1 the default solar-calibrated shape.
    if (U_uniforms.cooling_curve_mode != 2u && U_uniforms.cooling_curve_mode != COOLING_CURVE_TABULATED) {
        return 1.0;
    }
    let z = max(U_uniforms.cooling_metallicity, 0.0);
    let line_weight = 1.0 / (1.0 + pow(max(theta / 30.0, 0.0), 2.0));
    let metal = 0.18 + 0.82 * z;
    return mix(1.0, metal, line_weight);
}

fn segment_A(rate: f32, seg: CoolingSeg) -> f32 {
    return rate * seg.lambda_lo / pow(max(seg.theta_lo, 1.0e-12), seg.alpha);
}

fn theta_after_powerlaw(theta0: f32, dt: f32, rate: f32, seg: CoolingSeg) -> f32 {
    let A = segment_A(rate, seg);
    if (A <= 0.0 || dt <= 0.0) { return theta0; }
    let a = seg.alpha;
    if (abs(a - 1.0) < 1.0e-4) {
        return theta0 * exp(-A * dt);
    }
    let p = 1.0 - a;
    let y = pow(theta0, p) - p * A * dt;
    if (y <= 0.0 && p > 0.0) { return 0.0; }
    return pow(max(y, 1.0e-30), 1.0 / p);
}

fn time_to_theta(theta0: f32, theta1: f32, rate: f32, seg: CoolingSeg) -> f32 {
    if (theta1 >= theta0) { return 0.0; }
    let A = segment_A(rate, seg);
    if (A <= 0.0) { return 1.0e30; }
    let a = seg.alpha;
    if (abs(a - 1.0) < 1.0e-4) {
        return max(log(theta0 / max(theta1, 1.0e-30)) / A, 0.0);
    }
    let p = 1.0 - a;
    return max((pow(theta0, p) - pow(theta1, p)) / (A * p), 0.0);
}

fn cool_table_theta(theta0: f32, theta_floor: f32, dt: f32, rate: f32) -> f32 {
    var theta = max(theta0, theta_floor);
    var rem = dt;
    for (var iter: u32 = 0u; iter < 24u; iter = iter + 1u) {
        if (rem <= 0.0 || theta <= theta_floor) { break; }
        let seg0 = cooling_segment(theta);
        let seg = CoolingSeg(seg0.theta_lo,
                             seg0.lambda_lo * cooling_metal_scale(theta),
                             seg0.alpha);
        var lower = seg.theta_lo;
        if (theta_floor > lower) {
            lower = theta_floor;
        }
        if (seg.theta_lo <= 1.0e-4) {
            // Extend the first tabulated slope down to the configured floor.
            lower = theta_floor;
        }
        let dt_lower = time_to_theta(theta, lower, rate, seg);
        if (dt_lower <= 1.0e-12 || dt_lower != dt_lower) {
            theta = lower;
        } else if (rem < dt_lower) {
            let theta_trial = theta_after_powerlaw(theta, rem, rate, seg);
            theta = theta_trial;
            if (theta < lower) {
                theta = lower;
            }
            rem = 0.0;
        } else {
            theta = lower;
            rem = rem - dt_lower;
        }
    }
    return max(theta, theta_floor);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cooling_on = flag_set(U_uniforms.physics_flags, FLAG_COOLING)
                  && U_uniforms.cooling_lambda0 > 0.0;
    let heating_on = flag_set(U_uniforms.physics_flags, FLAG_HEATING)
                  && U_uniforms.heating_gamma0 > 0.0;
    if (!cooling_on && !heating_on) { return; }

    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    let c  = cell_idx_total(ix, iy, n_total);

    let u0 = U0[c];
    let u1 = U1[c];

    let rho = max(u0.x, DENSITY_FLOOR);
    let mx  = u0.y;
    let my  = u0.z;
    let mz  = u0.w;
    let E   = u1.x;
    let bz  = u1.y;

    let bx_c = 0.5 * (Bx_face[bx_face_idx(ix,      iy, n_total)]
                    + Bx_face[bx_face_idx(ix + 1u, iy, n_total)]);
    let by_c = 0.5 * (By_face[by_face_idx(ix, iy,      n_total)]
                    + By_face[by_face_idx(ix, iy + 1u, n_total)]);

    let ke = 0.5 * (mx*mx + my*my + mz*mz) / rho;
    let mb = 0.5 * (bx_c*bx_c + by_c*by_c + bz*bz);
    let p_floor = U_uniforms.pressure_floor;
    let p  = pressure_from_dual_energy(u0, u1, bx_c, by_c,
                                       U_uniforms.gamma, p_floor);

    // T = p / ρ in code units (Boltzmann's constant absorbed).
    let T = p / rho;

    let dT_excess = T - U_uniforms.cooling_T_floor;

    let T_ref = max(U_uniforms.cooling_T_ref, 1.0e-30);
    var T_new = T;
    if (cooling_on && dT_excess > 0.0) {
        if (U_uniforms.cooling_curve_mode == 0u) {
            let s0 = sqrt(dT_excess / T_ref);
            let C  = (U_uniforms.gamma - 1.0) * rho
                   * U_uniforms.cooling_lambda0 / (2.0 * T_ref);
            let s1 = max(s0 - C * dt_buf.dt, 0.0);
            T_new = U_uniforms.cooling_T_floor + T_ref * s1 * s1;
        } else {
            let theta0 = max(T / T_ref, 1.0e-8);
            let theta_floor = max(U_uniforms.cooling_T_floor / T_ref, 1.0e-8);
            let rate = (U_uniforms.gamma - 1.0) * rho * U_uniforms.cooling_lambda0 / T_ref;
            T_new = T_ref * cool_table_theta(theta0, theta_floor, dt_buf.dt, rate);
        }
    }
    let p_new = max(rho * T_new, p_floor);
    var E_new = ke + mb + p_new / (U_uniforms.gamma - 1.0);

    if (heating_on) {
        let rho_term = pow(rho, max(U_uniforms.heating_density_exp, 0.0));
        let cutoff = U_uniforms.heating_T_cut;
        let hot_suppression = select(
            1.0,
            1.0 / (1.0 + pow(max(T / max(cutoff, 1.0e-30), 0.0), 4.0)),
            cutoff > 0.0,
        );
        E_new = E_new + U_uniforms.heating_gamma0 * rho_term * hot_suppression * dt_buf.dt;
    }

    let p_final = max((U_uniforms.gamma - 1.0) * (E_new - ke - mb), p_floor);
    U1[c] = pack_u1_aux(E_new, bz, rho, p_final, U_uniforms.gamma, p_floor);
}

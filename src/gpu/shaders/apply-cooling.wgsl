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
// Integration (Session 15 — Townsend-style exact integration for the
// single-power-law cooling shape):
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

struct DtUniform {
    dt: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<uniform>             U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       U0:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> U1:         array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       Bx_face:    array<f32>;
@group(0) @binding(4) var<storage, read>       By_face:    array<f32>;
@group(0) @binding(5) var<uniform>             dt_buf:     DtUniform;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!flag_set(U_uniforms.physics_flags, FLAG_COOLING)) { return; }
    if (U_uniforms.cooling_lambda0 <= 0.0)               { return; }

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
    let p  = max((U_uniforms.gamma - 1.0) * (E - ke - mb), p_floor);

    // T = p / ρ in code units (Boltzmann's constant absorbed).
    let T = p / rho;

    let dT_excess = T - U_uniforms.cooling_T_floor;
    // Already at/below floor → cooling source is zero. Don't touch E.
    if (dT_excess <= 0.0) { return; }

    let T_ref = max(U_uniforms.cooling_T_ref, 1.0e-30);
    let s0 = sqrt(dT_excess / T_ref);
    let C  = (U_uniforms.gamma - 1.0) * rho
           * U_uniforms.cooling_lambda0 / (2.0 * T_ref);
    let s1 = max(s0 - C * dt_buf.dt, 0.0);
    let T_new = U_uniforms.cooling_T_floor + T_ref * s1 * s1;
    let p_new = max(rho * T_new, p_floor);
    let E_new = ke + mb + p_new / (U_uniforms.gamma - 1.0);

    U1[c] = vec4<f32>(E_new, bz, u1.z, u1.w);
}

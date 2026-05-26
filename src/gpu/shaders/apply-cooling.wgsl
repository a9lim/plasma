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
// Integration: explicit forward Euler over Δt = dt_hyp. For the
// breadth pass we accept that this is *not* unconditionally stable —
// when n²Λ Δt becomes a significant fraction of (γ-1)·p, the energy
// can over-shoot below the pressure floor in one step. The proper fix
// (Townsend 2009 exact integration with a piecewise-power-law Λ) is a
// later upgrade. For now we clamp the energy decrement so the cell
// can never drop below the pressure-floor energy in a single step.
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

    let dT = max(T - U_uniforms.cooling_T_floor, 0.0);
    let lambda_T = U_uniforms.cooling_lambda0 * sqrt(dT / U_uniforms.cooling_T_ref);

    // Λ(T) is energy/time/n². dE/dt = -n²Λ. With n = ρ:
    let cooling_rate = rho * rho * lambda_T;
    let dt = dt_buf.dt;
    var dE = cooling_rate * dt;

    // Clamp so we cannot drop below the pressure-floor energy in one
    // step. E_min = KE + ½|B|² + p_floor/(γ-1).
    let E_min = ke + mb + p_floor / (U_uniforms.gamma - 1.0);
    let max_dec = max(E - E_min, 0.0);
    if (dE > max_dec) { dE = max_dec; }

    U1[c] = vec4<f32>(E - dE, bz, u1.z, u1.w);
}

/**
 * @fileoverview GPU buffer allocator for Phase 4 (resistive MHD + ghost cells).
 *
 * Phase 4 changes vs Phase 3b:
 *   • All field buffers expand by GHOST_W per side.
 *     Cell-centered (U0, U1, edges, fluxes): (N+4) × (N+4).
 *     Bx_face:                                (N+5) × (N+4) — extra column.
 *     By_face:                                (N+4) × (N+5) — extra row.
 *     Ez_edge:                                (N+5) × (N+5) — corners.
 *   • Interior cell range: [GHOST_W, GHOST_W + N) = [2, N+2).
 *     Workgroup dispatch sizes target interior cells only; ghost cells
 *     are filled exclusively by apply-bcs.wgsl at the start of each
 *     RK3 stage. (compute-dt's reduce also sweeps interior only.)
 *   • New `bc_uniforms` storage buffer (48B padded to 64B) holds 4 per-
 *     edge BC mode IDs (N=top, S=bottom, E=right, W=left) plus an
 *     8-float driven inflow primitive state (ρ, vx, vy, vz, Bx, By, Bz, p).
 *   • Uniforms struct extended: adds `eta`, `grid_n_total`, `ghost_w`.
 *
 * Uniform consolidation (Round 2): a single shared `uniform` buffer
 * (64 B) replaces the legacy `uniform_x` / `uniform_y` pair. Sweep
 * direction is now in two tiny static buffers (`sweepDir_x`,
 * `sweepDir_y`, 16 B each) — only reconstruct-ppm and riemann-hlld
 * bind them. LIC render-pace fields (phase / intensity / drift) live
 * in a separate 16 B `licUniform` buffer rewritten per render frame
 * via `pushLicUniforms()`; the main `pushUniforms()` writes only on
 * physics-state changes.
 *
 * The face-ownership convention also changed (LEFT/DOWN owner instead of
 * RIGHT/UP). See shared-helpers.wgsl header for the full convention.
 *
 * Storage slot layout is otherwise identical:
 *   slot_n  — start-of-step state                       (A then B then A …)
 *   slot_1  — intermediate after stage 1                (always C)
 *   slot_2  — intermediate after stage 2                (always D)
 *   slot_next — destination of stage 3                  (B then A then B …)
 * After each step, swap A↔B so slot_next becomes slot_n for the next step.
 *
 * Cell-centered conserved state is packed into two parallel vec4 buffers
 * per slot:
 *   U0 = (ρ, ρvx, ρvy, ρvz)
 *   U1 = (E,  Bz,    _pad, _pad)
 *
 * PPM edge buffers store both left and right edge primitive states per
 * cell per direction (4 vec4 outputs per axis × 2 axes = 8 buffers).
 *
 * Bind-group layout policy (transpiler-friendly): one bind group per
 * dispatch, explicit static storage/uniform bindings, no dynamic
 * offsets / push constants / subgroup ops.
 *
 * No CPU readback in Phase 4 — every cross-pass dependency is GPU↔GPU.
 */

import {
    GRID_N, GHOST_WIDTH, UNIFORM_BUFFER_SIZE, BC_UNIFORM_BUFFER_SIZE,
    VIEW_DENSITY, BC_PERIODIC, ETA_DEFAULT, PRESSURE_FLOOR,
    LIC_NOISE_N, LIC_INTENSITY_DEFAULT, LIC_DRIFT_X, LIC_DRIFT_Y, LIC_NOISE_SEED,
} from '../config.js';
import { buildMicrophysicsTable, MICRO_TABLE_ENTRIES, MICRO_STRIDE } from '../microphysics.js';

const VEC4_BYTES = 16;
const F32_BYTES  = 4;
const STAGE_PARAMS_BYTES = 16; // (a0, a1, dt_w, _pad) f32×4

export class PlasmaBuffers {
    /**
     * @param {GPUDevice} device
     * @param {number} n interior grid resolution (square)
     */
    constructor(device, n = GRID_N) {
        this.device = device;
        this.n        = n;
        this.ghost    = GHOST_WIDTH;
        this.n_total  = n + 2 * GHOST_WIDTH;

        const N      = this.n_total;
        const cellsT = N * N;                      // cell-centered storage cells
        const xfaces = (N + 1) * N;                // Bx_face cells
        const yfaces = N * (N + 1);                // By_face cells
        const edges  = (N + 1) * (N + 1);          // Ez_edge cells

        const u_v4_cell_bytes   = cellsT * VEC4_BYTES;
        const u_f32_xface_bytes = xfaces * F32_BYTES;
        const u_f32_yface_bytes = yfaces * F32_BYTES;
        const u_f32_edge_bytes  = edges  * F32_BYTES;
        const u_f32_cell_bytes  = cellsT * F32_BYTES;
        const u_v4_edge_bytes   = edges  * VEC4_BYTES;
        this.cellScalarBytes = u_f32_cell_bytes;

        const mkStorage = (label, size, extra = 0) => device.createBuffer({
            label, size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra,
        });
        this._mkStorage = mkStorage;

        // ── RK3 storage: 4 slots × (U0, U1, Bx_face, By_face) ──────
        this.U0_a = mkStorage('plasma.U0_a', u_v4_cell_bytes);
        this.U1_a = mkStorage('plasma.U1_a', u_v4_cell_bytes);
        this.U0_b = mkStorage('plasma.U0_b', u_v4_cell_bytes);
        this.U1_b = mkStorage('plasma.U1_b', u_v4_cell_bytes);
        this.U0_1 = mkStorage('plasma.U0_1', u_v4_cell_bytes);
        this.U1_1 = mkStorage('plasma.U1_1', u_v4_cell_bytes);
        this.U0_2 = mkStorage('plasma.U0_2', u_v4_cell_bytes);
        this.U1_2 = mkStorage('plasma.U1_2', u_v4_cell_bytes);

        this.Bx_a = mkStorage('plasma.Bx_a', u_f32_xface_bytes);
        this.By_a = mkStorage('plasma.By_a', u_f32_yface_bytes);
        this.Bx_b = mkStorage('plasma.Bx_b', u_f32_xface_bytes);
        this.By_b = mkStorage('plasma.By_b', u_f32_yface_bytes);
        this.Bx_1 = mkStorage('plasma.Bx_1', u_f32_xface_bytes);
        this.By_1 = mkStorage('plasma.By_1', u_f32_yface_bytes);
        this.Bx_2 = mkStorage('plasma.Bx_2', u_f32_xface_bytes);
        this.By_2 = mkStorage('plasma.By_2', u_f32_yface_bytes);

        // Logical handles (renames). swap() flips A↔B.
        this._side = 'a';
        this.U0_n     = this.U0_a; this.U1_n     = this.U1_a;
        this.U0_next  = this.U0_b; this.U1_next  = this.U1_b;
        this.Bx_n     = this.Bx_a; this.By_n     = this.By_a;
        this.Bx_next  = this.Bx_b; this.By_next  = this.By_b;

        // Per-stage scratch: Ez_edge recomputed each stage.
        this.Ez_edge = mkStorage('plasma.Ez_edge', u_f32_edge_bytes);

        // ── RKL2 super-time-stepping buffers ────────────────────────
        // Resistive diffusion runs as ONE RKL2 super-step at the end of
        // each RK3 macro-step (Lie split). RKL2's recurrence (MDK 2014
        // eq 13) reads three field snapshots per substep — Y_init = U^n,
        // Y_{j-1}, Y_{j-2} — so we keep four buffer sets in play:
        //   - dst  (the destination buffer the RK3 step wrote into;
        //           overwritten at the end of the super-step with Y_s).
        //   - init  (frozen U^n for the duration of the super-step).
        //   - 3 rotating sets (pprev, prev, tmp) — see apply-resistivity*.wgsl
        //     headers for the role-rotation pattern.
        // 4 sets × 3 components = 12 new GPU buffers (init + pprev + prev
        // + tmp). At N=256: ~3 MB total. Negligible vs the main U/B
        // storage.
        this.Bx_res_init = mkStorage('plasma.Bx_res_init', u_f32_xface_bytes);
        this.By_res_init = mkStorage('plasma.By_res_init', u_f32_yface_bytes);
        this.U1_res_init = mkStorage('plasma.U1_res_init', u_v4_cell_bytes);

        this.Bx_res_pprev = mkStorage('plasma.Bx_res_pprev', u_f32_xface_bytes);
        this.By_res_pprev = mkStorage('plasma.By_res_pprev', u_f32_yface_bytes);
        this.U1_res_pprev = mkStorage('plasma.U1_res_pprev', u_v4_cell_bytes);

        this.Bx_res_prev = mkStorage('plasma.Bx_res_prev', u_f32_xface_bytes);
        this.By_res_prev = mkStorage('plasma.By_res_prev', u_f32_yface_bytes);
        this.U1_res_prev = mkStorage('plasma.U1_res_prev', u_v4_cell_bytes);

        this.Bx_res_tmp = mkStorage('plasma.Bx_res_tmp', u_f32_xface_bytes);
        this.By_res_tmp = mkStorage('plasma.By_res_tmp', u_f32_yface_bytes);
        this.U1_res_tmp = mkStorage('plasma.U1_res_tmp', u_v4_cell_bytes);

        // RKL2 coefficient buffer — packed (μ, ν, μ̃, γ̃) per substep.
        // Re-uploaded once per super-step (s coefficients × 4 f32).
        // STS_COEFFS_MAX_S = 100 caps in-browser super-step length —
        // Athena++ uses 200 in batch runs; 100 is safer for an
        // interactive sim (a single super-step never blocks the UI
        // for more than a few ms).
        const STS_COEFFS_MAX_S = 100;
        this.sts_coeffs_max_s = STS_COEFFS_MAX_S;
        this.sts_coeffs = device.createBuffer({
            label: 'plasma.sts_coeffs',
            size:  STS_COEFFS_MAX_S * 4 * F32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // RKL2 per-substep meta — (substep_idx, s_total, dt_super, _pad).
        //
        // Bug history (Session 13): originally a SINGLE 16 B uniform
        // rewritten via `queue.writeBuffer` inside the substep loop.
        // That collided with WebGPU's `writeBuffer`-vs-`submit` ordering:
        // all `writeBuffer` calls are ordered BEFORE the next `submit`,
        // so when the substep loop encodes `s` init+prev dispatches into
        // a single compute pass and then submits once, the queue applies
        // every writeBuffer first (with only the LAST value surviving)
        // and then runs the dispatches — meaning every substep ended up
        // reading `substep_idx = s`. RKL2 with j=s coefficients applied
        // s times is not RKL2; for s≥2 the effective Δt was ~1.25× too
        // large with the wrong L² term, manifesting as the field "jumping
        // suddenly" on Orszag-Tang once η crossed the s=1↔s=2 threshold
        // (~5e-2 at N=256).
        //
        // Fix: pre-allocate STS_COEFFS_MAX_S separate 16 B uniform
        // buffers (`sts_meta_per_j[j-1]`), each written ONCE at startup
        // with the constant (j, 1, 0, 0). The substep loop now selects
        // the appropriate buffer via the bind group rather than mutating
        // a shared buffer. `s_total` is the constant 1 — the shader only
        // uses it as an "is RKL2 active" gate, and CPU already guards
        // dispatch on `s > 0`. `dt_super` was retired in Session 10 (the
        // shader reads fresh from `dt_buf.dt_hyp` now), so we leave it
        // 0 here for layout compatibility.
        this.sts_meta_per_j = new Array(STS_COEFFS_MAX_S);
        const stsMetaSeed = new ArrayBuffer(16);
        const stsMetaSeedU32 = new Uint32Array(stsMetaSeed);
        const stsMetaSeedF32 = new Float32Array(stsMetaSeed);
        for (let j = 1; j <= STS_COEFFS_MAX_S; j++) {
            const buf = device.createBuffer({
                label: `plasma.sts_meta_j${j}`,
                size:  16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            stsMetaSeedU32[0] = j >>> 0;       // substep_idx
            stsMetaSeedU32[1] = 1 >>> 0;       // s_total > 0 — gate the shader on
            stsMetaSeedF32[2] = 0;             // dt_super — unused (Session 10)
            stsMetaSeedF32[3] = 0;             // pad
            device.queue.writeBuffer(buf, 0, stsMetaSeed);
            this.sts_meta_per_j[j - 1] = buf;
        }

        // ── PPM edge states — 4 buffers × 2 axes = 8 buffers ───────
        this.edge_l_x_0 = mkStorage('plasma.edge_l_x_0', u_v4_cell_bytes);
        this.edge_l_x_1 = mkStorage('plasma.edge_l_x_1', u_v4_cell_bytes);
        this.edge_r_x_0 = mkStorage('plasma.edge_r_x_0', u_v4_cell_bytes);
        this.edge_r_x_1 = mkStorage('plasma.edge_r_x_1', u_v4_cell_bytes);
        this.edge_l_y_0 = mkStorage('plasma.edge_l_y_0', u_v4_cell_bytes);
        this.edge_l_y_1 = mkStorage('plasma.edge_l_y_1', u_v4_cell_bytes);
        this.edge_r_y_0 = mkStorage('plasma.edge_r_y_0', u_v4_cell_bytes);
        this.edge_r_y_1 = mkStorage('plasma.edge_r_y_1', u_v4_cell_bytes);

        // ── Per-direction face fluxes ──────────────────────────────
        // Same shape as cell-centered: (N+4)×(N+4). flux_x[i,j] sits at
        // the LEFT face of cell (i,j); flux_y[i,j] at its BOTTOM face.
        this.flux_x_0 = mkStorage('plasma.flux_x_0', u_v4_cell_bytes);
        this.flux_x_1 = mkStorage('plasma.flux_x_1', u_v4_cell_bytes);
        this.flux_y_0 = mkStorage('plasma.flux_y_0', u_v4_cell_bytes);
        this.flux_y_1 = mkStorage('plasma.flux_y_1', u_v4_cell_bytes);

        // ── dt reduction ───────────────────────────────────────────
        this.wavespeed = device.createBuffer({
            label: 'plasma.wavespeed',
            size: 16,
            usage: GPUBufferUsage.STORAGE,
        });
        // dt buffer (32 B = 8 × f32):
        //   [0] dt_hyp        (hyperbolic dt used by RK3 + RKL2 super-step)
        //   [1] dt_parabolic  (forward-Euler resistive bound; diagnostic only)
        //   [2] eta_max       (per-cell anomalous-η maximum; diagnostic)
        //   [3] hall_rate     (max v_A·d_i/dx²; for sub-cycling)
        //   [4] cond_rate     (max 4·χ/dx²; for sub-cycling)
        //   [5..7] reserved   (future per-step diagnostic / sizing slots)
        // Consumers (update-conserved-weighted, update-b-weighted,
        // stats-display) all read slot 0 as before.
        this.dt = device.createBuffer({
            label: 'plasma.dt',
            size: 32,
            // STORAGE: written by compute-dt's atomicMax reduction.
            // UNIFORM: read by update-conserved-weighted (where dropping it
            //   from the storage count is what gets us under the 10-binding
            //   per-stage limit). Other consumers (update-b-weighted) still
            //   bind it as storage; either is fine.
            // COPY_SRC: stats-display reads dt back via ReadbackPool each
            //   sample (12 Hz at 256²). The RKL2 host-side substep count
            //   computation also reads dt_hyp + dt_parabolic from this
            //   buffer (via a separate readback path in sim.js).
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM
                 | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.dt_half = device.createBuffer({
            label: 'plasma.dt_half',
            size: 32,
            // scale-dt writes this as storage; gravity/cooling/geometry read the
            // first 16 B as a DtUniform during Strang source half-steps.
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM
                 | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.source_dt_params = device.createBuffer({
            label: 'plasma.source_dt_params',
            size: 32,
            // GPU source-dt pass reads inverse substep counts from here and
            // divides the fresh dt_half into per-operator dt buffers.
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.source_dt_params, 0,
            new Float32Array([1, 1, 1, 1, 1, 0, 0, 0]).buffer);

        // η_max reduction target — compute-dt's `reduce` writes the
        // per-cell anomalous-η maximum here via the same atomicMax-on-
        // bitcast<u32> pattern as wavespeed. compute-dt's `finalize`
        // reads it back to populate dt_buf[1] / [2].
        this.eta_max_buf = device.createBuffer({
            label: 'plasma.eta_max',
            size: 16,
            usage: GPUBufferUsage.STORAGE,
        });

        // Hall whistler-rate reduction target. Same atomic pattern as
        // eta_max_buf. compute-dt's finalize writes the result into
        // dt_buf[3] for host readback → Hall sub-cycle sizing.
        this.hall_speed_buf = device.createBuffer({
            label: 'plasma.hall_rate_max',
            size: 16,
            usage: GPUBufferUsage.STORAGE,
        });

        // Hall sub-step Δt. source-dt.wgsl writes dt_half / N_hall here
        // after compute-dt, so apply-ohm uses the smaller interval without
        // a CPU readback stall.
        this.hall_dt = device.createBuffer({
            label: 'plasma.hall_dt',
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM
                 | GPUBufferUsage.COPY_DST,
        });

        // Conduction sub-step Δt — mirror of hall_dt for the conduction
        // sub-cycle. source-dt.wgsl writes dt_half / N_cond here.
        this.cond_dt = device.createBuffer({
            label: 'plasma.cond_dt',
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM
                 | GPUBufferUsage.COPY_DST,
        });

        // Explicit viscosity and non-ideal induction sub-step Δt buffers.
        // These mirror hall_dt / cond_dt; the host chooses integer
        // substep counts, while source-dt.wgsl divides the fresh half-step.
        this.visc_dt = device.createBuffer({
            label: 'plasma.visc_dt',
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM
                 | GPUBufferUsage.COPY_DST,
        });
        this.nonideal_dt = device.createBuffer({
            label: 'plasma.nonideal_dt',
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM
                 | GPUBufferUsage.COPY_DST,
        });
        this.rad_dt = device.createBuffer({
            label: 'plasma.rad_dt',
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM
                 | GPUBufferUsage.COPY_DST,
        });

        // Conduction parabolic-rate reduction target. Same atomic
        // pattern as eta_max_buf and hall_speed_buf. compute-dt's
        // finalize writes the result into dt_buf[4] for host readback.
        this.cond_speed_buf = device.createBuffer({
            label: 'plasma.cond_rate_max',
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // ── Uniforms: single physics-state buffer ──────────────────
        // Sweep direction lives in two small SweepDir buffers; LIC
        // render-pace state lives in a separate licUniform buffer. The
        // main Uniforms buffer is rewritten only when a physics-state
        // parameter changes (preset/eta/cfl/gamma/view_mode/resolution).
        this.uniformHost = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
        this.uniformF32  = new Float32Array(this.uniformHost);
        this.uniformU32  = new Uint32Array(this.uniformHost);
        this.uniform = device.createBuffer({
            label: 'plasma.uniform',
            size: UNIFORM_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // ── SweepDir uniforms (16 B each, written once) ────────────
        // reconstruct-ppm and riemann-hlld are the only shaders that
        // read this; they bind one or the other based on sweep axis.
        const SWEEP_BUFFER_SIZE = 16;
        this.sweepDir_x = device.createBuffer({
            label: 'plasma.sweepDir_x',
            size: SWEEP_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.sweepDir_y = device.createBuffer({
            label: 'plasma.sweepDir_y',
            size: SWEEP_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        {
            const sweepBuf = new ArrayBuffer(SWEEP_BUFFER_SIZE);
            const sweepU32 = new Uint32Array(sweepBuf);
            sweepU32[0] = 0; // x
            device.queue.writeBuffer(this.sweepDir_x, 0, sweepBuf);
            sweepU32[0] = 1; // y
            device.queue.writeBuffer(this.sweepDir_y, 0, sweepBuf);
        }

        // ── LicUniforms (16 B, rewritten per render frame) ────────
        // Holds lic_phase / lic_intensity / lic_drift_{x,y}. Bound by
        // lic-advect (compute) and composite (fragment).
        const LIC_UNIFORM_SIZE = 16;
        this.licUniformHost = new ArrayBuffer(LIC_UNIFORM_SIZE);
        this.licUniformF32  = new Float32Array(this.licUniformHost);
        this.licUniform = device.createBuffer({
            label: 'plasma.licUniform',
            size: LIC_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // ── Perturbation uniforms (32 B, rewritten per pointer event) ──
        // Holds the center / drag-vector / sigma / amplitude payload for
        // pointer-driven drag (left) and excite (right) perturbations. See
        // perturb.wgsl for the field layout.
        const PERTURB_UNIFORM_SIZE = 32;
        this.perturbUniformHost = new ArrayBuffer(PERTURB_UNIFORM_SIZE);
        this.perturbUniformF32  = new Float32Array(this.perturbUniformHost);
        this.perturbUniform = device.createBuffer({
            label: 'plasma.perturbUniform',
            size: PERTURB_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // ── BC uniforms ────────────────────────────────────────────
        // Single storage buffer with per-edge mode IDs + per-edge driven states.
        // Layout matches struct BcUniforms in shared-helpers.wgsl:
        //   u32 mode_n, mode_s, mode_e, mode_w
        //   f32 driven_{N,S,E,W}_{rho,vx,vy,vz,bx,by,bz,p}
        this.bcUniformHost = new ArrayBuffer(BC_UNIFORM_BUFFER_SIZE);
        this.bcUniformU32  = new Uint32Array(this.bcUniformHost);
        this.bcUniformF32  = new Float32Array(this.bcUniformHost);
        this.bc_uniforms = device.createBuffer({
            label: 'plasma.bc_uniforms',
            size: BC_UNIFORM_BUFFER_SIZE,
            // Bound as a read-only storage buffer by apply-bcs.wgsl.
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // Initialize: all periodic, neutral driven state on every edge.
        this.pushBC({
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
            driven: { rho: 1, vx: 0, vy: 0, vz: 0, bx: 0, by: 0, bz: 0, p: 1 },
        });

        // ── Stage-params uniform buffers (one per RK3 stage) ───────
        const stageMk = (label) => device.createBuffer({
            label, size: STAGE_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.stage_1 = stageMk('plasma.stage_1');
        this.stage_2 = stageMk('plasma.stage_2');
        this.stage_3 = stageMk('plasma.stage_3');
        const writeStage = (buf, a0, a1, dtw) => {
            const arr = new Float32Array([a0, a1, dtw, 0]);
            device.queue.writeBuffer(buf, 0, arr.buffer);
        };
        writeStage(this.stage_1, 1.0,     0.0,     1.0);
        writeStage(this.stage_2, 3.0/4.0, 1.0/4.0, 1.0/4.0);
        writeStage(this.stage_3, 1.0/3.0, 2.0/3.0, 2.0/3.0);

        // ── LUT ────────────────────────────────────────────────────
        this.lut = device.createBuffer({
            label: 'plasma.lut',
            size: 256 * VEC4_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // ── Tabulated microphysics closures ────────────────────────
        // 144 vec4 rows: cooling, neutral fraction, Spitzer resistivity,
        // transport scale, and grey absorption/scattering opacity families.
        // See src/microphysics.js for the family layout and dimensional use.
        this.microphysics = device.createBuffer({
            label: 'plasma.microphysics',
            size: MICRO_TABLE_ENTRIES * MICRO_STRIDE * F32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.uploadMicrophysicsTable(buildMicrophysicsTable());

        // ── View field (interior cells only, packed contiguous) ────
        // The view-field shader writes only into interior cells. We
        // allocate (N+4)² so existing indexing works; viz reads from the
        // interior strip. Keeping the size identical to cell-centered
        // storage avoids a separate buffer dimension for the renderer.
        this.field = device.createBuffer({
            label: 'plasma.field',
            size: cellsT * F32_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });

        // ── Colored buffer ─────────────────────────────────────────
        this.colored = device.createBuffer({
            label: 'plasma.colored',
            size: cellsT * VEC4_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });

        // ── LIC: noise base + per-cell luminance output ─────────────
        // Noise buffer is resolution-independent — 1024×1024 f32. Sampled
        // by integer indexing with bilinear interpolation in WGSL math.
        // TODO(blue-noise): Phase 6 ships deterministic white noise
        // (mulberry32 PRNG). A future pass should replace this with true
        // blue noise (void-and-cluster or Mitchell's best-candidate) —
        // white noise is grainier than ideal but visually adequate.
        this.noise_n     = LIC_NOISE_N;
        this.noise_size  = LIC_NOISE_N * LIC_NOISE_N * F32_BYTES;
        this.noise = device.createBuffer({
            label: 'plasma.lic_noise',
            size: this.noise_size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._uploadNoise(LIC_NOISE_SEED);

        // LIC output luminance — one f32 per cell, ghost-padded storage
        // for indexing parity with `field` / `colored`. Interior writes only.
        // STORAGE | COPY_DST | COPY_SRC: lic-normalize reads-and-rewrites
        // it after lic-reduce computes the global min/max. COPY_DST gives
        // host-side clears a path; COPY_SRC mirrors the other ghost-padded
        // outputs in case a future debug readback wants the raw luminance.
        this.lic_out = device.createBuffer({
            label: 'plasma.lic_out',
            size: cellsT * F32_BYTES,
            usage: GPUBufferUsage.STORAGE
                 | GPUBufferUsage.COPY_DST
                 | GPUBufferUsage.COPY_SRC,
        });

        // LIC contrast-stretch global reduction target — 2 × u32 atomic
        // (min_bits, max_bits), 8 B. Written by lic-reduce, read by
        // lic-normalize. COPY_SRC is here purely for debug readback;
        // lic-reduce's `reset` entry seeds the values, so no host writes
        // are required at runtime.
        this.lic_minmax = device.createBuffer({
            label: 'plasma.lic_minmax',
            size: 8,
            usage: GPUBufferUsage.STORAGE
                 | GPUBufferUsage.COPY_DST
                 | GPUBufferUsage.COPY_SRC,
        });

        // ── Extended-physics scratch buffers ────────────────────────
        // Conduction and Hall are split into frozen-state compute/apply
        // passes. These scratch buffers carry the source deltas/EMFs across
        // dispatches so no invocation reads a neighbor that another invocation
        // is mutating in the same dispatch.
        this.conduction_dE = mkStorage('plasma.conduction_dE', u_f32_cell_bytes);
        this.hall_E        = mkStorage('plasma.hall_E',        u_v4_edge_bytes);
        this.hall_mb0      = mkStorage('plasma.hall_mb0',      u_f32_cell_bytes);
        this.nonideal_E    = mkStorage('plasma.nonideal_E',    u_v4_edge_bytes);
        this.viscosity_dU  = mkStorage('plasma.viscosity_dU',  u_v4_cell_bytes);
        this.radiation_E   = mkStorage('plasma.radiation_E',   u_f32_cell_bytes);
        this.radiation_dE  = mkStorage('plasma.radiation_dE',  u_v4_cell_bytes);

        // ── Self-gravity Poisson buffers (extended physics) ────────
        // phi / phi_next form a Jacobi ping-pong for ∇²φ = 4πGρ.
        // Same ghost-padded shape as cell-centered storage. Cleared at
        // init and stays effectively zero until gravity_G > 0.
        this.phi      = mkStorage('plasma.phi',      u_f32_cell_bytes);
        this.phi_next = mkStorage('plasma.phi_next', u_f32_cell_bytes);
        // Distinct writable dummies keep unused read_write bindings from
        // aliasing in multigrid bind groups (WebGPU validation forbids it).
        this.poisson_mg_dummy_a = mkStorage('plasma.poisson_mg_dummy_a', 16);
        this.poisson_mg_dummy_b = mkStorage('plasma.poisson_mg_dummy_b', 16);
        this.poisson_mg_dummy_c = mkStorage('plasma.poisson_mg_dummy_c', 16);
        this.poisson_mg_dummy_ro = mkStorage('plasma.poisson_mg_dummy_ro', 16);
        this.poisson_mg_dummy = this.poisson_mg_dummy_a;
        this.poissonMgLevels = [];
        // rho_mean — single f32 holding ρ̄ for the periodic Poisson
        // compatibility condition. rho_mean_partials holds one tile sum per
        // 8×8 workgroup before a tiny finalize pass accumulates the mean.
        this.rho_mean = device.createBuffer({
            label: 'plasma.rho_mean',
            size:  16,  // 1 f32 padded to 16 B for alignment
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.poisson_tiles_per_axis = Math.ceil(n / 8);
        this.rho_mean_partials = mkStorage(
            'plasma.rho_mean_partials',
            this.poisson_tiles_per_axis * this.poisson_tiles_per_axis * F32_BYTES,
        );

        // ── Conservation / stats diagnostics ───────────────────────
        // Two-pass reduction: per-tile partials → final 24-scalar packet.
        // Sized for the per-axis tile count at the current resolution
        // (ceil(N/WG)² × 24 × 4 B). At 256² this is ~96 KB; at
        // 1024² it's ~1.5 MB. This replaces multi-MB per-cadence field
        // readbacks with a tiny scalar readback.
        const CONS_WG = 8;
        this.cons_tiles_per_axis = Math.ceil(n / CONS_WG);
        const consTileCount = this.cons_tiles_per_axis * this.cons_tiles_per_axis;
        this.cons_tile_partials = device.createBuffer({
            label: 'plasma.cons_tile_partials',
            size: consTileCount * 24 * F32_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });
        // Final output: 21 live slots + 3 pad/reserved slots. STORAGE |
        // COPY_SRC so stats-display can pull it via ReadbackPool.
        this.cons_out = device.createBuffer({
            label: 'plasma.cons_out',
            size: 24 * F32_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // LIC animation + intensity state (host-side; sim.js pushes via uniforms).
        this._licPhase     = 0;
        this._licIntensity = LIC_INTENSITY_DEFAULT;
        this._licDriftX    = LIC_DRIFT_X;
        this._licDriftY    = LIC_DRIFT_Y;

        // Default eta — caller can override via pushUniforms.
        this._eta = ETA_DEFAULT;
        this._viewMode = VIEW_DENSITY;
        // Default CFL / pressure floor — pushUniforms reads these as
        // fallbacks if a particular slider hasn't fired yet.
        this._cfl = undefined;
        this._pressureFloor = PRESSURE_FLOOR;
        // Anomalous resistivity defaults — α = 0 means constant η_0
        // (RKL2 still runs but reduces to forward-Euler stability).
        this._etaAnomAlpha = 0;
        this._etaAnomJcrit = 10.0;
    }

    /**
     * Generate a deterministic white-noise field via mulberry32 PRNG and
     * upload to `this.noise`. Values are uniform in [0, 1).
     *
     * Called once at construction; reseeding is not exposed — the noise
     * is meant to be a fixed fingerprint that the LIC trace samples
     * different parts of as it advects.
     */
    _uploadNoise(seed) {
        const N    = this.noise_n;
        const arr  = new Float32Array(N * N);
        // mulberry32: 32-bit state, 7-line PRNG. Deterministic at any
        // seed; good enough mixing for visual noise.
        let s = (seed >>> 0) || 1;
        for (let i = 0; i < arr.length; i++) {
            s = (s + 0x6D2B79F5) >>> 0;
            let t = s;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            const v = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
            arr[i] = v;
        }
        this.device.queue.writeBuffer(this.noise, 0, arr.buffer);
    }

    /**
     * Flip the (slot_n, slot_next) handles after each step.
     */
    swap() {
        if (this._side === 'a') {
            this.U0_n     = this.U0_b; this.U1_n     = this.U1_b;
            this.U0_next  = this.U0_a; this.U1_next  = this.U1_a;
            this.Bx_n     = this.Bx_b; this.By_n     = this.By_b;
            this.Bx_next  = this.Bx_a; this.By_next  = this.By_a;
            this._side = 'b';
        } else {
            this.U0_n     = this.U0_a; this.U1_n     = this.U1_a;
            this.U0_next  = this.U0_b; this.U1_next  = this.U1_b;
            this.Bx_n     = this.Bx_a; this.By_n     = this.By_a;
            this.Bx_next  = this.Bx_b; this.By_next  = this.By_b;
            this._side = 'a';
        }
    }

    /**
     * Upload an MHD initial condition into the active slot_n. Presets
     * supply arrays sized for the GHOST-PADDED storage layout:
     *   U0:      Float32Array(4 · (N+4)²)
     *   U1:      Float32Array(4 · (N+4)²)
     *   Bx_face: Float32Array((N+5) · (N+4))
     *   By_face: Float32Array((N+4) · (N+5))
     * Interior data lives at indices [ghost, ghost+N) per axis; the
     * preset is responsible for either filling ghost strips or leaving
     * them zero (apply-bcs.wgsl re-fills them every stage anyway).
     */
    uploadInitialState({ U0, U1, Bx_face, By_face, radiation }) {
        const N      = this.n_total;
        const cellsT = N * N;
        const xfaces = (N + 1) * N;
        const yfaces = N * (N + 1);
        if (U0.length !== 4 * cellsT)
            throw new Error(`U0 length mismatch: got ${U0.length}, expected ${4 * cellsT}`);
        if (U1.length !== 4 * cellsT)
            throw new Error(`U1 length mismatch: got ${U1.length}, expected ${4 * cellsT}`);
        if (Bx_face.length !== xfaces)
            throw new Error(`Bx_face length mismatch: got ${Bx_face.length}, expected ${xfaces}`);
        if (By_face.length !== yfaces)
            throw new Error(`By_face length mismatch: got ${By_face.length}, expected ${yfaces}`);
        if (radiation !== undefined && radiation.length !== cellsT) {
            throw new Error(`radiation length mismatch: got ${radiation.length}, expected ${cellsT}`);
        }

        // Reset to side A.
        this._side = 'a';
        this.U0_n     = this.U0_a; this.U1_n     = this.U1_a;
        this.U0_next  = this.U0_b; this.U1_next  = this.U1_b;
        this.Bx_n     = this.Bx_a; this.By_n     = this.By_a;
        this.Bx_next  = this.Bx_b; this.By_next  = this.By_b;

        const q = this.device.queue;
        q.writeBuffer(this.U0_a, 0, U0.buffer, U0.byteOffset, U0.byteLength);
        q.writeBuffer(this.U0_b, 0, U0.buffer, U0.byteOffset, U0.byteLength);
        q.writeBuffer(this.U1_a, 0, U1.buffer, U1.byteOffset, U1.byteLength);
        q.writeBuffer(this.U1_b, 0, U1.buffer, U1.byteOffset, U1.byteLength);
        q.writeBuffer(this.Bx_a, 0, Bx_face.buffer, Bx_face.byteOffset, Bx_face.byteLength);
        q.writeBuffer(this.Bx_b, 0, Bx_face.buffer, Bx_face.byteOffset, Bx_face.byteLength);
        q.writeBuffer(this.By_a, 0, By_face.buffer, By_face.byteOffset, By_face.byteLength);
        q.writeBuffer(this.By_b, 0, By_face.buffer, By_face.byteOffset, By_face.byteLength);
        this.clearExtendedScratch();
        if (radiation !== undefined) {
            q.writeBuffer(this.radiation_E, 0,
                radiation.buffer, radiation.byteOffset, radiation.byteLength);
        } else {
            q.writeBuffer(this.radiation_E, 0, new Float32Array(cellsT).buffer);
        }
    }

    /**
     * Clear source-term scratch and Poisson state after preset/resolution loads.
     * This prevents a self-gravity solve from warm-starting against a previous
     * preset's potential or reusing stale Hall/conduction temporaries.
     */
    clearExtendedScratch() {
        const encoder = this.device.createCommandEncoder({ label: 'plasma.clearExtendedScratch' });
        encoder.clearBuffer(this.conduction_dE);
        encoder.clearBuffer(this.hall_E);
        encoder.clearBuffer(this.hall_mb0);
        encoder.clearBuffer(this.nonideal_E);
        encoder.clearBuffer(this.viscosity_dU);
        encoder.clearBuffer(this.radiation_dE);
        encoder.clearBuffer(this.phi);
        encoder.clearBuffer(this.phi_next);
        encoder.clearBuffer(this.poisson_mg_dummy_a);
        encoder.clearBuffer(this.poisson_mg_dummy_b);
        encoder.clearBuffer(this.poisson_mg_dummy_c);
        encoder.clearBuffer(this.poisson_mg_dummy_ro);
        for (const level of this.poissonMgLevels) {
            encoder.clearBuffer(level.phiA);
            encoder.clearBuffer(level.phiB);
            encoder.clearBuffer(level.rhs);
        }
        encoder.clearBuffer(this.rho_mean);
        encoder.clearBuffer(this.rho_mean_partials);
        // Session 15: sub-cycle dt buffers + their reduction targets.
        encoder.clearBuffer(this.hall_dt);
        encoder.clearBuffer(this.cond_dt);
        encoder.clearBuffer(this.visc_dt);
        encoder.clearBuffer(this.nonideal_dt);
        encoder.clearBuffer(this.rad_dt);
        encoder.clearBuffer(this.dt_half);
        encoder.clearBuffer(this.cond_speed_buf);
        this.device.queue.submit([encoder.finish()]);
    }

    ensurePoissonMultigridLevels() {
        if (this.poissonMgLevels.length > 0) return this.poissonMgLevels;

        for (let levelN = this.n, stride = 1; levelN >= 4; levelN = Math.floor(levelN / 2), stride *= 2) {
            const cells = levelN * levelN;
            const uniformHost = new ArrayBuffer(16);
            const uniformU32 = new Uint32Array(uniformHost);
            uniformU32[0] = levelN >>> 0;
            uniformU32[1] = stride >>> 0;
            uniformU32[2] = 0;
            uniformU32[3] = 0;
            const uniform = this.device.createBuffer({
                label: `plasma.poisson_mg_level_${levelN}.uniform`,
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.device.queue.writeBuffer(uniform, 0, uniformHost);
            this.poissonMgLevels.push({
                n: levelN,
                stride,
                uniform,
                phiA: this._mkStorage(`plasma.poisson_mg_${levelN}.phiA`, cells * F32_BYTES),
                phiB: this._mkStorage(`plasma.poisson_mg_${levelN}.phiB`, cells * F32_BYTES),
                rhs:  this._mkStorage(`plasma.poisson_mg_${levelN}.rhs`,  cells * F32_BYTES),
            });
        }

        return this.poissonMgLevels;
    }

    /**
     * Upload viridis (or any 256×RGBA8 LUT) as 256 × vec4<f32> normalized
     * to [0,1].  Uint8Array length must be 256 * 4 = 1024.
     */
    uploadLUT(uint8) {
        if (uint8.length !== 256 * 4) {
            throw new Error(`uploadLUT: expected 1024 bytes, got ${uint8.length}`);
        }
        const f32 = new Float32Array(256 * 4);
        for (let i = 0; i < uint8.length; i++) f32[i] = uint8[i] / 255;
        this.device.queue.writeBuffer(this.lut, 0, f32.buffer);
    }

    uploadMicrophysicsTable(table) {
        const expected = MICRO_TABLE_ENTRIES * MICRO_STRIDE;
        if (!(table instanceof Float32Array) || table.length !== expected) {
            throw new Error(`uploadMicrophysicsTable: expected Float32Array(${expected})`);
        }
        this.device.queue.writeBuffer(this.microphysics, 0,
            table.buffer, table.byteOffset, table.byteLength);
    }

    /**
     * Push the host physics-state uniform struct to GPU. LIC render-pace
     * fields (phase, intensity, drift) are owned by `pushLicUniforms`
     * — this method does not touch them. Sweep direction is owned by
     * the static `sweepDir_x` / `sweepDir_y` buffers.
     *
     * Layout (matches `struct Uniforms` in shared-helpers.wgsl):
     *   slot 0..7   f32: dx, gamma, view_min, view_max, eta,
     *                    _pad_lic_0, _pad_lic_1, _pad_lic_2
     *   slot 8..10  u32: grid_n, grid_n_total, ghost_w
     *   slot 11     f32: pressure_floor (UI slider, was _pad_sweep)
     *   slot 12     f32: cfl
     *   slot 13     u32: view_mode
     *   slot 14     f32: eta_anom_jcrit
     *   slot 15     u32: noise_n
     *   slot 30     u32: cooling_curve_mode
     *   slot 31     f32: hall_electron_pressure_frac
     *   slot 32..63 Session-17 source physics and reserved headroom
     *
     * Any field not passed in the args object falls back to the host-side
     * cache (so callers that only know about a subset don't clobber
     * other state).
     */
    pushUniforms({
        dx, gamma, viewMin, viewMax, gridN, viewMode, eta, cfl, pressureFloor,
        etaAnomAlpha, etaAnomJcrit,
        // Extended physics (any subset; falls back to host cache):
        hallDi, hallSubstepsMax,
        coolingLambda0, coolingTFloor, coolingTRef,
        conductionKappa, conductionIsoFrac, conductionSatFrac,
        gravityGx, gravityGy, gravityG, gravityPoissonIters,
        coolingCurveMode, hallElectronPressureFrac,
        coolingMetallicity, heatingGamma0, heatingDensityExp, heatingTCut,
        ambipolarEta, biermannCoeff, neutralFrac, ionizationT0,
        viscosityNu, viscosityBulk, viscosityAnisoFrac, viscosityShock,
        sourceSubstepsMax, geometryMode, geometryRMin,
        gravitySoftening, gravityPoissonOmega,
        spongeWidth, spongeStrength, coolingTableMix,
        radiationC, radiationKappaAbs, radiationKappaScat, radiationConst, radiationFloor,
        electronInertiaLength, electronInertiaDamping,
        gravityBoundaryMode,
        physicsFlags, emfMode,
    } = {}) {
        if (eta           !== undefined) this._eta           = eta;
        if (cfl           !== undefined) this._cfl           = cfl;
        if (pressureFloor !== undefined) this._pressureFloor = pressureFloor;
        if (etaAnomAlpha  !== undefined) this._etaAnomAlpha  = etaAnomAlpha;
        if (etaAnomJcrit  !== undefined) this._etaAnomJcrit  = etaAnomJcrit;
        // Cache extended physics scalars (defaults: all OFF / 0).
        if (hallDi              !== undefined) this._hallDi              = hallDi;
        if (hallSubstepsMax     !== undefined) this._hallSubstepsMax     = hallSubstepsMax;
        if (coolingLambda0      !== undefined) this._coolingLambda0      = coolingLambda0;
        if (coolingTFloor       !== undefined) this._coolingTFloor       = coolingTFloor;
        if (coolingTRef         !== undefined) this._coolingTRef         = coolingTRef;
        if (conductionKappa     !== undefined) this._conductionKappa     = conductionKappa;
        if (conductionIsoFrac   !== undefined) this._conductionIsoFrac   = conductionIsoFrac;
        if (conductionSatFrac   !== undefined) this._conductionSatFrac   = conductionSatFrac;
        if (gravityGx           !== undefined) this._gravityGx           = gravityGx;
        if (gravityGy           !== undefined) this._gravityGy           = gravityGy;
        if (gravityG            !== undefined) this._gravityG            = gravityG;
        if (gravityPoissonIters !== undefined) this._gravityPoissonIters = gravityPoissonIters;
        if (coolingCurveMode    !== undefined) this._coolingCurveMode    = coolingCurveMode;
        if (hallElectronPressureFrac !== undefined) this._hallElectronPressureFrac = hallElectronPressureFrac;
        if (coolingMetallicity  !== undefined) this._coolingMetallicity  = coolingMetallicity;
        if (heatingGamma0       !== undefined) this._heatingGamma0       = heatingGamma0;
        if (heatingDensityExp   !== undefined) this._heatingDensityExp   = heatingDensityExp;
        if (heatingTCut         !== undefined) this._heatingTCut         = heatingTCut;
        if (ambipolarEta        !== undefined) this._ambipolarEta        = ambipolarEta;
        if (biermannCoeff       !== undefined) this._biermannCoeff       = biermannCoeff;
        if (neutralFrac         !== undefined) this._neutralFrac         = neutralFrac;
        if (ionizationT0        !== undefined) this._ionizationT0        = ionizationT0;
        if (viscosityNu         !== undefined) this._viscosityNu         = viscosityNu;
        if (viscosityBulk       !== undefined) this._viscosityBulk       = viscosityBulk;
        if (viscosityAnisoFrac  !== undefined) this._viscosityAnisoFrac  = viscosityAnisoFrac;
        if (viscosityShock      !== undefined) this._viscosityShock      = viscosityShock;
        if (sourceSubstepsMax   !== undefined) this._sourceSubstepsMax   = sourceSubstepsMax;
        if (geometryMode        !== undefined) this._geometryMode        = geometryMode;
        if (geometryRMin        !== undefined) this._geometryRMin        = geometryRMin;
        if (gravitySoftening    !== undefined) this._gravitySoftening    = gravitySoftening;
        if (gravityPoissonOmega !== undefined) this._gravityPoissonOmega = gravityPoissonOmega;
        if (spongeWidth         !== undefined) this._spongeWidth         = spongeWidth;
        if (spongeStrength      !== undefined) this._spongeStrength      = spongeStrength;
        if (coolingTableMix     !== undefined) this._coolingTableMix     = coolingTableMix;
        if (radiationC          !== undefined) this._radiationC          = radiationC;
        if (radiationKappaAbs   !== undefined) this._radiationKappaAbs   = radiationKappaAbs;
        if (radiationKappaScat  !== undefined) this._radiationKappaScat  = radiationKappaScat;
        if (radiationConst      !== undefined) this._radiationConst      = radiationConst;
        if (radiationFloor      !== undefined) this._radiationFloor      = radiationFloor;
        if (electronInertiaLength !== undefined) this._electronInertiaLength = electronInertiaLength;
        if (electronInertiaDamping !== undefined) this._electronInertiaDamping = electronInertiaDamping;
        if (gravityBoundaryMode !== undefined) this._gravityBoundaryMode = gravityBoundaryMode;
        if (physicsFlags        !== undefined) this._physicsFlags        = physicsFlags;
        if (emfMode             !== undefined) this._emfMode             = emfMode;
        // ── Slots 0-15: original layout ────────────────────────────
        this.uniformF32[0] = dx;
        this.uniformF32[1] = gamma;
        this.uniformF32[2] = viewMin;
        this.uniformF32[3] = viewMax;
        this.uniformF32[4] = this._eta;
        this.uniformF32[5] = (this._etaAnomAlpha ?? 0);
        this.uniformF32[6] = 0; // _pad_lic_1
        this.uniformF32[7] = 0; // _pad_lic_2
        this.uniformU32[8]  = gridN >>> 0;
        this.uniformU32[9]  = (gridN + 2 * this.ghost) >>> 0;
        this.uniformU32[10] = this.ghost >>> 0;
        this.uniformF32[11] = (this._pressureFloor ?? PRESSURE_FLOOR);
        this.uniformF32[12] = (this._cfl ?? 0.4);
        this.uniformU32[13] = (viewMode ?? this._viewMode) >>> 0;
        if (viewMode !== undefined) this._viewMode = viewMode;
        this.uniformF32[14] = (this._etaAnomJcrit ?? 10.0);
        this.uniformU32[15] = this.noise_n >>> 0;
        // ── Slots 16-31: extended physics ──────────────────────────
        this.uniformF32[16] = (this._hallDi              ?? 0);
        this.uniformU32[17] = (this._hallSubstepsMax     ?? 8) >>> 0;
        this.uniformF32[18] = (this._coolingLambda0      ?? 0);
        this.uniformF32[19] = (this._coolingTFloor       ?? 1e-4);
        this.uniformF32[20] = (this._coolingTRef         ?? 1.0);
        this.uniformF32[21] = (this._conductionKappa     ?? 0);
        this.uniformF32[22] = (this._conductionIsoFrac   ?? 0);
        this.uniformF32[23] = (this._conductionSatFrac   ?? 0);
        this.uniformF32[24] = (this._gravityGx           ?? 0);
        this.uniformF32[25] = (this._gravityGy           ?? 0);
        this.uniformF32[26] = (this._gravityG            ?? 0);
        this.uniformU32[27] = (this._gravityPoissonIters ?? 30) >>> 0;
        this.uniformU32[28] = (this._physicsFlags        ?? 0) >>> 0;
        this.uniformU32[29] = (this._emfMode             ?? 0) >>> 0;
        this.uniformU32[30] = (this._coolingCurveMode    ?? 1) >>> 0;
        this.uniformF32[31] = (this._hallElectronPressureFrac ?? 0);
        // ── Slots 32-63: higher-fidelity source physics ────────────
        this.uniformF32[32] = (this._coolingMetallicity  ?? 1.0);
        this.uniformF32[33] = (this._heatingGamma0       ?? 0);
        this.uniformF32[34] = (this._heatingDensityExp   ?? 1.0);
        this.uniformF32[35] = (this._heatingTCut         ?? 0);
        this.uniformF32[36] = (this._ambipolarEta        ?? 0);
        this.uniformF32[37] = (this._biermannCoeff       ?? 0);
        this.uniformF32[38] = (this._neutralFrac         ?? 0);
        this.uniformF32[39] = (this._ionizationT0        ?? 1.0);
        this.uniformF32[40] = (this._viscosityNu         ?? 0);
        this.uniformF32[41] = (this._viscosityBulk       ?? 0);
        this.uniformF32[42] = (this._viscosityAnisoFrac  ?? 0);
        this.uniformF32[43] = (this._viscosityShock      ?? 0);
        this.uniformU32[44] = (this._sourceSubstepsMax   ?? 8) >>> 0;
        this.uniformU32[45] = (this._geometryMode        ?? 0) >>> 0;
        this.uniformF32[46] = (this._geometryRMin        ?? 0);
        this.uniformF32[47] = (this._gravitySoftening    ?? 0);
        this.uniformF32[48] = (this._gravityPoissonOmega ?? 1.0);
        this.uniformF32[49] = (this._spongeWidth         ?? 0);
        this.uniformF32[50] = (this._spongeStrength      ?? 0);
        this.uniformF32[51] = (this._coolingTableMix     ?? 0);
        this.uniformF32[52] = (this._radiationC          ?? 0);
        this.uniformF32[53] = (this._radiationKappaAbs   ?? 0);
        this.uniformF32[54] = (this._radiationKappaScat  ?? 0);
        this.uniformF32[55] = (this._radiationConst      ?? 1.0);
        this.uniformF32[56] = (this._radiationFloor      ?? 1.0e-12);
        this.uniformF32[57] = (this._electronInertiaLength ?? 0);
        this.uniformF32[58] = (this._electronInertiaDamping ?? 0);
        this.uniformU32[59] = (this._gravityBoundaryMode ?? 0) >>> 0;
        for (let i = 60; i < 64; i++) this.uniformF32[i] = 0;
        this.device.queue.writeBuffer(this.uniform, 0, this.uniformHost);
    }

    /**
     * Upload a packed RKL2 coefficient array. `arr` must be a Float32Array
     * of length 4 * s where s = sts_coeffs_max_s upper bound. Layout per
     * substep j: (μ_j, ν_j, μ̃_j, γ̃_j).
     */
    pushStsCoeffs(arr) {
        this.device.queue.writeBuffer(this.sts_coeffs, 0,
            arr.buffer, arr.byteOffset, arr.byteLength);
    }

    /**
     * Push the host LIC render-pace uniform struct to GPU. Called every
     * render frame (sim.render); writes only 16 bytes. Fields not passed
     * fall back to the host-side cache.
     */
    pushLicUniforms({ licPhase, licIntensity, licDriftX, licDriftY } = {}) {
        if (licPhase     !== undefined) this._licPhase     = licPhase;
        if (licIntensity !== undefined) this._licIntensity = licIntensity;
        if (licDriftX    !== undefined) this._licDriftX    = licDriftX;
        if (licDriftY    !== undefined) this._licDriftY    = licDriftY;
        this.licUniformF32[0] = this._licPhase;
        this.licUniformF32[1] = this._licIntensity;
        this.licUniformF32[2] = this._licDriftX;
        this.licUniformF32[3] = this._licDriftY;
        this.device.queue.writeBuffer(this.licUniform, 0, this.licUniformHost);
    }

    /**
     * Push the perturbation-uniform payload for the next applyPerturbation
     * dispatch. Fields are all f32 (cx, cy, dvec_x, dvec_y, sigma,
     * amplitude) plus two reserved u32 slots — see perturb.wgsl.
     */
    pushPerturbUniforms({ cx, cy, dvec_x, dvec_y, sigma, amplitude }) {
        this.perturbUniformF32[0] = cx;
        this.perturbUniformF32[1] = cy;
        this.perturbUniformF32[2] = dvec_x;
        this.perturbUniformF32[3] = dvec_y;
        this.perturbUniformF32[4] = sigma;
        this.perturbUniformF32[5] = amplitude;
        // Slots [6], [7] reserved (u32 _pad0/_pad1) — left at zero.
        this.device.queue.writeBuffer(this.perturbUniform, 0, this.perturbUniformHost);
    }

    /** Explicitly release GPU buffers owned by this allocation set. */
    destroy() {
        const seen = new Set();
        const destroyOne = (buf) => {
            if (!buf || typeof buf.destroy !== 'function' || typeof buf.size !== 'number') return;
            if (seen.has(buf)) return;
            seen.add(buf);
            try { buf.destroy(); } catch (e) { /* ignore best-effort teardown */ }
        };

        for (const level of this.poissonMgLevels || []) {
            destroyOne(level.uniform);
            destroyOne(level.phiA);
            destroyOne(level.phiB);
            destroyOne(level.rhs);
        }
        for (const value of Object.values(this)) {
            if (Array.isArray(value)) {
                for (const item of value) destroyOne(item);
            } else {
                destroyOne(value);
            }
        }
        this.poissonMgLevels = [];
    }

    /**
     * Update the BC mode + driven-state storage buffer.
     * @param {{modeN:number, modeS:number, modeE:number, modeW:number,
     *          driven?:{rho:number, vx:number, vy:number, vz:number,
     *                   bx:number, by:number, bz:number, p:number},
     *          drivenN?:object, drivenS?:object, drivenE?:object, drivenW?:object}} cfg
     */
    pushBC(cfg) {
        this.bcUniformU32[0] = cfg.modeN >>> 0;
        this.bcUniformU32[1] = cfg.modeS >>> 0;
        this.bcUniformU32[2] = cfg.modeE >>> 0;
        this.bcUniformU32[3] = cfg.modeW >>> 0;
        const fallback = { rho: 1, vx: 0, vy: 0, vz: 0, bx: 0, by: 0, bz: 0, p: 1 };
        const base = { ...fallback, ...(cfg.driven || {}) };
        const edgeState = (key) => ({ ...base, ...(cfg[key] || {}) });
        const writeDriven = (offset, d) => {
            this.bcUniformF32[offset + 0] = d.rho;
            this.bcUniformF32[offset + 1] = d.vx;
            this.bcUniformF32[offset + 2] = d.vy;
            this.bcUniformF32[offset + 3] = d.vz;
            this.bcUniformF32[offset + 4] = d.bx;
            this.bcUniformF32[offset + 5] = d.by;
            this.bcUniformF32[offset + 6] = d.bz;
            this.bcUniformF32[offset + 7] = d.p;
        };
        writeDriven(4,  edgeState('drivenN'));
        writeDriven(12, edgeState('drivenS'));
        writeDriven(20, edgeState('drivenE'));
        writeDriven(28, edgeState('drivenW'));
        this.device.queue.writeBuffer(this.bc_uniforms, 0, this.bcUniformHost);
    }
}

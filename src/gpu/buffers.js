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
    VIEW_DENSITY, BC_PERIODIC, ETA_DEFAULT,
    LIC_NOISE_N, LIC_INTENSITY_DEFAULT, LIC_DRIFT_X, LIC_DRIFT_Y, LIC_NOISE_SEED,
} from '../config.js';

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

        const mkStorage = (label, size, extra = 0) => device.createBuffer({
            label, size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra,
        });

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
        this.dt = device.createBuffer({
            label: 'plasma.dt',
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // ── Uniforms: two sweep-direction buffers ──────────────────
        this.uniformHost = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
        this.uniformF32  = new Float32Array(this.uniformHost);
        this.uniformU32  = new Uint32Array(this.uniformHost);
        this.uniform_x = device.createBuffer({
            label: 'plasma.uniform_x',
            size: UNIFORM_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.uniform_y = device.createBuffer({
            label: 'plasma.uniform_y',
            size: UNIFORM_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Legacy alias.
        this.uniform = this.uniform_x;

        // ── BC uniforms ────────────────────────────────────────────
        // Single storage buffer with per-edge mode IDs + driven state.
        // Layout matches struct BcUniforms in shared-helpers.wgsl:
        //   u32 mode_n, mode_s, mode_e, mode_w
        //   f32 driven_rho, driven_vx, driven_vy, driven_vz,
        //       driven_bx, driven_by, driven_bz, driven_p
        this.bcUniformHost = new ArrayBuffer(BC_UNIFORM_BUFFER_SIZE);
        this.bcUniformU32  = new Uint32Array(this.bcUniformHost);
        this.bcUniformF32  = new Float32Array(this.bcUniformHost);
        this.bc_uniforms = device.createBuffer({
            label: 'plasma.bc_uniforms',
            size: BC_UNIFORM_BUFFER_SIZE,
            // Bound as a read-only storage buffer by apply-bcs.wgsl.
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // Initialize: all periodic, neutral driven state.
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
        this.lic_out = device.createBuffer({
            label: 'plasma.lic_out',
            size: cellsT * F32_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });

        // LIC animation + intensity state (host-side; sim.js pushes via uniforms).
        this._licPhase     = 0;
        this._licIntensity = LIC_INTENSITY_DEFAULT;
        this._licDriftX    = LIC_DRIFT_X;
        this._licDriftY    = LIC_DRIFT_Y;

        // Default eta — caller can override via pushUniforms.
        this._eta = ETA_DEFAULT;
        this._viewMode = VIEW_DENSITY;
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
    uploadInitialState({ U0, U1, Bx_face, By_face }) {
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

    /**
     * Push the host uniform struct to GPU — writes BOTH sweep-direction
     * uniform buffers. Layout (matches `struct Uniforms` in
     * shared-helpers.wgsl):
     *   slot 0..7  f32: dx, gamma, view_min, view_max, eta,
     *                    lic_phase, lic_intensity, lic_drift_x
     *   slot 8..13 u32: grid_n, grid_n_total, ghost_w, sweep_dir,
     *                    step_parity, view_mode
     *   slot 14    f32: lic_drift_y
     *   slot 15    u32: noise_n
     *
     * Any field not passed in the args object falls back to the host-side
     * cache (so callers that only know about a subset don't clobber LIC
     * state).
     */
    pushUniforms({
        dx, gamma, viewMin, viewMax, gridN, stepParity, viewMode, eta,
        licPhase, licIntensity, licDriftX, licDriftY,
    } = {}) {
        if (eta          !== undefined) this._eta          = eta;
        if (licPhase     !== undefined) this._licPhase     = licPhase;
        if (licIntensity !== undefined) this._licIntensity = licIntensity;
        if (licDriftX    !== undefined) this._licDriftX    = licDriftX;
        if (licDriftY    !== undefined) this._licDriftY    = licDriftY;
        this.uniformF32[0] = dx;
        this.uniformF32[1] = gamma;
        this.uniformF32[2] = viewMin;
        this.uniformF32[3] = viewMax;
        this.uniformF32[4] = this._eta;
        this.uniformF32[5] = this._licPhase;
        this.uniformF32[6] = this._licIntensity;
        this.uniformF32[7] = this._licDriftX;
        this.uniformU32[8]  = gridN >>> 0;
        this.uniformU32[9]  = (gridN + 2 * this.ghost) >>> 0;
        this.uniformU32[10] = this.ghost >>> 0;
        this.uniformU32[11] = 0; // sweep_dir = 0 for x
        this.uniformU32[12] = stepParity >>> 0;
        this.uniformU32[13] = (viewMode ?? this._viewMode) >>> 0;
        if (viewMode !== undefined) this._viewMode = viewMode;
        this.uniformF32[14] = this._licDriftY;
        this.uniformU32[15] = this.noise_n >>> 0;
        this.device.queue.writeBuffer(this.uniform_x, 0, this.uniformHost);

        // y-sweep variant.
        this.uniformU32[11] = 1;
        this.device.queue.writeBuffer(this.uniform_y, 0, this.uniformHost);
    }

    /**
     * Update the BC mode + driven-state storage buffer.
     * @param {{modeN:number, modeS:number, modeE:number, modeW:number,
     *          driven:{rho:number, vx:number, vy:number, vz:number,
     *                  bx:number, by:number, bz:number, p:number}}} cfg
     */
    pushBC(cfg) {
        this.bcUniformU32[0] = cfg.modeN >>> 0;
        this.bcUniformU32[1] = cfg.modeS >>> 0;
        this.bcUniformU32[2] = cfg.modeE >>> 0;
        this.bcUniformU32[3] = cfg.modeW >>> 0;
        const d = cfg.driven || { rho: 1, vx: 0, vy: 0, vz: 0, bx: 0, by: 0, bz: 0, p: 1 };
        this.bcUniformF32[4]  = d.rho;
        this.bcUniformF32[5]  = d.vx;
        this.bcUniformF32[6]  = d.vy;
        this.bcUniformF32[7]  = d.vz;
        this.bcUniformF32[8]  = d.bx;
        this.bcUniformF32[9]  = d.by;
        this.bcUniformF32[10] = d.bz;
        this.bcUniformF32[11] = d.p;
        this.device.queue.writeBuffer(this.bc_uniforms, 0, this.bcUniformHost);
    }
}

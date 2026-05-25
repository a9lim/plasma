/**
 * @fileoverview GPU buffer allocator for Phase 3b (RK3 SSP + HLLD + PPM).
 *
 * Four RK3 storage slots for cell-centered conserved state + face-
 * centered transverse B. We ping-pong (n, n+1) and reuse (1, 2) as
 * stage scratch, because stage 3 needs to read U(n) while writing
 * U(n+1) — WebGPU forbids binding the same buffer as RO and RW in
 * one bind group, so the destination must be a distinct slot.
 *
 * Logical handles (these are renames, not separate allocations):
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
 * Face-centered Bx_face, By_face on their own f32 buffers per slot.
 * Edge-centered Ez lives on one f32 buffer (recomputed per stage).
 *
 * PPM produces *both* left and right edge primitive states per cell per
 * direction. Four vec4 outputs per axis:
 *   edge_l_x_0 = (ρ, vx, vy, vz)_L  edge_l_x_1 = (p, By, Bz, _)_L
 *   edge_r_x_0 = (ρ, vx, vy, vz)_R  edge_r_x_1 = (p, By, Bz, _)_R
 * Same for y with transverse-B mapping (Bx, Bz).
 *
 * Per-direction face fluxes match Phase 3a layout. The HLLD solver reads
 * `edge_r_*[i]` for QL and `edge_l_*[i+1]` for QR.
 *
 * Stage-params uniform buffers carry the SSP RK3 linear-combination
 * weights (a0, a1, dt_w, _pad) for each of the three stages. Written
 * once at init via writeBuffer; never changed.
 *
 * Sweep-direction uniform buffers carry the existing `Uniforms` struct
 * with sweep_dir = 0 or 1. Two buffers, both written via writeBuffer
 * whenever dx/gamma/view_* change. This lets one encoder cover the whole
 * RK3 step (no writeBuffer between stages).
 *
 * Bind-group layout policy (transpiler-friendly):
 *   - One bind group per (substage × ping-pong slot configuration).
 *   - Bind groups built per step. WebGPU bind-group creation is < 50 µs
 *     each — cheap enough to rebuild rather than maintain a cache.
 *   - All bindings are explicit, statically typed storage/uniform; no
 *     dynamic offsets, no push constants, no subgroup ops.
 *
 * No CPU readback in Phase 3b — every cross-pass dependency is GPU↔GPU.
 */

import { GRID_N, UNIFORM_BUFFER_SIZE, VIEW_DENSITY } from '../config.js';

const VEC4_BYTES = 16;
const F32_BYTES  = 4;
const STAGE_PARAMS_BYTES = 16; // (a0, a1, dt_w, _pad) f32×4

export class PlasmaBuffers {
    /**
     * @param {GPUDevice} device
     * @param {number} n grid resolution (square)
     */
    constructor(device, n = GRID_N) {
        this.device = device;
        this.n = n;
        const cells = n * n;

        const u_v4_bytes  = cells * VEC4_BYTES;
        const u_f32_bytes = cells * F32_BYTES;

        const mkStorage = (label, size, extra = 0) => device.createBuffer({
            label, size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra,
        });

        // ── RK3 storage: 4 slots × (U0, U1, Bx_face, By_face) ──────
        // Slots A/B ping-pong as (slot_n, slot_next). Slots 1, 2 are
        // scratch for intermediate RK3 stages.
        this.U0_a = mkStorage('plasma.U0_a', u_v4_bytes);
        this.U1_a = mkStorage('plasma.U1_a', u_v4_bytes);
        this.U0_b = mkStorage('plasma.U0_b', u_v4_bytes);
        this.U1_b = mkStorage('plasma.U1_b', u_v4_bytes);
        this.U0_1 = mkStorage('plasma.U0_1', u_v4_bytes);
        this.U1_1 = mkStorage('plasma.U1_1', u_v4_bytes);
        this.U0_2 = mkStorage('plasma.U0_2', u_v4_bytes);
        this.U1_2 = mkStorage('plasma.U1_2', u_v4_bytes);

        this.Bx_a = mkStorage('plasma.Bx_a', u_f32_bytes);
        this.By_a = mkStorage('plasma.By_a', u_f32_bytes);
        this.Bx_b = mkStorage('plasma.Bx_b', u_f32_bytes);
        this.By_b = mkStorage('plasma.By_b', u_f32_bytes);
        this.Bx_1 = mkStorage('plasma.Bx_1', u_f32_bytes);
        this.By_1 = mkStorage('plasma.By_1', u_f32_bytes);
        this.Bx_2 = mkStorage('plasma.Bx_2', u_f32_bytes);
        this.By_2 = mkStorage('plasma.By_2', u_f32_bytes);

        // Logical handles: slot_n is the start-of-step state; slot_next
        // is the destination for stage 3. We swap A↔B per step.
        this._side = 'a';
        this.U0_n     = this.U0_a; this.U1_n     = this.U1_a;
        this.U0_next  = this.U0_b; this.U1_next  = this.U1_b;
        this.Bx_n     = this.Bx_a; this.By_n     = this.By_a;
        this.Bx_next  = this.Bx_b; this.By_next  = this.By_b;

        // Per-stage scratch: Ez_edge recomputed in each stage; one buffer
        // reused. (Each stage uses Ez from the *current* L(U) eval.)
        this.Ez_edge = mkStorage('plasma.Ez_edge', u_f32_bytes);

        // ── PPM edge states — 4 buffers × 2 axes = 8 buffers ───────
        // Left + right primitive face states for each cell, both axes.
        this.edge_l_x_0 = mkStorage('plasma.edge_l_x_0', u_v4_bytes);
        this.edge_l_x_1 = mkStorage('plasma.edge_l_x_1', u_v4_bytes);
        this.edge_r_x_0 = mkStorage('plasma.edge_r_x_0', u_v4_bytes);
        this.edge_r_x_1 = mkStorage('plasma.edge_r_x_1', u_v4_bytes);
        this.edge_l_y_0 = mkStorage('plasma.edge_l_y_0', u_v4_bytes);
        this.edge_l_y_1 = mkStorage('plasma.edge_l_y_1', u_v4_bytes);
        this.edge_r_y_0 = mkStorage('plasma.edge_r_y_0', u_v4_bytes);
        this.edge_r_y_1 = mkStorage('plasma.edge_r_y_1', u_v4_bytes);

        // ── Per-direction face fluxes (unchanged from 3a) ──────────
        this.flux_x_0 = mkStorage('plasma.flux_x_0', u_v4_bytes);
        this.flux_x_1 = mkStorage('plasma.flux_x_1', u_v4_bytes);
        this.flux_y_0 = mkStorage('plasma.flux_y_0', u_v4_bytes);
        this.flux_y_1 = mkStorage('plasma.flux_y_1', u_v4_bytes);

        // ── dt reduction (unchanged) ───────────────────────────────
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
        // Each holds the full Uniforms struct (see shared-helpers.wgsl).
        // Written once per preset load (and any time dx/gamma/view changes).
        // Both share the same host-side scratch — we just rewrite each
        // with sweep_dir = 0 or 1 inside pushUniforms.
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
        // Legacy alias — `uniform` defaults to the x-sweep buffer so any
        // pass that doesn't care (compute_dt, view, etc.) just binds it.
        this.uniform = this.uniform_x;

        // ── Stage-params uniform buffers (one per RK3 stage) ───────
        //   stage 1: a0=1,    a1=0,    dt_w=1
        //   stage 2: a0=3/4,  a1=1/4,  dt_w=1/4
        //   stage 3: a0=1/3,  a1=2/3,  dt_w=2/3
        // Written once at init; never changed.
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

        // ── LUT (unchanged) ────────────────────────────────────────
        this.lut = device.createBuffer({
            label: 'plasma.lut',
            size: 256 * VEC4_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // ── View field (scalar f32 per cell) ───────────────────────
        this.field = device.createBuffer({
            label: 'plasma.field',
            size: cells * F32_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });

        // ── Colored buffer ─────────────────────────────────────────
        this.colored = device.createBuffer({
            label: 'plasma.colored',
            size: cells * VEC4_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });

        // Default view mode — caller can override via pushUniforms.
        this._viewMode = VIEW_DENSITY;
    }

    /**
     * Flip the (slot_n, slot_next) handles after each step. Slots 1 and 2
     * are scratch and stay put.
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
     * Upload an MHD initial condition into the active slot_n. Also seeds
     * slot_next with the same state (so an interrupted first step lands
     * sane data).
     */
    uploadInitialState({ U0, U1, Bx_face, By_face }) {
        const n = this.n;
        const cells = n * n;
        if (U0.length !== 4 * cells)       throw new Error(`U0 length mismatch: ${U0.length}`);
        if (U1.length !== 4 * cells)       throw new Error(`U1 length mismatch: ${U1.length}`);
        if (Bx_face.length !== cells)      throw new Error(`Bx_face length mismatch: ${Bx_face.length}`);
        if (By_face.length !== cells)      throw new Error(`By_face length mismatch: ${By_face.length}`);

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
     * uniform buffers. Single-submit RK3 needs the two sweep variants to
     * be ready before stage 1 dispatches.
     */
    pushUniforms({ dx, gamma, viewMin, viewMax, gridN, stepParity, viewMode }) {
        // x-sweep variant (sweep_dir = 0).
        this.uniformF32[0] = dx;
        this.uniformF32[1] = gamma;
        this.uniformF32[2] = viewMin;
        this.uniformF32[3] = viewMax;
        this.uniformU32[4] = gridN >>> 0;
        this.uniformU32[5] = 0;
        this.uniformU32[6] = stepParity >>> 0;
        this.uniformU32[7] = (viewMode ?? this._viewMode) >>> 0;
        if (viewMode !== undefined) this._viewMode = viewMode;
        this.device.queue.writeBuffer(this.uniform_x, 0, this.uniformHost);

        // y-sweep variant.
        this.uniformU32[5] = 1;
        this.device.queue.writeBuffer(this.uniform_y, 0, this.uniformHost);
    }
}

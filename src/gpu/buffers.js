/**
 * @fileoverview GPU buffer allocator for Phase 3a (2.5D ideal MHD + CT).
 *
 * Cell-centered conserved state is packed into two parallel ping-pong
 * vec4 buffers per slot:
 *   U0 = (ρ, ρvx, ρvy, ρvz)   — 4 components
 *   U1 = (E,  Bz,    _pad, _pad) — 2 active + 2 pad
 * That preserves the existing vec4 pipeline access pattern while
 * carrying the full 6-component MHD conservative state.
 *
 * Face-centered transverse magnetic field lives on its own pair of
 * f32 buffers (Bx_face on x-faces, By_face on y-faces); edge-centered
 * Ez lives on a single f32 buffer (corner index). Face/edge convention
 * documented in shared-helpers.wgsl.
 *
 * Per-direction slope + flux scratch buffers — we run PLM/HLL for both
 * sweep directions before doing the unsplit CT update, so x- and y-side
 * fluxes coexist (this is Stone+ 2008's directionally-unsplit CT style
 * even though the reconstruction & Riemann work itself is split-direction).
 *
 * No CPU readback in Phase 3a — every cross-pass dependency is GPU↔GPU.
 */

import { GRID_N, UNIFORM_BUFFER_SIZE, VIEW_DENSITY } from '../config.js';

const VEC4_BYTES = 16;
const F32_BYTES  = 4;

export class PlasmaBuffers {
    /**
     * @param {GPUDevice} device
     * @param {number} n grid resolution (square)
     */
    constructor(device, n = GRID_N) {
        this.device = device;
        this.n = n;
        const cells = n * n;

        const u_v4_bytes = cells * VEC4_BYTES;
        const u_f32_bytes = cells * F32_BYTES;

        const mkStorage = (label, size, extra = 0) => device.createBuffer({
            label, size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra,
        });

        // ── Ping-pong cell-centered conservative state ──────────────
        // Two slots × two vec4 buffers each.
        this.U0_a = mkStorage('plasma.U0_a', u_v4_bytes);
        this.U1_a = mkStorage('plasma.U1_a', u_v4_bytes);
        this.U0_b = mkStorage('plasma.U0_b', u_v4_bytes);
        this.U1_b = mkStorage('plasma.U1_b', u_v4_bytes);

        // Side-A is "current" on init. swap() flips logical roles.
        this._side = 'a';
        // Caller convenience handles.
        this.U0_current = this.U0_a;
        this.U1_current = this.U1_a;
        this.U0_next    = this.U0_b;
        this.U1_next    = this.U1_b;

        // ── Face-centered Bx, By + edge-centered Ez ─────────────────
        // Periodic BCs let us use Nx×Ny for both face buffers — face
        // index (i+½, j) is owned by cell (i,j) and wraps mod N. Same
        // for By and Ez (edge at (i+½, j+½) owned by cell (i,j)).
        //
        // Bx/By get a ping-pong pair so the curl(E) update doesn't have
        // to read+write the same buffer on the same pass.
        this.Bx_face_a = mkStorage('plasma.Bx_face_a', u_f32_bytes);
        this.Bx_face_b = mkStorage('plasma.Bx_face_b', u_f32_bytes);
        this.By_face_a = mkStorage('plasma.By_face_a', u_f32_bytes);
        this.By_face_b = mkStorage('plasma.By_face_b', u_f32_bytes);
        this.Bx_current = this.Bx_face_a;
        this.By_current = this.By_face_a;
        this.Bx_next    = this.Bx_face_b;
        this.By_next    = this.By_face_b;

        this.Ez_edge = mkStorage('plasma.Ez_edge', u_f32_bytes);

        // ── Per-direction PLM slopes (primitive-variable, 2 vec4 each) ──
        // slopes_x_0 = (ρ', vx', vy', vz')  slopes_x_1 = (p', By', Bz', _)
        // slopes_y_0 = (ρ', vx', vy', vz')  slopes_y_1 = (p', Bx', Bz', _)
        this.slopes_x_0 = mkStorage('plasma.slopes_x_0', u_v4_bytes);
        this.slopes_x_1 = mkStorage('plasma.slopes_x_1', u_v4_bytes);
        this.slopes_y_0 = mkStorage('plasma.slopes_y_0', u_v4_bytes);
        this.slopes_y_1 = mkStorage('plasma.slopes_y_1', u_v4_bytes);

        // ── Per-direction face fluxes (2 vec4 + 2 transverse-B floats) ──
        // We pack the transverse-B fluxes into the .y/.z lanes of flux_*_1
        // (currently flux_*_1.y already carries the f1 second slot for Bz;
        // we reuse the unused .z/.w slots for the transverse-B flux of the
        // *other* B component). See riemann-hll.wgsl for the precise pack.
        this.flux_x_0 = mkStorage('plasma.flux_x_0', u_v4_bytes);
        this.flux_x_1 = mkStorage('plasma.flux_x_1', u_v4_bytes);
        this.flux_y_0 = mkStorage('plasma.flux_y_0', u_v4_bytes);
        this.flux_y_1 = mkStorage('plasma.flux_y_1', u_v4_bytes);

        // ── dt reduction (unchanged) ────────────────────────────────
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

        // ── Uniforms ───────────────────────────────────────────────
        // Layout in shared-helpers.wgsl `Uniforms`:
        //   f32 dx, gamma, view_min, view_max,
        //   u32 grid_n, sweep_dir, step_parity, view_mode
        this.uniformHost = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
        this.uniformF32  = new Float32Array(this.uniformHost);
        this.uniformU32  = new Uint32Array(this.uniformHost);
        this.uniform = device.createBuffer({
            label: 'plasma.uniforms',
            size: UNIFORM_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // ── LUT (unchanged) ─────────────────────────────────────────
        this.lut = device.createBuffer({
            label: 'plasma.lut',
            size: 256 * VEC4_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // ── View field (scalar f32 per cell) ────────────────────────
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
     * Flip both cell-centered AND face-centered ping-pong handles. We
     * always re-derive .current/.next together so MHD updates that touch
     * cells AND faces stay in lockstep.
     */
    swap() {
        if (this._side === 'a') {
            this.U0_current = this.U0_b; this.U1_current = this.U1_b;
            this.U0_next    = this.U0_a; this.U1_next    = this.U1_a;
            this.Bx_current = this.Bx_face_b; this.By_current = this.By_face_b;
            this.Bx_next    = this.Bx_face_a; this.By_next    = this.By_face_a;
            this._side = 'b';
        } else {
            this.U0_current = this.U0_a; this.U1_current = this.U1_a;
            this.U0_next    = this.U0_b; this.U1_next    = this.U1_b;
            this.Bx_current = this.Bx_face_a; this.By_current = this.By_face_a;
            this.Bx_next    = this.Bx_face_b; this.By_next    = this.By_face_b;
            this._side = 'a';
        }
    }

    /**
     * Upload an MHD initial condition. The preset returns an object with:
     *   U0: Float32Array(4·N·N)   — (ρ, ρvx, ρvy, ρvz) per cell
     *   U1: Float32Array(4·N·N)   — (E, Bz, 0, 0) per cell
     *   Bx_face: Float32Array(N·N) — Bx on x-faces (cell i owns face at i+½)
     *   By_face: Float32Array(N·N) — By on y-faces (cell j owns face at j+½)
     */
    uploadInitialState({ U0, U1, Bx_face, By_face }) {
        const n = this.n;
        const cells = n * n;
        if (U0.length !== 4 * cells)       throw new Error(`U0 length mismatch: ${U0.length}`);
        if (U1.length !== 4 * cells)       throw new Error(`U1 length mismatch: ${U1.length}`);
        if (Bx_face.length !== cells)      throw new Error(`Bx_face length mismatch: ${Bx_face.length}`);
        if (By_face.length !== cells)      throw new Error(`By_face length mismatch: ${By_face.length}`);

        const q = this.device.queue;
        q.writeBuffer(this.U0_a, 0, U0.buffer, U0.byteOffset, U0.byteLength);
        q.writeBuffer(this.U0_b, 0, U0.buffer, U0.byteOffset, U0.byteLength);
        q.writeBuffer(this.U1_a, 0, U1.buffer, U1.byteOffset, U1.byteLength);
        q.writeBuffer(this.U1_b, 0, U1.buffer, U1.byteOffset, U1.byteLength);
        q.writeBuffer(this.Bx_face_a, 0, Bx_face.buffer, Bx_face.byteOffset, Bx_face.byteLength);
        q.writeBuffer(this.Bx_face_b, 0, Bx_face.buffer, Bx_face.byteOffset, Bx_face.byteLength);
        q.writeBuffer(this.By_face_a, 0, By_face.buffer, By_face.byteOffset, By_face.byteLength);
        q.writeBuffer(this.By_face_b, 0, By_face.buffer, By_face.byteOffset, By_face.byteLength);

        // Reset logical handles to side A.
        this._side = 'a';
        this.U0_current = this.U0_a; this.U1_current = this.U1_a;
        this.U0_next    = this.U0_b; this.U1_next    = this.U1_b;
        this.Bx_current = this.Bx_face_a; this.By_current = this.By_face_a;
        this.Bx_next    = this.Bx_face_b; this.By_next    = this.By_face_b;
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
     * Push the host uniform struct to GPU.
     */
    pushUniforms({ dx, gamma, viewMin, viewMax, gridN, sweepDir, stepParity, viewMode }) {
        this.uniformF32[0] = dx;
        this.uniformF32[1] = gamma;
        this.uniformF32[2] = viewMin;
        this.uniformF32[3] = viewMax;
        this.uniformU32[4] = gridN >>> 0;
        this.uniformU32[5] = sweepDir >>> 0;
        this.uniformU32[6] = stepParity >>> 0;
        this.uniformU32[7] = (viewMode ?? this._viewMode) >>> 0;
        if (viewMode !== undefined) this._viewMode = viewMode;
        this.device.queue.writeBuffer(this.uniform, 0, this.uniformHost);
    }
}

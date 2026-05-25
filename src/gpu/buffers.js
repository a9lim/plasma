/**
 * @fileoverview GPU buffer allocator for Phase 2.
 *
 * Owns the ping-pong U_cell buffers, the per-cell slope buffer, the
 * per-face flux buffer, the dt+wavespeed reduction buffers, the LUT,
 * the view-field buffer, and the colored buffer that the composite
 * pass samples. Uniforms live here too — one tightly-packed struct
 * holds dx/γ/grid/sweep_dir/etc.
 *
 * No CPU readback in Phase 2 — every cross-pass dependency is GPU↔GPU
 * via the storage buffers below.
 */

import { GRID_N, UNIFORM_BUFFER_SIZE } from '../config.js';

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

        // ── Conservative state, ping-pong ──────────────────────────
        // vec4<f32> per cell: (ρ, ρvx, ρvy, E)
        const uByteSize = cells * VEC4_BYTES;
        this.U_a = device.createBuffer({
            label: 'plasma.U_a',
            size: uByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.U_b = device.createBuffer({
            label: 'plasma.U_b',
            size: uByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Logical ping-pong: read from `current`, write to `next`. The two
        // sweeps per timestep ping twice, so after step() the pair is
        // back where it started (modulo content).
        this.current = this.U_a;
        this.next    = this.U_b;

        // ── Per-cell PLM slopes (primitive-variable space) ──────────
        // Stored vec4<f32> mirroring the U layout.
        this.slopes = device.createBuffer({
            label: 'plasma.slopes',
            size: uByteSize,
            usage: GPUBufferUsage.STORAGE,
        });

        // ── Per-cell flux at the i+1/2 face along the active sweep ──
        // Reset every dispatch so no clear is needed (every cell writes).
        this.flux = device.createBuffer({
            label: 'plasma.flux',
            size: uByteSize,
            usage: GPUBufferUsage.STORAGE,
        });

        // ── dt reduction: a single atomic<u32> bitcasted from the max
        //    wave speed, and a single f32 holding the resolved dt for
        //    the sweep passes to read.
        // 16-byte buffers for both — minimum useful storage binding size
        // on some adapters is 16 B, and we lose nothing by padding.
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

        // ── Uniforms (per-step constants) ──────────────────────────
        // Layout in shared-helpers.wgsl `Uniforms` struct:
        //   f32 dx, f32 gamma, f32 view_min, f32 view_max,
        //   u32 grid_n, u32 sweep_dir, u32 step_parity, u32 _pad
        this.uniformHost = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
        this.uniformF32  = new Float32Array(this.uniformHost);
        this.uniformU32  = new Uint32Array(this.uniformHost);
        this.uniform = device.createBuffer({
            label: 'plasma.uniforms',
            size: UNIFORM_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // ── LUT (256-entry RGBA, uploaded once) ─────────────────────
        // 256 × vec4<f32> = 256 × 16 = 4096 B.
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

        // ── Colored buffer (vec4<f32> per cell) ─────────────────────
        // Read by the composite render pass via read-only storage.
        this.colored = device.createBuffer({
            label: 'plasma.colored',
            size: cells * VEC4_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });
    }

    /**
     * Swap the current/next pointers — call between sweeps.
     */
    swap() {
        const tmp = this.current;
        this.current = this.next;
        this.next = tmp;
    }

    /**
     * Upload conservative-state IC from a Float32Array (length 4·N·N).
     * Writes to both ping-pong slots so the first sweep's `current` is
     * the IC regardless of starting orientation; the other slot is
     * overwritten in the same step but harmless to seed.
     */
    uploadInitialState(arr) {
        const n = this.n;
        if (arr.length !== 4 * n * n) {
            throw new Error(`uploadInitialState: expected ${4 * n * n} floats, got ${arr.length}`);
        }
        this.device.queue.writeBuffer(this.U_a, 0, arr.buffer, arr.byteOffset, arr.byteLength);
        this.device.queue.writeBuffer(this.U_b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
        this.current = this.U_a;
        this.next    = this.U_b;
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
     * Push the host uniform struct to GPU. Cheap (64 B); fine to call
     * every sweep.
     */
    pushUniforms({ dx, gamma, viewMin, viewMax, gridN, sweepDir, stepParity }) {
        this.uniformF32[0] = dx;
        this.uniformF32[1] = gamma;
        this.uniformF32[2] = viewMin;
        this.uniformF32[3] = viewMax;
        this.uniformU32[4] = gridN >>> 0;
        this.uniformU32[5] = sweepDir >>> 0;
        this.uniformU32[6] = stepParity >>> 0;
        this.uniformU32[7] = 0;
        this.device.queue.writeBuffer(this.uniform, 0, this.uniformHost);
    }
}

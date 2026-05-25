/**
 * @fileoverview Phase 2 initial conditions.
 *
 * One preset (Sod shock tube) — sufficient to verify the HLL+PLM+FE
 * pipeline. Phase 3 adds OT/Brio-Wu/Harris/etc.
 *
 * A preset returns a Float32Array of length 4·N·N laid out as `vec4<ρ,
 * ρvx, ρvy, E>` per cell (matches the GPU `array<vec4<f32>>` U buffer).
 * It may also override per-preset run parameters (γ, view window, etc.)
 * via the returned descriptor.
 */

import { GRID_N, DOMAIN_LENGTH, PRESSURE_FLOOR } from './config.js';

/**
 * Conservative state packer: takes primitive (ρ, vx, vy, p) and γ, returns
 * (ρ, ρvx, ρvy, E) with E = p/(γ-1) + 0.5·ρ·(vx² + vy²).
 */
function pack(rho, vx, vy, p, gamma) {
    const ke = 0.5 * rho * (vx * vx + vy * vy);
    const E  = Math.max(p, PRESSURE_FLOOR) / (gamma - 1) + ke;
    return [rho, rho * vx, rho * vy, E];
}

/**
 * Sod shock tube IC. The 1D classic embedded in a 2D grid (uniform in y).
 *
 *   Left  half (x < 0.5·L): ρ=1.0,   vx=0, vy=0, p=1.0
 *   Right half (x ≥ 0.5·L): ρ=0.125, vx=0, vy=0, p=0.1
 *
 * γ=1.4 (literature standard for Sod, not the MHD default 5/3). Expected
 * features at t≈0.2 (with L=1): shock at x≈0.85, contact at x≈0.69, head
 * of rarefaction near x≈0.26.
 */
export function makeSodPreset(n = GRID_N) {
    const gamma = 1.4;
    const arr = new Float32Array(4 * n * n);
    const half = Math.floor(n / 2);

    const left  = pack(1.0,   0, 0, 1.0, gamma);
    const right = pack(0.125, 0, 0, 0.1, gamma);

    for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
            const idx = 4 * (j * n + i);
            const cell = (i < half) ? left : right;
            arr[idx + 0] = cell[0];
            arr[idx + 1] = cell[1];
            arr[idx + 2] = cell[2];
            arr[idx + 3] = cell[3];
        }
    }

    return {
        id: 'sod',
        label: 'Sod shock tube',
        gamma,
        domainLength: DOMAIN_LENGTH,
        data: arr,
        // Display window tuned for Sod: ρ ∈ [0.125, 1.0] with margin.
        viewMin: 0.05,
        viewMax: 1.10,
        // Reference run time for verification. The sim doesn't stop on its
        // own — this is just metadata for any future "auto-stop at t" knob.
        verifyTime: 0.2,
    };
}

export const PRESETS = {
    sod: makeSodPreset,
};

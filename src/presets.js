/**
 * @fileoverview Phase 3a initial conditions.
 *
 * Each preset returns:
 *   {
 *     id, label,
 *     gamma, domainLength,
 *     data: {
 *       U0: Float32Array(4·N·N),    // (ρ, ρvx, ρvy, ρvz)
 *       U1: Float32Array(4·N·N),    // (E, Bz, 0, 0)
 *       Bx_face: Float32Array(N·N), // Bx on x-faces — cell (i,j) owns face (i+½, j)
 *       By_face: Float32Array(N·N), // By on y-faces — cell (i,j) owns face (i, j+½)
 *     },
 *     viewMin, viewMax, verifyTime,
 *   }
 */

import { GRID_N, DOMAIN_LENGTH, PRESSURE_FLOOR } from './config.js';

/**
 * Pack a primitive (ρ, vx, vy, vz, p, Bx_c, By_c, Bz) cell state into the
 * pair (U0, U1) at cell index `idx`. Bx_c, By_c are cell-centered values
 * (the staggered face values are written separately by the caller).
 *
 * E = p/(γ-1) + ½·ρ·|v|² + ½·|B|²
 */
function writeCell(U0, U1, idx, rho, vx, vy, vz, p, bx_c, by_c, bz, gamma) {
    const ke = 0.5 * rho * (vx*vx + vy*vy + vz*vz);
    const mb = 0.5 * (bx_c*bx_c + by_c*by_c + bz*bz);
    const E  = Math.max(p, PRESSURE_FLOOR) / (gamma - 1) + ke + mb;
    U0[4 * idx + 0] = rho;
    U0[4 * idx + 1] = rho * vx;
    U0[4 * idx + 2] = rho * vy;
    U0[4 * idx + 3] = rho * vz;
    U1[4 * idx + 0] = E;
    U1[4 * idx + 1] = bz;
    U1[4 * idx + 2] = 0;
    U1[4 * idx + 3] = 0;
}

/**
 * Sod shock tube (pure hydro, embedded in MHD framework: B ≡ 0).
 * γ = 1.4. Left half ρ=1, p=1; right half ρ=0.125, p=0.1. v = 0.
 */
export function makeSodPreset(n = GRID_N) {
    const gamma = 1.4;
    const cells = n * n;
    const U0 = new Float32Array(4 * cells);
    const U1 = new Float32Array(4 * cells);
    const Bx_face = new Float32Array(cells);   // all zero
    const By_face = new Float32Array(cells);   // all zero
    const half = Math.floor(n / 2);

    for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
            const idx = j * n + i;
            const left = (i < half);
            const rho = left ? 1.0 : 0.125;
            const p   = left ? 1.0 : 0.1;
            writeCell(U0, U1, idx, rho, 0, 0, 0, p, 0, 0, 0, gamma);
        }
    }

    return {
        id: 'sod',
        label: 'Sod shock tube',
        gamma,
        domainLength: DOMAIN_LENGTH,
        data: { U0, U1, Bx_face, By_face },
        viewMin: 0.05,
        viewMax: 1.10,
        verifyTime: 0.2,
    };
}

/**
 * Brio-Wu MHD shock tube. The canonical 1D MHD Riemann problem.
 * γ = 2.0.
 *
 *   Left  (x < 0.5·L): ρ=1.0,   vx=vy=vz=0, p=1.0, Bx=0.75, By=+1.0, Bz=0
 *   Right (x ≥ 0.5·L): ρ=0.125, vx=vy=vz=0, p=0.1, Bx=0.75, By=−1.0, Bz=0
 *
 * Bx is uniform across the entire grid (it's the face-normal of the
 * discontinuity, so ∇·B = 0 forces it to be constant in x). The y-faces
 * carry the discontinuous By; the x-faces uniformly hold 0.75. By_face
 * at the discontinuity belongs to the cells either side, not the
 * discontinuity surface itself.
 *
 * Expected at t ≈ 0.1 (with L=1, γ=2): fast rarefaction (left), slow
 * compound (left-of-center), contact, slow shock, fast rarefaction
 * (right). All four-wave structure visible in ρ.
 */
export function makeBrioWuPreset(n = GRID_N) {
    const gamma = 2.0;
    const cells = n * n;
    const U0 = new Float32Array(4 * cells);
    const U1 = new Float32Array(4 * cells);
    const Bx_face = new Float32Array(cells);
    const By_face = new Float32Array(cells);
    const half = Math.floor(n / 2);

    const Bx_uniform = 0.75;
    const ByL = 1.0;
    const ByR = -1.0;

    for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
            const idx  = j * n + i;
            const left = (i < half);
            const rho  = left ? 1.0   : 0.125;
            const p    = left ? 1.0   : 0.1;
            const by_c = left ? ByL   : ByR;
            const bz   = 0.0;

            writeCell(U0, U1, idx, rho, 0, 0, 0, p, Bx_uniform, by_c, bz, gamma);

            // Face-centered B init.
            // Bx_face owned by cell (i,j) sits at (i+½, j) — uniform 0.75.
            Bx_face[idx] = Bx_uniform;
            // By_face owned by cell (i,j) sits at (i, j+½). Since By is
            // uniform in y on each side, every y-face on the left side
            // = +1, every y-face on the right side = -1. The face at
            // (half-1, j+½) sits *inside* the left-cell column and gets
            // +1; the face at (half, j+½) sits inside the right column
            // and gets -1.
            By_face[idx] = left ? ByL : ByR;
        }
    }

    return {
        id: 'brio-wu',
        label: 'Brio-Wu MHD shock tube',
        gamma,
        domainLength: DOMAIN_LENGTH,
        data: { U0, U1, Bx_face, By_face },
        viewMin: 0.05,
        viewMax: 1.10,
        verifyTime: 0.1,
    };
}

export const PRESETS = {
    sod: makeSodPreset,
    'brio-wu': makeBrioWuPreset,
};

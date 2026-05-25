/**
 * @fileoverview Phase 4 initial conditions.
 *
 * Each preset returns:
 *   {
 *     id, label,
 *     gamma, domainLength,
 *     eta,                   // optional, default 0
 *     bc,                    // optional, defaults to all periodic
 *     data: {
 *       U0:      Float32Array(4 · (N+4)²),     // cell-centered (ρ, ρvx, ρvy, ρvz)
 *       U1:      Float32Array(4 · (N+4)²),     // cell-centered (E, Bz, _, _)
 *       Bx_face: Float32Array((N+5) · (N+4)),  // LEFT face of cell (i,j)
 *       By_face: Float32Array((N+4) · (N+5)),  // BOTTOM face of cell (i,j)
 *     },
 *     viewMin, viewMax, verifyTime,
 *   }
 *
 * Cell-centered storage convention: cell (i, j) occupies indices i, j
 * with i, j ∈ [0, N+4). The INTERIOR cells are i, j ∈ [ghost, ghost+N).
 * With ghost = 2 and N = 256, interior i, j ∈ [2, 258).
 *
 * Face convention (LEFT/DOWN ownership):
 *   Bx_face[i, j] sits on the LEFT face of cell (i, j) at x = (i-ghost)·dx.
 *   By_face[i, j] sits on the BOTTOM face of cell (i, j) at y = (j-ghost)·dx.
 * Interior x-faces: i ∈ [ghost, ghost+N+1), j ∈ [ghost, ghost+N).
 * Interior y-faces: i ∈ [ghost, ghost+N),   j ∈ [ghost, ghost+N+1).
 *
 * Ghost-strip values written by the preset are only ADVISORY — apply-
 * bcs.wgsl re-fills them at the start of each RK3 stage based on the
 * BC config. For periodic BCs, we still fill ghost strips here so the
 * INITIAL state on side-A buffers is sane before the first apply-bcs.
 */

import {
    GRID_N, GHOST_WIDTH, DOMAIN_LENGTH, PRESSURE_FLOOR, ETA_DEFAULT,
    BC_PERIODIC, BC_OUTFLOW,
} from './config.js';

/** Cell-centered flat index in ghost-padded storage. */
function cellIdx(i, j, nTotal) { return j * nTotal + i; }

/** Bx_face index: (n_total+1) wide × n_total tall. */
function bxFaceIdx(i, j, nTotal) { return j * (nTotal + 1) + i; }

/** By_face index: n_total wide × (n_total+1) tall. */
function byFaceIdx(i, j, nTotal) { return j * nTotal + i; }

/**
 * Pack a primitive (ρ, vx, vy, vz, p, Bx_c, By_c, Bz) state into the
 * (U0, U1) pair at cell index `idx` (already a ghost-padded flat index).
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
 * Allocate the four data arrays for a preset with ghost-padded sizing.
 */
function allocData(n) {
    const ghost  = GHOST_WIDTH;
    const nT     = n + 2 * ghost;
    const cellsT = nT * nT;
    const xfaces = (nT + 1) * nT;
    const yfaces = nT * (nT + 1);
    return {
        nT,
        ghost,
        U0:      new Float32Array(4 * cellsT),
        U1:      new Float32Array(4 * cellsT),
        Bx_face: new Float32Array(xfaces),
        By_face: new Float32Array(yfaces),
    };
}

/**
 * Replicate a function `fn(iInterior, jInterior)` over a single
 * interior cell + every ghost cell that wraps to it. Used to fill ghost
 * strips for periodic-BC presets.
 */
function fillCellGhostPeriodic(U0, U1, gamma, nT, ghost, nInterior,
                                primAt) {
    for (let j = 0; j < nT; j++) {
        for (let i = 0; i < nT; i++) {
            // Wrap (i, j) to the interior cell at (iw, jw).
            let iw = i - ghost;
            let jw = j - ghost;
            iw = ((iw % nInterior) + nInterior) % nInterior;
            jw = ((jw % nInterior) + nInterior) % nInterior;
            const idx = cellIdx(i, j, nT);
            const p = primAt(iw, jw);
            writeCell(U0, U1, idx,
                      p.rho, p.vx, p.vy, p.vz, p.p, p.bx_c, p.by_c, p.bz, gamma);
        }
    }
}

/**
 * Sod shock tube. γ = 1.4, B ≡ 0, periodic BCs.
 */
export function makeSodPreset(n = GRID_N) {
    const gamma = 1.4;
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);
    const half = Math.floor(n / 2);

    const primAt = (iw, jw) => {
        const left = (iw < half);
        return {
            rho: left ? 1.0 : 0.125, vx: 0, vy: 0, vz: 0,
            p:   left ? 1.0 : 0.1,   bx_c: 0, by_c: 0, bz: 0,
        };
    };
    fillCellGhostPeriodic(U0, U1, gamma, nT, ghost, n, primAt);
    // Bx_face / By_face all zero (B ≡ 0).

    return {
        id: 'sod', label: 'Sod shock tube',
        gamma,
        domainLength: DOMAIN_LENGTH,
        eta: 0,
        bc: {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        data: { U0, U1, Bx_face, By_face },
        viewMin: 0.05, viewMax: 1.10, verifyTime: 0.2,
    };
}

/**
 * Brio-Wu MHD shock tube. γ = 2, periodic BCs.
 */
export function makeBrioWuPreset(n = GRID_N) {
    const gamma = 2.0;
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);
    const half = Math.floor(n / 2);
    const Bx_uniform = 0.75;
    const ByL = 1.0, ByR = -1.0;

    const primAt = (iw, jw) => {
        const left = (iw < half);
        return {
            rho: left ? 1.0 : 0.125, vx: 0, vy: 0, vz: 0,
            p:   left ? 1.0 : 0.1,
            bx_c: Bx_uniform, by_c: left ? ByL : ByR, bz: 0,
        };
    };
    fillCellGhostPeriodic(U0, U1, gamma, nT, ghost, n, primAt);

    // Face B: Bx is uniform everywhere; By is +1 / -1 on the appropriate
    // side. We fill every face including ghost strips so the initial
    // state is sane. Bx_face has (nT+1) cols × nT rows.
    for (let j = 0; j < nT; j++) {
        for (let i = 0; i <= nT; i++) {
            Bx_face[bxFaceIdx(i, j, nT)] = Bx_uniform;
        }
    }
    // By_face has nT cols × (nT+1) rows. By_face[i, j] is the bottom
    // face of cell (i, j). For interior cell columns i ∈ [ghost, ghost+N),
    // the value matches "left half / right half" of the cell ABOVE it
    // (cell (i, j)). For ghost cells wrap periodically.
    for (let j = 0; j <= nT; j++) {
        for (let i = 0; i < nT; i++) {
            let iw = i - ghost;
            iw = ((iw % n) + n) % n;
            const left = (iw < half);
            By_face[byFaceIdx(i, j, nT)] = left ? ByL : ByR;
        }
    }

    return {
        id: 'brio-wu', label: 'Brio-Wu MHD shock tube',
        gamma,
        domainLength: DOMAIN_LENGTH,
        eta: 0,
        bc: {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        data: { U0, U1, Bx_face, By_face },
        viewMin: 0.05, viewMax: 1.10, verifyTime: 0.1,
    };
}

/**
 * Orszag-Tang vortex. γ = 5/3, [0, 2π]² periodic, B from vector potential.
 */
export function makeOrszagTangPreset(n = GRID_N) {
    const gamma = 5.0 / 3.0;
    const L = 2.0 * Math.PI;
    const dx = L / n;
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);

    const rho = gamma * gamma;
    const p   = gamma;

    // Cell-centered primitives at every cell (incl. ghosts). For
    // periodic IC we extend the analytic functions over ghost cells
    // directly — the trig functions are periodic with period 2π so we
    // can evaluate them at the actual extended coordinate.
    for (let j = 0; j < nT; j++) {
        const y_c = (j - ghost + 0.5) * dx;
        for (let i = 0; i < nT; i++) {
            const x_c = (i - ghost + 0.5) * dx;
            const vx   = -Math.sin(y_c);
            const vy   =  Math.sin(x_c);
            const bx_c = -Math.sin(y_c);
            const by_c =  Math.sin(2 * x_c);
            const bz   = 0.0;
            writeCell(U0, U1, cellIdx(i, j, nT),
                      rho, vx, vy, 0, p, bx_c, by_c, bz, gamma);
        }
    }

    // Bx_face[i, j] is at (x = (i-ghost)·dx, y = (j-ghost+0.5)·dx).
    for (let j = 0; j < nT; j++) {
        const y_xf = (j - ghost + 0.5) * dx;
        for (let i = 0; i <= nT; i++) {
            Bx_face[bxFaceIdx(i, j, nT)] = -Math.sin(y_xf);
        }
    }
    // By_face[i, j] at (x = (i-ghost+0.5)·dx, y = (j-ghost)·dx).
    for (let j = 0; j <= nT; j++) {
        for (let i = 0; i < nT; i++) {
            const x_yf = (i - ghost + 0.5) * dx;
            By_face[byFaceIdx(i, j, nT)] = Math.sin(2 * x_yf);
        }
    }

    return {
        id: 'orszag-tang', label: 'Orszag-Tang vortex',
        gamma,
        domainLength: L,
        eta: 0,
        bc: {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        data: { U0, U1, Bx_face, By_face },
        viewMin: 1.0, viewMax: 6.0, verifyTime: 0.5,
    };
}

/**
 * Harris current sheet — the canonical reconnection initial condition.
 *
 * γ = 5/3. Square domain [-1, 1]² (we keep square per the spec's
 * fallback option; the perturbation breaks the artificial mirror
 * symmetry to seed reconnection). Sheet half-width a = 0.1. Periodic
 * in x (W/E), outflow in y (N/S).
 *
 *   Bx(y) = B_0 · tanh(y / a),  with B_0 = 1, a = 0.1
 *   By = 0,  Bz = 0
 *   ρ(y)  = ρ_∞ + ρ_0 · sech²(y/a),  ρ_∞ = 0.2, ρ_0 = 1
 *   p(y)  = p_∞ + 0.5 · B_0² · sech²(y/a),  p_∞ = 0.1
 *     → total pressure p + ½|B|² = p_∞ + 0.5·B_0² (constant across sheet)
 *   v(x,y) perturbation: vy = 0.01 · sin(π·x) · sech²(y/a), vx = vz = 0
 *
 * Verification points:
 *   y=0:  Bx = 0, ρ = 1.2, p = 0.6
 *   y=a:  Bx ≈ 0.762, ρ ≈ 0.62, p ≈ 0.42
 *
 * BCs: periodic x (W=E=periodic), outflow y (N=S=outflow). η = 1e-3.
 */
export function makeHarrisPreset(n = GRID_N) {
    const gamma = 5.0 / 3.0;
    const L = 2.0;                // domain half-width is 1.0 (centered at 0)
    const dx = L / n;             // dy = dx (square mesh)
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);

    const a    = 0.1;
    const B0   = 1.0;
    const rhoI = 0.2;             // ρ_∞
    const rho0 = 1.0;
    const pI   = 0.1;             // p_∞

    const sech2 = (z) => {
        const c = Math.cosh(z);
        return 1.0 / (c * c);
    };

    // Position helper: interior cell (i, j) at (i-ghost+0.5)·dx - 1.
    // The "-1" centers the domain on (0, 0).
    const xOf = (i) => (i - ghost + 0.5) * dx - 1.0;
    const yOf = (j) => (j - ghost + 0.5) * dx - 1.0;
    const xFaceOf = (i) => (i - ghost) * dx - 1.0;
    const yFaceOf = (j) => (j - ghost) * dx - 1.0;

    // Cell-centered primitives.
    for (let j = 0; j < nT; j++) {
        const y_c = yOf(j);
        const s   = sech2(y_c / a);
        const rho = rhoI + rho0 * s;
        const p   = pI + 0.5 * B0 * B0 * s;
        const bxc = B0 * Math.tanh(y_c / a);
        for (let i = 0; i < nT; i++) {
            const x_c = xOf(i);
            const vy  = 0.01 * Math.sin(Math.PI * x_c) * s;
            writeCell(U0, U1, cellIdx(i, j, nT),
                      rho, 0, vy, 0, p, bxc, 0, 0, gamma);
        }
    }

    // Bx_face[i, j] at (x = xFaceOf(i), y = yOf(j)) — face value uses the
    // cell-row's y center; Bx only depends on y.
    for (let j = 0; j < nT; j++) {
        const y_xf = yOf(j);
        const bxf  = B0 * Math.tanh(y_xf / a);
        for (let i = 0; i <= nT; i++) {
            Bx_face[bxFaceIdx(i, j, nT)] = bxf;
        }
    }
    // By_face[i, j] at (x = xOf(i), y = yFaceOf(j)) — By ≡ 0.
    // Already zero from Float32Array initialization.

    return {
        id: 'harris', label: 'Harris current sheet',
        gamma,
        domainLength: L,
        eta: ETA_DEFAULT,     // 1e-3
        bc: {
            modeN: BC_OUTFLOW,  modeS: BC_OUTFLOW,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        data: { U0, U1, Bx_face, By_face },
        viewMin: -3.0, viewMax: 3.0,   // Jz view default
        verifyTime: 10.0,              // reconnection becomes visible
    };
}

export const PRESETS = {
    sod: makeSodPreset,
    'brio-wu': makeBrioWuPreset,
    'orszag-tang': makeOrszagTangPreset,
    'harris': makeHarrisPreset,
};

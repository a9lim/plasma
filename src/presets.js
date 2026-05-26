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
    BC_PERIODIC, BC_OUTFLOW, BASE_PHYSICS_FLAGS, EXTENDED_PHYSICS_FLAGS,
    FLAG_COOLING, FLAG_GRAVITY_SELF, FLAG_CONDUCTION, FLAG_HALL,
    COOLING_CURVE_TABLE,
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
 * Sod shock tube. γ = 1.4, B ≡ 0. Outflow in x, periodic transverse.
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
            modeE: BC_OUTFLOW,  modeW: BC_OUTFLOW,
        },
        physics: { physicsFlags: BASE_PHYSICS_FLAGS },
        data: { U0, U1, Bx_face, By_face },
        viewMin: 0.05, viewMax: 1.10, verifyTime: 0.2,
    };
}

/**
 * Brio-Wu MHD shock tube. γ = 2. Outflow in x, periodic transverse.
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
            modeE: BC_OUTFLOW,  modeW: BC_OUTFLOW,
        },
        physics: { physicsFlags: BASE_PHYSICS_FLAGS },
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
        physics: { physicsFlags: BASE_PHYSICS_FLAGS },
        data: { U0, U1, Bx_face, By_face },
        viewMin: 1.0, viewMax: 6.0, verifyTime: 0.5,
        // η floor coefficient — see sim.getEtaMin(). Set to 0: the
        // engine's defensive layer (apply-resistivity snapshot,
        // HLLD_BX_EPS2 = 1e-10, conserved-state sanitization in
        // update-conserved-weighted) handles thin-sheet NaN cascades
        // gracefully without needing an explicit grid Reynolds floor.
        //
        // Calibration data (post-fixes):
        //   N=256  critical η ≈ 8e-4   (dx ≈ 2.45e-2)
        //   N=1024 critical η ≈ 1e-4   (dx ≈ 6.14e-3, sim lasted long
        //                               before NaN — degrades gracefully)
        // Implied empirical coeff (C·v_char) ≈ 0.03 at N=256, ~0.016 at
        // N=1024 — i.e., super-linear scaling because OT concentrates
        // energy faster at finer grids. A static coeff would be wrong
        // at one end or the other; the defensive layer obviates the
        // tradeoff. If a future preset has worse degradation
        // characteristics, set a nonzero coeff here.
        etaFloorCoeff: 0,
    };
}

/**
 * Orszag-Tang with the opt-in extended physics stack enabled. Kept separate
 * from the canonical Orszag-Tang preset so validation remains adiabatic MHD.
 */
export function makeOrszagTangExtendedPreset(n = GRID_N) {
    const preset = makeOrszagTangPreset(n);
    return {
        ...preset,
        id: 'orszag-tang-extended',
        label: 'Orszag-Tang + extended physics',
        physics: {
            physicsFlags: EXTENDED_PHYSICS_FLAGS,
            hallDi: 0.02,
            hallSubstepsMax: 8,
            coolingLambda0: 0.01,
            coolingTFloor: 1.0e-4,
            coolingTRef: 1.0,
            coolingCurveMode: COOLING_CURVE_TABLE,
            conductionKappa: 1.0e-3,
            conductionIsoFrac: 0.1,
            conductionSatFrac: 0.3,
            gravityGx: 0.0,
            gravityGy: 0.0,
            gravityG: 1.0e-3,
            gravityPoissonIters: 30,
            hallElectronPressureFrac: 0.5,
        },
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
 *
 * Anomalous resistivity (Session 8): set α > 0 in the Advanced settings
 * dropdown to enable fast reconnection — η(|J|) = η_0 + α·max(0, |J|/J_crit−1)².
 * At Harris's J_z ≈ B_0/a = 10 at the sheet center, sensible defaults
 * are α ≈ 1e-3, J_crit ≈ 10–20: leaves the bulk plasma at η_0 = 1e-3
 * Sweet-Parker (slow) reconnection but spikes η at the X-point to seed
 * Petschek-like fast reconnection + plasmoid generation.
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
        physics: { physicsFlags: BASE_PHYSICS_FLAGS },
        data: { U0, U1, Bx_face, By_face },
        viewMin: -3.0, viewMax: 3.0,   // Jz view default
        verifyTime: 10.0,              // reconnection becomes visible
    };
}

/**
 * Circularly polarized Alfvén wave — canonical MHD convergence test
 * (Tóth 2000; Stone+ 2008 §4.2). Smooth, periodic, exact analytic
 * solution: the IC translates rigidly along k at the Alfvén speed v_A
 * and returns to itself every period T = λ/v_A.
 *
 *  ── Setup (Tóth 2000) ────────────────────────────────────────────
 *  Domain          : [0, 1]² square, all periodic BCs.
 *  Wave angle      : α = atan(2)  →  cos α = 1/√5, sin α = 2/√5.
 *                    Wave vector  k_hat = (cos α, sin α).
 *  Phase variable  : φ(x, y) = 2π(x + 2y).
 *                    Implicit wavelength λ = 1/√5 along k; this packs
 *                    exactly ONE wavelength along k between (0,0) and
 *                    (1, 2), so the field is periodic on [0,1]² with
 *                    period 1 in x and period 1/2 in y.
 *  Background      : ρ = 1, p = 0.1, γ = 5/3.
 *                    B = B_∥ k_hat with B_∥ = 1.
 *  Perturbation    : A = 0.1 (amplitude).
 *                    δB_⊥1 = A sin φ   in the (x,y)-plane ⊥ to k.
 *                    δB_⊥2 = A cos φ   along z.
 *                    Alfvén polarization for +k propagation:
 *                       δv_⊥ = -δB_⊥ / √ρ.
 *  Alfvén speed    : v_A = B_∥ / √ρ = 1.
 *  Period          : T = λ / v_A = 1/√5 ≈ 0.4472136.
 *
 *  ── Rotation to lab frame ─────────────────────────────────────────
 *  Unit vectors:  k̂   = ( cos α,  sin α, 0)
 *                 ⊥1  = (-sin α,  cos α, 0)
 *                 ⊥2  = ( 0,      0,     1)
 *
 *  B_x = cos α - sin α · A sin φ
 *  B_y = sin α + cos α · A sin φ
 *  B_z = A cos φ
 *  v_x = +sin α · A sin φ           (since δv_⊥1 = -A sin φ → v_x = -sin α · δv_⊥1 = +sin α · A sin φ)
 *  v_y = -cos α · A sin φ
 *  v_z = -A cos φ
 *
 *  Check ∇·B = ∂_x B_x + ∂_y B_y
 *            = -sin α · A cos φ · ∂_x φ + cos α · A cos φ · ∂_y φ
 *            = A cos φ · (-sin α · 2π + cos α · 4π)
 *            = A cos φ · 2π · (-2/√5 + 2/√5) = 0  ✓
 *
 *  ── Vector potential (for discretely divergence-free face B) ──────
 *  In 2.5D we have B_⊥z (cell-centered) and B_xy (face). A_z gives B_xy:
 *       B_x = +∂A_z/∂y,  B_y = -∂A_z/∂x.
 *  Solve:
 *       A_z(x, y) = y cos α - x sin α + (A / (2π √5)) · cos φ
 *  Verify:
 *       ∂A_z/∂y = cos α + (A/(2π√5)) · (-sin φ) · 4π
 *               = cos α - (2A/√5) sin φ = cos α - sin α · A sin φ  ✓
 *       -∂A_z/∂x = sin α + (A/(2π√5)) · sin φ · 2π
 *               = sin α + (A/√5) sin φ = sin α + cos α · A sin φ   ✓
 *  Face B from corner-sampled A_z:
 *       Bx_face[i, j] = (A_z(x_face_i, y_corner_top) - A_z(x_face_i, y_corner_bot)) / dy
 *       By_face[i, j] = -(A_z(x_corner_right, y_face_j) - A_z(x_corner_left, y_face_j)) / dx
 *  Cell-centered B_z is exact (no vector potential needed in 2D for B_z).
 *  Cell-centered B_x / B_y for the energy budget use the analytic
 *  expression evaluated at the cell center; this is consistent with the
 *  shader's downstream face-average reconstruction to O(dx²).
 */
export function makeAlfvenCpawPreset(n = GRID_N) {
    const gamma = 5.0 / 3.0;
    const L     = 1.0;
    const dx    = L / n;
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);

    // Wave geometry. atan(2) is the canonical Tóth choice — it makes the
    // box accommodate an integer wavelength along k.
    const alpha   = Math.atan(2);
    const cosA    = Math.cos(alpha);     // 1/√5
    const sinA    = Math.sin(alpha);     // 2/√5
    const A       = 0.1;                 // perturbation amplitude
    const B_par   = 1.0;
    const rho0    = 1.0;
    const p0      = 0.1;
    const v_A     = B_par / Math.sqrt(rho0);  // = 1
    const lambda  = 1.0 / Math.sqrt(5.0);     // along k
    const verifyTime = lambda / v_A;          // T = 1/√5 ≈ 0.4472136
    const TWO_PI  = 2.0 * Math.PI;
    // phase(x, y) = 2π · (x + 2 y) — equivalent to 2π ξ/λ.

    // Position helpers. Domain [0, 1]² in physical coords; interior cell
    // (i, j) sits at center ((i-ghost+½)·dx, (j-ghost+½)·dx).
    const xCellOf = (i) => (i - ghost + 0.5) * dx;
    const yCellOf = (j) => (j - ghost + 0.5) * dx;
    // Faces (LEFT/DOWN ownership): Bx_face[i, j] at (x = (i-ghost)·dx,
    //                                                y = (j-ghost+½)·dx).
    //                              By_face[i, j] at (x = (i-ghost+½)·dx,
    //                                                y = (j-ghost)·dx).
    const xFaceOf = (i) => (i - ghost) * dx;
    const yFaceOf = (j) => (j - ghost) * dx;

    // Analytic A_z at a point. Used to derive face B via finite differences
    // at the same locations the discrete CT update would interpret them —
    // this guarantees ∇·B = 0 to machine precision on the discrete face grid.
    const Az = (x, y) => {
        const phi = TWO_PI * (x + 2.0 * y);
        return y * cosA - x * sinA + (A / (TWO_PI * Math.sqrt(5.0))) * Math.cos(phi);
    };

    // Cell-centered primitives + total B (analytic, used for energy).
    for (let j = 0; j < nT; j++) {
        const y_c = yCellOf(j);
        for (let i = 0; i < nT; i++) {
            const x_c = xCellOf(i);
            const phi = TWO_PI * (x_c + 2.0 * y_c);
            const sphi = Math.sin(phi);
            const cphi = Math.cos(phi);
            const bx_c = cosA - sinA * A * sphi;
            const by_c = sinA + cosA * A * sphi;
            const bz   = A * cphi;
            const vx   =  sinA * A * sphi;
            const vy   = -cosA * A * sphi;
            const vz   = -A * cphi;
            writeCell(U0, U1, cellIdx(i, j, nT),
                      rho0, vx, vy, vz, p0, bx_c, by_c, bz, gamma);
        }
    }

    // Face B from discrete curl of A_z — guarantees ∇·B_face = 0 to fp32
    // noise. The "corner" locations are the four corners of the face's
    // owning cell, but for an edge-aligned face the relevant corners are
    // on either side along the perpendicular axis.
    //
    // Bx_face[i, j] lives at x = xFaceOf(i), spans y ∈ [yFaceOf(j),
    // yFaceOf(j+1)] = [(j-ghost)·dx, (j-ghost+1)·dx]:
    //   Bx = (A_z(x, y_top) - A_z(x, y_bot)) / dy
    for (let j = 0; j < nT; j++) {
        const y_bot = yFaceOf(j);
        const y_top = yFaceOf(j + 1);
        for (let i = 0; i <= nT; i++) {
            const x = xFaceOf(i);
            Bx_face[bxFaceIdx(i, j, nT)] = (Az(x, y_top) - Az(x, y_bot)) / dx;
        }
    }
    // By_face[i, j] at y = yFaceOf(j), spans x ∈ [xFaceOf(i), xFaceOf(i+1)]:
    //   By = -(A_z(x_right, y) - A_z(x_left, y)) / dx
    for (let j = 0; j <= nT; j++) {
        const y = yFaceOf(j);
        for (let i = 0; i < nT; i++) {
            const x_left  = xFaceOf(i);
            const x_right = xFaceOf(i + 1);
            By_face[byFaceIdx(i, j, nT)] = -(Az(x_right, y) - Az(x_left, y)) / dx;
        }
    }

    return {
        id: 'alfven-cpaw', label: 'Circularly polarized Alfvén wave',
        gamma,
        domainLength: L,
        eta: 0,
        bc: {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        physics: { physicsFlags: BASE_PHYSICS_FLAGS },
        data: { U0, U1, Bx_face, By_face },
        // |B| view ranges 0..√(1+A²) ≈ 1.005; bracket the steady value.
        viewMin: 0.95, viewMax: 1.05,
        verifyTime,
        // Convergence-test metadata for tests/alfven-convergence.html.
        cpaw: { alpha, cosA, sinA, A, lambda, B_par, rho0, p0, v_A, gamma },
    };
}

/**
 * Linear acoustic wave — pure hydro convergence test (B ≡ 0).
 *
 * Smooth periodic 1D acoustic wave on a 2D periodic domain. Used to
 * isolate Euler-side convergence behavior from MHD-only paths (HLLD full
 * 5-wave, BS-only EMF, CT face-B update). If CPAW (MHD) shows degraded
 * order but this test recovers textbook order, the bug is MHD-specific.
 *
 *  ── Setup ─────────────────────────────────────────────────────────────
 *  Domain          : [0, 1]² square, all periodic BCs.
 *  Background      : ρ₀ = 1, p₀ = 1, v = 0, B = 0.  γ = 5/3.
 *                    Sound speed c_s = √(γ p₀/ρ₀) = √(5/3) ≈ 1.2910.
 *  Wave direction  : axis-aligned, +x. Wavelength λ = 1 (one wave fits
 *                    in the box along x); axis-aligned is simpler than
 *                    the tilted CPAW geometry — we already know tilt
 *                    doesn't move the slope much, and axis-aligned makes
 *                    the comparison to the CPAW slope cleaner.
 *  Amplitude       : A = 1e-3. Smaller than CPAW's 0.1 to stay deeply
 *                    linear — hydro acoustic waves steepen into N-waves
 *                    much faster than Alfvén waves (Riemann invariant
 *                    J+ = v + 2c/(γ-1) goes nonlinear at amplitude ~ε³
 *                    for ε = A/p₀).
 *  Perturbation    : right-going linear acoustic mode (Riemann invariant
 *                    J+ constant; J- carries the perturbation):
 *                       ρ' = A · sin(2π x / λ)
 *                       v_x' = c_s · A / ρ₀ · sin(2π x / λ)
 *                            = c_s · A · sin(2π x / λ)
 *                       p' = c_s² · A · sin(2π x / λ)
 *                       v_y' = v_z' = 0
 *                       B = 0
 *  Period          : T = λ / c_s = 1/√(5/3) = √(3/5) ≈ 0.7746.
 *  Analytic at t=T : same as IC (returned to start by periodicity).
 *
 *  ── Why this isolates Euler ───────────────────────────────────────────
 *  B = 0 everywhere → face B = 0 → cell B = 0. HLLD Branch A's
 *  Bn²-comparison degenerates (Bn = 0 trivially), falling into HLLC.
 *  Compute-EMF reads cell-centered vy·Bx - vx·By = 0 and face Bx/By = 0,
 *  so Ez_corner = 0; the CT face-B update writes 0 - 0 = 0. The only
 *  active path is: PPM → HLLC → update-conserved-weighted → energy floor.
 */
export function makeAcousticWaveHydroPreset(n = GRID_N, amplitudeOverride) {
    const gamma = 5.0 / 3.0;
    const L     = 1.0;
    const dx    = L / n;
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);

    const rho0   = 1.0;
    const p0     = 1.0;
    const A      = (typeof amplitudeOverride === 'number' && isFinite(amplitudeOverride))
                       ? amplitudeOverride : 1e-3;
    const lambda = 1.0;
    const cs     = Math.sqrt(gamma * p0 / rho0);  // ≈ 1.2910
    const TWO_PI = 2.0 * Math.PI;
    const k      = TWO_PI / lambda;
    const verifyTime = lambda / cs;               // T = √(3/5) ≈ 0.7746

    const xCellOf = (i) => (i - ghost + 0.5) * dx;

    // Cell-centered primitives. B ≡ 0 everywhere.
    for (let j = 0; j < nT; j++) {
        for (let i = 0; i < nT; i++) {
            const x   = xCellOf(i);
            const s   = Math.sin(k * x);
            const rho = rho0 + A * s;
            const vx  = cs * A * s;
            const p   = p0 + cs * cs * A * s;
            writeCell(U0, U1, cellIdx(i, j, nT),
                      rho, vx, 0, 0, p, 0, 0, 0, gamma);
        }
    }

    // Face B already zero from Float32Array init — explicit no-op.

    return {
        id: 'acoustic-wave-hydro', label: 'Linear acoustic wave (hydro)',
        gamma,
        domainLength: L,
        eta: 0,
        bc: {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        physics: { physicsFlags: BASE_PHYSICS_FLAGS },
        data: { U0, U1, Bx_face, By_face },
        // ρ ranges 1 ± A → bracket the background tightly.
        viewMin: rho0 - 2 * A, viewMax: rho0 + 2 * A,
        verifyTime,
        // Convergence-test metadata for tests/acoustic-convergence.html.
        acoustic: { A, lambda, cs, rho0, p0, gamma },
    };
}

/**
 * Hall whistler dispersion test (Session 15).
 *
 * Right-hand circularly polarized whistler wave on a uniform background.
 * In pure Hall MHD, the dispersion relation is
 *
 *   ω² = k² v_A² (1 + (k · d_i)²)
 *
 * (Tóth, Ma, Gombosi 2008, eq 11). With v_A = 1 and k·d_i ~ 1 the whistler
 * branch deviates strongly from the Alfvén branch ω = k·v_A — that's the
 * signature the test is designed to expose.
 *
 * Setup:
 *   ρ = 1, p = 1, γ = 5/3 ⇒ c_s² = 5/3.
 *   B₀ = (1, 0, 0).  v_A = |B|/√ρ = 1.
 *   Perturbation: right-circular in y/z plane carried by a single mode
 *   with k_n wavelengths per box.
 *     δB_y =  A·cos(k·x)        δB_z =  A·sin(k·x)
 *     δv_y = -A·cos(k·x)/√ρ     δv_z = -A·sin(k·x)/√ρ
 *   A = 1e-3 (deep linear). k_n = 4. d_i = 0.05 ⇒ k·d_i ≈ 1.26 at k_n = 4.
 */
export function makeHallWhistlerPreset(n = GRID_N) {
    const gamma = 5.0 / 3.0;
    const L     = 1.0;
    const dx    = L / n;
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);

    const rho0 = 1.0;
    const p0   = 1.0;
    const B0   = 1.0;
    const A    = 1e-3;
    const k_n  = 4;
    const TWO_PI = 2.0 * Math.PI;
    const k    = TWO_PI * k_n / L;
    const d_i  = 0.05;
    const v_A  = B0 / Math.sqrt(rho0);
    const omega_whistler = k * v_A * Math.sqrt(1.0 + (k * d_i) ** 2);

    const xCellOf = (i) => (i - ghost + 0.5) * dx;
    const xFaceOf = (i) => (i - ghost) * dx;

    for (let j = 0; j < nT; j++) {
        for (let i = 0; i < nT; i++) {
            const x = xCellOf(i);
            const c = Math.cos(k * x);
            const s = Math.sin(k * x);
            const bx_c = B0;
            const by_c = A * c;
            const bz   = A * s;
            const vy   = -A * c / Math.sqrt(rho0);
            const vz   = -A * s / Math.sqrt(rho0);
            writeCell(U0, U1, cellIdx(i, j, nT),
                      rho0, 0, vy, vz, p0, bx_c, by_c, bz, gamma);
        }
    }

    // Face B: Bx is uniform B0; By is sinusoidal in x at face locations.
    for (let j = 0; j < nT; j++) {
        for (let i = 0; i <= nT; i++) {
            Bx_face[bxFaceIdx(i, j, nT)] = B0;
        }
    }
    for (let j = 0; j <= nT; j++) {
        for (let i = 0; i < nT; i++) {
            // By_face at corner-cell midpoint along x = xCellOf(i) is fine
            // because the perturbation is independent of y.
            const x = xCellOf(i);
            By_face[byFaceIdx(i, j, nT)] = A * Math.cos(k * x);
        }
    }

    return {
        id: 'hall-whistler', label: 'Hall whistler dispersion',
        gamma,
        domainLength: L,
        eta: 0,
        bc: {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        physics: {
            physicsFlags: BASE_PHYSICS_FLAGS | FLAG_HALL,
            hallDi: d_i,
            hallSubstepsMax: 8,
            hallElectronPressureFrac: 0.0,
            // Defaults for the rest stay quiet.
        },
        data: { U0, U1, Bx_face, By_face },
        viewMode: 3,  // VIEW_BMAG — wave amplitude is visible in |B|
        viewMin: B0 - 3 * A, viewMax: B0 + 3 * A,
        verifyTime: TWO_PI / omega_whistler,
        whistler: { k, d_i, v_A, omega_alfven: k * v_A, omega_whistler, A, k_n },
    };
}

/**
 * Thermal conduction front (Session 15).
 *
 * Isolated hot spot in an otherwise uniform medium with a uniform
 * magnetic field along x̂. Anisotropic Spitzer conduction with
 * κ_⊥ / κ_∥ = 0.1 should spread the spot ~10× faster along x than along
 * y. The 1D analytic Green's function for unbounded heat diffusion is
 * a Gaussian of width σ(t) = √(σ₀² + 2χ·t) with χ = (γ−1)·κ/ρ.
 *
 * Setup:
 *   ρ = 1, B = (1, 0, 0), v = 0, γ = 5/3.
 *   T_cold = 1, T_hot = 2.
 *   p(x, y) = ρ · T_cold + (T_hot − T_cold)·exp(−r²/2σ₀²), σ₀ = 0.05·L.
 */
export function makeConductionFrontPreset(n = GRID_N) {
    const gamma = 5.0 / 3.0;
    const L     = 1.0;
    const dx    = L / n;
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);

    const rho0 = 1.0;
    const T_cold = 1.0;
    const T_hot  = 2.0;
    const sigma0 = 0.05 * L;
    const B0     = 1.0;
    const cx = 0.5 * L;
    const cy = 0.5 * L;

    const xCellOf = (i) => (i - ghost + 0.5) * dx;
    const yCellOf = (j) => (j - ghost + 0.5) * dx;

    for (let j = 0; j < nT; j++) {
        const y = yCellOf(j);
        const dy = y - cy;
        for (let i = 0; i < nT; i++) {
            const x = xCellOf(i);
            const dxr = x - cx;
            const r2 = dxr * dxr + dy * dy;
            const T = T_cold + (T_hot - T_cold) * Math.exp(-r2 / (2 * sigma0 * sigma0));
            const p = rho0 * T;
            writeCell(U0, U1, cellIdx(i, j, nT),
                      rho0, 0, 0, 0, p, B0, 0, 0, gamma);
        }
    }

    for (let j = 0; j < nT; j++) {
        for (let i = 0; i <= nT; i++) {
            Bx_face[bxFaceIdx(i, j, nT)] = B0;
        }
    }
    // By_face stays zero.

    return {
        id: 'conduction-front', label: 'Anisotropic conduction front',
        gamma,
        domainLength: L,
        eta: 0,
        bc: {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        physics: {
            physicsFlags: BASE_PHYSICS_FLAGS | FLAG_CONDUCTION,
            conductionKappa:   1.0e-2,
            conductionIsoFrac: 0.1,
            conductionSatFrac: 0.3,
        },
        data: { U0, U1, Bx_face, By_face },
        viewMode: 5,  // VIEW_T — temperature spread is the test signal
        viewMin: T_cold, viewMax: T_hot,
        verifyTime: 0.1,
        conductionFront: {
            T_cold, T_hot, sigma0, kappa_par: 1.0e-2, iso_frac: 0.1,
        },
    };
}

/**
 * Thermal cooling instability (Session 15).
 *
 * Uniform gas with multi-mode density perturbations under bremsstrahlung-
 * shape cooling Λ(T) = Λ₀ · √(T/T_ref). For Λ above the local thermal
 * timescale the gas fragments: hotter regions cool faster (dE/dt = -ρ² Λ
 * but in the bremsstrahlung-shape we use, the rate goes as √T, so the
 * cooling time t_cool = p/(γ−1) / (ρ² Λ) = T/((γ−1) ρ Λ₀ √(T/T_ref))
 * decreases as √T — hotter loses energy faster, driving condensation).
 *
 * Setup:
 *   ρ = 1 + A·noise(x, y), p = 1, B = 0, v = 0, γ = 5/3.
 *   Λ₀ = 0.1 (supercritical for the box geometry).
 *
 * The noise is deterministic — a sum of three low-k modes — so the test
 * is reproducible.
 */
export function makeCoolingInstabilityPreset(n = GRID_N) {
    const gamma = 5.0 / 3.0;
    const L     = 1.0;
    const dx    = L / n;
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);

    const rho0 = 1.0;
    const p0   = 1.0;
    const A    = 0.05;
    const TWO_PI = 2.0 * Math.PI;
    const xCellOf = (i) => (i - ghost + 0.5) * dx;
    const yCellOf = (j) => (j - ghost + 0.5) * dx;

    const primAt = (iw, jw) => {
        const x = (iw + 0.5) * dx;
        const y = (jw + 0.5) * dx;
        // Three coprime modes — interferes into a clumpy field.
        const s = Math.sin(TWO_PI * 3 * x) * Math.cos(TWO_PI * 2 * y)
                + Math.sin(TWO_PI * 5 * (x + y)) * 0.7
                + Math.cos(TWO_PI * 4 * (x - y)) * 0.5;
        return {
            rho: rho0 * (1 + A * s),
            vx: 0, vy: 0, vz: 0,
            p: p0,
            bx_c: 0, by_c: 0, bz: 0,
        };
    };
    fillCellGhostPeriodic(U0, U1, gamma, nT, ghost, n, primAt);
    // B ≡ 0 → all face arrays stay zero.

    return {
        id: 'cooling-instability', label: 'Cooling instability',
        gamma,
        domainLength: L,
        eta: 0,
        bc: {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        physics: {
            physicsFlags: BASE_PHYSICS_FLAGS | FLAG_COOLING,
            coolingLambda0: 0.1,
            coolingTFloor:  1.0e-3,
            coolingTRef:    1.0,
            coolingCurveMode: COOLING_CURVE_TABLE,
        },
        data: { U0, U1, Bx_face, By_face },
        viewMode: 0,  // VIEW_DENSITY — fragmentation shows up as ρ clumps
        viewMin: rho0 * (1 - 2 * A), viewMax: rho0 * (1 + 2 * A),
        verifyTime: 0.5,
    };
}

/**
 * Jeans instability (Session 15).
 *
 * Small-amplitude density perturbation under self-gravity. Linear
 * dispersion for hydrodynamic Jeans modes:
 *
 *   ω² = k² c_s² − 4πG ρ₀
 *
 * Unstable (ω² < 0 ⇒ exponential growth) when
 *   k < k_J ≡ √(4πG ρ₀ / c_s²)    (wavelength > λ_J)
 *
 * Setup chosen so the chosen mode is solidly unstable:
 *   ρ = ρ₀ · (1 + A·sin(2π x/L)), A = 1e-2, λ = L = 1.
 *   c_s² = γ p₀ / ρ₀ = 5/3, k = 2π/L.
 *   With G = 10:
 *     4πG ρ₀ = 40π ≈ 125.7,    k² c_s² = (2π)²·5/3 ≈ 65.8.
 *     ω² = 65.8 − 125.7 ≈ −59.9  ⇒  growth rate γ_g = √59.9 ≈ 7.74.
 */
export function makeJeansInstabilityPreset(n = GRID_N) {
    const gamma = 5.0 / 3.0;
    const L     = 1.0;
    const dx    = L / n;
    const { nT, ghost, U0, U1, Bx_face, By_face } = allocData(n);

    const rho0 = 1.0;
    const p0   = 1.0;
    const A    = 1.0e-2;
    const TWO_PI = 2.0 * Math.PI;
    const k    = TWO_PI / L;
    const G    = 10.0;
    const cs2  = gamma * p0 / rho0;
    const omega2 = k * k * cs2 - 4 * Math.PI * G * rho0;
    const growthRate = omega2 < 0 ? Math.sqrt(-omega2) : 0;
    const lambdaJ = Math.sqrt(Math.PI * cs2 / (G * rho0));

    const xCellOf = (i) => (i - ghost + 0.5) * dx;

    for (let j = 0; j < nT; j++) {
        for (let i = 0; i < nT; i++) {
            const x = xCellOf(i);
            const rho = rho0 * (1 + A * Math.sin(k * x));
            writeCell(U0, U1, cellIdx(i, j, nT),
                      rho, 0, 0, 0, p0, 0, 0, 0, gamma);
        }
    }
    // B ≡ 0.

    return {
        id: 'jeans-instability', label: 'Jeans instability',
        gamma,
        domainLength: L,
        eta: 0,
        bc: {
            modeN: BC_PERIODIC, modeS: BC_PERIODIC,
            modeE: BC_PERIODIC, modeW: BC_PERIODIC,
        },
        physics: {
            physicsFlags: BASE_PHYSICS_FLAGS | FLAG_GRAVITY_SELF,
            gravityG: G,
            gravityPoissonIters: 64,
        },
        data: { U0, U1, Bx_face, By_face },
        viewMode: 0,  // VIEW_DENSITY — growth is in ρ
        viewMin: rho0 * (1 - 3 * A), viewMax: rho0 * (1 + 3 * A),
        // Should grow by ~e in t ≈ 1/γ_g.
        verifyTime: growthRate > 0 ? 1.0 / growthRate : 1.0,
        jeans: { k, G, cs2, growthRate, lambdaJ, A },
    };
}

export const PRESETS = {
    sod: makeSodPreset,
    'brio-wu': makeBrioWuPreset,
    'orszag-tang': makeOrszagTangPreset,
    'orszag-tang-extended': makeOrszagTangExtendedPreset,
    'harris': makeHarrisPreset,
    'alfven-cpaw': makeAlfvenCpawPreset,
    'acoustic-wave-hydro': makeAcousticWaveHydroPreset,
    'hall-whistler': makeHallWhistlerPreset,
    'conduction-front': makeConductionFrontPreset,
    'cooling-instability': makeCoolingInstabilityPreset,
    'jeans-instability': makeJeansInstabilityPreset,
};

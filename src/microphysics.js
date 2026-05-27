/**
 * @fileoverview Tabulated microphysics closures for the extended plasma model.
 *
 * The shader table is deliberately small and dimensionless. Temperature is
 * represented as theta = T / T_ref; values are relative coefficients consumed
 * by the WGSL source terms. The shape follows standard optically thin CIE
 * cooling, ionization turnover, Spitzer resistivity / conduction scaling, and
 * Braginskii-like viscosity scaling. Presets own the dimensional calibration
 * through T_ref and the source coefficients in Uniforms.
 */

export const MICRO_TABLE_ENTRIES = 64;
export const MICRO_STRIDE = 4;

const LOG10 = Math.log(10);

function slope(a, b) {
    return (b.logValue - a.logValue) / Math.max(b.logTheta - a.logTheta, 1e-12);
}

function fillRows(out, start, rows) {
    for (let i = 0; i < rows.length; i++) {
        const base = (start + i) * MICRO_STRIDE;
        const m = i + 1 < rows.length ? slope(rows[i], rows[i + 1]) : rows[i].tailSlope;
        out[base + 0] = rows[i].logTheta;
        out[base + 1] = rows[i].logValue;
        out[base + 2] = Number.isFinite(m) ? m : 0;
        out[base + 3] = rows[i].aux ?? 0;
    }
}

function row(theta, value, tailSlope = 0, aux = 0) {
    return {
        logTheta: Math.log(theta) / LOG10,
        logValue: Math.log(Math.max(value, 1e-30)) / LOG10,
        tailSlope,
        aux,
    };
}

/**
 * Layout, 16 rows per family:
 *   0..15   cooling Lambda(theta) relative to cooling_lambda0
 *   16..31  neutral fraction multiplier f_n(theta)
 *   32..47  Spitzer magnetic diffusivity eta(theta) relative to eta
 *   48..63  transport scale: kappa~T^(5/2), viscosity~T^(5/2)
 */
export function buildMicrophysicsTable() {
    const out = new Float32Array(MICRO_TABLE_ENTRIES * MICRO_STRIDE);

    fillRows(out, 0, [
        row(1.0e-4, 0.004, 0.40),
        row(3.0e-4, 0.007, 0.85),
        row(1.0e-3, 0.020, 1.45),
        row(3.0e-3, 0.100, 1.35),
        row(1.0e-2, 0.500, 1.45),
        row(3.0e-2, 2.500, 0.58),
        row(1.0e-1, 5.000, -0.63),
        row(3.0e-1, 2.500, -0.47),
        row(1.0e0,  1.400, -0.31),
        row(3.0e0,  1.000, 0.19),
        row(1.0e1,  1.250, 0.33),
        row(3.0e1,  1.800, 0.43),
        row(1.0e2,  3.000, 0.50),
        row(3.0e2,  5.200, 0.50),
        row(1.0e3,  9.000, 0.50),
        row(3.0e3, 15.600, 0.50),
    ]);

    fillRows(out, 16, [
        row(1.0e-4, 1.000, -0.02),
        row(3.0e-4, 0.980, -0.05),
        row(1.0e-3, 0.920, -0.14),
        row(3.0e-3, 0.780, -0.35),
        row(1.0e-2, 0.520, -0.75),
        row(3.0e-2, 0.220, -1.20),
        row(1.0e-1, 0.055, -1.55),
        row(3.0e-1, 0.010, -1.80),
        row(1.0e0,  0.0015, -2.00),
        row(3.0e0,  2.0e-4, -2.00),
        row(1.0e1,  2.0e-5, -2.00),
        row(3.0e1,  2.2e-6, -2.00),
        row(1.0e2,  2.5e-7, -2.00),
        row(3.0e2,  2.8e-8, -2.00),
        row(1.0e3,  3.0e-9, -2.00),
        row(3.0e3,  3.0e-10, -2.00),
    ]);

    fillRows(out, 32, [
        row(1.0e-4, 1.0e6, -1.5),
        row(3.0e-4, 1.9e5, -1.5),
        row(1.0e-3, 3.2e4, -1.5),
        row(3.0e-3, 6.1e3, -1.5),
        row(1.0e-2, 1.0e3, -1.5),
        row(3.0e-2, 1.9e2, -1.5),
        row(1.0e-1, 3.2e1, -1.5),
        row(3.0e-1, 6.1, -1.5),
        row(1.0e0,  1.0, -1.5),
        row(3.0e0,  0.192, -1.5),
        row(1.0e1,  0.0316, -1.5),
        row(3.0e1,  0.0061, -1.5),
        row(1.0e2,  0.0010, -1.5),
        row(3.0e2,  1.92e-4, -1.5),
        row(1.0e3,  3.16e-5, -1.5),
        row(3.0e3,  6.09e-6, -1.5),
    ]);

    fillRows(out, 48, [
        row(1.0e-4, 1.0e-10, 2.5),
        row(3.0e-4, 1.56e-9, 2.5),
        row(1.0e-3, 3.16e-8, 2.5),
        row(3.0e-3, 4.93e-7, 2.5),
        row(1.0e-2, 1.0e-5, 2.5),
        row(3.0e-2, 1.56e-4, 2.5),
        row(1.0e-1, 3.16e-3, 2.5),
        row(3.0e-1, 4.93e-2, 2.5),
        row(1.0e0,  1.0, 2.5),
        row(3.0e0,  15.6, 2.5),
        row(1.0e1,  316.0, 2.5),
        row(3.0e1,  4930.0, 2.5),
        row(1.0e2,  1.0e5, 2.5),
        row(3.0e2,  1.56e6, 2.5),
        row(1.0e3,  3.16e7, 2.5),
        row(3.0e3,  4.93e8, 2.5),
    ]);

    return out;
}

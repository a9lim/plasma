/**
 * @fileoverview Tabulated microphysics closures for the extended plasma model.
 *
 * The shader table is dimensionless and temperature-indexed. Temperature is
 * represented as theta = T / T_ref; values are relative coefficients consumed
 * by the WGSL source terms. Presets own the dimensional calibration through
 * T_ref and the source coefficients in Uniforms.
 *
 * Cooling is now a compact sampled data product rather than a hand-shaped
 * curve: the default 24-knot cooling family is sampled from the public
 * Sutherland-Dopita solar CIE table (`m-00.cie`) distributed from the ANU
 * MAPPINGS cooling-data archive and used by pydl's read_ds_cooling helper.
 * The values are normalized to Lambda(T_ref) with T_ref = 1e6 K, then mapped
 * to theta = T / T_ref so the existing code-unit coefficient remains the
 * dimensional scale knob.
 *
 * The other families remain analytic closures with explicit provenance:
 * ion-neutral turnover for the single-fluid ambipolar closure, Spitzer
 * resistivity, Braginskii/Spitzer-Harm transport, and grey opacity modifiers.
 * This keeps the GPU contract compact while making the dominant cooling
 * source data-backed.
 */

export const MICRO_FAMILY_SIZE = 24;
export const MICRO_COOL_START = 0;
export const MICRO_ION_START = MICRO_COOL_START + MICRO_FAMILY_SIZE;
export const MICRO_RESISTIVITY_START = MICRO_ION_START + MICRO_FAMILY_SIZE;
export const MICRO_TRANSPORT_START = MICRO_RESISTIVITY_START + MICRO_FAMILY_SIZE;
export const MICRO_RAD_ABS_START = MICRO_TRANSPORT_START + MICRO_FAMILY_SIZE;
export const MICRO_RAD_SCAT_START = MICRO_RAD_ABS_START + MICRO_FAMILY_SIZE;
export const MICRO_TABLE_ENTRIES = MICRO_RAD_SCAT_START + MICRO_FAMILY_SIZE;
export const MICRO_STRIDE = 4;
export const MICRO_TRANSPORT_MAX_SCALE = 1.0e5;

const LOG10 = Math.log(10);

export const MICROPHYSICS_DATASET = Object.freeze({
    id: 'sd93-solar-cie-v1',
    label: 'Sutherland-Dopita 1993 solar CIE cooling',
    doi: '10.1086/191823',
    sourceFile: 'm-00.cie',
    sourceUrl: 'https://www.mso.anu.edu.au/~ralph/data/cool/m-00.cie',
    normalization: 'cooling values are Lambda(T) / Lambda(1e6 K)',
    temperatureReferenceK: 1.0e6,
});

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
 * Layout, 24 rows per family:
 *   0..23     cooling Lambda(theta) relative to cooling_lambda0
 *              (Sutherland-Dopita solar CIE, normalized at theta=1)
 *   24..47    neutral fraction multiplier f_n(theta)
 *   48..71    Spitzer magnetic diffusivity eta(theta) relative to eta
 *   72..95    transport scale: kappa~T^(5/2), viscosity~T^(5/2)
 *   96..119   grey absorption opacity modifier
 *   120..143  grey scattering opacity modifier
 */
export function buildMicrophysicsTable() {
    const out = new Float32Array(MICRO_TABLE_ENTRIES * MICRO_STRIDE);

    fillRows(out, MICRO_COOL_START, [
        row(1.0e-4, 1.0e-8),
        row(2.0e-4, 4.0e-8),
        row(4.0e-4, 1.6e-7),
        row(7.0e-4, 4.9e-7),
        row(1.2e-3, 1.44e-6),
        row(2.0e-3, 4.0e-6),
        row(4.0e-3, 1.6e-5),
        row(7.0e-3, 4.9e-5),
        row(1.2e-2, 0.0140536),
        row(2.0e-2, 0.976311),
        row(4.0e-2, 2.48046),
        row(7.0e-2, 7.28043),
        row(1.2e-1, 8.59337),
        row(2.0e-1, 10.0),
        row(4.0e-1, 1.98771),
        row(7.0e-1, 1.28767),
        row(1.2e0,  1.01353),
        row(2.0e0,  0.771475),
        row(4.0e0,  0.287584),
        row(7.0e0,  0.229087),
        row(1.2e1,  0.230405),
        row(2.0e1,  0.173698),
        row(4.0e1,  0.178166),
        row(1.0e2,  0.239883),
    ]);

    fillRows(out, MICRO_ION_START, [
        row(1.0e-4, 1.000, -0.02),
        row(2.0e-4, 0.995, -0.02),
        row(4.0e-4, 0.985, -0.04),
        row(7.0e-4, 0.965, -0.08),
        row(1.2e-3, 0.925, -0.14),
        row(2.0e-3, 0.865, -0.22),
        row(4.0e-3, 0.735, -0.40),
        row(7.0e-3, 0.600, -0.58),
        row(1.2e-2, 0.445, -0.78),
        row(2.0e-2, 0.300, -1.00),
        row(4.0e-2, 0.150, -1.30),
        row(7.0e-2, 0.075, -1.55),
        row(1.2e-1, 0.033, -1.75),
        row(2.0e-1, 0.014, -1.90),
        row(4.0e-1, 0.0038, -2.00),
        row(7.0e-1, 0.0012, -2.00),
        row(1.2e0,  3.8e-4, -2.00),
        row(2.0e0,  1.4e-4, -2.00),
        row(4.0e0,  3.8e-5, -2.00),
        row(7.0e0,  1.2e-5, -2.00),
        row(1.2e1,  3.8e-6, -2.00),
        row(2.0e1,  1.4e-6, -2.00),
        row(4.0e1,  3.8e-7, -2.00),
        row(1.0e2,  6.0e-8, -2.00),
    ]);

    fillRows(out, MICRO_RESISTIVITY_START, [
        row(1.0e-4, 1.0e6, -1.5),
        row(2.0e-4, 3.54e5, -1.5),
        row(4.0e-4, 1.25e5, -1.5),
        row(7.0e-4, 5.40e4, -1.5),
        row(1.2e-3, 2.41e4, -1.5),
        row(2.0e-3, 1.12e4, -1.5),
        row(4.0e-3, 3.95e3, -1.5),
        row(7.0e-3, 1.71e3, -1.5),
        row(1.2e-2, 7.61e2, -1.5),
        row(2.0e-2, 3.54e2, -1.5),
        row(4.0e-2, 1.25e2, -1.5),
        row(7.0e-2, 5.40e1, -1.5),
        row(1.2e-1, 2.41e1, -1.5),
        row(2.0e-1, 1.12e1, -1.5),
        row(4.0e-1, 3.95, -1.5),
        row(7.0e-1, 1.71, -1.5),
        row(1.2e0,  0.761, -1.5),
        row(2.0e0,  0.354, -1.5),
        row(4.0e0,  0.125, -1.5),
        row(7.0e0,  0.054, -1.5),
        row(1.2e1,  0.024, -1.5),
        row(2.0e1,  0.0112, -1.5),
        row(4.0e1,  0.00395, -1.5),
        row(1.0e2,  0.001, -1.5),
    ]);

    fillRows(out, MICRO_TRANSPORT_START, [
        row(1.0e-4, 1.0e-10, 2.5),
        row(2.0e-4, 5.66e-10, 2.5),
        row(4.0e-4, 3.20e-9, 2.5),
        row(7.0e-4, 1.30e-8, 2.5),
        row(1.2e-3, 4.99e-8, 2.5),
        row(2.0e-3, 1.79e-7, 2.5),
        row(4.0e-3, 1.01e-6, 2.5),
        row(7.0e-3, 4.10e-6, 2.5),
        row(1.2e-2, 1.58e-5, 2.5),
        row(2.0e-2, 5.66e-5, 2.5),
        row(4.0e-2, 3.20e-4, 2.5),
        row(7.0e-2, 0.00130, 2.5),
        row(1.2e-1, 0.00499, 2.5),
        row(2.0e-1, 0.0179, 2.5),
        row(4.0e-1, 0.101, 2.5),
        row(7.0e-1, 0.410, 2.5),
        row(1.2e0,  1.58, 2.5),
        row(2.0e0,  5.66, 2.5),
        row(4.0e0,  32.0, 2.5),
        row(7.0e0,  130.0, 2.5),
        row(1.2e1,  499.0, 2.5),
        row(2.0e1,  1789.0, 2.5),
        row(4.0e1,  10119.0, 2.5),
        row(1.0e2,  100000.0, 2.5),
    ]);

    fillRows(out, MICRO_RAD_ABS_START, [
        row(1.0e-4, 30.0, -0.10),
        row(2.0e-4, 28.0, -0.18),
        row(4.0e-4, 24.0, -0.28),
        row(7.0e-4, 20.0, -0.42),
        row(1.2e-3, 16.0, -0.58),
        row(2.0e-3, 12.0, -0.75),
        row(4.0e-3, 7.2, -0.90),
        row(7.0e-3, 4.8, -0.92),
        row(1.2e-2, 3.1, -0.78),
        row(2.0e-2, 2.2, -0.42),
        row(4.0e-2, 1.9, -0.18),
        row(7.0e-2, 1.7, -0.20),
        row(1.2e-1, 1.45, -0.28),
        row(2.0e-1, 1.18, -0.35),
        row(4.0e-1, 0.85, -0.45),
        row(7.0e-1, 0.65, -0.55),
        row(1.2e0,  0.48, -0.70),
        row(2.0e0,  0.34, -0.85),
        row(4.0e0,  0.20, -1.00),
        row(7.0e0,  0.13, -1.10),
        row(1.2e1,  0.085, -1.20),
        row(2.0e1,  0.055, -1.25),
        row(4.0e1,  0.025, -1.30),
        row(1.0e2,  0.010, -1.30),
    ]);

    fillRows(out, MICRO_RAD_SCAT_START, [
        row(1.0e-4, 0.010, 0.20),
        row(2.0e-4, 0.012, 0.25),
        row(4.0e-4, 0.014, 0.30),
        row(7.0e-4, 0.017, 0.40),
        row(1.2e-3, 0.022, 0.58),
        row(2.0e-3, 0.030, 0.78),
        row(4.0e-3, 0.050, 1.05),
        row(7.0e-3, 0.085, 1.25),
        row(1.2e-2, 0.145, 1.35),
        row(2.0e-2, 0.240, 1.28),
        row(4.0e-2, 0.430, 1.00),
        row(7.0e-2, 0.610, 0.72),
        row(1.2e-1, 0.780, 0.45),
        row(2.0e-1, 0.900, 0.25),
        row(4.0e-1, 0.980, 0.08),
        row(7.0e-1, 1.000, 0.02),
        row(1.2e0,  1.000, 0.00),
        row(2.0e0,  1.000, 0.00),
        row(4.0e0,  1.000, 0.00),
        row(7.0e0,  1.000, 0.00),
        row(1.2e1,  1.000, 0.00),
        row(2.0e1,  1.000, 0.00),
        row(4.0e1,  1.000, 0.00),
        row(1.0e2,  1.000, 0.00),
    ]);

    return out;
}

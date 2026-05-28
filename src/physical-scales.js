/**
 * @fileoverview Physical-scale and dimensionless-number calibration helpers.
 *
 * The WebGPU solver evolves nondimensional MHD variables, but the extended
 * source stack becomes much more defensible when presets state the physical
 * regime they are imitating and derive code coefficients from dimensionless
 * targets. These helpers keep that conversion in one place.
 */

export const CGS = Object.freeze({
    kB: 1.380649e-16,          // erg K^-1
    mP: 1.67262192369e-24,     // g
    eStatC: 4.80320471257e-10, // statcoulomb
    c: 2.99792458e10,          // cm s^-1
    pi: Math.PI,
    pc: 3.085677581491367e18,  // cm
    microGauss: 1.0e-6,        // G
    year: 3.15576e7,           // s
});

export function makePhysicalScale(opts) {
    const lengthCm = positive(opts.lengthCm, 'lengthCm');
    const numberDensity = positive(opts.numberDensityCm3, 'numberDensityCm3');
    const temperatureK = positive(opts.temperatureK, 'temperatureK');
    const mu = positive(opts.meanMolecularWeight ?? 0.62, 'meanMolecularWeight');
    const magneticFieldG = Math.max(0, opts.magneticFieldG ?? 0);
    const rhoCgs = numberDensity * mu * CGS.mP;
    const pressureCgs = numberDensity * CGS.kB * temperatureK;
    const velocityCms = Math.sqrt(pressureCgs / rhoCgs);
    const timeS = lengthCm / velocityCms;
    const magneticFieldUnitG = Math.sqrt(4 * CGS.pi * rhoCgs) * velocityCms;
    const ionInertialLengthCm = CGS.c
        * Math.sqrt(mu * CGS.mP / (4 * CGS.pi * numberDensity * CGS.eStatC * CGS.eStatC));
    return Object.freeze({
        label: opts.label ?? 'physical scale',
        lengthCm,
        numberDensityCm3: numberDensity,
        temperatureK,
        meanMolecularWeight: mu,
        rhoCgs,
        pressureCgs,
        velocityCms,
        timeS,
        magneticFieldG,
        magneticFieldUnitG,
        beta: magneticFieldG > 0
            ? 8 * CGS.pi * pressureCgs / (magneticFieldG * magneticFieldG)
            : Infinity,
        alfvenMachForCodeVelocity1: magneticFieldG > 0
            ? velocityCms / (magneticFieldG / Math.sqrt(4 * CGS.pi * rhoCgs))
            : Infinity,
        ionInertialLengthCm,
        ionInertialLengthCode: ionInertialLengthCm / lengthCm,
    });
}

export function deriveCodePhysicsFromTargets(target) {
    const rho = positive(target.rhoRef ?? 1, 'rhoRef');
    const p = positive(target.pRef ?? 1, 'pRef');
    const gamma = positive(target.gamma ?? (5 / 3), 'gamma');
    const velocity = positive(target.velocityRef ?? 1, 'velocityRef');
    const length = positive(target.lengthRef ?? 1, 'lengthRef');
    const crossingTime = length / velocity;
    const coeffs = {};

    if (target.magneticReynolds !== undefined) {
        coeffs.eta = velocity * length / positive(target.magneticReynolds, 'magneticReynolds');
    }
    if (target.peclet !== undefined) {
        const chi = velocity * length / positive(target.peclet, 'peclet');
        coeffs.conductionKappa = chi * rho / Math.max(gamma - 1, 1e-30);
    }
    if (target.reynolds !== undefined) {
        coeffs.viscosityNu = velocity * length / positive(target.reynolds, 'reynolds');
    }
    if (target.bulkReynolds !== undefined) {
        coeffs.viscosityBulk = velocity * length / positive(target.bulkReynolds, 'bulkReynolds');
    }
    if (target.shockReynolds !== undefined) {
        coeffs.viscosityShock = velocity * length / positive(target.shockReynolds, 'shockReynolds');
    }
    if (target.ambipolarRm !== undefined) {
        coeffs.ambipolarEta = velocity * length / positive(target.ambipolarRm, 'ambipolarRm');
    }
    if (target.coolingTimeCrossings !== undefined) {
        const tCool = positive(target.coolingTimeCrossings, 'coolingTimeCrossings') * crossingTime;
        coeffs.coolingLambda0 = (p / Math.max(gamma - 1, 1e-30)) / (rho * rho * tCool);
    }
    if (target.heatingBalanceFrac !== undefined && coeffs.coolingLambda0 !== undefined) {
        coeffs.heatingGamma0 = Math.max(0, target.heatingBalanceFrac) * rho * rho * coeffs.coolingLambda0;
    }
    if (target.hallLengthFrac !== undefined) {
        coeffs.hallDi = Math.max(0, target.hallLengthFrac) * length;
    }
    if (target.biermannFieldGrowthTimeCrossings !== undefined) {
        coeffs.biermannCoeff = 1 / positive(target.biermannFieldGrowthTimeCrossings,
                                            'biermannFieldGrowthTimeCrossings') / crossingTime;
    }
    return Object.freeze({
        coefficients: Object.freeze(coeffs),
        dimensionless: Object.freeze({
            magneticReynolds: target.magneticReynolds,
            peclet: target.peclet,
            reynolds: target.reynolds,
            bulkReynolds: target.bulkReynolds,
            shockReynolds: target.shockReynolds,
            ambipolarRm: target.ambipolarRm,
            coolingTimeCrossings: target.coolingTimeCrossings,
            hallLengthFrac: target.hallLengthFrac,
            biermannFieldGrowthTimeCrossings: target.biermannFieldGrowthTimeCrossings,
        }),
    });
}

export function applyInteractiveBounds(coeffs, bounds = {}) {
    const out = { ...coeffs };
    for (const [key, limit] of Object.entries(bounds)) {
        if (out[key] === undefined) continue;
        const min = limit.min ?? -Infinity;
        const max = limit.max ?? Infinity;
        out[key] = Math.min(max, Math.max(min, out[key]));
    }
    return out;
}

export function serializableScale(scale) {
    return {
        label: scale.label,
        lengthCm: scale.lengthCm,
        numberDensityCm3: scale.numberDensityCm3,
        temperatureK: scale.temperatureK,
        meanMolecularWeight: scale.meanMolecularWeight,
        magneticFieldG: scale.magneticFieldG,
        beta: scale.beta,
        velocityCms: scale.velocityCms,
        timeS: scale.timeS,
        magneticFieldUnitG: scale.magneticFieldUnitG,
        ionInertialLengthCode: scale.ionInertialLengthCode,
    };
}

function positive(value, name) {
    const x = Number(value);
    if (!Number.isFinite(x) || x <= 0) {
        throw new Error(`${name} must be positive, got ${value}`);
    }
    return x;
}

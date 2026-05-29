import { Sim } from '../src/sim.js';
import { PlasmaBuffers } from '../src/gpu/buffers.js';
import { PlasmaRenderer } from '../src/gpu/render.js';
import { VIRIDIS } from '../src/colormaps.js';
import { ReadbackPool, readbackBatch } from '../src/gpu/readback.js';
import { DENSITY_FLOOR } from '../src/config.js';

export function fmt(v, digits = 4) {
    if (!Number.isFinite(v)) return String(v);
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 0.001 && a < 10000) return v.toFixed(digits);
    return v.toExponential(digits);
}

export function createTestCanvas() {
    const canvas = document.createElement('canvas');
    canvas.width = 16;
    canvas.height = 16;
    canvas.style.display = 'none';
    document.body.appendChild(canvas);
    return canvas;
}

export async function createSimAtResolution(device, format, n, preset) {
    const canvas = createTestCanvas();
    const context = canvas.getContext('webgpu');
    context.configure({ device, format, alphaMode: 'premultiplied' });

    const sim = new Sim(device, context, format, { hasTimestamp: false });
    await sim.init();
    sim.setRunning(false);

    sim.n = n;
    sim.n_total = n + 2 * sim.ghost;
    sim.dx = sim.domainLength / n;
    sim.buffers = new PlasmaBuffers(device, n);
    sim.renderer = new PlasmaRenderer(device, context, sim.pipelines, sim.buffers);
    sim.buffers.uploadLUT(VIRIDIS);
    device.queue.writeBuffer(sim.buffers.dt, 0, new Float32Array([1e-4]).buffer);
    sim._pushUniforms();
    sim._pushLicUniforms();

    sim.loadPreset(preset);
    sim._buildBindGroupCache();
    return { sim, canvas };
}

export function stateSizes(n, ghost) {
    const nT = n + 2 * ghost;
    return {
        nT,
        cellsT: nT * nT,
        xfaces: (nT + 1) * nT,
        yfaces: nT * (nT + 1),
    };
}

export async function readState(device, pool, sim) {
    const { cellsT, xfaces, yfaces } = stateSizes(sim.n, sim.ghost);
    const side = sim.buffers._side;
    const U0 = side === 'a' ? sim.buffers.U0_a : sim.buffers.U0_b;
    const U1 = side === 'a' ? sim.buffers.U1_a : sim.buffers.U1_b;
    const Bx = side === 'a' ? sim.buffers.Bx_a : sim.buffers.Bx_b;
    const By = side === 'a' ? sim.buffers.By_a : sim.buffers.By_b;
    const out = await readbackBatch(device, pool, [
        { buf: U0, byteOffset: 0, byteSize: 4 * 4 * cellsT },
        { buf: U1, byteOffset: 0, byteSize: 4 * 4 * cellsT },
        { buf: Bx, byteOffset: 0, byteSize: 4 * xfaces },
        { buf: By, byteOffset: 0, byteSize: 4 * yfaces },
        { buf: sim.buffers.dt, byteOffset: 0, byteSize: 32 },
        { buf: sim.buffers.radiation_E, byteOffset: 0, byteSize: 4 * cellsT },
        { buf: sim.buffers.phi, byteOffset: 0, byteSize: 4 * cellsT },
    ]);
    return {
        U0: new Float32Array(out[0]),
        U1: new Float32Array(out[1]),
        Bx: new Float32Array(out[2]),
        By: new Float32Array(out[3]),
        dt: new Float32Array(out[4]),
        radiation: new Float32Array(out[5]),
        phi: new Float32Array(out[6]),
    };
}

export async function readDt(device, pool, sim) {
    const out = await readbackBatch(device, pool, [
        { buf: sim.buffers.dt, byteOffset: 0, byteSize: 32 },
    ]);
    return new Float32Array(out[0]);
}

export function mirrorDtReadbackIntoSim(sim, dt) {
    if (sim && typeof sim.syncDtFeedback === 'function') {
        sim.syncDtFeedback(dt);
        return;
    }
    if (Number.isFinite(dt[0]) && dt[0] > 0) {
        sim._lastDtHyp = dt[0];
        sim.lastDt = dt[0];
    }
    if (Number.isFinite(dt[1]) && dt[1] > 0) sim._lastDtParabolic = dt[1];
    if (Number.isFinite(dt[2])) {
        sim._lastEtaMax = dt[2];
        sim._lastEtaMaxValid = true;
    }
    if (Number.isFinite(dt[3]) && dt[3] >= 0) {
        sim._lastHallRateMax = dt[3];
        sim._lastHallRateValid = true;
    }
    if (Number.isFinite(dt[4]) && dt[4] >= 0) {
        sim._lastCondRateMax = dt[4];
        sim._lastCondRateValid = true;
    }
}

export async function runForSteps(device, pool, sim, steps, onSample) {
    let t = 0;
    for (let step = 0; step < steps; step++) {
        sim.step();
        await device.queue.onSubmittedWorkDone();
        const dt = await readDt(device, pool, sim);
        mirrorDtReadbackIntoSim(sim, dt);
        if (Number.isFinite(dt[0]) && dt[0] > 0) t += dt[0];
        if (onSample) await onSample({ step: step + 1, t, dt });
    }
    return t;
}

export async function runForTime(device, pool, sim, targetTime, opts = {}) {
    const maxSteps = opts.maxSteps ?? 2000;
    let t = 0;
    let steps = 0;
    while (t < targetTime && steps < maxSteps) {
        sim.step();
        await device.queue.onSubmittedWorkDone();
        const dt = await readDt(device, pool, sim);
        mirrorDtReadbackIntoSim(sim, dt);
        if (Number.isFinite(dt[0]) && dt[0] > 0) t += dt[0];
        steps++;
        if (opts.onSample) await opts.onSample({ step: steps, t, dt });
    }
    return { t, steps };
}

export async function runExtendedSourceOnly(device, sim, dt, steps, opts = {}) {
    const n = Math.max(1, opts.substeps ?? 1);
    const dtSub = dt / n;
    const seed = new Float32Array([dtSub, 0, 0, 0]);
    let t = 0;
    for (let step = 0; step < steps; step++) {
        device.queue.writeBuffer(sim.buffers.hall_dt, 0, seed.buffer);
        device.queue.writeBuffer(sim.buffers.cond_dt, 0, seed.buffer);
        device.queue.writeBuffer(sim.buffers.visc_dt, 0, seed.buffer);
        device.queue.writeBuffer(sim.buffers.nonideal_dt, 0, seed.buffer);
        device.queue.writeBuffer(sim.buffers.rad_dt, 0, seed.buffer);
        device.queue.writeBuffer(sim.buffers.dt_half, 0, seed.buffer);
        const encoder = device.createCommandEncoder({ label: 'plasma.test.sourceOnly.enc' });
        sim._encodeExtendedPhysics(encoder, sim.buffers._side, n, n, n, n, {
            target: 'src',
            dtBuffer: sim.buffers.dt_half,
            radiationSubsteps: n,
            labelPrefix: 'plasma.test.sourceOnly',
        });
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        t += dt;
    }
    return { t, steps };
}

export function summarizeState(sim, state) {
    const n = sim.n;
    const ghost = sim.ghost;
    const nT = n + 2 * ghost;
    const gamma = sim.gamma;
    const pFloor = sim.pressureFloor ?? 1e-6;
    const dx = sim.dx;
    const cIdx = (i, j) => j * nT + i;
    const bxIdx = (i, j) => j * (nT + 1) + i;
    const byIdx = (i, j) => j * nT + i;

    let finite = true;
    let firstBad = null;
    let nanCount = 0;
    let rhoMin = Infinity, rhoMax = -Infinity;
    let pMin = Infinity, pMax = -Infinity, pRawMin = Infinity;
    let vMax = 0, bMax = 0, jMax = 0;
    let divbAbsAvg = 0, divbAbsMax = 0;
    let pFloorCount = 0, rhoFloorCount = 0;
    let mass = 0, energy = 0, eint = 0, entropy = 0, radiation = 0;

    for (let j = ghost; j < ghost + n; j++) {
        for (let i = ghost; i < ghost + n; i++) {
            const c = cIdx(i, j);
            const rhoRaw = state.U0[4 * c + 0];
            const mx = state.U0[4 * c + 1];
            const my = state.U0[4 * c + 2];
            const mz = state.U0[4 * c + 3];
            const E = state.U1[4 * c + 0];
            const bz = state.U1[4 * c + 1];
            const auxE = state.U1[4 * c + 2];
            const auxK = state.U1[4 * c + 3];
            const bx = 0.5 * (state.Bx[bxIdx(i, j)] + state.Bx[bxIdx(i + 1, j)]);
            const by = 0.5 * (state.By[byIdx(i, j)] + state.By[byIdx(i, j + 1)]);
            const vals = [rhoRaw, mx, my, mz, E, bz, bx, by, auxE, auxK];
            for (const v of vals) {
                if (!Number.isFinite(v)) {
                    finite = false;
                    nanCount++;
                    if (!firstBad) firstBad = { i: i - ghost, j: j - ghost, value: v };
                    break;
                }
            }
            if (!Number.isFinite(rhoRaw) || !Number.isFinite(E)) continue;

            const rho = Math.max(rhoRaw, DENSITY_FLOOR);
            const vx = mx / rho;
            const vy = my / rho;
            const vz = mz / rho;
            const ke = 0.5 * (mx * mx + my * my + mz * mz) / rho;
            const mb = 0.5 * (bx * bx + by * by + bz * bz);
            const pRaw = (gamma - 1) * (E - ke - mb);
            const p = Math.max(pRaw, pFloor);
            const vmag = Math.hypot(vx, vy, vz);
            const bmag = Math.hypot(bx, by, bz);
            const jz = ((state.By[byIdx(i + 1, j)] - state.By[byIdx(i - 1, j)])
                      - (state.Bx[bxIdx(i, j + 1)] - state.Bx[bxIdx(i, j - 1)]))
                     / (2 * dx);
            const divb = ((state.Bx[bxIdx(i + 1, j)] - state.Bx[bxIdx(i, j)])
                        + (state.By[byIdx(i, j + 1)] - state.By[byIdx(i, j)])) / dx;

            rhoMin = Math.min(rhoMin, rho);
            rhoMax = Math.max(rhoMax, rho);
            pMin = Math.min(pMin, p);
            pMax = Math.max(pMax, p);
            pRawMin = Math.min(pRawMin, pRaw);
            vMax = Math.max(vMax, vmag);
            bMax = Math.max(bMax, bmag);
            jMax = Math.max(jMax, Math.abs(jz));
            divbAbsAvg += Math.abs(divb);
            divbAbsMax = Math.max(divbAbsMax, Math.abs(divb));
            if (rho <= 1.001 * DENSITY_FLOOR) rhoFloorCount++;
            if (p <= 1.001 * pFloor) pFloorCount++;
            mass += rhoRaw;
            energy += E;
            eint += auxE;
            entropy += auxK;
            if (state.radiation) radiation += state.radiation[c];
        }
    }

    const cellCount = n * n;
    return {
        finite,
        firstBad,
        nanCount,
        rhoMin,
        rhoMax,
        pMin,
        pMax,
        pRawMin,
        vMax,
        bMax,
        jMax,
        divbAbsAvg: divbAbsAvg / cellCount,
        divbAbsMax,
        rhoFloorCount,
        pFloorCount,
        mass,
        energy,
        eint,
        entropy,
        radiation,
    };
}

export function cosineModeAmplitude(sim, state, field, mode = 1) {
    const n = sim.n;
    const ghost = sim.ghost;
    const nT = n + 2 * ghost;
    const cIdx = (i, j) => j * nT + i;
    const twoPi = 2 * Math.PI;
    let csum = 0;
    let ssum = 0;
    for (let j = ghost; j < ghost + n; j++) {
        for (let i = ghost; i < ghost + n; i++) {
            const c = cIdx(i, j);
            const x = (i - ghost + 0.5) / n;
            let value = 0;
            if (field === 'rho') value = state.U0[4 * c + 0];
            else if (field === 'by') {
                const byIdx = (ii, jj) => jj * nT + ii;
                value = 0.5 * (state.By[byIdx(i, j)] + state.By[byIdx(i, j + 1)]);
            } else if (field === 'bz') value = state.U1[4 * c + 1];
            const phase = twoPi * mode * x;
            csum += value * Math.cos(phase);
            ssum += value * Math.sin(phase);
        }
    }
    return 2 * Math.hypot(csum, ssum) / (n * n);
}

export function temperatureMoments(sim, state) {
    const n = sim.n;
    const ghost = sim.ghost;
    const nT = n + 2 * ghost;
    const gamma = sim.gamma;
    const pFloor = sim.pressureFloor ?? 1e-6;
    const cIdx = (i, j) => j * nT + i;
    const bxIdx = (i, j) => j * (nT + 1) + i;
    const byIdx = (i, j) => j * nT + i;
    let wsum = 0, xsum = 0, ysum = 0;
    let x2sum = 0, y2sum = 0;
    let tMin = Infinity, tMax = -Infinity;
    for (let j = ghost; j < ghost + n; j++) {
        const y = (j - ghost + 0.5) / n;
        for (let i = ghost; i < ghost + n; i++) {
            const x = (i - ghost + 0.5) / n;
            const c = cIdx(i, j);
            const rho = Math.max(state.U0[4 * c + 0], DENSITY_FLOOR);
            const mx = state.U0[4 * c + 1];
            const my = state.U0[4 * c + 2];
            const mz = state.U0[4 * c + 3];
            const E = state.U1[4 * c + 0];
            const bz = state.U1[4 * c + 1];
            const bx = 0.5 * (state.Bx[bxIdx(i, j)] + state.Bx[bxIdx(i + 1, j)]);
            const by = 0.5 * (state.By[byIdx(i, j)] + state.By[byIdx(i, j + 1)]);
            const ke = 0.5 * (mx * mx + my * my + mz * mz) / rho;
            const mb = 0.5 * (bx * bx + by * by + bz * bz);
            const p = Math.max((gamma - 1) * (E - ke - mb), pFloor);
            const T = p / rho;
            const w = Math.max(0, T - 1);
            wsum += w;
            xsum += w * x;
            ysum += w * y;
            x2sum += w * x * x;
            y2sum += w * y * y;
            tMin = Math.min(tMin, T);
            tMax = Math.max(tMax, T);
        }
    }
    const inv = wsum > 0 ? 1 / wsum : 0;
    const xbar = xsum * inv;
    const ybar = ysum * inv;
    return {
        tMin,
        tMax,
        weight: wsum,
        varX: x2sum * inv - xbar * xbar,
        varY: y2sum * inv - ybar * ybar,
    };
}

export function makePool(device) {
    return new ReadbackPool(device);
}

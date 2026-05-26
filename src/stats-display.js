/**
 * @fileoverview Stats tab — energy / β / ∇·B / reconnection-rate display.
 *
 * Phase 5 takes a pragmatic approach: instead of building dedicated
 * GPU reduction kernels for every aggregate (which the LIC work in
 * Phase 6 will piggyback on), we readback the interior `U0_n` + `U1_n`
 * + face-B buffers at a low cadence (every 5 frames ≈ 12 Hz) and
 * compute the aggregates on the CPU. At 256² interior the slice is
 * ~640 KB total — comfortable for the 12 Hz cadence, and zero on the
 * 53.33 ms not-readback-frame budget.
 *
 * For 512² and 1024² the readback grows; we drop the cadence to every
 * 10 frames (6 Hz) at 512 and every 20 frames (3 Hz) at 1024.
 *
 * The conservation-drift baseline is captured at the first successful
 * readback and reset whenever the user changes preset/resolution.
 *
 * No `innerHTML` is used; the DOM is created via createElement and
 * mutated via `textContent` per the repo rule.
 *
 * Sparklines: one per primary aggregate (total energy, β-mean, ∇·B).
 * Capacity 240 samples ≈ 20 seconds at 12 Hz. Rendered every readback
 * tick onto a 160×32 DPR-aware canvas.
 */

import { readbackBatch, ReadbackPool } from './gpu/readback.js';
import { GHOST_WIDTH, DENSITY_FLOOR, DT_MIN } from './config.js';

const SPARK_CAP = 240;

/** Helper: create a DOM element with optional class + text. */
function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

/** Stat row builder: returns { row, value }. */
function statRow(label, opts) {
    const cls = opts && opts.sub ? 'stat-row stat-sub' : 'stat-row';
    const row = el('div', cls);
    row.append(el('span', 'stat-label', label));
    const value = el('span', 'stat-value', '—');
    row.append(value);
    return { row, value };
}

/** Group label (matches geon's `.group-label`). */
function groupLabel(text) {
    return el('h2', 'group-label', text);
}

/** Build a sparkline canvas + ring buffer. */
function buildSpark() {
    const wrap = el('div', 'spark-wrap');
    const canvas = el('canvas', 'spark-canvas');
    canvas.width = 320;     // 2× DPR target
    canvas.height = 64;
    canvas.style.width = '160px';
    canvas.style.height = '32px';
    wrap.append(canvas);
    return {
        wrap,
        canvas,
        ctx: canvas.getContext('2d'),
        hist: createSparkHistory(SPARK_CAP),
    };
}

export class StatsDisplay {
    /**
     * @param {{device: GPUDevice, buffers: any, sim: any}} ctx
     * @param {HTMLElement} root  the Stats tab panel into which we mount
     */
    constructor(ctx, root) {
        this.device = ctx.device;
        this.buffers = ctx.buffers;
        this.sim = ctx.sim;
        this.root = root;

        this.pool = new ReadbackPool(this.device);
        this._readbackBusy = false;
        this._frameCounter = 0;
        this._baseline = null;   // { Etot, mass } captured first tick after reset

        this._build();
    }

    /** Re-read buffer handles from sim after a resolution change. */
    bindBuffers(buffers) {
        this.buffers = buffers;
        this._resetBaseline();
    }

    _resetBaseline() {
        this._baseline = null;
        this._consBaseline = null;
        resetSparkHistory(this._sparkEnergy.hist);
        resetSparkHistory(this._sparkBeta.hist);
        resetSparkHistory(this._sparkDivB.hist);
        for (const s of this._consSparks) resetSparkHistory(s.hist);
    }

    _build() {
        // Energy section
        this.root.append(groupLabel('Energy'));
        const eTot   = statRow('Total');
        this._sparkEnergy = buildSpark();
        eTot.row.append(this._sparkEnergy.wrap);
        this.root.append(eTot.row);
        const eKin   = statRow('Kinetic',  { sub: true });
        const eMag   = statRow('Magnetic', { sub: true });
        const eInt   = statRow('Internal', { sub: true });
        const eDrift = statRow('Drift %',  { sub: true });
        this.root.append(eKin.row, eMag.row, eInt.row, eDrift.row);

        // β section
        this.root.append(groupLabel('Plasma β'));
        const bMean = statRow('Mean');
        this._sparkBeta = buildSpark();
        bMean.row.append(this._sparkBeta.wrap);
        const bMin  = statRow('Min', { sub: true });
        const bMax  = statRow('Max', { sub: true });
        this.root.append(bMean.row, bMin.row, bMax.row);

        // Maxima section
        this.root.append(groupLabel('Maxima'));
        const maxB  = statRow('|B|_max');
        const maxV  = statRow('|v|_max');
        const maxJ  = statRow('|J_z|_max');
        this.root.append(maxB.row, maxV.row, maxJ.row);

        // Health counters — the diagnostic page caught these first; keeping
        // them in the main UI makes numerical death visible before the field
        // view turns into a misleading color patch.
        this.root.append(groupLabel('Health'));
        const nanCells = statRow('Nonfinite cells');
        const rhoFloor = statRow('ρ floor cells');
        const pFloor   = statRow('p floor cells');
        const dtMin    = statRow('dt min');
        this.root.append(nanCells.row, rhoFloor.row, pFloor.row, dtMin.row);
        this._healthRefs = {
            nanCells: nanCells.value,
            rhoFloor: rhoFloor.value,
            pFloor: pFloor.value,
            dtMin: dtMin.value,
        };

        // ∇·B section
        this.root.append(groupLabel('Divergence'));
        const divB = statRow('‖∇·B‖₂');
        this._sparkDivB = buildSpark();
        divB.row.append(this._sparkDivB.wrap);
        this.root.append(divB.row);

        // Conservation diagnostics — GPU-side per-step reduction over
        // interior cells. Seven quantities: mass, three momentum
        // components, total energy, magnetic energy, |∇·B| L1.
        // First six show drift % vs. the first sampled baseline (the
        // truth-teller for a research MHD code); divB shows the raw
        // L1 value (CT preserves this at fp32 machine eps, so any
        // drift away from ~1e-6 signals a CT bug). Baseline resets on
        // preset / resolution change via _resetBaseline.
        this.root.append(groupLabel('Conservation'));
        const consSpecs = [
            { key: 'mass',  label: '∫ρ',         drift: true },
            { key: 'momx',  label: '∫ρv_x',      drift: true },
            { key: 'momy',  label: '∫ρv_y',      drift: true },
            { key: 'momz',  label: '∫ρv_z',      drift: true },
            { key: 'eTot',  label: '∫E',         drift: true },
            { key: 'eMag',  label: '∫½|B|²',     drift: true },
            { key: 'divB',  label: '⟨|∇·B|⟩',    drift: false },
        ];
        this._consRefs   = {};
        this._consSparks = [];
        for (const s of consSpecs) {
            const r = statRow(s.label);
            const spark = buildSpark();
            r.row.append(spark.wrap);
            this.root.append(r.row);
            if (s.drift) {
                const d = statRow('Δ %', { sub: true });
                this.root.append(d.row);
                this._consRefs[s.key + 'Drift'] = d.value;
            }
            this._consRefs[s.key] = r.value;
            this._consSparks.push({ key: s.key, ...spark });
        }

        // Reconnection rate (visible only when Harris)
        this._reconWrap = el('div');
        this._reconWrap.append(groupLabel('Reconnection'));
        const rrate = statRow('dΨ/dt');
        this._reconWrap.append(rrate.row);
        this._reconWrap.hidden = true;
        this.root.append(this._reconWrap);

        // Time + step (always visible footer)
        this.root.append(groupLabel('Clock'));
        const tStep = statRow('Step');
        const tCfl  = statRow('CFL');
        const tGpu  = statRow('GPU step');
        const tHall = statRow('Hall substeps');
        const tCond = statRow('Cond substeps');
        this.root.append(tStep.row, tCfl.row, tGpu.row, tHall.row, tCond.row);

        this._refs = {
            eTot: eTot.value, eKin: eKin.value, eMag: eMag.value,
            eInt: eInt.value, eDrift: eDrift.value,
            bMean: bMean.value, bMin: bMin.value, bMax: bMax.value,
            maxB: maxB.value, maxV: maxV.value, maxJ: maxJ.value,
            divB: divB.value, rrate: rrate.value,
            tStep: tStep.value, tCfl: tCfl.value, tGpu: tGpu.value,
            tHall: tHall.value, tCond: tCond.value,
        };
    }

    /** Show/hide the reconnection-rate section based on current preset. */
    updatePresetVisibility(presetName) {
        this._reconWrap.hidden = presetName !== 'harris';
        if (this._reconWrap.hidden) this._reconPsiPrev = null;
        this._resetBaseline();
    }

    /** Called by the UI render loop. Bumps the frame counter; readback at cadence. */
    tick() {
        this._frameCounter += 1;
        const cadence = this._cadence();
        if (this._frameCounter % cadence !== 0) return;
        if (this._readbackBusy) return;
        this._readbackBusy = true;
        this._doReadback().finally(() => { this._readbackBusy = false; });
    }

    _cadence() {
        if (this.sim.n >= 1024) return 20;
        if (this.sim.n >= 512)  return 10;
        return 5;
    }

    async _doReadback() {
        const b = this.buffers;
        const n = this.sim.n;
        const nT = this.sim.n_total;
        const cellsT = nT * nT;
        const xfaceCells = (nT + 1) * nT;
        const yfaceCells = nT * (nT + 1);
        const F32 = 4, V4 = 16;

        const specs = [
            { buf: b.U0_n, byteOffset: 0, byteSize: cellsT * V4 },
            { buf: b.U1_n, byteOffset: 0, byteSize: cellsT * V4 },
            { buf: b.Bx_n, byteOffset: 0, byteSize: xfaceCells * F32 },
            { buf: b.By_n, byteOffset: 0, byteSize: yfaceCells * F32 },
            { buf: b.dt,   byteOffset: 0, byteSize: F32 },
            // Conservation diagnostics — 8 × f32 (7 live + 1 pad). The
            // GPU-side reduction runs at the end of every sim step; we
            // pull the current value at the same cadence as everything
            // else here. Negligible bandwidth: 32 B vs ~640 KB for U/B.
            { buf: b.cons_out, byteOffset: 0, byteSize: 8 * F32 },
        ];
        // Timestamp resolve buffer (2 × u64 = 16 B) when the device
        // supports `timestamp-query`. Batched alongside the other reads so
        // we keep one round-trip per cadence tick.
        const tsIdx = (this.sim._tsResolve) ? specs.length : -1;
        if (tsIdx >= 0) {
            specs.push({ buf: this.sim._tsResolve, byteOffset: 0, byteSize: 16 });
        }

        let bufs;
        try {
            bufs = await readbackBatch(this.device, this.pool, specs);
        } catch (e) {
            console.warn('[plasma.stats] readback failed:', e);
            return;
        }
        const U0 = new Float32Array(bufs[0]);
        const U1 = new Float32Array(bufs[1]);
        const Bxf = new Float32Array(bufs[2]);
        const Byf = new Float32Array(bufs[3]);
        const dtArr = new Float32Array(bufs[4]);
        // Conservation: 8 f32 — [mass, mom_x, mom_y, mom_z, E_tot,
        // E_mag, divB_L1, _pad]. Sums (NOT pre-multiplied by dx²).
        const consArr = new Float32Array(bufs[5]);

        // Decode the two 64-bit timestamps (ns since some device epoch) into
        // a step time in ms. We read as two BigUint64s — the high bits of
        // BigInt cap at 53-bit safe but the absolute timestamp values can
        // exceed that; we subtract first as BigInt and only convert to
        // Number after the diff fits.
        let gpuMs = null;
        if (tsIdx >= 0) {
            try {
                const ts = new BigUint64Array(bufs[tsIdx]);
                const diffNs = ts[1] - ts[0];
                gpuMs = Number(diffNs) / 1_000_000;
                if (Number.isFinite(gpuMs) && gpuMs >= 0) {
                    this.sim._tsLastMs = gpuMs;
                }
            } catch (e) {
                // BigUint64Array might not be available; drop silently.
            }
        }

        this._compute(U0, U1, Bxf, Byf, dtArr[0], n, nT, gpuMs, consArr);
    }

    _compute(U0, U1, Bxf, Byf, dt, n, nT, gpuMs, consArr) {
        const ghost = GHOST_WIDTH;
        const dx = this.sim.dx;
        const dxInv = 1 / dx;
        const cellArea = dx * dx;
        const gammaM1 = this.sim.gamma - 1;

        // Aggregates
        let Ekin = 0, Emag = 0, Eint = 0;
        let betaSum = 0, betaMin = Infinity, betaMax = -Infinity;
        let bMagMax = 0, vMagMax = 0, jzMax = 0;
        let divBSq = 0;
        let nCells = 0;
        let nanCount = 0, rhoFloorCount = 0, pFloorCount = 0;
        const pFloor = this.sim.pressureFloor ?? 1e-6;

        for (let j = ghost; j < ghost + n; j++) {
            for (let i = ghost; i < ghost + n; i++) {
                const idx = j * nT + i;
                const rhoRaw = U0[idx * 4 + 0];
                const rho = Math.max(rhoRaw, DENSITY_FLOOR);
                const mx  = U0[idx * 4 + 1];
                const my  = U0[idx * 4 + 2];
                const mz  = U0[idx * 4 + 3];
                const E   = U1[idx * 4 + 0];
                const Bz  = U1[idx * 4 + 1];

                // Cell-centered face-B average.
                const bxL = Bxf[j * (nT + 1) + i];
                const bxR = Bxf[j * (nT + 1) + (i + 1)];
                const byD = Byf[j * nT + i];
                const byU = Byf[(j + 1) * nT + i];
                const Bx = 0.5 * (bxL + bxR);
                const By = 0.5 * (byD + byU);

                if (![rhoRaw, mx, my, mz, E, Bz, Bx, By].every(Number.isFinite)) {
                    nanCount += 1;
                    continue;
                }

                if (rhoRaw <= 1.001 * DENSITY_FLOOR) rhoFloorCount += 1;

                const vx = mx / rho, vy = my / rho, vz = mz / rho;
                const ke = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
                const mb = 0.5 * (Bx * Bx + By * By + Bz * Bz);
                const pRaw = gammaM1 * (E - ke - mb);
                const p  = Math.max(pRaw, pFloor);
                if (p <= 1.001 * pFloor) pFloorCount += 1;

                Ekin += ke * cellArea;
                Emag += mb * cellArea;
                Eint += (p / gammaM1) * cellArea;

                const beta = 2 * p / Math.max(2 * mb, 1e-12);
                betaSum += beta;
                if (beta < betaMin) betaMin = beta;
                if (beta > betaMax) betaMax = beta;

                const Bmag = Math.sqrt(Bx * Bx + By * By + Bz * Bz);
                if (Bmag > bMagMax) bMagMax = Bmag;
                const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
                if (vMag > vMagMax) vMagMax = vMag;

                // ∇·B at cell — uses face values directly (CT-exact).
                const divB = (bxR - bxL) * dxInv + (byU - byD) * dxInv;
                divBSq += divB * divB;

                // J_z = ∂By/∂x - ∂Bx/∂y, central difference on cell-centered values.
                // We approximate with face deltas: ((By_iR_avg - By_iL_avg)/dx - (Bx_jU_avg - Bx_jD_avg)/dx).
                // For a Phase-5 magnitude estimate this is sufficient; the
                // shader-side J_z uses neighbour cell averages — close enough.
                if (i > ghost && i < ghost + n - 1 && j > ghost && j < ghost + n - 1) {
                    const idxR = j * nT + (i + 1);
                    const idxL = j * nT + (i - 1);
                    const idxU = (j + 1) * nT + i;
                    const idxD = (j - 1) * nT + i;
                    const ByR = 0.5 * (Byf[j * nT + (i + 1)] + Byf[(j + 1) * nT + (i + 1)]);
                    const ByL = 0.5 * (Byf[j * nT + (i - 1)] + Byf[(j + 1) * nT + (i - 1)]);
                    const BxU = 0.5 * (Bxf[(j + 1) * (nT + 1) + i] + Bxf[(j + 1) * (nT + 1) + (i + 1)]);
                    const BxD = 0.5 * (Bxf[(j - 1) * (nT + 1) + i] + Bxf[(j - 1) * (nT + 1) + (i + 1)]);
                    const jz = (ByR - ByL) * 0.5 * dxInv - (BxU - BxD) * 0.5 * dxInv;
                    const ajz = Math.abs(jz);
                    if (ajz > jzMax) jzMax = ajz;
                }

                nCells++;
            }
        }

        const Etot = Ekin + Emag + Eint;
        const divBNorm = Math.sqrt(divBSq * cellArea);

        if (!this._baseline) this._baseline = { Etot };
        const drift = this._baseline.Etot !== 0
            ? 100 * (Etot - this._baseline.Etot) / Math.abs(this._baseline.Etot)
            : 0;

        // Reconnection rate (Harris only): ψ(t) = ∫ Bx dy at x=0, y∈[0, L/2].
        // We approximate by summing Bx along the column at the center of
        // the domain over the upper half-plane interior.
        let psi = 0;
        if (!this._reconWrap.hidden) {
            const iCol = ghost + (n >> 1);
            const jMid = ghost + (n >> 1);
            for (let j = jMid; j < ghost + n; j++) {
                psi += Bxf[j * (nT + 1) + iCol] * dx;
            }
        }
        let rrate = 0;
        if (this._reconPsiPrev !== undefined && this._reconPsiPrev !== null && !this._reconWrap.hidden) {
            // dΨ/dt approximated by (Δψ / Δsteps · dt). We don't have
            // sim-step-time-between-readbacks; use last dt × cadence as
            // a coarse proxy. The number is for "is reconnection
            // happening" not for paper-grade rate.
            const tSpan = Math.max(dt * this._cadence(), 1e-12);
            rrate = (psi - this._reconPsiPrev) / tSpan;
        }
        this._reconPsiPrev = psi;

        // ── Update DOM ───────────────────────────────────────────
        const fmt   = (v) => v.toExponential(3);
        const fmtFx = (v) => v.toFixed(4);

        this._refs.eTot.textContent   = fmt(Etot);
        this._refs.eKin.textContent   = fmt(Ekin);
        this._refs.eMag.textContent   = fmt(Emag);
        this._refs.eInt.textContent   = fmt(Eint);
        this._refs.eDrift.textContent = (drift >= 0 ? '+' : '') + drift.toFixed(2) + '%';

        const betaMean = betaSum / Math.max(nCells, 1);
        this._refs.bMean.textContent = fmtFx(betaMean);
        this._refs.bMin.textContent  = fmt(betaMin);
        this._refs.bMax.textContent  = fmt(betaMax);

        this._refs.maxB.textContent = fmtFx(bMagMax);
        this._refs.maxV.textContent = fmtFx(vMagMax);
        this._refs.maxJ.textContent = fmtFx(jzMax);

        this._refs.divB.textContent = fmt(divBNorm);
        this._refs.rrate.textContent = (rrate >= 0 ? '+' : '') + rrate.toExponential(3);
        this._healthRefs.nanCells.textContent = String(nanCount);
        this._healthRefs.rhoFloor.textContent = String(rhoFloorCount);
        this._healthRefs.pFloor.textContent = String(pFloorCount);
        this._healthRefs.dtMin.textContent = dt <= 1.001 * DT_MIN ? 'yes' : 'no';
        if (nanCount > 0 && this.sim.running) {
            this.sim.setRunning(false);
            console.warn(`[plasma.stats] paused after detecting ${nanCount} nonfinite cells`);
        }

        this._refs.tStep.textContent = String(this.sim.stepCount);
        this._refs.tCfl.textContent  = dt.toExponential(3);
        this._refs.tHall.textContent = String(this.sim._lastHallSubsteps ?? 1);
        this._refs.tCond.textContent = String(this.sim._lastCondSubsteps ?? 1);
        // GPU step time. When timestamp-query isn't supported, we never
        // populate `gpuMs`; fall back to em-dash via the cached refs.
        if (gpuMs != null && Number.isFinite(gpuMs)) {
            this._refs.tGpu.textContent = gpuMs.toFixed(2) + ' ms';
        } else if (this.sim._tsLastMs > 0) {
            this._refs.tGpu.textContent = this.sim._tsLastMs.toFixed(2) + ' ms';
        }

        // Update sparklines.
        pushSparkSample(this._sparkEnergy.hist, Etot);
        pushSparkSample(this._sparkBeta.hist, betaMean);
        pushSparkSample(this._sparkDivB.hist, divBNorm);

        // ── Conservation diagnostics (GPU-reduced) ───────────────
        // consArr layout (from conservation-finalize.wgsl): seven
        // straight sums over interior cells (NOT pre-multiplied by
        // dx²) followed by a pad slot. Scale by cellArea here so the
        // numbers match the "∫" framing in the UI labels and the
        // CPU-computed Etot above (post-scale they're in the same
        // unit system).
        const massSum = consArr[0] * cellArea;
        const momX    = consArr[1] * cellArea;
        const momY    = consArr[2] * cellArea;
        const momZ    = consArr[3] * cellArea;
        const eGpu    = consArr[4] * cellArea;
        const eMagGpu = consArr[5] * cellArea;
        // ∇·B reported as mean L1 per cell — divides the summed |∇·B|
        // by the interior cell count. Independent of dx (the divB
        // shader formula already carries the 1/dx factor). Reads as
        // ~machine-eps when CT is healthy.
        const divBMean = consArr[6] / Math.max(nCells, 1);

        // Capture baseline on the first sample where mass is non-zero —
        // before any sim.step() has run the cons_out buffer is still
        // the WebGPU zero-init, so latching that would make every
        // subsequent drift % read as ±∞.
        if (!this._consBaseline && massSum !== 0 && Number.isFinite(massSum)) {
            this._consBaseline = {
                mass: massSum, momx: momX, momy: momY, momz: momZ,
                eTot: eGpu, eMag: eMagGpu,
            };
        }
        const driftPct = (cur, base) =>
            (base !== 0 && Number.isFinite(base))
                ? 100 * (cur - base) / Math.abs(base)
                : 0;
        const consVals = {
            mass: massSum, momx: momX, momy: momY, momz: momZ,
            eTot: eGpu, eMag: eMagGpu, divB: divBMean,
        };
        for (const k of ['mass', 'momx', 'momy', 'momz', 'eTot', 'eMag']) {
            this._consRefs[k].textContent = fmt(consVals[k]);
            if (this._consBaseline) {
                const d = driftPct(consVals[k], this._consBaseline[k]);
                this._consRefs[k + 'Drift'].textContent = (d >= 0 ? '+' : '') + d.toFixed(3) + '%';
            } else {
                this._consRefs[k + 'Drift'].textContent = '—';
            }
        }
        this._consRefs.divB.textContent = fmt(divBMean);
        for (const s of this._consSparks) pushSparkSample(s.hist, consVals[s.key]);

        this._drawSparks();
    }

    _drawSparks() {
        const style = getComputedStyle(document.documentElement);
        const color = (style.getPropertyValue('--accent') || '#e11107').trim();
        const dim   = (style.getPropertyValue('--text-muted') || '#888').trim();
        const draw = (s) => {
            const w = s.canvas.width, h = s.canvas.height;
            drawSparkline(s.ctx, s.hist, w, h, color, dim);
        };
        draw(this._sparkEnergy);
        draw(this._sparkBeta);
        draw(this._sparkDivB);
        for (const s of this._consSparks) draw(s);
    }
}

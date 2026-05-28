/**
 * @fileoverview Stats tab — energy / β / ∇·B / reconnection-rate display.
 *
 * The heavy aggregates are reduced on the GPU by
 * conservation-{reduce,finalize}.wgsl. This panel reads back only dt,
 * the scalar diagnostics packet, and optional timestamps at a low cadence
 * instead of copying U/B field buffers to JS.
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
import { DT_MIN } from './config.js';

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
        this._bufferGeneration = 0;
        this._baseline = null;   // { Etot, mass } captured first tick after reset

        this._build();
    }

    /** Re-read buffer handles from sim after a resolution change. */
    bindBuffers(buffers) {
        this.buffers = buffers;
        this._bufferGeneration += 1;
        this.pool.destroy();
        this.pool = new ReadbackPool(this.device);
        this._readbackBusy = false;
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
        const tVisc = statRow('Visc substeps');
        const tNonideal = statRow('Nonideal substeps');
        this.root.append(tStep.row, tCfl.row, tGpu.row, tHall.row, tCond.row,
                         tVisc.row, tNonideal.row);

        this._refs = {
            eTot: eTot.value, eKin: eKin.value, eMag: eMag.value,
            eInt: eInt.value, eDrift: eDrift.value,
            bMean: bMean.value, bMin: bMin.value, bMax: bMax.value,
            maxB: maxB.value, maxV: maxV.value, maxJ: maxJ.value,
            divB: divB.value, rrate: rrate.value,
            tStep: tStep.value, tCfl: tCfl.value, tGpu: tGpu.value,
            tHall: tHall.value, tCond: tCond.value,
            tVisc: tVisc.value, tNonideal: tNonideal.value,
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
        const generation = this._bufferGeneration;
        const b = this.buffers;
        const F32 = 4;

        const specs = [
            { buf: b.dt,   byteOffset: 0, byteSize: F32 },
            { buf: b.cons_out, byteOffset: 0, byteSize: 24 * F32 },
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
        if (generation !== this._bufferGeneration) return;
        const dtArr = new Float32Array(bufs[0]);
        const statsArr = new Float32Array(bufs[1]);

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

        this._compute(dtArr[0], gpuMs, statsArr);
    }

    _compute(dt, gpuMs, consArr) {
        const dx = this.sim.dx;
        const cellArea = dx * dx;
        const nCells = Math.max(0, Math.round(consArr[20] || 0));
        const Ekin = consArr[7] * cellArea;
        const Emag = consArr[5] * cellArea;
        const Eint = consArr[8] * cellArea;
        const Etot = Ekin + Emag + Eint;
        const betaSum = consArr[9];
        const betaMean = betaSum / Math.max(nCells, 1);
        const betaMin = nCells > 0 ? consArr[10] : 0;
        const betaMax = nCells > 0 ? consArr[11] : 0;
        const bMagMax = consArr[12];
        const vMagMax = consArr[13];
        const jzMax = consArr[14];
        const divBNorm = Math.sqrt(Math.max(consArr[15] * cellArea, 0));
        const nanCount = Math.round(consArr[16] || 0);
        const rhoFloorCount = Math.round(consArr[17] || 0);
        const pFloorCount = Math.round(consArr[18] || 0);

        if (!this._baseline) this._baseline = { Etot };
        const drift = this._baseline.Etot !== 0
            ? 100 * (Etot - this._baseline.Etot) / Math.abs(this._baseline.Etot)
            : 0;

        // Harris reconnection proxy: shader slot 19 sums Bx along the
        // center column over the upper half-plane; scale by dx here.
        const psi = this._reconWrap.hidden ? 0 : consArr[19] * dx;
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
        this._refs.tVisc.textContent = String(this.sim._lastViscSubsteps ?? 1);
        this._refs.tNonideal.textContent = String(this.sim._lastNonidealSubsteps ?? 1);
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
        // consArr slots 0..6 preserve the conservation-panel contract:
        // straight sums over interior cells (NOT pre-multiplied by dx²).
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

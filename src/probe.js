/**
 * @fileoverview Probe tab — hover-driven cell sampling.
 *
 * The probe shows the local primitive state at whichever interior cell
 * the pointer is currently hovering over the simulation canvas. No
 * click-to-pin, no shift-drag, no crosshair, no sparkline / time-series
 * — those responsibilities moved to the pointer perturbation system in
 * `ui.js` (left-click drag = momentum injection, right-click drag =
 * B-field excite). The Probe owns:
 *
 *   1. Pointer-move handlers on the sim canvas that update the hover
 *      cell index in interior coords.
 *   2. A 10 Hz readback that fetches a 3-row window around the hover
 *      cell, derives the primitive state + J_z + β, and updates DOM.
 *
 * Per-cell readback shape (~30 KB at N=256): a 3-row window of
 * U0/U1/Bx/By in one batched submit. Trivial cost at 10 Hz.
 *
 * No `innerHTML`; DOM built via createElement.
 */

import { readbackBatch, ReadbackPool } from './gpu/readback.js';
import { GHOST_WIDTH } from './config.js';

const PROBE_HZ = 10;          // 100 ms readback interval
const DUAL_ENERGY_FRACTION = 1e-3;

const FIELDS = [
    { id: 'rho',     label: 'ρ' },
    { id: 'vx',      label: 'v_x' },
    { id: 'vy',      label: 'v_y' },
    { id: 'vz',      label: 'v_z' },
    { id: 'bx',      label: 'B_x' },
    { id: 'by',      label: 'B_y' },
    { id: 'bz',      label: 'B_z' },
    { id: 'p',       label: 'p' },
    { id: 'T',       label: 'T' },
    { id: 'jz',      label: 'J_z' },
    { id: 'bmag',    label: '|B|' },
    { id: 'vmag',    label: '|v|' },
    { id: 'beta',    label: 'β' },
];

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

function pressureFromDualEnergy({ E, eAux, ke, mb, gammaM1, pFloor }) {
    const ethFloor = pFloor / Math.max(gammaM1, 1e-6);
    const ethTotal = E - ke - mb;
    const totalOk = Number.isFinite(ethTotal)
        && ethTotal > Math.max(ethFloor, DUAL_ENERGY_FRACTION * Math.max(Math.abs(E), ethFloor));
    const eth = totalOk ? ethTotal : Math.max(eAux, ethFloor);
    return Math.max(gammaM1 * eth, pFloor);
}

export class Probe {
    /**
     * @param {{device: GPUDevice, buffers: any, sim: any, canvas: HTMLCanvasElement}} ctx
     * @param {HTMLElement} root  the Probe tab panel
     */
    constructor(ctx, root) {
        this.device = ctx.device;
        this.buffers = ctx.buffers;
        this.sim = ctx.sim;
        this.simCanvas = ctx.canvas;
        this.root = root;

        this.pool = new ReadbackPool(this.device);
        this._tickHandle = 0;
        this._readbackBusy = false;
        this._bufferGeneration = 0;

        // Hover cell in interior coords [0, n).
        this.cellI = Math.floor(this.sim.n / 2);
        this.cellJ = Math.floor(this.sim.n / 2);
        // rAF coalescer for mousemove → cell updates.
        this._rafPending = false;
        this._pendingClientX = 0;
        this._pendingClientY = 0;

        this._build();
        this._wireInput();
    }

    bindBuffers(buffers) {
        this.buffers = buffers;
        this._bufferGeneration += 1;
        this.pool.destroy();
        this.pool = new ReadbackPool(this.device);
        this._readbackBusy = false;
        // Resolution may have changed; clamp the hover cell into the new
        // interior range so the next readback's stencil math stays valid.
        const n = this.sim.n;
        this.cellI = Math.max(0, Math.min(n - 1, this.cellI));
        this.cellJ = Math.max(0, Math.min(n - 1, this.cellJ));
    }

    /** Convert screen-pixel coords to interior cell index. */
    screenToCell(px, py) {
        const rect = this.simCanvas.getBoundingClientRect();
        const x = px - rect.left;
        const y = py - rect.top;
        if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
            return { i: -1, j: -1, in_bounds: false };
        }
        const n = this.sim.n;
        // Composite blits interior cells full-canvas with y up (MHD layout).
        const i = Math.floor((x / rect.width)  * n);
        const j = Math.floor((1 - y / rect.height) * n);
        return {
            i: Math.max(0, Math.min(n - 1, i)),
            j: Math.max(0, Math.min(n - 1, j)),
            in_bounds: true,
        };
    }

    getCell() { return { i: this.cellI, j: this.cellJ }; }

    start() {
        if (this._tickHandle) return;
        this._tickHandle = setInterval(() => this._tick(), 1000 / PROBE_HZ);
    }

    stop() {
        if (!this._tickHandle) return;
        clearInterval(this._tickHandle);
        this._tickHandle = 0;
    }

    /** Force a single readback now. */
    readback() { this._tick(); }

    _build() {
        // Header: cell coords + position.
        const head = el('div', 'panel-section');
        head.append(el('h2', 'group-label', 'Hover Cell'));
        const cellRow = el('div', 'stat-row');
        cellRow.append(el('span', 'stat-label', 'Index'));
        const cellVal = el('span', 'stat-value', '—');
        cellRow.append(cellVal);
        head.append(cellRow);

        const posRow = el('div', 'stat-row');
        posRow.append(el('span', 'stat-label', 'Position'));
        const posVal = el('span', 'stat-value', '—');
        posRow.append(posVal);
        head.append(posRow);

        const hint = el('p', 'panel-hint',
            'Hover the canvas to sample. Left-drag pushes the plasma; right-drag twists the field.');
        head.append(hint);
        this.root.append(head);

        // State readout block.
        const state = el('div', 'panel-section');
        state.append(el('h2', 'group-label', 'Local State'));
        const refs = {};
        for (const f of FIELDS) {
            const r = el('div', 'stat-row');
            r.append(el('span', 'stat-label', f.label));
            const v = el('span', 'stat-value', '—');
            r.append(v);
            state.append(r);
            refs[f.id] = v;
        }
        this.root.append(state);

        this._refs = { ...refs, cell: cellVal, pos: posVal };
    }

    _wireInput() {
        // pointermove: coalesce via rAF so 60-240 Hz pointer streams don't
        // fight the existing 10 Hz readback. We only update the cell index
        // here; readback is on its own timer.
        const onMove = (e) => {
            this._pendingClientX = e.clientX;
            this._pendingClientY = e.clientY;
            if (this._rafPending) return;
            this._rafPending = true;
            requestAnimationFrame(() => {
                this._rafPending = false;
                const r = this.screenToCell(this._pendingClientX, this._pendingClientY);
                if (!r.in_bounds) return;
                this.cellI = r.i;
                this.cellJ = r.j;
                this._refs.cell.textContent = `(${this.cellI}, ${this.cellJ})`;
            });
        };
        this.simCanvas.addEventListener('pointermove', onMove);
    }

    async _tick() {
        if (this._readbackBusy) return;
        this._readbackBusy = true;
        try {
            await this._doReadback();
        } catch (e) {
            console.warn('[plasma.probe] readback failed:', e);
        } finally {
            this._readbackBusy = false;
        }
    }

    async _doReadback() {
        const generation = this._bufferGeneration;
        const b = this.buffers;
        const ghost = GHOST_WIDTH;
        const n = this.sim.n, nT = this.sim.n_total;
        const i0 = this.cellI + ghost;
        const j0 = this.cellJ + ghost;
        const clamp = (v) => Math.max(0, Math.min(nT - 1, v));
        const ci = clamp(i0), cj = clamp(j0);

        const rowsStart = Math.max(0, cj - 1);
        const rowsEnd   = Math.min(nT - 1, cj + 1);
        const rowsN     = rowsEnd - rowsStart + 1;

        const V4 = 16, F32 = 4;
        const cellRowBytes = nT * V4;
        const bxRowBytes   = (nT + 1) * F32;
        const byRowBytes   = nT * F32;

        const specs = [
            { buf: b.U0_n, byteOffset: rowsStart * cellRowBytes, byteSize: rowsN * cellRowBytes },
            { buf: b.U1_n, byteOffset: rowsStart * cellRowBytes, byteSize: rowsN * cellRowBytes },
            { buf: b.Bx_n, byteOffset: rowsStart * bxRowBytes,   byteSize: rowsN * bxRowBytes },
            { buf: b.By_n, byteOffset: rowsStart * byRowBytes,
              byteSize: Math.min((rowsN + 1) * byRowBytes, (nT + 1) * byRowBytes - rowsStart * byRowBytes) },
        ];

        const bufs = await readbackBatch(this.device, this.pool, specs);
        if (generation !== this._bufferGeneration) return;
        const U0 = new Float32Array(bufs[0]);
        const U1 = new Float32Array(bufs[1]);
        const Bxf = new Float32Array(bufs[2]);
        const Byf = new Float32Array(bufs[3]);

        const jLocal = cj - rowsStart;
        const iC = ci;
        const k = jLocal * nT + iC;
        const u0_c = [U0[k*4], U0[k*4+1], U0[k*4+2], U0[k*4+3]];
        const u1_c = [U1[k*4], U1[k*4+1], U1[k*4+2], U1[k*4+3]];

        const rho = Math.max(u0_c[0], 1e-12);
        const vx  = u0_c[1] / rho;
        const vy  = u0_c[2] / rho;
        const vz  = u0_c[3] / rho;
        const Bz  = u1_c[1];

        const bxRowStride = nT + 1;
        const byRowStride = nT;
        const bxL = Bxf[jLocal * bxRowStride + iC];
        const bxR = Bxf[jLocal * bxRowStride + iC + 1];
        const byD = Byf[jLocal * byRowStride + iC];
        const byU = Byf[(jLocal + 1) * byRowStride + iC];
        const Bx = 0.5 * (bxL + bxR);
        const By = 0.5 * (byD + byU);

        const ke = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
        const mb = 0.5 * (Bx * Bx + By * By + Bz * Bz);
        const gammaM1 = this.sim.gamma - 1;
        const pFloor = this.sim.pressureFloor ?? 1e-6;
        const p = pressureFromDualEnergy({
            E: u1_c[0],
            eAux: u1_c[2],
            ke,
            mb,
            gammaM1,
            pFloor,
        });
        const T = p / rho;
        const beta = 2 * p / Math.max(2 * mb, 1e-12);
        const bmag = Math.sqrt(Bx * Bx + By * By + Bz * Bz);
        const vmag = Math.sqrt(vx * vx + vy * vy + vz * vz);

        // J_z needs the 4 face-derivative neighbors.
        const dx = this.sim.dx;
        const dxInv = 1 / dx;
        let jz = 0;
        if (jLocal >= 0 && jLocal < rowsN) {
            const ByCellLeft  = (iC > 0)        ? 0.5 * (Byf[jLocal * byRowStride + (iC - 1)] + Byf[(jLocal + 1) * byRowStride + (iC - 1)]) : By;
            const ByCellRight = (iC < nT - 1)   ? 0.5 * (Byf[jLocal * byRowStride + (iC + 1)] + Byf[(jLocal + 1) * byRowStride + (iC + 1)]) : By;
            const jU = jLocal + 1, jD = jLocal - 1;
            const BxCellUp   = (jU < rowsN) ? 0.5 * (Bxf[jU * bxRowStride + iC] + Bxf[jU * bxRowStride + (iC + 1)]) : Bx;
            const BxCellDown = (jD >= 0)    ? 0.5 * (Bxf[jD * bxRowStride + iC] + Bxf[jD * bxRowStride + (iC + 1)]) : Bx;
            jz = (ByCellRight - ByCellLeft) * 0.5 * dxInv - (BxCellUp - BxCellDown) * 0.5 * dxInv;
        }

        // Update DOM
        const fmt   = (v) => v.toExponential(3);
        const fmtFx = (v) => v.toFixed(4);
        this._refs.rho.textContent  = fmtFx(rho);
        this._refs.vx.textContent   = fmtFx(vx);
        this._refs.vy.textContent   = fmtFx(vy);
        this._refs.vz.textContent   = fmtFx(vz);
        this._refs.bx.textContent   = fmtFx(Bx);
        this._refs.by.textContent   = fmtFx(By);
        this._refs.bz.textContent   = fmtFx(Bz);
        this._refs.p.textContent    = fmtFx(p);
        this._refs.T.textContent    = fmtFx(T);
        this._refs.jz.textContent   = fmtFx(jz);
        this._refs.bmag.textContent = fmtFx(bmag);
        this._refs.vmag.textContent = fmtFx(vmag);
        this._refs.beta.textContent = fmt(beta);

        const x = (this.cellI + 0.5) * dx;
        const y = (this.cellJ + 0.5) * dx;
        this._refs.pos.textContent = `(${x.toFixed(3)}, ${y.toFixed(3)})`;
    }
}

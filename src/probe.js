/**
 * @fileoverview Probe tab — local cell state sampling + crosshair overlay.
 *
 * The probe owns three responsibilities:
 *
 *   1. A 2D `<canvas id="probeOverlay">` absolutely positioned over the
 *      main WebGPU canvas, transparent except for a small crosshair at
 *      the current probe cell.
 *   2. Click-to-place / shift-drag-to-move cell selection. The probe
 *      cell is stored as interior indices (i, j) ∈ [0, n)² and a UI
 *      cosmetic position in CSS pixels for the overlay.
 *   3. A 10 Hz readback that copies the small handful of bytes for
 *      the probe cell from each storage buffer, derives the visible
 *      state (ρ, vx, vy, vz, Bx_c, By_c, Bz, p, β, J_z, vorticity),
 *      and updates DOM nodes — plus a mini sparkline of one
 *      user-selected field over time.
 *
 * Per-cell readback shape (per tick, total ~52 bytes raw):
 *   • U0 cell (16 bytes)
 *   • U1 cell (16 bytes)
 *   • Bx_face left + right (8 bytes)
 *   • Bx_face above + below at (i, j±1) for J_z (8 bytes)
 *   • By_face up + down (8 bytes)
 *   • By_face left + right at (i±1, j) for J_z (8 bytes)
 *   • v_y at (i+1, j) and (i-1, j) for vorticity (8 bytes)
 *   • v_x at (i, j+1) and (i, j-1) for vorticity (8 bytes)
 *
 * For simplicity in Phase 5 we readback a small 3×3 stencil window
 * around the probe cell from `U0_n`, `U1_n`, `Bx_n`, `By_n`. That's
 * 12 × 16 + 12 × 16 + 16 × 4 + 16 × 4 = 512 bytes total, all in one
 * batched submit. Trivial cost at 10 Hz.
 *
 * No `innerHTML`; DOM built via createElement.
 */

import { readbackBatch, ReadbackPool } from './gpu/readback.js';
import { GHOST_WIDTH } from './config.js';

const PROBE_HZ = 10;          // 100 ms readback interval
const SPARK_CAP = 240;        // 24 seconds at 10 Hz

const FIELDS = [
    { id: 'rho',  label: 'ρ' },
    { id: 'vx',   label: 'v_x' },
    { id: 'vy',   label: 'v_y' },
    { id: 'vz',   label: 'v_z' },
    { id: 'bx',   label: 'B_x' },
    { id: 'by',   label: 'B_y' },
    { id: 'bz',   label: 'B_z' },
    { id: 'p',    label: 'p' },
    { id: 'beta', label: 'β' },
    { id: 'jz',   label: 'J_z' },
    { id: 'vort', label: 'ω' },
];

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

export class Probe {
    /**
     * @param {{device: GPUDevice, buffers: any, sim: any, canvas: HTMLCanvasElement}} ctx
     * @param {HTMLElement} root  the Probe tab panel
     * @param {HTMLCanvasElement} overlay  the probe overlay 2D canvas
     */
    constructor(ctx, root, overlay) {
        this.device = ctx.device;
        this.buffers = ctx.buffers;
        this.sim = ctx.sim;
        this.simCanvas = ctx.canvas;
        this.root = root;
        this.overlay = overlay;
        this.octx = overlay.getContext('2d');

        this.pool = new ReadbackPool(this.device);
        this._tickHandle = 0;
        this._readbackBusy = false;
        this._sparkField = 'jz';
        this._spark = createSparkHistory(SPARK_CAP);

        // Cell index in interior coords [0, n).
        this.cellI = Math.floor(this.sim.n / 2);
        this.cellJ = Math.floor(this.sim.n / 2);

        this._build();
        this._wireInput();
        this._syncOverlaySize();
        window.addEventListener('resize', () => this._syncOverlaySize(), { passive: true });
        this._draw();
    }

    bindBuffers(buffers) {
        this.buffers = buffers;
        resetSparkHistory(this._spark);
    }

    /** Sample a specific interior cell index. */
    setCell(i, j) {
        const n = this.sim.n;
        this.cellI = Math.max(0, Math.min(n - 1, i | 0));
        this.cellJ = Math.max(0, Math.min(n - 1, j | 0));
        this._refs.cell.textContent = `(${this.cellI}, ${this.cellJ})`;
        this._draw();
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

    /** Force a single readback now (used for one-shots after clicks). */
    readback() { this._tick(); }

    _build() {
        // Header: cell coords + reset-to-center button.
        const head = el('div', 'panel-section');
        head.append(el('h2', 'group-label', 'Probe Cell'));
        const cellRow = el('div', 'stat-row');
        cellRow.append(el('span', 'stat-label', 'Index'));
        const cellVal = el('span', 'stat-value', `(${this.cellI}, ${this.cellJ})`);
        cellRow.append(cellVal);
        head.append(cellRow);

        const posRow = el('div', 'stat-row');
        posRow.append(el('span', 'stat-label', 'Position'));
        const posVal = el('span', 'stat-value', '—');
        posRow.append(posVal);
        head.append(posRow);

        const hint = el('p', 'panel-hint',
            'Click the canvas to place; Shift-drag to move.');
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

        // Sparkline block.
        const spark = el('div', 'panel-section');
        spark.append(el('h2', 'group-label', 'History'));
        const fieldSelRow = el('div', 'ctrl-row');
        fieldSelRow.append(el('span', 'stat-label', 'Field'));
        const sel = document.createElement('select');
        sel.className = 'sim-select probe-field-select';
        for (const f of FIELDS) {
            const opt = document.createElement('option');
            opt.value = f.id;
            opt.textContent = f.label;
            if (f.id === this._sparkField) opt.selected = true;
            sel.append(opt);
        }
        sel.addEventListener('change', () => {
            this._sparkField = sel.value;
            resetSparkHistory(this._spark);
        });
        fieldSelRow.append(sel);
        spark.append(fieldSelRow);

        const canvas = el('canvas', 'probe-spark');
        canvas.width = 480;     // 2× DPR
        canvas.height = 96;
        canvas.style.width = '240px';
        canvas.style.height = '48px';
        spark.append(canvas);
        this.root.append(spark);

        this._refs = { ...refs, cell: cellVal, pos: posVal, sparkCanvas: canvas, sparkCtx: canvas.getContext('2d'), fieldSel: sel };
    }

    _wireInput() {
        // Click-to-place on the main canvas.
        const onClick = (e) => {
            const { i, j } = this._screenToCell(e.clientX, e.clientY);
            if (i < 0) return;
            this.setCell(i, j);
            this.readback();
        };
        // Shift-drag to move continuously.
        let dragging = false;
        const onDown = (e) => { if (e.shiftKey) { dragging = true; onClick(e); } };
        const onMove = (e) => { if (dragging) onClick(e); };
        const onUp   = () => { dragging = false; };

        this.simCanvas.addEventListener('click', onClick);
        this.simCanvas.addEventListener('pointerdown', onDown);
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
    }

    /** Map screen-pixel coords to interior cell index. */
    _screenToCell(px, py) {
        const rect = this.simCanvas.getBoundingClientRect();
        const x = px - rect.left;
        const y = py - rect.top;
        if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
            return { i: -1, j: -1 };
        }
        const n = this.sim.n;
        // y is screen-down; sim y points up in the typical MHD layout.
        // The composite pass maps interior cells to the full canvas, so
        // we use a direct linear mapping with y-flip.
        const i = Math.floor((x / rect.width)  * n);
        const j = Math.floor((1 - y / rect.height) * n);
        return { i: Math.max(0, Math.min(n - 1, i)),
                 j: Math.max(0, Math.min(n - 1, j)) };
    }

    _syncOverlaySize() {
        const rect = this.simCanvas.getBoundingClientRect();
        const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        this.overlay.style.width  = rect.width  + 'px';
        this.overlay.style.height = rect.height + 'px';
        this.overlay.width  = Math.floor(rect.width  * dpr);
        this.overlay.height = Math.floor(rect.height * dpr);
        this._draw();
    }

    /** Render the crosshair at the probe cell. */
    _draw() {
        const ctx = this.octx;
        const w = this.overlay.width;
        const h = this.overlay.height;
        ctx.clearRect(0, 0, w, h);
        const n = this.sim.n;
        const cellW = w / n, cellH = h / n;
        const cx = (this.cellI + 0.5) * cellW;
        const cy = (n - 0.5 - this.cellJ) * cellH;
        const style = getComputedStyle(document.documentElement);
        const color = (style.getPropertyValue('--accent') || '#e11107').trim();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        // Crosshair lines
        ctx.beginPath();
        ctx.moveTo(0, cy); ctx.lineTo(w, cy);
        ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
        ctx.stroke();
        // Cell box
        ctx.strokeStyle = color;
        ctx.strokeRect(cx - cellW / 2, cy - cellH / 2, cellW, cellH);
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
        const b = this.buffers;
        const ghost = GHOST_WIDTH;
        const n = this.sim.n, nT = this.sim.n_total;
        const i0 = this.cellI + ghost;
        const j0 = this.cellJ + ghost;

        // Pull a 3×3 cell window around (i0, j0) — wraps via index math
        // for fields living on ghost-padded buffers. Out-of-range stencil
        // neighbours fall back to the cell value via min/max clamps.
        const clamp = (v) => Math.max(0, Math.min(nT - 1, v));
        const ci = clamp(i0), cj = clamp(j0);

        // For cell-centered (U0, U1) we read the cell + its 4 neighbours
        // (5 cells total, 80 bytes each buffer).
        // We'll issue 5 small range copies, but the easier and faster
        // path is to copy a contiguous row (nT cells) for the centre row
        // and the rows above and below. That's 3 rows × nT cells × 16 B
        // at 256 = 12 KB — still trivial.
        // Even simpler: copy 3-row block = (3 × nT) × 16 = 3·256·16 = 12 KB
        // for U0 and again for U1; 12 + 12 + 3·257·4 + 3·256·4 ≈ 30 KB.
        // We use this layout because face buffers have different widths.

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
            // Bx_face row indexing matches the cell row index (one row per j).
            { buf: b.Bx_n, byteOffset: rowsStart * bxRowBytes,   byteSize: rowsN * bxRowBytes },
            // By_face: need rows j-1, j, j+1, j+2 (one extra row because
            // By_face has (nT+1) rows). Cap at the available range.
            { buf: b.By_n, byteOffset: rowsStart * byRowBytes,
              byteSize: Math.min((rowsN + 1) * byRowBytes, (nT + 1) * byRowBytes - rowsStart * byRowBytes) },
        ];

        const bufs = await readbackBatch(this.device, this.pool, specs);
        const U0 = new Float32Array(bufs[0]);
        const U1 = new Float32Array(bufs[1]);
        const Bxf = new Float32Array(bufs[2]);
        const Byf = new Float32Array(bufs[3]);

        // Local-row index of the probe cell within the 3-row window.
        const jLocal = cj - rowsStart;
        const iC = ci;

        const cellAt = (di, dj) => {
            const lj = jLocal + dj;
            const li = iC + di;
            const k = lj * nT + li;
            return { U0: [U0[k*4], U0[k*4+1], U0[k*4+2], U0[k*4+3]],
                     U1: [U1[k*4], U1[k*4+1], U1[k*4+2], U1[k*4+3]] };
        };

        const c = cellAt(0, 0);
        const rho = Math.max(c.U0[0], 1e-12);
        const vx  = c.U0[1] / rho;
        const vy  = c.U0[2] / rho;
        const vz  = c.U0[3] / rho;
        const Bz  = c.U1[1];

        // Face-B for cell-center average:
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
        const p  = Math.max((this.sim.gamma - 1) * (c.U1[0] - ke - mb), 1e-12);
        const beta = 2 * p / Math.max(2 * mb, 1e-12);

        // J_z and vorticity require neighbours.
        const dx = this.sim.dx;
        const dxInv = 1 / dx;
        let jz = 0, vort = 0;
        // Neighbours: rely on the 3-row window. dj=±1 valid only when
        // jLocal stayed in [0, rowsN-1] after add — which it does for
        // interior cells.
        if (jLocal >= 0 && jLocal < rowsN) {
            // Cell-centered By at (i+1, j) and (i-1, j); requires the
            // central row only.
            const ByCellLeft  = (iC > 0)        ? 0.5 * (Byf[jLocal * byRowStride + (iC - 1)] + Byf[(jLocal + 1) * byRowStride + (iC - 1)]) : By;
            const ByCellRight = (iC < nT - 1)   ? 0.5 * (Byf[jLocal * byRowStride + (iC + 1)] + Byf[(jLocal + 1) * byRowStride + (iC + 1)]) : By;
            // Cell-centered Bx at (i, j+1) and (i, j-1); requires top/bot rows.
            const jU = jLocal + 1, jD = jLocal - 1;
            const BxCellUp   = (jU < rowsN) ? 0.5 * (Bxf[jU * bxRowStride + iC] + Bxf[jU * bxRowStride + (iC + 1)]) : Bx;
            const BxCellDown = (jD >= 0)    ? 0.5 * (Bxf[jD * bxRowStride + iC] + Bxf[jD * bxRowStride + (iC + 1)]) : Bx;
            jz = (ByCellRight - ByCellLeft) * 0.5 * dxInv - (BxCellUp - BxCellDown) * 0.5 * dxInv;

            // ω_z = ∂v_y/∂x − ∂v_x/∂y. We have U0 for centre row only
            // (jLocal). For dj ±1 we'd need rows we don't have direct
            // access to here unless they're in the 3-row window.
            const cellL = (iC > 0)       ? { mx: U0[(jLocal * nT + (iC - 1)) * 4 + 1],
                                              my: U0[(jLocal * nT + (iC - 1)) * 4 + 2],
                                              rho: Math.max(U0[(jLocal * nT + (iC - 1)) * 4 + 0], 1e-12) } : null;
            const cellR = (iC < nT - 1)  ? { mx: U0[(jLocal * nT + (iC + 1)) * 4 + 1],
                                              my: U0[(jLocal * nT + (iC + 1)) * 4 + 2],
                                              rho: Math.max(U0[(jLocal * nT + (iC + 1)) * 4 + 0], 1e-12) } : null;
            const cellU = (jU < rowsN)   ? { mx: U0[(jU * nT + iC) * 4 + 1],
                                              my: U0[(jU * nT + iC) * 4 + 2],
                                              rho: Math.max(U0[(jU * nT + iC) * 4 + 0], 1e-12) } : null;
            const cellD = (jD >= 0)      ? { mx: U0[(jD * nT + iC) * 4 + 1],
                                              my: U0[(jD * nT + iC) * 4 + 2],
                                              rho: Math.max(U0[(jD * nT + iC) * 4 + 0], 1e-12) } : null;
            const vyR = cellR ? cellR.my / cellR.rho : vy;
            const vyL = cellL ? cellL.my / cellL.rho : vy;
            const vxU = cellU ? cellU.mx / cellU.rho : vx;
            const vxD = cellD ? cellD.mx / cellD.rho : vx;
            vort = (vyR - vyL) * 0.5 * dxInv - (vxU - vxD) * 0.5 * dxInv;
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
        this._refs.beta.textContent = fmt(beta);
        this._refs.jz.textContent   = fmtFx(jz);
        this._refs.vort.textContent = fmtFx(vort);

        // Physical position (centered on domain).
        const x = (this.cellI + 0.5) * dx;
        const y = (this.cellJ + 0.5) * dx;
        this._refs.pos.textContent = `(${x.toFixed(3)}, ${y.toFixed(3)})`;

        // Push selected field into sparkline.
        const fieldVal = ({ rho, vx, vy, vz, bx: Bx, by: By, bz: Bz, p, beta, jz, vort })[this._sparkField];
        pushSparkSample(this._spark, fieldVal);
        this._drawSpark();
    }

    _drawSpark() {
        const style = getComputedStyle(document.documentElement);
        const color = (style.getPropertyValue('--accent') || '#e11107').trim();
        const dim   = (style.getPropertyValue('--text-muted') || '#888').trim();
        const c = this._refs.sparkCanvas;
        drawSparkline(this._refs.sparkCtx, this._spark, c.width, c.height, color, dim);
    }
}

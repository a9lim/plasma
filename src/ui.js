/**
 * @fileoverview plasma Phase-5 UI entry point.
 *
 * Wires the topbar (play/pause, step, speed, save/load, settings,
 * about, theme), the three-tab sidebar (Settings / Stats / Probe),
 * preset/view/η/resolution/BC controls, advanced settings dropdown,
 * keyboard shortcuts, and the stats + probe readback loops to the
 * Sim instance owned by main.js.
 *
 * Public entry: `setupUI(simShell)` where `simShell` is the top-level
 * `PlasmaSim` from main.js (owns the canvas + `sim` orchestrator).
 *
 * No `innerHTML` assignments — all DOM built via createElement /
 * textContent / append. shared-icons.js auto-renders `data-icon`
 * attributes; shared-dropdown.js auto-enhances `select.sim-select`.
 */

import {
    BC_PERIODIC, BC_OUTFLOW, BC_REFLECTING, BC_DRIVEN,
    VIEW_DENSITY, VIEW_PRESSURE, VIEW_VMAG, VIEW_BMAG, VIEW_JZ,
    VIEW_T, VIEW_QMAG, VIEW_PHI, VIEW_ENTROPY,
    FLAG_POSITIVITY, EMF_MODE_BS_MEAN, EMF_MODE_GS_UPWIND,
    SLIDER_BOUNDS, ETA_SLIDER_MIN_FLOOR, SPEED_OPTIONS,
    PERTURB_SIGMA_CELLS, DRAG_VSCALE, EXCITE_BSCALE, PERTURB_MAX_DCELL,
} from './config.js';
import { StatsDisplay } from './stats-display.js';
import { Probe } from './probe.js';
import { buildPhysicsPanel } from './physics-panel.js';
import { section, sliderRow, epSlider, modeRow, toggleRow } from './panel-ui.js';

const SAVE_KEY = 'plasma.state.v1';
const THEME_KEY = 'plasma-theme';
const B = SLIDER_BOUNDS;

const BC_NAME_TO_ID = {
    periodic: BC_PERIODIC,
    outflow: BC_OUTFLOW,
    reflecting: BC_REFLECTING,
    driven: BC_DRIVEN,
};
const BC_ID_TO_NAME = ['periodic', 'outflow', 'reflecting', 'driven'];

/** Show a transient toast — uses shared-utils.js if loaded, else logs. */
function toast(msg) {
    if (typeof showToast === 'function') showToast(msg);
    else console.log('[plasma]', msg);
}

export function setupUI(simShell) {
    const sim = simShell.sim;
    const playBtn   = document.getElementById('playBtn');
    const stepBtn   = document.getElementById('stepBtn');
    const speedBtn  = document.getElementById('speedBtn');
    const resetBtn  = document.getElementById('clearBtn');
    const saveBtn   = document.getElementById('saveBtn');
    const loadBtn   = document.getElementById('loadBtn');
    const themeBtn  = document.getElementById('themeToggleBtn');
    const aboutBtn  = document.getElementById('about-btn');
    const panelToggle = document.getElementById('panelToggle');
    const panel       = document.getElementById('control-panel');
    const panelClose  = document.getElementById('panelClose');

    if (panelToggle && panel) {
        _toolbar.initSidebar(panelToggle, panel, panelClose);
    }

    // ── Stats + Probe (must be built before wireSettings refs them) ──
    const statsRoot   = document.getElementById('tab-stats');
    const probeRoot   = document.getElementById('tab-probe');
    const physicsRoot = document.getElementById('tab-physics');

    const stats = new StatsDisplay({ device: sim.device, buffers: sim.buffers, sim }, statsRoot);
    const probe = new Probe({ device: sim.device, buffers: sim.buffers, sim, canvas: simShell.canvas },
                            probeRoot);
    probe.start();
    if (physicsRoot) buildPhysicsPanel(physicsRoot, sim);

    const uiCtx = {
        simShell, sim, stats, probe,
        playBtn, stepBtn, speedBtn, resetBtn, saveBtn, loadBtn, themeBtn,
    };

    // ── Pointer perturbation (left-drag = grab, right-drag = excite) ──
    wirePointerPerturbation(uiCtx);

    // ── Settings tab controls ─────────────────────────────────
    wireSettings(uiCtx);

    // ── Topbar ────────────────────────────────────────────────
    wireTopbar(uiCtx);

    // ── Stats readback hook into render loop ──────────────────
    const origRender = simShell.render.bind(simShell);
    simShell.render = function() {
        origRender();
        stats.tick();
    };

    // ── About panel ───────────────────────────────────────────
    if (typeof initAboutPanel === 'function') {
        const aboutHandle = initAboutPanel({
            title: 'Plasma',
            lastUpdated: '2026-05-27',
            description: 'WebGPU-native 2D resistive MHD plasma simulator. Hover the canvas to sample the local state; left-drag pushes the plasma, right-drag twists the field. Settings holds preset / view / resistivity (η + anomalous) / numerics / render / resolution / boundaries; Physics holds the extended source layer (Hall, cooling, conduction, radiation, viscosity, non-ideal Ohm, gravity, geometry); Stats and Probe surface the live diagnostics.',
            controls: [
                { label: 'Sample cell',  value: 'Hover canvas' },
                { label: 'Push plasma',  value: 'Left-click drag' },
                { label: 'Twist field',  value: 'Right-click drag' },
                { label: 'Pause/play',   value: 'Space' },
                { label: 'Step',         value: '/' },
                { label: 'Reset',        value: 'R' },
                { label: 'View mode',    value: 'V' },
            ],
            shortcuts: _buildShortcuts(simShell, sim, stats, probe),
            repo: 'https://github.com/a9lim/plasma',
        });
        if (aboutBtn) aboutBtn.addEventListener('click', () => aboutHandle.show && aboutHandle.show());
    }

    // ── Keyboard shortcuts ────────────────────────────────────
    if (typeof initShortcuts === 'function') {
        initShortcuts(_buildShortcuts(simShell, sim, stats, probe),
                      { helpTitle: 'Keyboard Shortcuts' });
    }

    // ── Auto-restore last state from localStorage on init ─────
    try {
        const saved = localStorage.getItem(SAVE_KEY);
        if (saved) {
            // Don't auto-apply on first load — too disruptive. Only the
            // user-driven Load button reads it. We still verify it
            // parses by leaving the try/catch.
        }
    } catch (e) { /* ignore */ }

    return { stats, probe };
}

// ── topbar wiring ─────────────────────────────────────────────

function wireTopbar(ctx) {
    const { simShell, sim, playBtn, stepBtn, speedBtn, resetBtn,
            saveBtn, loadBtn, themeBtn, stats, probe } = ctx;

    // Play / pause
    const togglePause = () => {
        sim.setRunning(!sim.running);
        simShell.running = sim.running;
        if (sim.running) simShell.lastTime = performance.now();
        _toolbar.updatePlayBtn(playBtn, sim.running);
    };
    _toolbar.updatePlayBtn(playBtn, sim.running);
    playBtn.addEventListener('click', togglePause);

    // Step
    stepBtn.addEventListener('click', () => {
        if (!sim.running) sim.step();
    });

    // Speed cycle
    let speedIdx = 0;
    sim.setSpeedScale(SPEED_OPTIONS[speedIdx]);
    _toolbar.updateSpeedBtn(speedBtn, SPEED_OPTIONS[speedIdx]);
    const cycleSpeed = (delta) => {
        speedIdx = (speedIdx + delta + SPEED_OPTIONS.length) % SPEED_OPTIONS.length;
        sim.setSpeedScale(SPEED_OPTIONS[speedIdx]);
        _toolbar.updateSpeedBtn(speedBtn, SPEED_OPTIONS[speedIdx]);
    };
    speedBtn.addEventListener('click', () => cycleSpeed(1));
    speedBtn.addEventListener('contextmenu', (e) => { e.preventDefault(); cycleSpeed(-1); });

    // Reset preset
    resetBtn.addEventListener('click', () => {
        sim.setPreset(sim.presetName);
        toast('Preset reset');
    });

    // Save / Load
    saveBtn.addEventListener('click', () => {
        try {
            localStorage.setItem(SAVE_KEY, sim.saveState());
            toast('State saved');
        } catch (e) { console.warn('[plasma] save failed:', e); }
    });
    loadBtn.addEventListener('click', () => {
        try {
            const s = localStorage.getItem(SAVE_KEY);
            if (s) {
                const result = sim.loadState(s);
                if (result && result.buffersChanged) {
                    stats.bindBuffers(sim.buffers);
                    probe.bindBuffers(sim.buffers);
                    ctx._refreshEtaSlider?.();
                }
                stats.updatePresetVisibility(sim.presetName);
                toast(result && result.ok === false ? 'State load failed' : 'State loaded');
            } else {
                toast('No saved state');
            }
        } catch (e) { console.warn('[plasma] load failed:', e); }
    });

    // Theme toggle
    _toolbar.initTheme(THEME_KEY);
    themeBtn.addEventListener('click', () => _toolbar.toggleTheme(THEME_KEY));
}

// ── settings tab ─────────────────────────────────────────────

function wireSettings(ctx) {
    const { simShell, sim, stats, probe } = ctx;
    const root = document.getElementById('tab-settings');

    // Preset
    const presetSel = root.querySelector('#ctrl-preset');
    presetSel.value = sim.presetName;
    presetSel.addEventListener('change', () => {
        sim.setPreset(presetSel.value);
        stats.updatePresetVisibility(presetSel.value);
        // Push current BC dropdowns back to match the preset's BCs.
        syncBCDropdowns();
        // Preset switch may change the η floor (OT has one; others don't).
        ctx._refreshEtaSlider?.();
        toast(`Loaded ${presetSel.value}`);
    });

    // View mode
    const viewSel = root.querySelector('#ctrl-view');
    viewSel.value = String(sim.viewMode);
    viewSel.addEventListener('change', () => {
        sim.setViewMode(parseInt(viewSel.value, 10));
    });

    // η slider (log scale; snap-to-0 only when the active preset has no
    // grid-Reynolds floor — when it does, the slider's lower bound moves
    // up to log10(etaMin) and the hint reflects the floor.)
    const etaSlider = root.querySelector('#ctrl-eta');
    const etaLabel  = root.querySelector('#ctrl-eta-val');
    const etaHint   = root.querySelector('#ctrl-eta-hint');
    const SLIDER_MIN_FLOOR = ETA_SLIDER_MIN_FLOOR;   // snap-to-0 floor (config.js)
    const etaFromSlider = (v) => {
        if (v <= -6.05) return 0;        // snap-to-0 below 1e-6
        return Math.pow(10, v);
    };
    const etaToSlider = (eta) => (eta <= 0 ? SLIDER_MIN_FLOOR : Math.log10(eta));

    // Refresh slider geometry + label + hint based on sim's current floor.
    // Called on init and whenever preset or resolution changes (both
    // affect getEtaMin: preset selects the coefficient, resolution sets dx).
    function refreshEtaSlider() {
        const etaMin = sim.getEtaMin();
        if (etaMin > 0) {
            etaSlider.min = String(Math.log10(etaMin));
            etaHint.textContent =
                `Min η = ${etaMin.toExponential(2)} ` +
                `(grid Reynolds limit — prevents NaN cascade from sub-grid current sheets).`;
        } else {
            etaSlider.min = String(SLIDER_MIN_FLOOR);
            etaHint.textContent = 'Below 1e-6 snaps to 0 (ideal MHD).';
        }
        etaSlider.value = String(etaToSlider(sim.eta));
        etaLabel.textContent = sim.eta === 0 ? '0' : sim.eta.toExponential(2);
    }
    refreshEtaSlider();
    _forms.bindSlider(etaSlider, null, (v) => {
        sim.setEta(etaFromSlider(v));
        // sim.setEta may clamp upward — reflect the actual value back.
        etaLabel.textContent = sim.eta === 0 ? '0' : sim.eta.toExponential(2);
    });
    // Expose to the preset / resolution handlers below.
    ctx._refreshEtaSlider = refreshEtaSlider;

    // Resolution mode group
    const resGroup = root.querySelector('#ctrl-res-toggles');
    _forms.bindModeGroup(resGroup, 'res', (val) => {
        const n = parseInt(val, 10);
        sim.setResolution(n);
        stats.bindBuffers(sim.buffers);
        probe.bindBuffers(sim.buffers);
        // After re-instantiation, the renderer was rebuilt inside
        // sim.setResolution; the simShell still holds the original
        // sim object so further calls into sim.render() use the new
        // renderer through sim.renderer.
        // Resolution change scales dx → η floor scales inversely with N.
        ctx._refreshEtaSlider?.();
        toast(`Resolution ${n}²`);
    });

    // Boundaries: 4 dropdowns (N/S/E/W) + driven state reveal
    const bcN = root.querySelector('#ctrl-bc-n');
    const bcS = root.querySelector('#ctrl-bc-s');
    const bcE = root.querySelector('#ctrl-bc-e');
    const bcW = root.querySelector('#ctrl-bc-w');
    const bcDropdowns = [bcN, bcS, bcE, bcW];
    const bcEdges = ['N', 'S', 'E', 'W'];

    const drivenPanel = root.querySelector('#ctrl-driven-panel');

    function anyDriven() {
        return bcDropdowns.some(d => d.value === 'driven');
    }
    function syncBCDropdowns() {
        // Pull current sim.bcConfig (which the preset may have set) into the selects.
        bcN.value = BC_ID_TO_NAME[sim.bcConfig.modeN] || 'periodic';
        bcS.value = BC_ID_TO_NAME[sim.bcConfig.modeS] || 'periodic';
        bcE.value = BC_ID_TO_NAME[sim.bcConfig.modeE] || 'periodic';
        bcW.value = BC_ID_TO_NAME[sim.bcConfig.modeW] || 'periodic';
        // Fire change events so shared-dropdown's custom UI re-syncs labels.
        bcDropdowns.forEach(d => d.dispatchEvent(new Event('change', { bubbles: false })));
        updateBCDeps();
    }

    const updateBCDeps = _forms.bindDeps([
        { target: drivenPanel, show: () => anyDriven() },
    ]);

    function drivenEdges() {
        return bcDropdowns
            .map((sel, idx) => sel.value === 'driven' ? bcEdges[idx] : null)
            .filter(Boolean);
    }

    bcDropdowns.forEach((sel, idx) => {
        sel.addEventListener('change', () => {
            const mode = BC_NAME_TO_ID[sel.value] ?? BC_PERIODIC;
            sim.setBC(bcEdges[idx], mode);
            updateBCDeps();
        });
    });

    // Driven state inputs (8 sliders / number inputs)
    const drivenIds = ['rho','vx','vy','vz','bx','by','bz','p'];
    for (const k of drivenIds) {
        const input = root.querySelector('#driven-' + k);
        if (!input) continue;
        input.value = String(sim.bcConfig.driven[k] ?? 0);
        const label = root.querySelector('#driven-' + k + '-val');
        if (label) label.textContent = parseFloat(input.value).toFixed(3);
        input.addEventListener('input', () => {
            const v = parseFloat(input.value);
            if (label) label.textContent = v.toFixed(3);
            const edges = drivenEdges();
            if (edges.length === 0) {
                sim.setDrivenState({ [k]: v });
            } else {
                for (const edge of edges) sim.setDrivenState({ [k]: v }, edge);
            }
        });
    }

    // ── η-anomalous (Birn 2001) — appended into the Resistivity section ──
    // α = 0 (off) keeps the constant-η₀ baseline; α > 0 activates
    // |J| > J_crit enhanced resistivity for fast reconnection.
    const etaSection = etaSlider.closest('.panel-section');
    epSlider(etaSection, 'η-anom α (log10)', {
        ...B.etaAnomAlpha, sim,
        getScalar: () => sim.etaAnomAlpha,
        setScalar: v => sim.setEtaAnomAlpha(v),
        hint: 'α > 0 enables |J|>J_crit enhanced resistivity; off = constant η₀.',
    });
    sliderRow(etaSection, 'η-anom J_crit', {
        ...B.etaAnomJcrit,
        value: sim.etaAnomJcrit,
        format: v => v.toFixed(0),
        onChange: v => sim.setEtaAnomJcrit(v),
    });

    // ── Numerics + Render — folded in from the former gear dropdown.
    // Built in JS (like the Physics tab) and inserted between the
    // Resistivity and Resolution sections so the tab reads
    // preset → view → resistivity → numerics → render → resolution → BCs.
    const resolutionSection = resGroup.closest('.panel-section');

    const numerics = section('Numerics');
    sliderRow(numerics, 'CFL', {
        ...B.cfl, value: sim.cfl,
        format: v => v.toFixed(2),
        onChange: v => sim.setCFL(v),
    });
    sliderRow(numerics, 'γ', {
        ...B.gamma, value: sim.gamma,
        format: v => v.toFixed(2),
        onChange: v => sim.setGamma(v),
    });
    sliderRow(numerics, 'p-floor (log10)', {
        ...B.pressureFloorLog, value: Math.log10(sim.pressureFloor),
        format: v => '1e' + v.toFixed(1),
        onChange: v => sim.setPressureFloor(Math.pow(10, v)),
    });
    sliderRow(numerics, 'Source substeps cap', {
        ...B.sourceSubstepsCap, value: sim.sourceSubstepsMax,
        format: v => String(v | 0),
        onChange: v => sim.setSourceSubstepsMax(v | 0),
    });
    modeRow(numerics, 'EMF', {
        dataAttr: 'emf-mode',
        buttons: [
            { value: String(EMF_MODE_BS_MEAN),   label: 'BS mean',
              active: sim.emfMode === EMF_MODE_BS_MEAN },
            { value: String(EMF_MODE_GS_UPWIND), label: 'GS upwind',
              active: sim.emfMode === EMF_MODE_GS_UPWIND },
        ],
        onChange: v => sim.setEmfMode(parseInt(v, 10)),
    });
    toggleRow(numerics, 'Positivity guard', {
        checked: (sim.physicsFlags & FLAG_POSITIVITY) !== 0,
        onChange: on => sim.setPhysicsFlag(FLAG_POSITIVITY, on),
    });

    const render = section('Render');
    sliderRow(render, 'LIC intensity', {
        ...B.licIntensity, value: sim.licIntensity,
        format: v => v.toFixed(2),
        onChange: v => sim.setLicIntensity(v),
    });
    sliderRow(render, 'LIC drift', {
        ...B.licDrift, value: sim.licDriftX,
        format: v => v.toFixed(1) + ' px/s',
        onChange: v => sim.setLicDrift(v, sim.licDriftY),
    });

    resolutionSection.before(numerics, render);

    // Initial sync
    syncBCDropdowns();
    stats.updatePresetVisibility(sim.presetName);

    // Expose for testing
    ctx._syncBCDropdowns = syncBCDropdowns;
}

// ── pointer perturbation ─────────────────────────────────────
//
// Left-click drag deposits a Gaussian-weighted momentum bump along the
// drag vector ("grab the plasma"). Right-click drag launches a curl-of-Az
// magnetic perturbation in the drag direction ("twist the field"). Both
// modes use the same WGSL pipeline; only the kind enum and the dvec
// scaling differ. See plasma/src/gpu/shaders/perturb.wgsl for the math.
//
// Coalescing: we don't apply per-pointermove event. Each event records
// the new position and accumulates a pending Δcell; the actual GPU
// dispatch fires on the next animation frame. This keeps the per-frame
// command-buffer count predictable and gives the dispatch a non-tiny
// drag vector even on high-Hz pointers (240 Hz Magic Trackpad would
// otherwise see ~Δcell/4 deposits which feel anaemic).

function wirePointerPerturbation(ctx) {
    const { simShell, sim, probe } = ctx;
    const canvas = simShell.canvas;

    // Pointer state. `button` follows MouseEvent.button: 0 = left = drag,
    // 2 = right = excite. `lastFrac` is the previous pointermove position
    // in canvas-fractional coords ([0, 1] × [0, 1], y already flipped to
    // sim-up).
    let active = false;
    let kind = null;       // 'drag' | 'excite'
    let lastFrac = null;   // { x, y } in [0, 1]
    let pending = null;    // { centerFrac, dCellX, dCellY }
    let rafPending = false;

    const screenToFrac = (clientX, clientY) => {
        const rect = canvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        if (x < 0 || x > rect.width || y < 0 || y > rect.height) return null;
        return { x: x / rect.width, y: 1 - y / rect.height };
    };

    const flushPending = () => {
        rafPending = false;
        if (!pending || !active || !kind) { pending = null; return; }
        const { centerFrac, dCellX, dCellY } = pending;
        pending = null;
        // Clamp huge per-frame drags so flicks don't blow up the cell.
        const clamp = (v) => Math.max(-PERTURB_MAX_DCELL,
                                       Math.min(PERTURB_MAX_DCELL, v));
        const dcx = clamp(dCellX);
        const dcy = clamp(dCellY);
        if (Math.abs(dcx) + Math.abs(dcy) < 1e-3) return;
        const L = sim.domainLength;
        const cx = centerFrac.x * L;
        const cy = centerFrac.y * L;
        const sigma = PERTURB_SIGMA_CELLS * sim.dx;
        const scale = kind === 'drag' ? DRAG_VSCALE : EXCITE_BSCALE;
        sim.applyPerturbation({
            kind,
            cx, cy,
            dvec_x: scale * dcx,
            dvec_y: scale * dcy,
            sigma,
            amplitude: 1.0,
        });
    };

    const scheduleFlush = () => {
        if (rafPending) return;
        rafPending = true;
        requestAnimationFrame(flushPending);
    };

    canvas.addEventListener('pointerdown', (e) => {
        // We only care about primary (button 0) and secondary (button 2).
        // Middle-click / extra buttons are ignored — let other things use them.
        if (e.button !== 0 && e.button !== 2) return;
        const frac = screenToFrac(e.clientX, e.clientY);
        if (!frac) return;
        active = true;
        kind = e.button === 0 ? 'drag' : 'excite';
        lastFrac = frac;
        // Capture so pointermove keeps firing if the cursor leaves the
        // canvas mid-drag.
        try { canvas.setPointerCapture(e.pointerId); } catch (_) { /* older browsers */ }
        e.preventDefault();
    });

    // Right-click would otherwise show the OS context menu. Suppress it
    // over the canvas so excite-drag works.
    canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); });

    canvas.addEventListener('pointermove', (e) => {
        if (!active || !kind || !lastFrac) return;
        const frac = screenToFrac(e.clientX, e.clientY);
        if (!frac) return;
        // Δ in cell units (canvas spans n interior cells).
        const n = sim.n;
        const dCellX = (frac.x - lastFrac.x) * n;
        const dCellY = (frac.y - lastFrac.y) * n;
        lastFrac = frac;
        if (pending) {
            pending.centerFrac = frac;
            pending.dCellX += dCellX;
            pending.dCellY += dCellY;
        } else {
            pending = { centerFrac: frac, dCellX, dCellY };
        }
        scheduleFlush();
    });

    const endDrag = (e) => {
        if (!active) return;
        active = false;
        kind = null;
        lastFrac = null;
        pending = null;
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    // Touch support: a single touch maps to left-drag automatically (the
    // browser dispatches synthetic pointerdown with button=0). Two-finger
    // gestures fall through to OS — no excite-on-touch by design.
}

// ── keyboard shortcuts ───────────────────────────────────────

function _buildShortcuts(simShell, sim, stats, probe) {
    const togglePause = () => {
        sim.setRunning(!sim.running);
        if (sim.running) simShell.lastTime = performance.now();
        const btn = document.getElementById('playBtn');
        if (btn) _toolbar.updatePlayBtn(btn, sim.running);
    };
    const stepOnce = () => { if (!sim.running) sim.step(); };
    const cycleView = () => {
        const order = [
            VIEW_JZ, VIEW_BMAG, VIEW_DENSITY, VIEW_VMAG, VIEW_PRESSURE,
            VIEW_T, VIEW_QMAG, VIEW_PHI, VIEW_ENTROPY,
        ];
        const idx = order.indexOf(sim.viewMode);
        const next = order[(idx + 1) % order.length];
        sim.setViewMode(next);
        const sel = document.getElementById('ctrl-view');
        if (sel) { sel.value = String(next); sel.dispatchEvent(new Event('change', { bubbles: false })); }
    };
    const resetPreset = () => { sim.setPreset(sim.presetName); };
    const saveState = () => { try { localStorage.setItem(SAVE_KEY, sim.saveState()); toast('State saved'); } catch (e) {} };
    const loadState = () => { try { const s = localStorage.getItem(SAVE_KEY); if (s) { sim.loadState(s); toast('State loaded'); } } catch (e) {} };
    const switchTab = (tabName) => {
        const btn = document.querySelector(`.sidebar-tabs .tab-btn[data-tab="${tabName}"]`);
        if (btn) btn.click();
    };
    return [
        { key: 'space',  label: 'Pause / play',           group: 'Simulation', action: togglePause },
        { key: '/',      label: 'Step forward',           group: 'Simulation', action: stepOnce },
        { key: 'r',      label: 'Reset preset',           group: 'Simulation', action: resetPreset },
        { key: 'v',      label: 'Cycle view mode',        group: 'View',       action: cycleView },
        { key: 's',      label: 'Save state',             group: 'State',      action: saveState },
        { key: 'l',      label: 'Load state',             group: 'State',      action: loadState },
        { key: '1',      label: 'Settings tab',           group: 'Sidebar',    action: () => switchTab('settings') },
        { key: '2',      label: 'Physics tab',            group: 'Sidebar',    action: () => switchTab('physics') },
        { key: '3',      label: 'Stats tab',              group: 'Sidebar',    action: () => switchTab('stats') },
        { key: '4',      label: 'Probe tab',              group: 'Sidebar',    action: () => switchTab('probe') },
    ];
}

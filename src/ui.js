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
} from './config.js';
import { StatsDisplay } from './stats-display.js';
import { Probe } from './probe.js';

const SAVE_KEY = 'plasma.state.v1';
const THEME_KEY = 'plasma-theme';
const SPEED_OPTIONS = [1, 2, 4, 8, 16];

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
    const settingsBtn = document.getElementById('settings-btn');
    const panelToggle = document.getElementById('panelToggle');
    const panel       = document.getElementById('control-panel');
    const panelClose  = document.getElementById('panelClose');

    // ── Topbar ────────────────────────────────────────────────
    wireTopbar({ simShell, sim, playBtn, stepBtn, speedBtn, resetBtn,
                 saveBtn, loadBtn, themeBtn });

    if (panelToggle && panel) {
        _toolbar.initSidebar(panelToggle, panel, panelClose);
    }

    // ── Stats + Probe (must be built before wireSettings refs them) ──
    const statsRoot = document.getElementById('tab-stats');
    const probeRoot = document.getElementById('tab-probe');
    const probeOverlay = document.getElementById('probeOverlay');

    const stats = new StatsDisplay({ device: sim.device, buffers: sim.buffers, sim }, statsRoot);
    const probe = new Probe({ device: sim.device, buffers: sim.buffers, sim, canvas: simShell.canvas },
                            probeRoot, probeOverlay);
    probe.start();

    // ── Settings tab controls ─────────────────────────────────
    wireSettings({ simShell, sim, stats, probe });

    // ── Stats readback hook into render loop ──────────────────
    const origRender = simShell.render.bind(simShell);
    simShell.render = function() {
        origRender();
        stats.tick();
    };

    // ── Advanced settings dropdown ────────────────────────────
    if (settingsBtn && typeof _settings !== 'undefined') {
        _settings.create(settingsBtn, [
            { type: 'slider', label: 'CFL', min: 0.1, max: 0.8, step: 0.05,
              value: sim.cfl, format: v => v.toFixed(2),
              onChange: v => sim.setCFL(v) },
            { type: 'slider', label: 'γ', min: 1.1, max: 2.0, step: 0.05,
              value: sim.gamma, format: v => v.toFixed(2),
              onChange: v => sim.setGamma(v) },
            { type: 'slider', label: 'p-floor (log10)', min: -8, max: -3, step: 0.5,
              value: Math.log10(sim.pressureFloor), format: v => '1e' + v.toFixed(1),
              onChange: v => sim.setPressureFloor(Math.pow(10, v)) },
            { type: 'slider', label: 'LIC intensity', min: 0, max: 2, step: 0.05,
              value: sim.licIntensity, format: v => v.toFixed(2),
              onChange: v => sim.setLicIntensity(v) },
            { type: 'slider', label: 'LIC drift', min: 0, max: 4, step: 0.1,
              value: sim.licDriftX, format: v => v.toFixed(1) + ' px/s',
              onChange: v => sim.setLicDrift(v, sim.licDriftY) },
        ], { width: 300 });
    }

    // ── About panel ───────────────────────────────────────────
    if (typeof initAboutPanel === 'function') {
        const aboutHandle = initAboutPanel({
            title: 'Plasma',
            lastUpdated: '2026-05-24',
            description: 'WebGPU-native 2D resistive MHD plasma simulator. Click to place the probe; use Settings to switch preset, view mode, resistivity, and boundary conditions.',
            controls: [
                { label: 'Probe cell',  value: 'Click canvas' },
                { label: 'Drag probe',  value: 'Shift + drag' },
                { label: 'Pause/play',  value: 'Space' },
                { label: 'Step',        value: '/' },
                { label: 'Reset',       value: 'R' },
                { label: 'View mode',   value: 'V' },
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
            saveBtn, loadBtn, themeBtn } = ctx;

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
            if (s) { sim.loadState(s); toast('State loaded'); }
            else   { toast('No saved state'); }
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
    const SLIDER_MIN_FLOOR = -6.5;   // matches HTML's default min (snap-to-0)
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
            sim.setDrivenState({ [k]: v });
        });
    }

    // Initial sync
    syncBCDropdowns();
    stats.updatePresetVisibility(sim.presetName);

    // Expose for testing
    ctx._syncBCDropdowns = syncBCDropdowns;
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
        const order = [VIEW_JZ, VIEW_BMAG, VIEW_DENSITY, VIEW_VMAG, VIEW_PRESSURE];
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
        { key: '2',      label: 'Stats tab',              group: 'Sidebar',    action: () => switchTab('stats') },
        { key: '3',      label: 'Probe tab',              group: 'Sidebar',    action: () => switchTab('probe') },
    ];
}

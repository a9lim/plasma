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
    FLAG_COOLING, FLAG_GRAVITY_SELF, FLAG_CONDUCTION, FLAG_HALL,
    FLAG_AMBIPOLAR, FLAG_BIERMANN, FLAG_VISCOSITY, FLAG_HEATING,
    FLAG_SPONGE, FLAG_GEOMETRY,
    FLAG_POSITIVITY, EMF_MODE_BS_MEAN, EMF_MODE_GS_UPWIND,
    COOLING_CURVE_BREMS, COOLING_CURVE_TABLE, COOLING_CURVE_CIE, COOLING_CURVE_TABULATED,
    GEOMETRY_CARTESIAN, GEOMETRY_CYLINDRICAL,
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
        // Snap-to-0 log slider helper for the extended-physics scalars.
        // The shaders early-return when their corresponding scalar is 0
        // OR their flag bit is clear, so we drive both: the value sets
        // the scalar, and "off" at the bottom of the range additionally
        // clears the flag. Raising a slider out of "off" re-arms the
        // corresponding source-physics flag so the preset's choice of
        // FLAG_* doesn't trap the user.
        const epSlider = (label, getScalar, setScalar, flag, opts = {}) => {
            const { lo = -6, hi = 0, step = 0.25 } = opts;
            const cur = getScalar();
            const offBoundary = lo - 0.05;
            const valueAt = cur > 0 ? Math.log10(cur) : (lo - 0.5);
            return {
                type: 'slider', label,
                min: lo - 0.5, max: hi, step,
                value: valueAt,
                format: v => (v <= offBoundary ? 'off' : '1e' + v.toFixed(2)),
                onChange: v => {
                    if (v <= offBoundary) {
                        setScalar(0);
                        sim.setPhysicsFlag(flag, false);
                    } else {
                        setScalar(Math.pow(10, v));
                        sim.setPhysicsFlag(flag, true);
                    }
                },
            };
        };

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
            // Anomalous resistivity — α = 0 disables the boost (sim runs
            // with constant η_0). α > 0 activates Birn-2001-style
            // |J|>J_crit enhanced resistivity for fast reconnection.
            // Slider is log10 in [−6, 0] = [1e-6, 1.0] plus an explicit
            // "off" snap at the bottom.
            { type: 'slider', label: 'η-anom α (log10)', min: -6, max: 0, step: 0.25,
              value: (sim.etaAnomAlpha > 0 ? Math.log10(sim.etaAnomAlpha) : -6.5),
              format: v => (v <= -6.05 ? 'off' : '1e' + v.toFixed(2)),
              onChange: v => sim.setEtaAnomAlpha(v <= -6.05 ? 0 : Math.pow(10, v)) },
            { type: 'slider', label: 'η-anom J_crit', min: 1, max: 100, step: 1,
              value: sim.etaAnomJcrit, format: v => v.toFixed(0),
              onChange: v => sim.setEtaAnomJcrit(v) },

            // ── Extended physics (Session 15) ───────────────────
            // Each scalar slider is log10 with a snap-to-0/"off" at the
            // bottom that also clears the corresponding FLAG_* bit. EMF
            // mode is a separate two-button group; positivity guard is
            // a toggle (no scalar).
            epSlider('Hall d_i (log10)',
                () => sim.hallDi, v => sim.setHallDi(v),
                FLAG_HALL, { lo: -4 }),
            { type: 'slider', label: 'Hall p_e / p', min: 0, max: 1, step: 0.05,
              value: sim.hallElectronPressureFrac, format: v => v.toFixed(2),
              onChange: v => sim.setHallElectronPressureFrac(v) },
            epSlider('Cooling Λ₀ (log10)',
                () => sim.coolingLambda0, v => sim.setCoolingLambda0(v),
                FLAG_COOLING, { lo: -4 }),
            { type: 'mode', label: 'Cooling curve', dataAttr: 'cooling-curve',
              buttons: [
                  { value: String(COOLING_CURVE_TABULATED), label: 'tab',   active: sim.coolingCurveMode === COOLING_CURVE_TABULATED },
                  { value: String(COOLING_CURVE_CIE),   label: 'CIE',   active: sim.coolingCurveMode === COOLING_CURVE_CIE },
                  { value: String(COOLING_CURVE_TABLE), label: 'table', active: sim.coolingCurveMode === COOLING_CURVE_TABLE },
                  { value: String(COOLING_CURVE_BREMS), label: 'brems', active: sim.coolingCurveMode === COOLING_CURVE_BREMS },
              ],
              onChange: v => sim.setCoolingCurveMode(parseInt(v, 10)) },
            { type: 'slider', label: 'Metallicity Z', min: 0, max: 3, step: 0.1,
              value: sim.coolingMetallicity, format: v => v.toFixed(1) + '×',
              onChange: v => sim.setCoolingMetallicity(v) },
            epSlider('Heating Γ (log10)',
                () => sim.heatingGamma0, v => sim.setHeatingGamma0(v),
                FLAG_HEATING, { lo: -6, hi: 0 }),
            { type: 'slider', label: 'Heating ρ exponent', min: 0, max: 2, step: 0.1,
              value: sim.heatingDensityExp, format: v => v.toFixed(1),
              onChange: v => sim.setHeatingDensityExp(v) },
            { type: 'slider', label: 'Heating T cutoff', min: -4.5, max: 2, step: 0.25,
              value: sim.heatingTCut > 0 ? Math.log10(sim.heatingTCut) : -4.5,
              format: v => (v <= -4.45 ? 'off' : '1e' + v.toFixed(2)),
              onChange: v => sim.setHeatingTCut(v <= -4.45 ? 0 : Math.pow(10, v)) },
            epSlider('Conduction κ∥ (log10)',
                () => sim.conductionKappa, v => sim.setConductionKappa(v),
                FLAG_CONDUCTION, { lo: -6 }),
            { type: 'slider', label: 'κ⊥ / κ∥', min: 0, max: 1, step: 0.05,
              value: sim.conductionIsoFrac, format: v => v.toFixed(2),
              onChange: v => sim.setConductionIsoFrac(v) },
            { type: 'slider', label: 'q_sat φ', min: 0, max: 1, step: 0.05,
              value: sim.conductionSatFrac, format: v => (v <= 0 ? 'off' : v.toFixed(2)),
              onChange: v => sim.setConductionSatFrac(v) },
            epSlider('Viscosity ν (log10)',
                () => sim.viscosityNu, v => sim.setViscosityNu(v),
                FLAG_VISCOSITY, { lo: -7, hi: -1 }),
            { type: 'slider', label: 'Bulk viscosity', min: -7.5, max: -1, step: 0.25,
              value: sim.viscosityBulk > 0 ? Math.log10(sim.viscosityBulk) : -7.5,
              format: v => (v <= -7.45 ? 'off' : '1e' + v.toFixed(2)),
              onChange: v => sim.setViscosityBulk(v <= -7.45 ? 0 : Math.pow(10, v)) },
            { type: 'slider', label: 'ν B-aligned frac', min: 0, max: 1, step: 0.05,
              value: sim.viscosityAnisoFrac, format: v => v.toFixed(2),
              onChange: v => sim.setViscosityAnisoFrac(v) },
            { type: 'slider', label: 'Shock viscosity', min: -7.5, max: -1, step: 0.25,
              value: sim.viscosityShock > 0 ? Math.log10(sim.viscosityShock) : -7.5,
              format: v => (v <= -7.45 ? 'off' : '1e' + v.toFixed(2)),
              onChange: v => sim.setViscosityShock(v <= -7.45 ? 0 : Math.pow(10, v)) },
            epSlider('Ambipolar η_A (log10)',
                () => sim.ambipolarEta, v => sim.setAmbipolarEta(v),
                FLAG_AMBIPOLAR, { lo: -7, hi: -1 }),
            { type: 'slider', label: 'Neutral fraction', min: 0, max: 1, step: 0.05,
              value: sim.neutralFrac, format: v => v.toFixed(2),
              onChange: v => sim.setNeutralFrac(v) },
            { type: 'slider', label: 'Ionization T₀', min: -4, max: 2, step: 0.25,
              value: Math.log10(sim.ionizationT0), format: v => '1e' + v.toFixed(2),
              onChange: v => sim.setIonizationT0(Math.pow(10, v)) },
            epSlider('Biermann C_B (log10)',
                () => Math.abs(sim.biermannCoeff), v => sim.setBiermannCoeff(v),
                FLAG_BIERMANN, { lo: -8, hi: -1 }),
            epSlider('Self-gravity G (log10)',
                () => sim.gravityG, v => sim.setGravityG(v),
                FLAG_GRAVITY_SELF, { lo: -4, hi: 2 }),
            { type: 'slider', label: 'Poisson iters', min: 0, max: 128, step: 1,
              value: sim.gravityPoissonIters, format: v => String(v | 0),
              onChange: v => sim.setGravityPoissonIters(v | 0) },
            { type: 'slider', label: 'Gravity softening', min: 0, max: 0.2, step: 0.005,
              value: sim.gravitySoftening, format: v => (v <= 0 ? 'off' : v.toFixed(3)),
              onChange: v => sim.setGravitySoftening(v) },
            { type: 'slider', label: 'Jacobi ω', min: 0.2, max: 1.8, step: 0.05,
              value: sim.gravityPoissonOmega, format: v => v.toFixed(2),
              onChange: v => sim.setGravityPoissonOmega(v) },
            { type: 'mode', label: 'Geometry', dataAttr: 'geometry-mode',
              buttons: [
                  { value: String(GEOMETRY_CARTESIAN),   label: 'cart', active: sim.geometryMode === GEOMETRY_CARTESIAN },
                  { value: String(GEOMETRY_CYLINDRICAL), label: 'cyl',  active: sim.geometryMode === GEOMETRY_CYLINDRICAL },
              ],
              onChange: v => {
                  const mode = parseInt(v, 10);
                  sim.setGeometryMode(mode);
                  sim.setPhysicsFlag(FLAG_GEOMETRY, mode === GEOMETRY_CYLINDRICAL);
              } },
            { type: 'slider', label: 'r-axis guard', min: 0, max: 0.25, step: 0.005,
              value: sim.geometryRMin, format: v => v.toFixed(3),
              onChange: v => sim.setGeometryRMin(v) },
            { type: 'slider', label: 'Sponge width', min: 0, max: 32, step: 1,
              value: sim.spongeWidth, format: v => (v <= 0 ? 'off' : v.toFixed(0) + ' cells'),
              onChange: v => {
                  sim.setSpongeWidth(v);
                  sim.setPhysicsFlag(FLAG_SPONGE, v > 0 && sim.spongeStrength > 0);
              } },
            epSlider('Sponge strength (log10)',
                () => sim.spongeStrength, v => sim.setSpongeStrength(v),
                FLAG_SPONGE, { lo: -3, hi: 1 }),
            { type: 'slider', label: 'Source substeps cap', min: 1, max: 64, step: 1,
              value: sim.sourceSubstepsMax, format: v => String(v | 0),
              onChange: v => sim.setSourceSubstepsMax(v | 0) },
            { type: 'mode', label: 'EMF', dataAttr: 'emf-mode',
              buttons: [
                  { value: String(EMF_MODE_BS_MEAN),   label: 'BS mean',  active: sim.emfMode === EMF_MODE_BS_MEAN },
                  { value: String(EMF_MODE_GS_UPWIND), label: 'GS upwind', active: sim.emfMode === EMF_MODE_GS_UPWIND },
              ],
              onChange: v => sim.setEmfMode(parseInt(v, 10)) },
            { type: 'toggle', label: 'Positivity guard',
              checked: (sim.physicsFlags & FLAG_POSITIVITY) !== 0,
              onChange: on => sim.setPhysicsFlag(FLAG_POSITIVITY, on) },

            { type: 'slider', label: 'LIC intensity', min: 0, max: 2, step: 0.05,
              value: sim.licIntensity, format: v => v.toFixed(2),
              onChange: v => sim.setLicIntensity(v) },
            { type: 'slider', label: 'LIC drift', min: 0, max: 4, step: 0.1,
              value: sim.licDriftX, format: v => v.toFixed(1) + ' px/s',
              onChange: v => sim.setLicDrift(v, sim.licDriftY) },
        ], { width: 320 });
    }

    // ── About panel ───────────────────────────────────────────
    if (typeof initAboutPanel === 'function') {
        const aboutHandle = initAboutPanel({
            title: 'Plasma',
            lastUpdated: '2026-05-27',
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
        { key: '2',      label: 'Stats tab',              group: 'Sidebar',    action: () => switchTab('stats') },
        { key: '3',      label: 'Probe tab',              group: 'Sidebar',    action: () => switchTab('probe') },
    ];
}

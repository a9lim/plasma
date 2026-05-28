/**
 * @fileoverview Physics tab — extended source-physics controls.
 *
 * Until the Session 22 coherence pass these knobs lived in the gear
 * dropdown as ~35 flat slider rows. They're grouped here into eight
 * subsections (Hall, Cooling & Heating, Conduction, Radiation,
 * Viscosity, Non-ideal Ohm, Gravity, Geometry & sponge) using the
 * standard plasma sidebar idioms — `.panel-section` + `.group-label`
 * + `.control-group` rows — so the visual structure carries the same
 * information that the AGENTS.md table does.
 *
 * The "snap-to-off below floor" log-slider behavior (originally written
 * inline in ui.js as `epSlider`) is replicated as `epSlider()` here —
 * raising a slider above the off boundary re-arms the corresponding
 * `FLAG_*` bit and writes the scalar; falling below it clears both.
 * Shaders early-return when either is unset, so the user can never
 * leave a half-engaged source physics state behind.
 *
 * Mode groups go through `_forms.bindModeGroup` for the sliding
 * indicator. Toggles use the standard `.tog-wrap` markup so they pick
 * up shared-base.css automatically. No `innerHTML`.
 */

import {
    FLAG_COOLING, FLAG_GRAVITY_SELF, FLAG_CONDUCTION, FLAG_HALL,
    FLAG_AMBIPOLAR, FLAG_BIERMANN, FLAG_VISCOSITY, FLAG_HEATING,
    FLAG_SPONGE, FLAG_GEOMETRY, FLAG_RADIATION, FLAG_ELECTRON_INERTIA,
    COOLING_CURVE_BREMS, COOLING_CURVE_TABLE, COOLING_CURVE_CIE,
    COOLING_CURVE_TABULATED,
    GEOMETRY_CARTESIAN, GEOMETRY_CYLINDRICAL,
    GRAVITY_BOUNDARY_PERIODIC, GRAVITY_BOUNDARY_ISOLATED,
    GRAVITY_SOLVER_MULTIGRID, GRAVITY_SOLVER_JACOBI,
} from './config.js';

/* ─── small DOM helpers (match the existing index.html Settings idiom) ─── */

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

function section(label) {
    const sec = el('div', 'panel-section');
    sec.append(el('h2', 'group-label', label));
    return sec;
}

/**
 * Standard plasma slider row: label on top with right-aligned value
 * span, native range underneath. Mirrors the #ctrl-eta block in
 * index.html so the styling lines up without per-row CSS.
 */
function sliderRow(parent, label, opts) {
    const { min, max, step, value, format, onChange, hint } = opts;
    const wrap = el('div', 'control-group');
    const lbl = document.createElement('label');
    lbl.append(el('span', null, label));
    const valSpan = el('span', 'slider-value', format(value));
    lbl.append(valSpan);
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.addEventListener('input', () => {
        const v = parseFloat(slider.value);
        valSpan.textContent = format(v);
        onChange(v);
    });
    wrap.append(lbl, slider);
    if (hint) wrap.append(el('p', 'panel-hint', hint));
    parent.append(wrap);
    return { slider, valSpan };
}

/**
 * Log-scale slider with a snap-to-off floor that also clears (or
 * arms) the corresponding FLAG_* bit. `lo` is the lowest active
 * decade — actual slider range extends 0.5 below it; values at or
 * below `lo - 0.05` render as "off" and disable the flag.
 *
 * `getScalar` reads the current sim value to seed the slider position
 * so re-opening the tab reflects whatever the preset set.
 */
function epSlider(parent, label, opts) {
    const { lo = -6, hi = 0, step = 0.25,
            getScalar, setScalar, flag, sim, hint } = opts;
    const offBoundary = lo - 0.05;
    const cur = getScalar();
    const valueAt = cur > 0 ? Math.log10(cur) : (lo - 0.5);
    return sliderRow(parent, label, {
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
        hint,
    });
}

/** Mode group built on `.mode-toggles` so shared-forms wires the indicator. */
function modeRow(parent, label, opts) {
    const { dataAttr, buttons, onChange } = opts;
    const wrap = el('div', 'control-group');
    wrap.append(el('label', null, label));
    const group = el('div', 'mode-toggles');
    group.setAttribute('data-scope', dataAttr);
    for (const b of buttons) {
        const btn = document.createElement('button');
        btn.className = 'mode-btn' + (b.active ? ' active' : '');
        btn.setAttribute('data-' + dataAttr, b.value);
        btn.textContent = b.label;
        group.append(btn);
    }
    wrap.append(group);
    parent.append(wrap);
    if (typeof _forms !== 'undefined' && _forms.bindModeGroup) {
        _forms.bindModeGroup(group, dataAttr, onChange);
    }
    return group;
}

/* ─── panel builder ──────────────────────────────────────────── */

/**
 * Build the Physics tab DOM under `root`. `sim` is the orchestrator
 * from main.js — same one the gear dropdown talks to.
 *
 * Stateless — there's nothing to tear down on resolution change since
 * every slider's onChange just calls a sim setter, and the sim layer
 * absorbs the resolution swap. The seed values may go slightly stale
 * if a preset reload changes the scalar without going through the
 * panel, but the gear dropdown has the same property and it hasn't
 * been a problem in practice.
 */
export function buildPhysicsPanel(root, sim) {

    /* Hall */
    {
        const s = section('Hall');
        epSlider(s, 'd_i (log10)', {
            lo: -4, hi: 0, sim,
            getScalar: () => sim.hallDi,
            setScalar: v => sim.setHallDi(v),
            flag: FLAG_HALL,
        });
        sliderRow(s, 'p_e / p', {
            min: 0, max: 1, step: 0.05,
            value: sim.hallElectronPressureFrac,
            format: v => v.toFixed(2),
            onChange: v => sim.setHallElectronPressureFrac(v),
        });
        root.append(s);
    }

    /* Cooling & Heating */
    {
        const s = section('Cooling & Heating');
        epSlider(s, 'Λ₀ (log10)', {
            lo: -4, hi: 0, sim,
            getScalar: () => sim.coolingLambda0,
            setScalar: v => sim.setCoolingLambda0(v),
            flag: FLAG_COOLING,
        });
        modeRow(s, 'Curve', {
            dataAttr: 'cooling-curve',
            buttons: [
                { value: String(COOLING_CURVE_TABULATED), label: 'tab',
                  active: sim.coolingCurveMode === COOLING_CURVE_TABULATED },
                { value: String(COOLING_CURVE_CIE),   label: 'CIE',
                  active: sim.coolingCurveMode === COOLING_CURVE_CIE },
                { value: String(COOLING_CURVE_TABLE), label: 'table',
                  active: sim.coolingCurveMode === COOLING_CURVE_TABLE },
                { value: String(COOLING_CURVE_BREMS), label: 'brems',
                  active: sim.coolingCurveMode === COOLING_CURVE_BREMS },
            ],
            onChange: v => sim.setCoolingCurveMode(parseInt(v, 10)),
        });
        sliderRow(s, 'Metallicity Z', {
            min: 0, max: 3, step: 0.1,
            value: sim.coolingMetallicity,
            format: v => v.toFixed(1) + '×',
            onChange: v => sim.setCoolingMetallicity(v),
        });
        epSlider(s, 'Heating Γ (log10)', {
            lo: -6, hi: 0, sim,
            getScalar: () => sim.heatingGamma0,
            setScalar: v => sim.setHeatingGamma0(v),
            flag: FLAG_HEATING,
        });
        sliderRow(s, 'Heating ρ exponent', {
            min: 0, max: 2, step: 0.1,
            value: sim.heatingDensityExp,
            format: v => v.toFixed(1),
            onChange: v => sim.setHeatingDensityExp(v),
        });
        sliderRow(s, 'Heating T cutoff', {
            min: -4.5, max: 2, step: 0.25,
            value: sim.heatingTCut > 0 ? Math.log10(sim.heatingTCut) : -4.5,
            format: v => (v <= -4.45 ? 'off' : '1e' + v.toFixed(2)),
            onChange: v => sim.setHeatingTCut(v <= -4.45 ? 0 : Math.pow(10, v)),
        });
        root.append(s);
    }

    /* Conduction */
    {
        const s = section('Conduction');
        epSlider(s, 'κ∥ (log10)', {
            lo: -6, hi: 0, sim,
            getScalar: () => sim.conductionKappa,
            setScalar: v => sim.setConductionKappa(v),
            flag: FLAG_CONDUCTION,
        });
        sliderRow(s, 'κ⊥ / κ∥', {
            min: 0, max: 1, step: 0.05,
            value: sim.conductionIsoFrac,
            format: v => v.toFixed(2),
            onChange: v => sim.setConductionIsoFrac(v),
        });
        sliderRow(s, 'q_sat φ', {
            min: 0, max: 1, step: 0.05,
            value: sim.conductionSatFrac,
            format: v => (v <= 0 ? 'off' : v.toFixed(2)),
            onChange: v => sim.setConductionSatFrac(v),
        });
        root.append(s);
    }

    /* Radiation */
    {
        const s = section('Radiation');
        epSlider(s, 'c (log10)', {
            lo: -2, hi: 2, sim,
            getScalar: () => sim.radiationC,
            setScalar: v => sim.setRadiationC(v),
            flag: FLAG_RADIATION,
        });
        sliderRow(s, 'κ_abs', {
            min: -4.5, max: 2, step: 0.25,
            value: sim.radiationKappaAbs > 0 ? Math.log10(sim.radiationKappaAbs) : -4.5,
            format: v => (v <= -4.45 ? 'off' : '1e' + v.toFixed(2)),
            onChange: v => sim.setRadiationKappaAbs(v <= -4.45 ? 0 : Math.pow(10, v)),
        });
        sliderRow(s, 'κ_scat', {
            min: -4.5, max: 2, step: 0.25,
            value: sim.radiationKappaScat > 0 ? Math.log10(sim.radiationKappaScat) : -4.5,
            format: v => (v <= -4.45 ? 'off' : '1e' + v.toFixed(2)),
            onChange: v => sim.setRadiationKappaScat(v <= -4.45 ? 0 : Math.pow(10, v)),
        });
        sliderRow(s, 'a_r (log10)', {
            min: -4, max: 1, step: 0.25,
            value: Math.log10(Math.max(sim.radiationConst, 1e-30)),
            format: v => '1e' + v.toFixed(2),
            onChange: v => sim.setRadiationConst(Math.pow(10, v)),
        });
        root.append(s);
    }

    /* Viscosity */
    {
        const s = section('Viscosity');
        epSlider(s, 'ν (log10)', {
            lo: -7, hi: -1, sim,
            getScalar: () => sim.viscosityNu,
            setScalar: v => sim.setViscosityNu(v),
            flag: FLAG_VISCOSITY,
        });
        sliderRow(s, 'Bulk', {
            min: -7.5, max: -1, step: 0.25,
            value: sim.viscosityBulk > 0 ? Math.log10(sim.viscosityBulk) : -7.5,
            format: v => (v <= -7.45 ? 'off' : '1e' + v.toFixed(2)),
            onChange: v => sim.setViscosityBulk(v <= -7.45 ? 0 : Math.pow(10, v)),
        });
        sliderRow(s, 'B-aligned frac', {
            min: 0, max: 1, step: 0.05,
            value: sim.viscosityAnisoFrac,
            format: v => v.toFixed(2),
            onChange: v => sim.setViscosityAnisoFrac(v),
        });
        sliderRow(s, 'Shock', {
            min: -7.5, max: -1, step: 0.25,
            value: sim.viscosityShock > 0 ? Math.log10(sim.viscosityShock) : -7.5,
            format: v => (v <= -7.45 ? 'off' : '1e' + v.toFixed(2)),
            onChange: v => sim.setViscosityShock(v <= -7.45 ? 0 : Math.pow(10, v)),
        });
        root.append(s);
    }

    /* Non-ideal Ohm */
    {
        const s = section('Non-ideal Ohm');
        epSlider(s, 'Ambipolar η_A (log10)', {
            lo: -7, hi: -1, sim,
            getScalar: () => sim.ambipolarEta,
            setScalar: v => sim.setAmbipolarEta(v),
            flag: FLAG_AMBIPOLAR,
        });
        sliderRow(s, 'Neutral fraction', {
            min: 0, max: 1, step: 0.05,
            value: sim.neutralFrac,
            format: v => v.toFixed(2),
            onChange: v => sim.setNeutralFrac(v),
        });
        sliderRow(s, 'Ionization T₀', {
            min: -4, max: 2, step: 0.25,
            value: Math.log10(sim.ionizationT0),
            format: v => '1e' + v.toFixed(2),
            onChange: v => sim.setIonizationT0(Math.pow(10, v)),
        });
        epSlider(s, 'Biermann C_B (log10)', {
            lo: -8, hi: -1, sim,
            getScalar: () => Math.abs(sim.biermannCoeff),
            setScalar: v => sim.setBiermannCoeff(v),
            flag: FLAG_BIERMANN,
        });
        epSlider(s, 'Electron d_e (log10)', {
            lo: -5, hi: -1, sim,
            getScalar: () => sim.electronInertiaLength,
            setScalar: v => sim.setElectronInertiaLength(v),
            flag: FLAG_ELECTRON_INERTIA,
        });
        sliderRow(s, 'Electron damping', {
            min: -4.5, max: 0, step: 0.25,
            value: sim.electronInertiaDamping > 0 ? Math.log10(sim.electronInertiaDamping) : -4.5,
            format: v => (v <= -4.45 ? 'off' : '1e' + v.toFixed(2)),
            onChange: v => {
                sim.setElectronInertiaDamping(v <= -4.45 ? 0 : Math.pow(10, v));
                sim.setPhysicsFlag(
                    FLAG_ELECTRON_INERTIA,
                    sim.electronInertiaLength > 0 && v > -4.45,
                );
            },
        });
        root.append(s);
    }

    /* Gravity */
    {
        const s = section('Gravity');
        epSlider(s, 'Self-gravity G (log10)', {
            lo: -4, hi: 2, sim,
            getScalar: () => sim.gravityG,
            setScalar: v => sim.setGravityG(v),
            flag: FLAG_GRAVITY_SELF,
        });
        sliderRow(s, 'Poisson iters', {
            min: 0, max: 128, step: 1,
            value: sim.gravityPoissonIters,
            format: v => String(v | 0),
            onChange: v => sim.setGravityPoissonIters(v | 0),
        });
        sliderRow(s, 'Softening', {
            min: 0, max: 0.2, step: 0.005,
            value: sim.gravitySoftening,
            format: v => (v <= 0 ? 'off' : v.toFixed(3)),
            onChange: v => sim.setGravitySoftening(v),
        });
        sliderRow(s, 'Jacobi ω', {
            min: 0.2, max: 1.8, step: 0.05,
            value: sim.gravityPoissonOmega,
            format: v => v.toFixed(2),
            onChange: v => sim.setGravityPoissonOmega(v),
        });
        modeRow(s, 'Boundary', {
            dataAttr: 'gravity-boundary',
            buttons: [
                { value: String(GRAVITY_BOUNDARY_PERIODIC), label: 'periodic',
                  active: sim.gravityBoundaryMode === GRAVITY_BOUNDARY_PERIODIC },
                { value: String(GRAVITY_BOUNDARY_ISOLATED), label: 'isolated',
                  active: sim.gravityBoundaryMode === GRAVITY_BOUNDARY_ISOLATED },
            ],
            onChange: v => sim.setGravityBoundaryMode(v | 0),
        });
        modeRow(s, 'Solver', {
            dataAttr: 'gravity-solver',
            buttons: [
                { value: String(GRAVITY_SOLVER_MULTIGRID), label: 'multi',
                  active: sim.gravitySolverMode === GRAVITY_SOLVER_MULTIGRID },
                { value: String(GRAVITY_SOLVER_JACOBI), label: 'jacobi',
                  active: sim.gravitySolverMode === GRAVITY_SOLVER_JACOBI },
            ],
            onChange: v => sim.setGravitySolverMode(v | 0),
        });
        root.append(s);
    }

    /* Geometry & sponge */
    {
        const s = section('Geometry & sponge');
        modeRow(s, 'Geometry', {
            dataAttr: 'geometry-mode',
            buttons: [
                { value: String(GEOMETRY_CARTESIAN),   label: 'cart',
                  active: sim.geometryMode === GEOMETRY_CARTESIAN },
                { value: String(GEOMETRY_CYLINDRICAL), label: 'cyl',
                  active: sim.geometryMode === GEOMETRY_CYLINDRICAL },
            ],
            onChange: v => {
                const mode = parseInt(v, 10);
                sim.setGeometryMode(mode);
                sim.setPhysicsFlag(FLAG_GEOMETRY, mode === GEOMETRY_CYLINDRICAL);
            },
        });
        sliderRow(s, 'r-axis guard', {
            min: 0, max: 0.25, step: 0.005,
            value: sim.geometryRMin,
            format: v => v.toFixed(3),
            onChange: v => sim.setGeometryRMin(v),
        });
        sliderRow(s, 'Sponge width', {
            min: 0, max: 32, step: 1,
            value: sim.spongeWidth,
            format: v => (v <= 0 ? 'off' : v.toFixed(0) + ' cells'),
            onChange: v => {
                sim.setSpongeWidth(v);
                sim.setPhysicsFlag(FLAG_SPONGE, v > 0 && sim.spongeStrength > 0);
            },
        });
        epSlider(s, 'Sponge strength (log10)', {
            lo: -3, hi: 1, sim,
            getScalar: () => sim.spongeStrength,
            setScalar: v => sim.setSpongeStrength(v),
            flag: FLAG_SPONGE,
        });
        root.append(s);
    }
}

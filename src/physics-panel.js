/**
 * @fileoverview Physics tab — extended source-physics controls.
 *
 * Eight subsections (Hall, Cooling & Heating, Conduction, Radiation,
 * Viscosity, Non-ideal Ohm, Gravity, Geometry & sponge) built from the
 * shared sidebar builders in panel-ui.js, so the visual structure carries
 * the same information as the AGENTS.md table. Every slider range comes
 * from config.js `SLIDER_BOUNDS` — no inline min/max/step here.
 *
 * epSlider arms/clears the matching `FLAG_*` bit as it crosses its off
 * boundary, so a source physics term can't be left half-engaged (shaders
 * early-return when the flag or scalar is unset). Linear sliders that snap
 * to 0 at the bottom treat `bounds.min + 0.05` as their off boundary. Mode
 * groups use `_forms.bindModeGroup` for the sliding indicator. No
 * `innerHTML`.
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
    SLIDER_BOUNDS,
} from './config.js';
import { section, sliderRow, epSlider, modeRow } from './panel-ui.js';

const B = SLIDER_BOUNDS;

/**
 * Build the Physics tab DOM under `root`. `sim` is the orchestrator from
 * main.js — the same one the Settings-tab controls talk to.
 *
 * Stateless: every slider's onChange just calls a sim setter and the sim
 * layer absorbs resolution swaps, so there's nothing to tear down. Seed
 * values may go slightly stale if a preset reload changes a scalar without
 * going through the panel — acceptable, same as the rest of the UI.
 */
export function buildPhysicsPanel(root, sim) {

    /* Hall */
    {
        const s = section('Hall');
        epSlider(s, 'd_i (log10)', {
            ...B.hallDi, sim,
            getScalar: () => sim.hallDi,
            setScalar: v => sim.setHallDi(v),
            flag: FLAG_HALL,
        });
        sliderRow(s, 'p_e / p', {
            ...B.hallElectronPe,
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
            ...B.coolingLambda0, sim,
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
            ...B.coolingMetallicity,
            value: sim.coolingMetallicity,
            format: v => v.toFixed(1) + '×',
            onChange: v => sim.setCoolingMetallicity(v),
        });
        epSlider(s, 'Heating Γ (log10)', {
            ...B.heatingGamma0, sim,
            getScalar: () => sim.heatingGamma0,
            setScalar: v => sim.setHeatingGamma0(v),
            flag: FLAG_HEATING,
        });
        sliderRow(s, 'Heating ρ exponent', {
            ...B.heatingDensityExp,
            value: sim.heatingDensityExp,
            format: v => v.toFixed(1),
            onChange: v => sim.setHeatingDensityExp(v),
        });
        {
            const off = B.heatingTCut.min + 0.05;
            sliderRow(s, 'Heating T cutoff', {
                ...B.heatingTCut,
                value: sim.heatingTCut > 0 ? Math.log10(sim.heatingTCut) : B.heatingTCut.min,
                format: v => (v <= off ? 'off' : '1e' + v.toFixed(2)),
                onChange: v => sim.setHeatingTCut(v <= off ? 0 : Math.pow(10, v)),
            });
        }
        root.append(s);
    }

    /* Conduction */
    {
        const s = section('Conduction');
        epSlider(s, 'κ∥ (log10)', {
            ...B.conductionKappa, sim,
            getScalar: () => sim.conductionKappa,
            setScalar: v => sim.setConductionKappa(v),
            flag: FLAG_CONDUCTION,
        });
        sliderRow(s, 'κ⊥ / κ∥', {
            ...B.conductionIsoFrac,
            value: sim.conductionIsoFrac,
            format: v => v.toFixed(2),
            onChange: v => sim.setConductionIsoFrac(v),
        });
        sliderRow(s, 'q_sat φ', {
            ...B.conductionSatFrac,
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
            ...B.radiationC, sim,
            getScalar: () => sim.radiationC,
            setScalar: v => sim.setRadiationC(v),
            flag: FLAG_RADIATION,
        });
        {
            const off = B.radiationKappaAbs.min + 0.05;
            sliderRow(s, 'κ_abs', {
                ...B.radiationKappaAbs,
                value: sim.radiationKappaAbs > 0 ? Math.log10(sim.radiationKappaAbs) : B.radiationKappaAbs.min,
                format: v => (v <= off ? 'off' : '1e' + v.toFixed(2)),
                onChange: v => sim.setRadiationKappaAbs(v <= off ? 0 : Math.pow(10, v)),
            });
        }
        {
            const off = B.radiationKappaScat.min + 0.05;
            sliderRow(s, 'κ_scat', {
                ...B.radiationKappaScat,
                value: sim.radiationKappaScat > 0 ? Math.log10(sim.radiationKappaScat) : B.radiationKappaScat.min,
                format: v => (v <= off ? 'off' : '1e' + v.toFixed(2)),
                onChange: v => sim.setRadiationKappaScat(v <= off ? 0 : Math.pow(10, v)),
            });
        }
        sliderRow(s, 'a_r (log10)', {
            ...B.radiationConst,
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
            ...B.viscosityNu, sim,
            getScalar: () => sim.viscosityNu,
            setScalar: v => sim.setViscosityNu(v),
            flag: FLAG_VISCOSITY,
        });
        {
            const off = B.viscosityBulk.min + 0.05;
            sliderRow(s, 'Bulk', {
                ...B.viscosityBulk,
                value: sim.viscosityBulk > 0 ? Math.log10(sim.viscosityBulk) : B.viscosityBulk.min,
                format: v => (v <= off ? 'off' : '1e' + v.toFixed(2)),
                onChange: v => sim.setViscosityBulk(v <= off ? 0 : Math.pow(10, v)),
            });
        }
        sliderRow(s, 'B-aligned frac', {
            ...B.viscosityAnisoFrac,
            value: sim.viscosityAnisoFrac,
            format: v => v.toFixed(2),
            onChange: v => sim.setViscosityAnisoFrac(v),
        });
        {
            const off = B.viscosityShock.min + 0.05;
            sliderRow(s, 'Shock', {
                ...B.viscosityShock,
                value: sim.viscosityShock > 0 ? Math.log10(sim.viscosityShock) : B.viscosityShock.min,
                format: v => (v <= off ? 'off' : '1e' + v.toFixed(2)),
                onChange: v => sim.setViscosityShock(v <= off ? 0 : Math.pow(10, v)),
            });
        }
        root.append(s);
    }

    /* Non-ideal Ohm */
    {
        const s = section('Non-ideal Ohm');
        epSlider(s, 'Ambipolar η_A (log10)', {
            ...B.ambipolarEta, sim,
            getScalar: () => sim.ambipolarEta,
            setScalar: v => sim.setAmbipolarEta(v),
            flag: FLAG_AMBIPOLAR,
        });
        sliderRow(s, 'Neutral fraction', {
            ...B.neutralFrac,
            value: sim.neutralFrac,
            format: v => v.toFixed(2),
            onChange: v => sim.setNeutralFrac(v),
        });
        sliderRow(s, 'Ionization T₀', {
            ...B.ionizationT0,
            value: Math.log10(sim.ionizationT0),
            format: v => '1e' + v.toFixed(2),
            onChange: v => sim.setIonizationT0(Math.pow(10, v)),
        });
        epSlider(s, 'Biermann C_B (log10)', {
            ...B.biermannCoeff, sim,
            getScalar: () => Math.abs(sim.biermannCoeff),
            setScalar: v => sim.setBiermannCoeff(v),
            flag: FLAG_BIERMANN,
        });
        epSlider(s, 'Electron d_e (log10)', {
            ...B.electronInertiaLen, sim,
            getScalar: () => sim.electronInertiaLength,
            setScalar: v => sim.setElectronInertiaLength(v),
            flag: FLAG_ELECTRON_INERTIA,
        });
        {
            const off = B.electronDamping.min + 0.05;
            sliderRow(s, 'Electron damping', {
                ...B.electronDamping,
                value: sim.electronInertiaDamping > 0 ? Math.log10(sim.electronInertiaDamping) : B.electronDamping.min,
                format: v => (v <= off ? 'off' : '1e' + v.toFixed(2)),
                onChange: v => {
                    sim.setElectronInertiaDamping(v <= off ? 0 : Math.pow(10, v));
                    sim.setPhysicsFlag(
                        FLAG_ELECTRON_INERTIA,
                        sim.electronInertiaLength > 0 && v > off,
                    );
                },
            });
        }
        root.append(s);
    }

    /* Gravity */
    {
        const s = section('Gravity');
        epSlider(s, 'Self-gravity G (log10)', {
            ...B.gravityG, sim,
            getScalar: () => sim.gravityG,
            setScalar: v => sim.setGravityG(v),
            flag: FLAG_GRAVITY_SELF,
        });
        sliderRow(s, 'Poisson iters', {
            ...B.gravityPoissonIters,
            value: sim.gravityPoissonIters,
            format: v => String(v | 0),
            onChange: v => sim.setGravityPoissonIters(v | 0),
        });
        sliderRow(s, 'Softening', {
            ...B.gravitySoftening,
            value: sim.gravitySoftening,
            format: v => (v <= 0 ? 'off' : v.toFixed(3)),
            onChange: v => sim.setGravitySoftening(v),
        });
        sliderRow(s, 'Jacobi ω', {
            ...B.gravityPoissonOmega,
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
            ...B.geometryRMin,
            value: sim.geometryRMin,
            format: v => v.toFixed(3),
            onChange: v => sim.setGeometryRMin(v),
        });
        sliderRow(s, 'Sponge width', {
            ...B.spongeWidth,
            value: sim.spongeWidth,
            format: v => (v <= 0 ? 'off' : v.toFixed(0) + ' cells'),
            onChange: v => {
                sim.setSpongeWidth(v);
                sim.setPhysicsFlag(FLAG_SPONGE, v > 0 && sim.spongeStrength > 0);
            },
        });
        epSlider(s, 'Sponge strength (log10)', {
            ...B.spongeStrength, sim,
            getScalar: () => sim.spongeStrength,
            setScalar: v => sim.setSpongeStrength(v),
            flag: FLAG_SPONGE,
        });
        root.append(s);
    }
}

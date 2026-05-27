# Session 17 — second realism layer

Goal: push the Session 14-16 extended-physics stack beyond "breadth-pass
toggles" toward more realistic source physics while preserving the
canonical MHD verification presets.

## Landed

* **Uniform contract expanded to 256 B.** Slots 32-63 now hold
  metallicity/heating, ambipolar/Biermann, viscosity, source substep,
  geometry, gravity-softening, and sponge controls.
* **Cooling/heating.** `apply-cooling.wgsl` keeps the exact brems and
  Session-16 piecewise modes, and adds `cooling_curve_mode = 2`: a
  broader CIE-inspired, metallicity-scaled power-law table. Optional
  volumetric heating (`FLAG_HEATING`) supports thermal-balance runs.
* **Partial ionization / generalized Ohm.** New
  `apply-nonideal.wgsl` adds ambipolar diffusion (`η_A J_perp`) and a
  Biermann battery source for `B_z`. It is split into frozen-state
  `compute_emf` + `apply_update`, and is sub-cycled separately from Hall.
* **Viscous transport.** New `apply-viscosity.wgsl` adds shear, bulk,
  compression-triggered shock viscosity, optional B-aligned projection,
  momentum diffusion, and viscous work/heating in total energy.
* **Gravity/geometry/boundaries.** `solve-poisson.wgsl` supports
  weighted Jacobi and a screened softening length. New
  `apply-geometry.wgsl` adds a cylindrical axisymmetric source layer
  (`x=r`, `y=z`) and a pressure-preserving boundary sponge.
* **Host wiring.** `sim.js` now saves/loads/pushes all new controls,
  sizes viscosity/non-ideal substeps host-side, seeds dedicated
  `visc_dt` / `nonideal_dt` buffers, and tracks their last substep counts.
* **UI/stats.** Advanced settings expose the new knobs; Stats reports
  Hall, conduction, viscosity, and non-ideal substep counts.
* **Preset defaults.** Canonical presets still use `BASE_PHYSICS_FLAGS`.
  `orszag-tang-extended` now exercises the richer stack with modest
  nonzero source values; geometry and sponge remain manual.

## Caveats

* The CIE curve is an in-shader code-unit fit, not a loaded CHIANTI or
  Sutherland-Dopita data product. Good shape, not publication data.
* Ambipolar and Biermann terms need dedicated validation presets before
  treating them as quantitative.
* Viscosity is explicit and sub-cycled, not RKL2/implicit.
* The cylindrical mode is a source-layer approximation over the existing
  Cartesian finite-volume/CT core, not a full r-weighted solver.

## Verification

* `node --check` on modified JS modules.
* `node tests/wgsl-transpile/run.js plasma --quiet`
* `node tests/wgsl-transpile/smoke.js`
* `node tests/wgsl-transpile/runner-smoke.js`
* `node _build.mjs`
* `node tests/wgsl-transpile/build-smoke.js`
* Browser load at `http://127.0.0.1:8765/plasma/` with no warnings or
  errors reported by Playwright console capture.

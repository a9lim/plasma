# Session 21 — data-backed cooling and source-cap safety

This pass took the first three next-realism items from the physics roadmap and
landed the bounded versions that fit the current WebGPU architecture.

## Landed

- Replaced the hand-shaped uploaded cooling row in `src/microphysics.js` with
  sampled public Sutherland-Dopita 1993 solar CIE data (`m-00.cie`), normalized
  to `Lambda(1e6 K)`. The table layout stays unchanged, so
  `apply-cooling.wgsl` and `apply-radiation.wgsl` keep the same storage-buffer
  contract.
- Added dataset provenance via `MICROPHYSICS_DATASET` and tightened the
  microphysics validation row to require the expected SD93 curve shape.
- Added a source-substep sizing validation row that exercises the new
  high-stiffness path without a visual preset.
- Changed source-rate feedback from "diagnostic-only" to a macro-dt safety
  limiter when the configured soft cap would not cover the explicit stability
  requirement.
- Made `hallSubstepsMax` and `sourceSubstepsMax` soft performance targets. The
  host now takes the required Hall/conduction/viscosity/non-ideal/radiation
  substeps up to an internal hard safety ceiling while the GPU dt reducer
  shrinks the next macro step.
- Integrated cylindrical toroidal momentum and `B_phi` curvature with an exact
  frozen-coefficient update for the shared `-v_r y/r` dilution term. That
  matches the existing exact continuity treatment and reduces first-order
  angular-momentum / toroidal-field drift near the axis.
- Regenerated the parent repo's WGSL-transpiled artifacts for
  `compute-dt.wgsl` and `apply-geometry.wgsl`.

## Verification

- `node --check src/sim.js && node --check src/microphysics.js && node --check src/gpu/pipelines.js`
- Host microphysics sanity import: dataset `sd93-solar-cie-v1`, table length
  `576`.
- `node tests/wgsl-transpile/smoke.js` — `118 passed, 0 failed`.
- `node tests/wgsl-transpile/runner-smoke.js` — `6 passed, 0 failed`.
- `node tests/wgsl-transpile/run.js plasma --quiet` — all 34 plasma shaders
  passed tokenize/parse/resolve/compile.
- `node _build.mjs` — final run wrote the four compute-dt shader artifacts
  affected by the source-cap limiter.
- `node tests/wgsl-transpile/build-smoke.js` — `11 passed, 0 failed`.
- `python3 tests/physics-validation.py --static-root --port 8090 --timeout 600 --out /tmp/plasma-physics-validation-session21.json`
- Browser validation result: `21 passed, 0 failed`.
- New rows: microphysics `sd93-solar-cie-v1`, `cool peak/trough=11.139`,
  `tail/low=2.399e+5`; source substep sizing `expected 66, Hall=66, cond=66`.

## Still future work

- The cooling data is now externally sourced for the solar CIE row, but the
  table is still one-dimensional in temperature. A real production pass wants
  metallicity/density grids and vetted opacity/heating products behind the same
  upload contract.
- Explicit source subcycling is safer, not architecturally final. HDS/IMEX for
  Hall/non-ideal terms and RKL2/implicit diffusion are still the next jump for
  stiff workloads.
- Cylindrical mode has better toroidal balance, but strongly radial rotating
  or magnetized flows still deserve a higher-order angular-momentum and
  magnetic-energy audit.

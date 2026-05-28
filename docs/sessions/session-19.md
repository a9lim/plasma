# Session 19 — validation, calibration, and source-coupling hardening

This pass turned the extended physics layer from "feature-present" into
something closer to a regression-gated physics engine.

## Landed

- Added `tests/physics-validation.html` plus `tests/physics-validation.py`,
  a production-WebGPU matrix that currently covers 11 checks: Sod, Brio-Wu,
  Orszag-Tang, Harris, driven wind/cloud, per-edge driven BC fill, cooling
  monotonicity, anisotropic conduction, Hall whistler dispersion, Jeans
  growth, and cylindrical expansion.
- Added `tests/physics-test-utils.js` for reusable GPU readback/state
  summaries and source-only stepping.
- Added dimensioned/dimensionless calibration helpers in
  `src/physical-scales.js`; the extended OT and driven wind/cloud presets now
  derive coefficients from target Reynolds/Peclet/cooling/Hall/ambipolar/
  Biermann/viscosity scales instead of unlabelled literals.
- Fixed the Hall whistler validation preset to use the Toth/Ma/Gombosi
  right-going branch speed `c_w = w/2 + sqrt(v_A^2 + w^2/4)` and eigenvector
  ratio `|delta v|/|delta B| = |B0|/(c_w rho)`. The old IC mixed branches and
  made the phase test mostly meaningless.
- Replaced the old Lie-split extended source step with Strang-style source
  bracketing: `S(dt/2) H(dt) RKL2(dt) S(dt/2)`. Cooling, gravity, and geometry
  read the fresh GPU `dt_half`; Hall/conduction/viscosity/non-ideal substeps
  now get GPU-divided from the same fresh half-step through `source-dt.wgsl`.
- Replaced the all-or-nothing conserved-state positivity fallback with a local
  theta limiter that scales only as much of the flux update as needed before
  the existing magnetic-pressure-aware energy floor pass.
- Made cylindrical geometry's continuity source positivity-preserving by
  integrating `d rho / dt = -rho v_r / r` exactly for frozen `v_r/r`.
- Expanded driven boundary uniforms from one global primitive state to per-edge
  N/S/E/W primitive states, while keeping the old `driven` fallback for UI/API
  compatibility.

## Verification

Latest browser/Playwright matrix on the static server:

- `python3 tests/physics-validation.py --port 8090 --timeout 180 --out /tmp/plasma-physics-validation.json`
- Result: `11 passed, 0 failed`.
- Hall whistler row after the eigenmode fix: phase error `0.005`, amplitude
  ratio `0.985`.
- Cylindrical row: inner density drop `0.0310`, outer drop `0.0064`,
  exact-source relative error `4.795e-4`.

## Still future work

- Source splitting is now Strang-style but not fully IMEX. Large
  conduction/viscosity workloads may still want RKL2/STS folding or an
  implicit operator.
- Periodic self-gravity is still Jacobi, not FFT/multigrid, and non-periodic
  gravity is not implemented.
- Cylindrical geometry remains a source-layer approximation, not a fully
  r-weighted finite-volume rewrite.
- Resistivity still uses uniform/anomalous current-dependent eta; the table
  contains a Spitzer eta family but the RKL2 resistivity kernels are at the
  storage-binding cap and do not yet consume local temperature.

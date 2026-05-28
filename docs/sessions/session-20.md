# Session 20 — next realism layer

This pass pushed the source stack closer to research-code behavior while
keeping the canonical MHD presets on the same baseline flags.

## Landed

- Added opt-in grey radiation: a radiation-energy reservoir with local
  gas/radiation exchange, flux-limited diffusion, subcycled `rad_dt`, and a
  `radiative-relaxation` validation preset.
- Expanded `src/microphysics.js` to 24 knots per family for cooling,
  ionization, resistivity, transport, absorption opacity, and scattering
  opacity. The cooling, conduction, viscosity, Ohm, and radiation shaders now
  consume the expanded table offsets.
- Added kinetic-scale current smoothing through the unified Ohm layer:
  an electron-inertia / hyper-resistive closure behind
  `FLAG_ELECTRON_INERTIA`, plus the `kinetic-current-smoothing` validation
  preset.
- Added isolated zero-potential self-gravity boundaries through
  `gravityBoundaryMode`, plus the `isolated-gravity-pulse` validation preset.
- Moved cylindrical continuity into the conservative hyperbolic update as an
  r-weighted finite-volume radial divergence.
- Added cylindrical CT for the axial field via the matching
  `1/r d(r E_phi)/dr` update, with a cylindrical divergence validation row.
- Added a cylindrical Poisson stencil
  `(1/r) d_r(r d_r phi) + d_zz phi` for self-gravity under cylindrical
  geometry, plus the `cylindrical-gravity-column` residual validation row.
- Added a Cartesian geometric multigrid V-cycle (`solve-poisson-mg.wgsl`)
  for periodic/isolated self-gravity. `gravitySolverMode` defaults to
  multigrid for Cartesian runs; the weighted-Jacobi path remains the
  cylindrical solver and explicit fallback.
- Corrected the cylindrical radial momentum source to `T_phi_phi/r`,
  including gas pressure and magnetic pressure, and stopped injecting a
  separate geometry-source total-energy term. The new static-equilibrium row
  verifies uniform pressure plus axial field remains static.
- Switched conductive heat-flux divergence to
  `1/r d(r q_r)/dr + d q_z/dz` in cylindrical mode, with a log-temperature
  balance validation row.

## Verification

Latest static-server browser matrix:

- `node _build.mjs`
- `node tests/wgsl-transpile/smoke.js`
- `python3 tests/physics-validation.py --static-root --port 8090 --timeout 600 --out /tmp/plasma-physics-validation.json`
- Result: `20 passed, 0 failed`.
- New Cartesian multigrid row: same 48-step solver budget gives residual
  `0.008` vs Jacobi `0.040` (`ratio=0.210`, relative residual `9.522e-4`).
- New cylindrical static-equilibrium row: `vmax=2.429e-6`, `pmin=1.0000`.
- New cylindrical conduction row: `DeltaTmax=3.179e-7`,
  `DeltaTavg=5.887e-9`.
- New cylindrical Poisson row: residual cylindrical/cartesian
  `0.850/1.909`, relative residual `0.149`, center potential `-0.8084`
  after the validation-only 2200 Jacobi sweeps.

## Still future work

- Cartesian Poisson now has multigrid. Cylindrical self-gravity is still
  weighted Jacobi; a cylindrical multigrid operator or spectral/Green-function
  path is the next real gravity jump.
- Hall, electron-inertia, and conduction remain explicit/subcycled. HDS,
  RKL2, or implicit operators are still the path for stiffer regimes.
- Cylindrical geometry now has r-weighted continuity, axial CT, Poisson,
  static pressure/magnetic balance, and r-weighted conduction. Strongly
  radial rotating/magnetized flows still deserve a higher-order angular-
  momentum and magnetic-energy audit.
- Grey radiation is useful as a single-group approximation, not a replacement
  for multi-group transport or externally vetted opacity products.

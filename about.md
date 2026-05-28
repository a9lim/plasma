# Plasma

Plasma is a WebGPU-native simulator for two-dimensional resistive
magnetohydrodynamics. It treats the fluid as a conducting gas threaded by a
magnetic field, then evolves density, momentum, magnetic field, and energy on
a Cartesian grid. The point is to make plasma behavior visible at the scale
where fluid motion and field topology are inseparable: shocks compress the
field, current sheets thin out and reconnect, Alfven waves move information
along magnetic tension, and pressure balance decides which structures survive.

The core model is 2.5D ideal MHD with optional source physics. Each cell stores
three velocity components, three magnetic-field components, density, total
energy, and dual-energy auxiliaries for internal energy and an entropy proxy.
The numerical method uses PPM reconstruction, an HLLD Riemann solver, RK3 SSP
time integration, and constrained transport on a staggered Yee-style magnetic
mesh. The constrained-transport update is the important structural choice:
face-centered magnetic fluxes are advanced by the curl of a corner electric
field, so the discrete divergence of B cancels by construction.

The default Orszag-Tang preset is a compact turbulence and shock test. It
starts smooth, then folds into interacting shocks, magnetic islands, and dense
current sheets. Harris current sheet is the reconnection test: a pressure-
balanced sheet with a small perturbation, resistivity, and open vertical
boundaries. Brio-Wu and Sod are shock tubes, useful because they isolate wave
fan structure in one dimension. The extended Orszag-Tang and driven wind/cloud
presets turn on the richer source layer: tabulated cooling and heating,
anisotropic conduction, Hall and ambipolar terms, Biermann battery generation,
viscosity, gravity, geometry sources, and sponge or driven boundaries.

This is still a browser simulation, not a production plasma code. The cooling
table is a compact code-unit model, the self-gravity solve is periodic Jacobi,
and source terms are split after the main hyperbolic update. The implementation
is built to keep those compromises explicit. Canonical verification presets
default to just the base MHD numerics and guard terms, while extended presets
opt into the source physics.

## Hidden presets

The preset dropdown intentionally surfaces only the ten presets that are
useful as exploratory starting points. Nine more are kept in
`src/presets.js` for the validation harness and for hand-driven testing:

* `alfven-cpaw` and `acoustic-wave-hydro` are linear convergence rigs.
* `radiative-relaxation` and `kinetic-current-smoothing` isolate the
  grey radiation and hyper-resistive Ohm closures.
* `isolated-gravity-pulse` exercises the isolated Poisson boundary.
* The five `cylindrical-*` presets verify the r-weighted finite-volume
  update, cylindrical CT induction, conduction, and Poisson operators.

They're reachable from the JS console via
`sim.setPreset('cylindrical-static-equilibrium')` etc., and they drive
the `tests/physics-validation.{html,py}` matrix.

## Learning Outcomes

* Connect magnetic pressure, thermal pressure, and plasma beta in an MHD flow.
* Identify current sheets from the out-of-plane current density diagnostic.
* Explain why constrained transport preserves the discrete divergence of B.
* Compare hydrodynamic and MHD shock structure using Sod and Brio-Wu.
* Watch magnetic reconnection convert field energy into thermal and kinetic
  structure in the Harris current sheet.
* Relate anisotropic conduction and Hall terms to field-aligned transport and
  dispersive whistler behavior.
* Use the entropy proxy and pressure diagnostics to spot where total-energy
  recovery becomes numerically delicate.

## Prerequisites

The simulator is written for readers with some exposure to vector calculus,
partial differential equations, and electromagnetism. Fluid dynamics helps,
but the controls are also useful for exploratory play: change the preset,
switch scalar diagnostics, click a probe point, and compare how the fields
organize themselves.

## Accessibility

The app uses keyboard-reachable controls, ARIA labels on the canvas and panel,
high-contrast theme support through the site design system, and text labels
for all major controls. The simulation is animated and can be paused, stepped,
or reset. It does not use sound.

## References

* Miyoshi and Kusano, "A multi-state HLL approximate Riemann solver for ideal
  magnetohydrodynamics", Journal of Computational Physics, 2005.
  DOI: <https://doi.org/10.1016/j.jcp.2005.02.017>
* Gardiner and Stone, "An unsplit Godunov method for ideal MHD via constrained
  transport", Journal of Computational Physics, 2005.
  DOI: <https://doi.org/10.1016/j.jcp.2004.11.016>
* Stone, Gardiner, Teuben, Hawley, and Simon, "Athena: A New Code for
  Astrophysical MHD", The Astrophysical Journal Supplement Series, 2008.
  DOI: <https://doi.org/10.1086/588755>
* Colella and Woodward, "The Piecewise Parabolic Method (PPM) for
  gas-dynamical simulations", Journal of Computational Physics, 1984.
  DOI: <https://doi.org/10.1016/0021-9991(84)90143-8>

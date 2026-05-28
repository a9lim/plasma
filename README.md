# Plasma

Interactive 2D resistive magnetohydrodynamics simulator that runs entirely in the browser. It treats the plasma as a conducting gas threaded by a magnetic field, then evolves density, momentum, magnetic field, and energy on a Cartesian grid in real time. The point is to make plasma behavior visible at the scale where fluid motion and field topology are inseparable: shocks compress the field, current sheets thin out and reconnect, Alfven waves carry information along magnetic tension, and pressure balance decides which structures survive. Everything runs on a WebGPU compute backend.

**[Try it](https://a9l.im/plasma)** | Part of the [a9l.im](https://a9l.im) portfolio

## Physics

2.5D ideal MHD with resistive and optional source-term extensions. Each cell stores density, three momentum components, total energy, three magnetic-field components, and dual-energy auxiliaries for internal energy and an entropy proxy. The display is two-dimensional, but the velocity and magnetic field carry their out-of-plane components, so guide-field and out-of-plane shear dynamics still participate in the energy budget.

The base system is the resistive MHD equations in conservative form, coupling fluid advection, thermal pressure, magnetic pressure, and magnetic tension. On top of that there is an optional source layer that you can turn on per preset.

### Source physics

Each source term early-returns when its flag is clear or its coefficient is zero, so features can be toggled independently:

- **Radiative cooling and heating**: exact-integrated cooling against a tabulated microphysics curve, paired with density-law volumetric heating
- **Anisotropic conduction**: field-aligned Spitzer heat flux with smooth Cowie-McKee saturation
- **Hall MHD**: the `(d_i / rho)(J x B - grad p_e)` electromotive force, sub-cycled against the whistler timescale
- **Ambipolar diffusion**: ion-neutral drift in partially ionized gas
- **Biermann battery**: magnetic field generation from misaligned density and pressure gradients
- **Electron-inertia smoothing**: a hyper-resistive Ohm closure that damps high-wavenumber current
- **Viscosity**: explicit shear, bulk, and shock viscosity with optional B-aligned projection
- **Self-gravity**: a periodic or isolated Poisson solve via geometric multigrid, plus external gravity
- **Grey radiation**: a separate radiation-energy reservoir with flux-limited diffusion and absorption exchange
- **Cylindrical geometry**: axisymmetric curvature sources with r-weighted constrained transport, plus a boundary sponge

## Numerical method

The hyperbolic step is finite-volume MHD. Primitive states are reconstructed with PPM (Colella and Woodward 1984), limited in characteristic variables (Stone et al. 2008, the Athena and Athena++ default for MHD). Intercell fluxes come from the HLLD approximate Riemann solver (Miyoshi and Kusano 2005), with HLLC and HLL fallbacks for degenerate wave fans and a positivity fallback for the star states. Time integration is three-stage SSP RK3, with one macro timestep computed from the start of the step and reused across all three stages.

The magnetic field uses constrained transport on a Yee-style staggered grid. The in-plane components live on cell faces and are advanced by the curl of a corner-centered electric field, so the discrete divergence of B cancels by construction. The EMF has two modes: the Balsara-Spicer arithmetic mean and the Gardiner-Stone 2005 upwind form, with the upwind form as the default.

Resistivity is applied as `curl(eta J)` on the same staggered grid, so it shares the divergence-preserving structure of constrained transport. It runs once per macro step as an RKL2 super-time-step after the RK3 hyperbolic stages. Source terms are Strang-bracketed around the hyperbolic core: a half-step of sources, then the hyperbolic and resistive update, then a second half-step.

Boundaries are per-edge selectable: periodic, outflow, reflecting, or driven inflow. The grid is 256² by default, with a sidebar selector for 256, 512, and 1024. Every field buffer carries two ghost cells per side to feed PPM's five-point stencil at the edges.

## Presets

The dropdown surfaces ten starting points. Six are canonical numerical tests on the base MHD numerics:

- **Orszag-Tang**: a smooth periodic vortex that folds into interacting shocks, magnetic islands, and dense current sheets by t ≈ 0.5
- **Harris current sheet**: a pressure-balanced sheet with a small perturbation, resistivity, and open vertical boundaries; reconnection develops with plasmoid chains along the sheet
- **Brio-Wu**: the standard MHD shock tube, isolating slow shock, compound, contact, and slow shock structure
- **Sod**: a hydrodynamic shock tube with a shock, contact, and rarefaction
- **Alfven CPAW**: a circularly-polarized Alfven wave that should translate exactly, used as a convergence test
- **Acoustic wave**: a linear acoustic mode, a hydrodynamic convergence test

The rest opt into the source layer. The extended Orszag-Tang and driven wind/cloud presets turn on a representative source stack (cooling, conduction, Hall, ambipolar, Biermann, viscosity, gravity), and a set of isolation presets each exercise a single feature: Hall whistler, conduction front, cooling instability, radiative relaxation, kinetic current smoothing, the Jeans and isolated-gravity-pulse self-gravity tests, and five cylindrical-geometry validation presets. The validation presets stay out of the dropdown but are reachable from the console via `sim.setPreset('...')`, and they drive the `tests/physics-validation.{html,py}` matrix.

## Visualization

The default scalar view is out-of-plane current density J_z, because it makes reconnection and current-sheet formation visible. Other views show density, pressure, velocity magnitude, magnetic-field strength, temperature, heat flux, gravitational potential, and the dual-energy entropy proxy `K = p / rho^gamma`. Animated line integral convolution stays active over every scalar view, so the magnetic-field topology remains visible while you inspect a thermodynamic or transport diagnostic.

## Controls

The four-tab sidebar carries the full control surface:

- **Settings**: preset, view mode, resistivity and grid Reynolds floor, resolution, and per-edge boundary conditions with driven inflow states
- **Physics**: the extended source layer, with one section each for Hall, cooling and heating, conduction, radiation, viscosity, the non-ideal Ohm terms, gravity, and geometry and sponge
- **Stats**: energy, plasma beta, field maxima, the divergence of B, reconnection rate on the Harris sheet, and conservation drift
- **Probe**: click a cell to sample its state, with a small time-series

Keyboard shortcuts 1 through 4 switch tabs. Play, pause, step, speed, reset, and theme live on the top toolbar, and the gear dropdown holds the numerics and render knobs (CFL, gamma, pressure floor, anomalous resistivity, source-substep cap, EMF mode, positivity guard, and LIC intensity and drift).

## Running Locally

```bash
cd path/to/a9lim.github.io && python -m http.server
# -> http://localhost:8000/plasma/
```

Serve from the repository root, because the shared design files load via absolute paths. There is no build step and no dependencies, and ES6 modules require HTTP rather than `file://`.

Plasma needs WebGPU. Please use a recent Chrome or Edge, or Safari 26+, on a machine with a working GPU; the simulator shows a fallback notice if WebGPU is unavailable. A CPU fallback compiled from the WGSL shaders by the parent repo's transpiler is planned but not yet wired in.

## Tech

Vanilla JavaScript with no dependencies. ES6 modules for the host side and around 35 WGSL compute and render shaders for the GPU side. Canvas is rendered through a WebGPU device; all of the MHD numerics, source physics, and diagnostics are written from scratch.

The whole macro step encodes as a single command submit. Stage weights, sweep directions, and boundary configuration live in pre-written uniform buffers, so there are no `writeBuffer` calls in the hot path. The compute pipelines are written to one bind group each, with workgroup barriers only at the top level, so they stay compilable by the parent repo's `shared-wgsl-transpile.js` for the future CPU path. The composite pass is a fragment shader and stays GPU-only.

PPM caches per-cell primitives in a workgroup-shared tile with a halo, the dt reduction and the LIC contrast stretch use per-tile shared-atomic min/max reductions, and the self-gravity solve uses a Cartesian geometric multigrid V-cycle with a weighted-Jacobi fallback for cylindrical geometry.

This is a browser simulation, not a production plasma code. The cooling table is a compact code-unit model, source terms are operator-split after the hyperbolic and resistive step, and the cylindrical closure does not yet cover every source coupling. The implementation is built to keep those compromises explicit: the canonical verification presets default to just the base MHD numerics and the guard flags, and the extended presets opt into the source physics.

## Sibling Projects

- [Geon](https://a9l.im/geon) ([GitHub](https://github.com/a9lim/geon))
- [Cyano](https://a9l.im/cyano) ([GitHub](https://github.com/a9lim/cyano))
- [Gerry](https://a9l.im/gerry) ([GitHub](https://github.com/a9lim/gerry))
- [Scripture](https://a9l.im/scripture) ([GitHub](https://github.com/a9lim/scripture))
- [Shoals](https://a9l.im/shoals) ([GitHub](https://github.com/a9lim/shoals))

## License

[AGPL-3.0](LICENSE)

# Plasma handoff

Plasma v1.0 is shipped on the `dev` branch. The current product is a
WebGPU-only 2.5D resistive MHD laboratory with a base finite-volume/CT engine,
an opt-in extended source layer, animated LIC visualization, four-tab HUD,
pointer perturbations, and a production-engine validation matrix. This file is
the current handoff; [`sessions/`](sessions/) preserves the implementation
history.

The live code and [`../AGENTS.md`](../AGENTS.md) are authoritative. The old
external implementation plan and session retrospectives explain why choices
were made, but they do not override current source behavior.

## Current shape

- Base state is 2.5D MHD: density, three momentum components, total energy,
  out-of-plane magnetic field, internal-energy auxiliary, and entropy proxy,
  with face-centered in-plane magnetic fields.
- PPM reconstruction with characteristic limiting feeds HLLD, with HLLC/HLL
  degenerate branches and positivity fallbacks.
- RK3 SSP advances the hyperbolic step. Constrained transport updates the
  staggered magnetic field using either Balsara-Spicer mean EMF or the default
  Gardiner-Stone upwind EMF.
- Resistivity is a divergence-preserving `curl(eta J)` update integrated by an
  RKL2 super-step after the RK3 core.
- Extended sources are Strang-bracketed around that core. The source layer
  includes cooling/heating, anisotropic conduction, grey radiation, Hall,
  ambipolar/Biermann/electron-inertia Ohm terms, viscosity, external and
  self-gravity, cylindrical geometry, and a boundary sponge.
- Cartesian self-gravity uses geometric multigrid by default; weighted Jacobi
  remains available and is used by cylindrical geometry.
- Per-edge boundaries support periodic, outflow, reflecting, and driven modes,
  with distinct N/S/E/W driven primitive states.
- The default `sandbox` preset enables the full stack on a quiescent uniform
  state. Left-drag injects momentum; right-drag applies a
  divergence-preserving magnetic perturbation.

The whole macro step is encoded into one queue submit. GPU-to-CPU feedback is
asynchronous: dt/source-rate feedback is consumed by the simulation, while
stats and the hover probe use pooled staging buffers at resolution-adaptive
cadences.

## UI contract

The sidebar has four tabs:

1. **Settings** — preset, scalar view, resistivity/anomalous resistivity,
   numerics, LIC render controls, resolution, and per-edge boundaries.
2. **Physics** — Hall, cooling/heating, conduction, radiation, viscosity,
   non-ideal Ohm, gravity, geometry, and sponge controls.
3. **Stats** — energy, plasma beta, extrema, divergence, Harris reconnection
   rate, and conservation drift. NaN auto-pause remains a safety path even
   though the old debug-only Health/Clock groups are gone.
4. **Probe** — a 10 Hz local-state readout for the cell under the pointer. It
   has no click-to-pin, crosshair, sparkline, or time-series mode.

There is no topbar settings dropdown and no `shared-settings.js` dependency.
Keyboard keys 1-4 select the tabs. Space toggles play/pause, `/` single-steps,
`R` resets, and `V` cycles scalar views.

## Presets

`src/presets.js` exports 21 presets. Eleven are visible in the dropdown:

- `sandbox`
- `orszag-tang`
- `orszag-tang-extended`
- `driven-wind-cloud`
- `harris`
- `brio-wu`
- `sod`
- `hall-whistler`
- `conduction-front`
- `cooling-instability`
- `jeans-instability`

Ten are validation/diagnostic presets reachable through
`sim.setPreset(name)`:

- `alfven-cpaw`
- `acoustic-wave-hydro`
- `radiative-relaxation`
- `kinetic-current-smoothing`
- `isolated-gravity-pulse`
- `cylindrical-expansion`
- `cylindrical-static-equilibrium`
- `cylindrical-conduction-balance`
- `cylindrical-magnetic-compression`
- `cylindrical-gravity-column`

Canonical Sod, Brio-Wu, Orszag-Tang, Harris, Alfvén CPAW, and acoustic presets
use `BASE_PHYSICS_FLAGS`: positivity plus upwind EMF, without the extended
source stack. Extended and isolation presets opt into their source terms
explicitly.

## Verification

The main regression gate is `tests/physics-validation.html`, driven headlessly
by `tests/physics-validation.py`. It instantiates the production `Sim` and has
21 rows covering:

- five finite-state preset smokes;
- distinct per-edge driven boundary fill;
- uploaded microphysics/opacity table shape and source substep sizing;
- cooling, conduction, radiation, electron-inertia smoothing, Hall whistler,
  and Jeans response;
- isolated gravity and Cartesian multigrid convergence;
- cylindrical expansion, static balance, conduction, CT divergence, and
  Poisson residual convergence.

Current verification: **21/21 passed on 2026-07-17** through the headless
production-WebGPU driver at the committed default test resolutions.

Additional focused pages cover circularly polarized Alfvén and acoustic
convergence plus the Harris diagnostic. See [`../tests/README.md`](../tests/README.md)
for commands and interpretation.

For a full production-style run, serve the parent repository and execute:

```bash
# from the parent a9lim.github.io repository
python -m http.server 8090

# from plasma/, in another shell
python3 tests/physics-validation.py --port 8090 --timeout 180 \
  --out /tmp/plasma-physics-validation.json
```

The driver needs an installed Playwright browser with WebGPU available. For a
quick static check when GPU execution is unavailable, syntax-check the host
modules, verify shader paths/bindings by inspection, and run `git diff --check`.

## Metadata and SEO

`about.md` is the canonical educational/SEO summary consumed by the parent
build. `index.html` owns the live head metadata, crawlable educational content,
and JSON-LD graph. The parent build synchronizes `about.md` frontmatter into
the submodule metadata and root discovery artifacts, so a parent rebuild is
required after changing `about.md`.

Current structured data uses four externally verified Wikidata entities:
magnetohydrodynamics (`Q2549249`), plasma (`Q10251`), magnetic reconnection
(`Q287506`), and Alfvén wave (`Q645813`). The four DOI links in `about.md` and
`index.html` resolve to the named HLLD, constrained-transport, Athena, and PPM
papers. Do not add or change identifiers without live verification.

The parent WGSL transpiler and runner are retained libraries but are not wired
into the build or runtime. Plasma has no CPU fallback; unsupported browsers see
the `#no-webgpu` notice.

## Known limits and next work

These are architecture/fidelity boundaries, not incomplete wiring:

1. Hall and the explicit diffusion-like source terms use sub-cycling with a
   hard safety ceiling and macro-dt backpressure. HDS, IMEX, or implicit paths
   would be the next step for genuinely stiff workloads.
2. Conduction still uses O(N) sub-cycling. An RKL2 integration would require a
   bind-group/shader-layout refactor and is only compelling when source
   substep counts become large.
3. The uploaded cooling/opacity table is a compact code-unit model anchored to
   sampled Sutherland-Dopita solar CIE data. Production astrophysical work
   needs a vetted metallicity/density grid and opacity products.
4. Cylindrical self-gravity uses the r-weighted Jacobi path; cylindrical
   multigrid or a Green-function approach remains open.
5. Cylindrical continuity, axial-field CT, Poisson, radial force balance, and
   conduction use r-weighted operators, but the closure is not a complete
   production axisymmetric MHD formulation for strongly radial flows.
6. Radiation is grey and electron inertia is represented by a practical
   hyper-resistive closure; multi-group radiation and a kinetic model are out
   of scope for the current browser engine.

## Change checklist

- Keep the main 256-byte `Uniforms` layout synchronized between
  `config.js`, `buffers.js`, and `shared-helpers.wgsl`.
- Preserve the face-ownership and two-ghost-cell conventions documented in
  `AGENTS.md`.
- Keep all compute pipelines on one bind group and barriers at top level so
  the retained transpiler can still parse the shaders.
- Rebind Stats and Probe after a resolution change replaces GPU buffers.
- Do not add `innerHTML` assignments in new Plasma code.
- Run the 21-row validation gate after numerical, shader, preset, or buffer
  changes; run the relevant focused convergence/diagnostic page as well.
- Update `about.md`, the `index.html` metadata/content mirrors, and the parent
  generated artifacts together for a user-visible model or capability change.
- Live-verify every new Wikidata QID, DOI, or scholarly URL.

## Documentation map

- [`../AGENTS.md`](../AGENTS.md) — full architecture and implementation
  contracts.
- [`../README.md`](../README.md) — public project overview and local run
  instructions.
- [`../about.md`](../about.md) — canonical SEO/educational summary.
- [`../tests/README.md`](../tests/README.md) — test commands and numerical
  interpretation.
- [`sessions/`](sessions/) — historical implementation retrospectives. Keep
  them historical; do not rewrite old observations as current claims.

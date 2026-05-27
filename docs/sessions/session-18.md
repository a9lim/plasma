# Session 18 — tabulated microphysics and dual-energy hardening

Goal: take the Session 17 realism layer from "many source toggles" toward
a more internally consistent multiphysics engine.

## Landed

* **Tabulated microphysics buffer.** New `src/microphysics.js` builds a
  64-row `vec4` table uploaded by `PlasmaBuffers`. Cooling, neutral
  fraction, and transport closures now consume table-backed/log-slope
  families where their binding budgets allow it.
* **Dual-energy auxiliaries.** `U1.zw` now store internal energy and
  entropy proxy `K = p/rho^gamma`. Primitive recovery falls back to
  the auxiliary internal energy when total-energy pressure is dominated
  by kinetic/magnetic subtraction error.
* **Thermodynamic repacking.** Hyperbolic updates, floors, cooling,
  conduction, viscosity, gravity, geometry, and dissipative Ohm updates
  refresh the dual-energy auxiliaries after changing total energy,
  momentum, or magnetic energy.
* **Unified generalized Ohm layer.** New `apply-ohm.wgsl` evaluates
  Hall, ambipolar, and Biermann terms from one frozen state. Hall still
  gets nondissipative magnetic-energy repair; ambipolar/Biermann updates
  keep total energy fixed and repack pressure so magnetic losses/gains
  exchange with the gas internal-energy budget.
* **Transport stability.** Conduction timestep reduction and the `|q|`
  view now include the default Spitzer-like `T^(5/2)` transport scaling.
  Conduction uses the shared source substep cap rather than the Hall-only
  cap.
* **Diagnostics and presets.** Added entropy view/probe support, dual-
  aware CPU stats/probe pressure recovery, tabulated cooling UI mode, and
  a `driven-wind-cloud` preset with a driven west inflow and open
  boundaries.
* **Docs and metadata.** Refreshed `about.md`, the crawlable
  `edu-content` section, JSON-LD, about-panel `lastUpdated`, `HANDOFF`,
  `AGENTS`, and the docs session index so the visible documentation
  matches the Session 18 physics surface.

## Caveats

* The uploaded table is still a compact built-in code-unit model, not a
  CHIANTI/Sutherland-Dopita data product. The architecture now supports
  replacing it without shader edits.
* Resistive RKL2 is still uniform/anomalous-eta because that shader is
  already at the storage-binding ceiling and does not bind density/temperature.
* Source coupling is still first-order split after the hyperbolic/RKL2
  step. Strang splitting would require a source-dt buffer path that can
  scale the freshly GPU-computed dt without host readback staleness.

## Verification

* `node --check` on modified JS modules.
* ESM import/shape check for every preset at `N=16`.
* Microphysics table finite-value/length check.
* JSON-LD parse check, plus live verification of Wikidata entities,
  Crossref DOI titles, and educational-alignment URLs before committing
  the structured data.
* Local server load at `http://127.0.0.1:8765/plasma/`; all WGSL modules
  fetched with shader version 30 and the live canvas rendered nonblank.

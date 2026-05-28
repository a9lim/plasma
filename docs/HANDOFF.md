# HANDOFF.md — plasma

**v1.0 shipped.** Sim built across one orchestration session (Claude
as orchestrator, 6 dispatched sub-agents for Phases 1–6) plus
follow-up sessions for Phases 7-8 polish and the Session 14-22
extended-physics arc. Engine, UI, LIC visualization, docs/metadata,
pointer perturbation, OG art, and parent-repo wiring are all in.
Phase 9 extended physics is partial (structural sharp edges closed;
remaining items are heavier physics architecture, not patching) —
see the Phase 9 section below.

The Probe tab was rewritten in Session 22 alongside the pointer
perturbation: no more click-to-place / shift-drag / sparkline /
time-series. Hovering the canvas shows the local primitive state at
the cursor cell — left-drag and right-drag are now the perturbation
gestures (drag = grab the plasma, right-drag = excite / twist field).

This doc is next-instance / next-agent context for picking up where
we left off. The implementation plan lives at
`~/.claude/plans/geon-currently-uses-cpu-abstract-cat.md` — that's the
source of truth for design decisions; this doc is specifically about
what's left to do and what to watch for.

Per-session retrospectives live alongside this file in
[`sessions/`](sessions/). Each captures the bugs surfaced, the fixes
applied, and the open concerns flagged for the next session. Shader-
level comments referring to "HANDOFF Session N" or just "Session N"
point at the corresponding `sessions/session-N.md`.

## Status

| Phase | Description                                | Status   |
|-------|--------------------------------------------|----------|
| 1     | WebGPU init, fullscreen-quad render        | done     |
| 2     | Pure hydro (Euler) with HLL + PLM + FE     | done     |
| 3a    | MHD state + CT + HLL-MHD                   | done     |
| 3b    | HLLD + PPM + RK3 SSP                       | done     |
| 4     | Resistivity + per-edge BCs                 | done     |
| 5     | UI scaffolding (sidebar, topbar, tabs)     | done     |
| 6     | Animated LIC visualization                 | done     |
| 7     | Polish (content, perturbation, JSON-LD, OG)| done     |
| 8     | Parent-repo wiring                         | done     |
| 9     | Extended physics (Hall, cooling/heating, conduction, self-gravity, Cartesian multigrid, ambipolar/Biermann, electron-inertia smoothing, radiation, viscosity, geometry, sponge, EMF toggle, positivity guard) | **partial** — see [Session 14](sessions/session-14.md), [Session 15](sessions/session-15.md), [Session 16](sessions/session-16.md), [Session 17](sessions/session-17.md), [Session 18](sessions/session-18.md), [Session 19](sessions/session-19.md), [Session 20](sessions/session-20.md), [Session 21](sessions/session-21.md), and the "Phase 9" section below |

**Verification status**: OT live-verified at N=256 and N=1024 across
the full η range up to 1.0 (Session 13 — the RKL2 substep-count fixes
make high-η OT smoothly diffusive instead of speckled). Harris ran
clean through 400+ steps as of Session 12. Session 19 added a browser
validation matrix that runs the production WebGPU `Sim`; the current matrix
passes 21/21 checks at small N: Sod, Brio-Wu, OT, Harris, driven
wind/cloud, per-edge driven BC fill, microphysics/opacity table shape,
source substep rate sizing, cooling, conduction, grey radiation,
electron-inertia current smoothing, Hall whistler, Jeans growth, isolated
gravity potential, Cartesian
Poisson multigrid convergence, and cylindrical
r-weighted expansion/static balance/conduction/CT induction/Poisson. N=512
is still not directly live-verified.

**Session 15 update**: the post-Session 14 caveat is gone. Canonical
verification presets (Sod, Brio-Wu, OT, Harris, Alfvén CPAW, acoustic)
now opt into `BASE_PHYSICS_FLAGS` only (positivity + GS upwind EMF) and
explicitly skip the extended-physics source terms. The full Session 14
stack ships behind the new `orszag-tang-extended` preset. To exercise
the breadth pass on a known IC, select that preset; to re-run the
Session 13 baseline, select any of the canonical presets — no manual
flag-zeroing needed.

**Session 17 update**: the extended stack now has a second realism layer:
CIE-inspired metallicity-scaled cooling plus optional heating, ambipolar
diffusion, Biermann battery, explicit shear/bulk/shock viscosity,
source-specific substep caps, weighted/softened Poisson, a cylindrical
axisymmetric source layer, and a boundary sponge. Canonical presets still
stay on `BASE_PHYSICS_FLAGS`; `orszag-tang-extended` opts into
representative nonzero values except geometry/sponge, which remain manual.

**Session 18 update**: the source stack now has a table-backed
microphysics/opacity buffer, dual-energy pressure recovery, unified Hall +
ambipolar + Biermann evaluation, dual-aware diagnostics, and a driven
wind/cloud preset for open-boundary experiments. Canonical presets still
stay on `BASE_PHYSICS_FLAGS`.

**Session 19 update**: validation is now a real gate rather than a loose
visual smoke. The extended presets gained dimensioned/dimensionless
calibration metadata, the source stack is Strang-bracketed with GPU-fresh
half-step subcycle dt, Hall whistler ICs now use the Toth/Ma/Gombosi
right-going eigenmode, positivity uses a theta limiter, cylindrical
continuity is exact-positive for frozen `v_r/r`, and driven boundaries
can carry distinct N/S/E/W primitive states.

**Session 21 update**: the uploaded cooling row is now anchored to the public
Sutherland-Dopita 1993 solar CIE table (`m-00.cie`, normalized at `1e6 K`);
explicit Hall/conduction/viscosity/non-ideal/radiation substep caps are soft
performance targets with a hard safety ceiling and GPU macro-dt backpressure;
and cylindrical toroidal momentum / `B_phi` curvature now integrates the
shared `-v_r y/r` dilution exactly for frozen `v_r/r`.

## Phase 7 — Polish

Estimated: 2 days. In recommended order:

### 1. Verify live (partially done — see Session 2)

OT verified at N=256 and N=1024. Remaining:

* Sod / Brio-Wu / Harris — confirm each loads and runs without
  regressions from the Session 2 sanitization changes. Cycle view
  modes; confirm J_z view shows current sheets in Harris.
* N=512 resolution selector smoke test (jump in between the two
  verified resolutions to confirm no surprises at the missing point).
* Confirm LIC visibly traces field topology across all presets. Tune
  intensity/drift if the default values are too subtle or too loud.
* Confirm stats and probe readbacks are sensible for Harris. Walk
  through the reconnection rate display.
* Confirm save/load round-trips parameters correctly.

If a regression shows up, the Session 2 fixes are the first place to
look — particularly sanitization (could over-clamp legitimate physics
in a shocked region) and the apply-resistivity snapshot pass (adds
synchronization that could interact with other passes oddly).

### 2. Engine fixes flagged by Phase 6 agent

Five small things worth doing in polish, before or after content
writing (any order):

1. ✅ **`simTime` ratchet is broken** (still applies, but downstream cause
   was masked). `Sim.step()` submits but never reads back `b.dt`, so
   `simTime` stays at 0. Stats panel works around it by doing its own
   readback. Session 2 fix #2 (dt buffer COPY_SRC) was a related
   latent bug — verify the stats dt readback is now correct before
   working around it again. If stats now shows a real dt, the simpler
   fix is to consume that value into `simTime` rather than build a
   second readback path.
   _Resolved by Session 3 #2 — Stats panel does the dt readback and it
   now reports a real dt; no separate ratchet path was added._

2. ✅ **`step_parity` uniform slot is dead.** No shader reads it. Either
   reclaim the slot for something useful (LIC drift_z? perturbation
   sequence number?) or stop pushing it.
   _Resolved by Session 3 #2 — slot 12 reclaimed as `cfl: f32`._

3. ✅ **`stage_params` clarity.** The three stage uniform buffers are
   written once at init and immutable thereafter, but the code
   structure made it look like they're rewritten every step.
   _Resolved by Phase 7 wrap-up — `buffers.js` now carries an explicit
   "written ONCE here at init and never touched again" comment block
   that also names the no-per-step-writeBuffer contract and points at
   `_build.mjs RK3_STAGE_WEIGHTS` for the transpiler-side mirror._

4. **CFL / pressure-floor sliders are non-functional.** Phase 5's
   slider for CFL and pressure floor updates `sim.cfl` and
   `sim.pressureFloor` for save/load capture, but the shader
   constants are baked in via `config.js`. Either wire them through
   to compute-dt and HLLD shaders as uniforms, or remove the
   sliders.
   _CFL slider wired by Session 3 #4. Pressure-floor slider wired this
   session — `pressure_floor` now occupies slot 11 of the main Uniforms
   and `setPressureFloor` pushes it; six shaders thread it through their
   cons↔prim helpers and floor checks._

5. ✅ **LIC contrast normalization.** Output is whatever average noise
   the trace finds — flat field-free regions render with a single
   luminance (~0.5). Contrast-stretching the LIC output (subtract min,
   divide by max−min) makes the texture pop more in low-intensity
   regions.
   _Landed in Session 5 — two new compute passes (`lic-reduce.wgsl`
   per-tile shared-atomic min/max reduce → `lic-normalize.wgsl`
   in-place stretch) chained between `lic-advect` and `composite`.
   Min/max formulation (canonical research-vis default) over mean/std.
   The transpiler-contract worry resolved earlier in the round: the
   2D shared-tile + halo + top-level-barrier pattern is now
   transpiler-verified (`testTwoDSharedTileHaloBarrier` in
   `tests/wgsl-transpile/smoke.js`); lic-reduce just mirrors
   compute-dt's reduce shape on the [0, 1] luminance range._

### 3. Pointer-drag velocity perturbation ✅

Landed in Session 22:

* **`src/gpu/shaders/perturb.wgsl`** — three entry points sharing one
  bind group:
  * `apply_drag` — Gaussian-weighted momentum deposit. Each interior
    cell gets δm = amplitude · ρ · (dvec_x, dvec_y) · exp(−r²/2σ²),
    with the KE change folded into E. No race (each cell touches only
    its own U0/U1).
  * `apply_excite_b` — divergence-preserving B perturbation via an
    analytic vector potential `Az(x,y) = amp · (dvec_x·dy − dvec_y·dx)
    · exp(−r²/2σ²)`. The curl at the click center evaluates to amp ·
    (dvec_x, dvec_y) exactly, and the discrete curl on the Yee grid is
    identically ∇·B = 0 by the same telescoping argument as CT. Each
    interior cell writes only its own LEFT Bx + BOTTOM By face; the
    rightmost/topmost dispatched row/col cover the boundary faces.
  * `apply_excite_energy` — re-syncs cell E with the updated |B|² by
    re-deriving δB at cell center from the analytic Az gradients and
    reading the post-update faces. Runs as a second dispatch in the
    same compute pass (inter-dispatch ordering guarantee).
* **`buffers.js`** — new 32 B `perturbUniform` buffer + `pushPerturbUniforms`
  helper (cx, cy, dvec_x, dvec_y, sigma, amplitude).
* **`pipelines.js`** — new `perturbBGL` (6 bindings: Uniforms + U0/U1/Bx/By
  RW + PerturbUniforms) and three compute pipelines.
* **`sim.js`** — `_perturbBG` builder, A/B cache entry under
  `_bgCache.perturb`, public `applyPerturbation({ kind, cx, cy, dvec_x,
  dvec_y, sigma, amplitude })`. Submits its own command buffer at call
  time; queue ordering guarantees the perturbed state lands in U_n
  before the next step's compute-dt reads it.
* **`ui.js`** — `wirePointerPerturbation` ties pointerdown/move/up on
  the canvas. Button 0 → drag, button 2 → excite; contextmenu is
  suppressed over the canvas. Per-pointermove deltas are coalesced via
  rAF — each frame deposits at most one dispatch, and per-event clamps
  prevent flick-induced overflow. Sigma defaults to 4·dx, drag scale
  0.25 (code-velocity per cell-of-drag), excite scale 0.20 (code-B per
  cell-of-drag) — all hardcoded.
* Validation: 21/21 production tests still passing; in-browser
  smoke test fires direct + synthetic-UI events with no console errors.
* Transpiler-compatible: no shared memory, no atomics, only the
  standard cell/face indexing helpers from `shared-helpers.wgsl`.

### 4. Educational content — about.md and edu-content ✅

Landed after Session 18:

* **`plasma/about.md`** now has a 400+ word technical overview with
  model scope, presets, numerical method, accessibility, learning
  outcomes, prerequisites, and DOI-backed references.
* **`<details class="edu-content">`** in `index.html` now has 500+
  words of crawlable technical-reference content: equations / model
  scope, HLLD / PPM / RK3 / CT notes, source-physics caveats,
  diagnostics, accessibility, DOI-backed references, and sibling
  links.
* Note: the parent `AGENTS.md` and the writing skill both exclude sim
  `about.md` / edu-content from a9-voice rewriting. These docs stay in
  technical-reference register.

### 5. SEO — JSON-LD blocks ✅

Landed after Session 18:

* `index.html` now ships `WebApplication` + `LearningResource`,
  `FAQPage`, `BreadcrumbList`, and `HowTo` JSON-LD in one `@graph`.
* Verified Wikidata IDs used:
  - magnetohydrodynamics: `Q2549249`
  - plasma: `Q10251`
  - magnetic reconnection: `Q287506`
  - Alfven wave: `Q645813`
* Verified DOIs used:
  - Miyoshi-Kusano HLLD: `10.1016/j.jcp.2005.02.017`
  - Gardiner-Stone CT: `10.1016/j.jcp.2004.11.016`
  - Athena MHD: `10.1086/588755`
  - Colella-Woodward PPM: `10.1016/0021-9991(84)90143-8`
* Educational alignments now point at verified reachable NGSS, AP
  Physics C: E&M, and AAPT undergraduate laboratory guidance URLs.

### 6. OG image ✅

Landed in Phase 7 wrap-up:

* `og/plasma.html` mirrors the `og/pile.html` boilerplate (hardcoded
  colors, no shared-tokens dep). The flair is a Harris reconnection
  topology — a horizontal red current sheet with three magnetic
  islands (O-points), two X-points between them, and curving open
  field lines arching anti-parallel above/below the sheet against a
  dot-grid NERV background.
* Added to `og/generate.js` CARDS array.
* `node og/generate.js` produces `plasma/og-image.webp` (1200×630,
  quality 90, ~21 KB).
* `plasma/index.html` now carries the standard `og:image` +
  `og:image:width|height|type|alt` + `twitter:image` meta block.

### 7. visibilitychange wiring (verify) ✅

Phase 1 added `visibilitychange` pause logic. Verify it actually
pauses the simulation when the tab is hidden — easy to test in dev
with a console log on visibility transitions.
_Verified by code-trace. `_hidden` flag gates `_scheduleLoop()` at the
end of `loop()`, so once `visibilitychange` flips it the rAF chain dies
after one more frame. Unhide resets `lastTime` and reschedules. Comment
in `main.js` documents the contract._

### 8. Sim metadata bump ✅

`dateModified` in JSON-LD and `lastUpdated` in `initAboutPanel` are
both `2026-05-27` for the Session 18 docs / metadata refresh.

## Phase 8 — Parent-repo wiring ✅

All landed in the Phase 8 wrap-up. Files touched in
`/Users/a9lim/Work/a9lim.github.io/`:

1. ✅ **`.gitmodules`** — plasma submodule entry was already present
   from the initial `git submodule add`. The parent submodule
   pointer still needs to be advanced once the plasma submodule has
   its v1.0 tip pushed; that's a one-line git operation, not a file
   edit.

2. ✅ **`src/projects.js`** — `plasma` entry added between Pile and
   Raiko: `kind: 'sim'`, all four i18n fields, `_ICON.projPlasma`.

3. ✅ **`_worker.js`** — `SIMS_SSR` carries a plasma card; the `/sims`
   `ItemList` JSON-LD now includes position 8 (Plasma).

4. ✅ **`_routes.json`** — `/plasma/*` is in the `exclude` array.

5. ✅ **`_headers`** — `/plasma/*` carries `Cross-Origin-Embedder-Policy:
   credentialless` (mirrors geon, needed for WebGPU compute timing on
   some browsers). Cache rules for `/plasma/*.js`, `/plasma/*.css`, and
   `/plasma/*.wgsl` are in. Early Hints are intentionally omitted —
   COEP: credentialless invalidates 103 preloads (fetched pre-policy),
   same as geon; the comment in `_headers` documents this.

6. ✅ **`_build.mjs`** — plasma is in `IMAGE_MAP`, `IMAGE_CAPTIONS`,
   `aboutFiles`, and `staticRoutes`. `homeData.stats.sims` bumped from
   7 to 8. (Plasma's WGSL transpile was already wired in
   `WGSL_SHADER_DIRS` from earlier work.)

7. ✅ **`index.html` (parent)** — plasma in the homepage `<li>` Sims
   list, the `Course` JSON-LD `@graph` (position 8), the Speculation
   Rules prefetch `href_matches` array, and the Course `description`
   string (now "eight … and magnetohydrodynamics").

8. ✅ **`llms.txt`** — one-line description added; `updated` bumped to
   2026-05-27.

9. ✅ **`manifest.json`** — `Plasma` shortcut entry added.

10. ✅ **`og/generate.js`** — `plasma.html` added to `CARDS`; produces
    `plasma/og-image.webp` on the next `node og/generate.js`.

11. ✅ **`shared-icons.js`** — `projPlasma` SVG added (two crossing
    S-curves around an O-point — reconnection topology in the NERV
    register).

12. ✅ **`i18n.js`** — `home.sims.plasma` added in both English and
    Japanese registers so the homepage Sims list translates cleanly
    (wasn't in the original 11-item punch list but lives on the same
    wiring slope).

After all 12: `node _build.mjs` from the parent regenerates sitemap,
feeds, llms-full.txt, and home-data.json (also re-runs the WGSL
transpiler). Smoke test via `./dev.sh`; then push parent.

`dev.sh` requires no changes (auto-symlinks all top-level
directories). `robots.txt` requires no changes (general `Allow: /`
covers it).

## Transpiler hookup

`shared-wgsl-transpile.js` at the parent repo can compile our compute
shaders to JS for CPU fallback. The compute kernels are written to
its supported subset (see the Transpiler Contract section in
`AGENTS.md`). After Phase 8 deploys, the transpiler agent (separate
context) can plug in:

* Replace the `no-webgpu` landing page with a CPU-fallback path that
  compiles each compute kernel via `compileWGSL` and runs them in
  JS.
* Composite stays GPU-only — needs a Canvas 2D fallback or graceful
  degradation.
* `compute-dt`'s workgroup-shared atomic reduction may need special
  handling — the barrier-at-top-level pattern needs to be respected.
* `apply-resistivity`'s in-place read-modify-write on `read_write`
  storage assumes neighbor reads return pre-dispatch values; CPU
  emulation should explicitly double-buffer (snapshot + write) for
  correctness, per the comment in that shader.

## Things worth flagging for the next instance

* **Don't skip broad live verification.** Session 18 rendered
  nonblank in-browser after the source-physics pass, but the full
  preset / resolution / stats / probe validation matrix is still the
  next meaningful gate.

* **The agent reports embedded in each phase commit message are
  useful context for debugging.** Each Phase agent flagged its
  decisions beyond spec, pre-flight concerns for the next phase,
  and engine critique. Read them via `git log -p` if something
  feels off.

* **The `shared-wgsl-transpile.js` constraint is real.** If you
  write new shaders, keep them transpilable (no nested barriers,
  no textures, no exotic types). The Phase 6 agent's transpiler
  audit table is the working pattern.

* **Sim docs are technical-reference register.** Parent `AGENTS.md`
  excludes sim `about.md` and `<details class="edu-content">` from the
  a9-voice writing workflow; keep them factual, direct, and current.

* **Wikidata QIDs and DOIs MUST be verified against the live
  source.** 88% hallucination rate per parent AGENTS.md. Use the
  API endpoints documented there. Don't generate identifiers from
  memory.

* **`HLLD_BX_EPS2` is now `1e-10`** (was 1e-24). See Session 2 #4.
  If you see the degenerate-Alfvén branch triggering often (debug
  logs would help), it's now expected at thin sheets and that's
  desirable behavior. Branch B (slow/fast wave coincidence) is
  extremely rare — if you see it firing in normal operation,
  something else is wrong.

* **Resolution change is destructive.** `Sim.setResolution(n)`
  re-instantiates buffers and reloads the preset. Any consumer
  caching GPU buffer refs (currently just `stats-display` and
  `probe`) needs to call `bindBuffers(sim.buffers)` after.

* **The unsplit CT update** in Phase 3a deviated from the plan's
  "keep dimensional splitting in structure" guidance, because CT
  EMF needs both-direction face fluxes simultaneously at each
  corner. This is the right call (matches Stone+ 2008) but worth
  knowing if reading the plan and code together.

## Phase 9 — Extended physics iteration

Session 14 landed first-pass implementations of Hall, cooling,
anisotropic conduction, self-gravity, positivity guard, and the
GS-upwind EMF toggle — all ON by default. Session 15 (Codex pass +
follow-up) addressed the structural sharp edges: race-elimination
splits, real Poisson reduction, BC consistency, per-preset opt-in.
Session 16 made a targeted realism pass: saturated heat flux,
piecewise exact cooling, an optional electron-pressure Hall term,
sub-step diagnostics, and higher-order self-gravity force recovery.
Session 17 extended that into partial-ionization, transport, heating,
geometry, and boundary realism. Session 18 added tabulated microphysics,
dual-energy recovery, a unified generalized-Ohm layer, and a driven
wind/cloud boundary preset. Session 19 added a production-WebGPU
validation matrix, physical coefficient calibration, Strang source
bracketing, per-edge driven states, and harder positivity/geometry guards.
Session 20 started the next realism layer with an opt-in grey radiation
energy reservoir: flux-limited diffusion plus gas/radiation thermal exchange,
sub-cycled inside the existing Strang source bracket. It also moved
cylindrical continuity into the hyperbolic conserved update as a true
r-weighted radial finite-volume divergence, leaving `apply-geometry` for the
remaining curvature terms; added a kinetic-scale electron-inertia
hyper-resistive Ohm closure for unresolved current-sheet smoothing; expanded
the uploaded microphysics table to 24 knots per family with grey
absorption/scattering opacity modifiers consumed by the radiation source;
added an isolated zero-potential Poisson boundary mode for non-periodic
self-gravity experiments; added r-weighted cylindrical CT induction for the
axial field; and switched self-gravity to the matching cylindrical Poisson
stencil when geometry mode is cylindrical. It also corrected the cylindrical
radial momentum source to use `T_phi_phi/r`, so uniform pressure plus axial
field cancels the r-weighted radial flux divergence instead of
self-accelerating, and made anisotropic conduction use the same radial
`1/r d(r q_r)/dr` geometry. A follow-up in the same Session 20 arc added
a Cartesian geometric multigrid V-cycle for periodic/isolated self-gravity,
with the existing weighted-Jacobi path retained for cylindrical geometry and
manual fallback. Session 21 anchored the cooling table to the public
Sutherland-Dopita solar CIE curve, made explicit source caps stability-aware
instead of truncating required substeps, and tightened cylindrical toroidal
curvature with an exact frozen-coefficient update. The remaining work is now
the heavier physics architecture: HDS/IMEX vs sub-cycling for Hall/non-ideal
terms, RKL2/implicit solvers for large diffusion workloads,
metallicity/density-dependent cooling/heating and externally-vetted opacity
products, cylindrical multigrid/FFT gravity, broader r-weighted source
coupling audits, and multi-frequency radiation beyond the current grey
closure.

### 1. Validation presets and view modes ✅ (Session 15)

Without dedicated tests / views, none of the new features were
*visible* enough to know whether they're doing the right thing.
`view-field.wgsl` gained T, |q|, φ view modes; `presets.js` gained
four isolation presets:

* **Hall whistler dispersion** — single-mode plane wave, measure
  phase velocity, compare to the Tóth/Ma/Gombosi right-going branch
  `c_w = w/2 + sqrt(v_A² + w²/4)`, `w = d_i k |B₀|/ρ`, and initialize
  with the matching `|δv|/|δB| = |B₀|/(c_w ρ)` eigenvector.
* **Thermal conduction front** — isolated hot spot, watch it spread
  along a uniform B (anisotropic test); compare to the analytic
  diffusion solution.
* **Cooling instability** — uniform gas at supercritical Λ;
  watch fragmentation timescale.
* **Grey radiation relaxation** — gas/radiation reservoir exchange plus
  FLD redistribution on a periodic box.
* **Electron-inertia current smoothing** — high-k transverse magnetic mode
  damping through the hyper-resistive Ohm closure while CT keeps divergence
  tiny.
* **Jeans instability** — small density perturbation under
  self-gravity; measure growth rate vs. Jeans λ.
* **Isolated gravity pulse** — compact central mass with a zero-φ exterior;
  verifies the non-periodic Poisson path forms a negative central well.
* **Cylindrical expansion** — r-weighted radial finite-volume continuity;
  compare inner-radius density dilution against the local frozen `v_r/r`
  trend.
* **Cylindrical magnetic compression** — z-varying radial inflow over a
  uniform axial field; verifies r-weighted CT keeps cylindrical ∇·B tiny.
* **Cylindrical static equilibrium** — uniform gas pressure plus axial field;
  verifies `T_phi_phi/r` cancels the r-weighted radial flux divergence.
* **Cylindrical conduction balance** — logarithmic temperature profile;
  verifies conduction uses `1/r ∂(r q_r)/∂r` rather than Cartesian `∂r q_r`.
* **Cylindrical gravity column** — compact axisymmetric mass with isolated
  boundaries; verifies the cylindrical r-weighted Poisson operator converges
  to a smaller residual than the Cartesian stencil on the same state.
* **Per-edge driven BC fill** — distinct N/S/E/W primitive states;
  verifies the expanded boundary uniform layout and driven face values.

### 2. apply-bcs after extended physics ✅ (Session 15 — Codex pass)

`_encodeApplyBcsDst` now wraps `_encodeExtendedPhysics` on both
sides — once before so the Poisson / Hall stencils read consistent
ghosts, once after so the post-step state is canonical for the
next RK3 stage. Targets the right destination side via the bind-
group cache.

### 3. Hall sub-cycling ✅ (Session 15)

Tóth 2008-style explicit Hall sub-cycle inside each Strang half-step.
The Hall whistler-like parabolic rate `v_A·d_i/dx²` is reduced separately in
`compute-dt.wgsl`; Session 21 also recasts the configured soft substep cap as
an equivalent macro signal speed so stiff Hall settings shrink the next
macro step instead of silently truncating the explicit stability requirement.
Host reads `dt_buf[3] = hall_rate_max` one-step-lagged, computes the required
`N_hall = ceil(dt_half · hall_rate_max / safety)`, allows it to exceed
`hallSubstepsMax` up to the internal hard safety ceiling, writes the inverse
substep count into `b.source_dt_params`, and `source-dt.wgsl` divides the
fresh GPU `dt_half` into `b.hall_dt` before the source pass.
`_encodeExtendedPhysics` then loops the `compute_emf →
apply_update → repair_energy` 3-pass sequence `N_hall` times.

The `hall_dt` buffer is bound at apply-hall's binding 5 instead of
the macro `dt_buf`, so the same shader code runs at sub-step
intervals without per-iteration buffer writes. Within a single
compute pass, WebGPU's dispatch ordering ensures each iteration's
`compute_emf` sees the prior iteration's `apply_update` writes.

Open question: O'Sullivan & Downes 2006 Hall Diffusion Scheme
(hyperbolizes the Hall term so the standard CFL covers it without
sub-cycling) — more code but cleaner. Deferred until a workload
actually saturates the sub-cycle cap.

### 4. Conduction sub-cycling ✅ (Session 15 — see note)

Note: the original plan in this slot was to fold conduction
parabolic L(E) into the existing RKL2 super-step alongside
resistivity. That ran into the 10-storage-binding cap on
`apply-resistivity-init.wgsl` — adding T/U0 inputs for `χ = (γ-1)
κ/ρ` would have required a multi-shader BGL reshuffle, and the
moving target Y_{j-1}.E means L(E) can't be precomputed once per
macro step. The fold remains a future option if the
implementation finds room.

What landed instead: conduction sub-cycling, exactly the same
shape as Hall sub-cycling. `compute-dt.wgsl` reduces `4·χ/dx²`
separately and adds a macro-dt backpressure term whenever the configured
`sourceSubstepsMax` soft cap would be insufficient. Host computes
`N_cond = ceil(dt_half · cond_rate_max / safety)`, allows the required
substeps up to the internal hard safety ceiling, and `_encodeExtendedPhysics`
loops the `compute_delta + apply_delta` pair `N_cond` times with
`source-dt.wgsl` seeding `b.cond_dt` from the fresh GPU half step.

For typical κ values N_cond stays small (5-20), so the O(N_cond)
sub-cycle cost is competitive with RKL2's O(√N_cond) without the
binding-cap complexity. If a future workload pushes N_cond above
~30, RKL2 folding becomes worth the implementation cost.

### 5. Townsend cooling integration ✅ (Session 16)

Explicit FE biases the cooling timestep. Townsend 2009 exact
integration with a piecewise-power-law Λ(T) is the canonical fix.
Session 15 removed the FE bias for the single `√T` bremsstrahlung
shape. Session 16 keeps that mode (`cooling_curve_mode = 0`) and adds
`cooling_curve_mode = 1`: a compact dimensionless piecewise power-law
cooling curve with exact per-segment integration. The table includes a
low-temperature rise, line-cooling peak, trough, and high-temperature
bremsstrahlung tail.

Session 17 adds `cooling_curve_mode = 2`, a broader CIE-inspired
metallicity-scaled table, plus a density-law volumetric heating term.
Session 18 adds `cooling_curve_mode = 3`, sourced from the uploaded
microphysics storage table rather than hard-coded WGSL constants. Session 20
expands that table to 24 knots per family and adds grey absorption/scattering
opacity families consumed by `apply-radiation.wgsl`. Session 21 replaces the
cooling row with sampled public Sutherland-Dopita 1993 solar CIE data
(`m-00.cie`, normalized to `Lambda(1e6 K)`) while keeping the existing compact
buffer contract. The next realism step is loading a full grid of vetted
metallicity/density-dependent cooling/heating and opacity data products into
the same table path.

### 6. Real ρ̄ reduction for Poisson ✅ (Session 15 — Codex pass)

`solve-poisson.wgsl` now exposes `reduce_mean` (8×8 tile sums via
workgroup-shared accumulator) + `finalize_mean` (small serial sum
of tile partials → `rho_mean[0]`). New `rho_mean_partials` storage
buffer holds the per-tile sums. The center-cell stub is gone;
periodic compatibility holds for asymmetric mass distributions.

The Jacobi iterate also moved to inline periodic indexing rather
than relying on ghost cells, so the Poisson solve no longer
depends on `apply-bcs` filling ghosts of `phi`.

### 7. Hall split into corner-buffer + CT-update ✅ (Session 15 — Codex pass)

Three ordered dispatches per Hall update (sharing one bind group +
two new scratch buffers `hall_E`, `hall_mb0`):

* `compute_emf` — corner-centered E_H written into `hall_E`;
  pre-Hall cell magnetic energy snapshotted into `hall_mb0`.
* `apply_update` — reads frozen `hall_E`, applies CT curl to face
  Bx/By and the 2.5D Bz update.
* `repair_energy` — adds Δ(½|B|²) = (½|B|²_new − `hall_mb0`) to
  total E so the Hall B update doesn't masquerade as spurious
  heating/cooling.

The race-prone in-place EMF evaluation is gone. Session 16 also adds
an optional electron-pressure term, so the corner EMF is now
`E_H = (d_i/ρ)·(J×B − ∇p_e)`, with `p_e/p` exposed as a scalar closure.
Conduction got the same frozen-state treatment (`compute_delta` +
`apply_delta` with the `conduction_dE` scratch buffer). Session 19 also
moved the substep dt write from stale host `dt_macro/N` to a GPU
division of the freshly reduced `dt_half`, so source half-steps integrate
the current macro interval exactly even though substep counts remain
one-reduction-lagged.

Session 18 supersedes the separate Hall/non-ideal dispatch path for the
main step with `apply-ohm.wgsl`: `compute_emf` evaluates Hall,
ambipolar, Biermann, and electron-inertia terms from one frozen state,
`apply_hall_update` + `repair_hall_energy` handles the nondissipative Hall
part, and `apply_dissipative_update` then applies the fixed-total-energy
non-Hall terms so magnetic diffusion heats/cools the gas through the
conserved energy budget. The older `apply-hall.wgsl` /
`apply-nonideal.wgsl` modules still compile as legacy/reference kernels.

### 8. UI surface

Sliders + toggles in the advanced settings dropdown:
* "Hall d_i" log slider, with snap-to-0
* "Hall p_e / p" linear slider
* "Cooling Λ_0" log slider, snap-to-0
* Cooling-curve mode group (uploaded table / CIE / piecewise table / brems)
* Metallicity, heating Γ, heating density exponent, heating T cutoff
* "Conduction κ_∥" log slider, snap-to-0
* "Conduction κ_⊥/κ_∥" linear 0–1
* "q_sat φ" linear slider for the Cowie-McKee saturation limiter
* Radiation c, absorption/scattering opacity, and radiation constant
* Viscosity ν, bulk viscosity, B-aligned fraction, shock viscosity
* Electron d_e and electron damping for hyper-resistive current smoothing
* Ambipolar η_A, neutral fraction, ionization T₀, Biermann C_B
* "Self-gravity G" log slider, snap-to-0
* Poisson iterations, gravity softening, Jacobi ω
* Geometry mode, r-axis guard, sponge width/strength, source substep cap
* Mode group for EMF mode (BS / GS upwind)
* Toggle row for positivity guard

### 9. Per-preset extended-physics defaults ✅ (Session 15 — Codex pass)

Preset schema gained an optional `physics` field. All canonical
verification presets (`sod`, `brio-wu`, `orszag-tang`, `harris`,
`alfven-cpaw`, `acoustic-wave-hydro`) declare `physics: {
physicsFlags: BASE_PHYSICS_FLAGS }` to keep just the numerical
guards (positivity + GS upwind). The new `orszag-tang-extended`
preset opts into the full Session 14 stack with the same scalars
that were the old global defaults. `Sim._applyPhysicsConfig`
absorbs the preset's `physics` block on every `loadPreset`; save/
load round-trips the full physics state. Adding per-preset values
to the four new validation presets above closes the remaining
"every preset gets the same defaults" gap.

### 10. Energy-floor cleanup pass (Session 15 — Codex pass, bonus)

Not on the original punch list but landed: a final `energy-floor`
dispatch at the end of `_encodeExtendedPhysics` clamps E against
the final B so that any combination of cooling + gravity + Hall +
conduction sources leaves the cell in a physically-consistent
state. Also: gravity's energy-work term now uses the time-centered
velocity (`v_mid = v + ½ g Δt`) instead of `v_old`, eliminating
the spurious heating that pure forward Euler introduces. And
`buffers.clearExtendedScratch()` runs on preset/resolution load
so the Poisson solve doesn't warm-start against a previous
preset's φ.

### 11. Session 16 realism pass

What landed:

* **Transport:** `conduction_sat_frac` is live. `apply-conduction.wgsl`
  applies a smooth Cowie-McKee saturated heat-flux limiter, and the
  `|q|` view uses the same limiter so the diagnostic matches the source.
* **Cooling:** `apply-cooling.wgsl` supports both the legacy exact
  `√T` brems mode and the new exact piecewise power-law table mode.
  The cooling-instability and extended presets use table mode.
* **Hall / generalized Ohm:** `apply-hall.wgsl` can include
  `−∇p_e`; the extended preset sets `p_e/p = 0.5`, while the whistler
  validation preset keeps it at 0 to preserve the analytic Hall-only
  dispersion comparison.
* **Gravity:** `apply-gravity.wgsl` now recovers `−∇φ` with a fourth-
  order central difference. Cartesian self-gravity uses a geometric
  multigrid V-cycle by default for either the periodic mean-subtracted box
  or an isolated zero-φ exterior. The weighted-Jacobi path remains the
  cylindrical r-weighted axisymmetric solver and explicit fallback.
* **Diagnostics:** Stats show the last Hall and conduction sub-cycle
  counts, so stiff regimes are visible instead of hidden in the step.

### 12. Session 18 microphysics / dual-energy pass

What landed:

* **Microphysics:** `src/microphysics.js` builds an uploaded table used
  by tabulated cooling, neutral fraction, transport scaling, and Session 20
  grey absorption/scattering opacity modifiers.
* **Dual energy:** `U1.zw` carry internal energy and entropy proxy; GPU
  primitive recovery plus CPU stats/probe paths now use the same fallback
  pressure logic.
* **Unified Ohm:** `apply-ohm.wgsl` replaces the main-step split between
  Hall and non-ideal kernels with one frozen generalized-Ohm evaluation
  (now including the Session 20 electron-inertia smoothing term).
* **Open-boundary preset:** `driven-wind-cloud` exercises a driven west
  inflow, NSCBC outflow, tabulated cooling/heating, conduction, Hall,
  ambipolar/Biermann, and viscosity in one scenario.

### 13. Session 19 validation / coupling pass

What landed:

* **Validation matrix:** `tests/physics-validation.html` and
  `tests/physics-validation.py` run the production WebGPU `Sim` through
  21 finite-state/source/boundary checks. Latest result: 21 passed, 0
  failed on the static-server Playwright driver.
* **Physical calibration:** `src/physical-scales.js` derives code-unit
  coefficients from physical scales and dimensionless targets; extended
  OT and driven wind/cloud no longer rely on unlabelled coefficient
  literals.
* **Source coupling:** extended source physics is now Strang-bracketed
  around the hyperbolic/RKL2 step, and `source-dt.wgsl` derives
  subcycle dt buffers from the fresh GPU `dt_half`. Session 21 makes the
  user-facing source substep caps soft performance targets: the host takes the
  required explicit substeps up to an internal hard ceiling, while
  `compute-dt.wgsl` feeds source-cap backpressure into the next macro step.
* **Positivity:** the conserved update uses a theta limiter instead of
  dropping the whole local flux update whenever a positivity constraint
  would be crossed.
* **Geometry:** cylindrical continuity uses an r-weighted radial
  finite-volume divergence in the conserved hyperbolic update, and CT uses
  the matching `1/r ∂(rEφ)/∂r` update for axial B. The self-gravity Poisson
  iterate also switches to the cylindrical
  `(1/r) ∂r(r ∂rφ) + ∂zzφ` stencil under cylindrical geometry. The radial
  momentum source now uses `T_phi_phi/r`, including gas and magnetic
  pressure, so the r-weighted flux/source pair preserves static cylindrical
  balance. Conductive heat flux also uses
  `1/r ∂r(r q_r) + ∂z q_z` in cylindrical mode.
* **Gravity boundaries:** self-gravity has a `gravityBoundaryMode` selector:
  periodic mean-subtracted by default, isolated zero-φ exterior for compact
  non-periodic mass distributions.
* **Cartesian multigrid:** self-gravity also has a host-side
  `gravitySolverMode` selector. The default Cartesian path now builds a
  compact geometric V-cycle pyramid (`solve-poisson-mg.wgsl`) and copies the
  finest potential back into the main ghost-padded `phi`; Jacobi remains
  available for comparison and is still used for cylindrical geometry.
* **Data-backed cooling:** the uploaded cooling row now samples the public
  Sutherland-Dopita 1993 solar CIE `m-00.cie` table, normalized at `1e6 K`,
  instead of a hand-shaped code-unit curve.
* **Boundaries:** driven BCs now support distinct N/S/E/W primitive
  states; the old `driven` state remains the fallback for UI/API
  compatibility.

## References

* Plan: `~/.claude/plans/geon-currently-uses-cpu-abstract-cat.md`
* Parent repo `AGENTS.md`: design system + shared module reference
* `shared-wgsl-transpile.js` at parent repo root: CPU fallback
  contract
* Stone, Gardiner, Teuben, Hawley, Simon (2008) — Athena++ paper
  (canonical CT + PPM + HLLD recipe)
* Miyoshi & Kusano (2005) — HLLD original paper
* Colella & Woodward (1984) — PPM original paper
* Gardiner & Stone (2005) — upwind CT EMF (landed in Session 4;
  eqns 41-45 with the HLLD contact velocity as the upwind selector)

### Extended physics (Session 14)

* Tóth, Ma, Gombosi (2008) — Hall MHD method + whistler test
* O'Sullivan & Downes (2006) — Hall Diffusion Scheme (HDS)
* Braginskii (1965) — anisotropic transport in magnetized plasma
* Spitzer & Härm (1953) — parallel thermal conductivity
* Cowie & McKee (1977) — saturated heat-flux limiter
* Townsend (2009) — exact integration of optically-thin cooling
* Sutherland & Dopita (1993) — solar collisional-ionization-equilibrium
  cooling table used by the Session 21 uploaded cooling row
* Hu, Adams & Shu (2013) — positivity-preserving limiters for
  finite-volume MHD
* Balsara & Spicer (1999) — arithmetic-mean CT EMF baseline

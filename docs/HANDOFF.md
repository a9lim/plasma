# HANDOFF.md — plasma

Sim built across one orchestration session (Claude as orchestrator, 6
dispatched sub-agents for Phases 1–6). Engine, UI, and LIC
visualization are in. The Phase 7 docs / metadata slice is now in;
pointer perturbation, OG art, broader live preset validation, and
parent-repo wiring remain.

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
| 7     | Polish (content, perturbation, JSON-LD, OG)| **partial** |
| 8     | Parent-repo wiring                         | **todo** |
| 9     | Extended physics (Hall, cooling/heating, conduction, self-gravity, ambipolar/Biermann, viscosity, geometry, sponge, EMF toggle, positivity guard) | **partial** — see [Session 14](sessions/session-14.md), [Session 15](sessions/session-15.md), [Session 16](sessions/session-16.md), [Session 17](sessions/session-17.md), [Session 18](sessions/session-18.md), and the "Phase 9" section below |

**Verification status**: OT live-verified at N=256 and N=1024 across
the full η range up to 1.0 (Session 13 — the RKL2 substep-count fixes
make high-η OT smoothly diffusive instead of speckled). Harris ran
clean through 400+ steps as of Session 12. Sod / Brio-Wu not yet
retested after Sessions 2-13 — should be safe (Sod is pure hydro, no
RKL2 path; Brio-Wu is η=0, no RKL2 path) but worth a smoke test.
N=512 also not directly verified. Session 18's large source-physics
pass was syntax/import checked and browser-smoked at 256² with a
nonblank rendered canvas, but that does not replace the per-preset
validation matrix above.

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
microphysics buffer, dual-energy pressure recovery, unified Hall +
ambipolar + Biermann evaluation, dual-aware diagnostics, and a driven
wind/cloud preset for open-boundary experiments. Canonical presets still
stay on `BASE_PHYSICS_FLAGS`.

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

3. **`stage_params` clarity.** The three stage uniform buffers are
   written once at init and immutable thereafter, but the code
   structure makes it look like they're rewritten every step. Add a
   clear comment or consolidate into a single buffer with offset
   views.

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

### 3. Pointer-drag velocity perturbation

New shader `src/gpu/shaders/perturb.wgsl` + integration:

* Compute shader, runs once per pointer event (not per frame).
* Reads U0 (ρ, ρvx, ρvy), applies a Gaussian bump at the click
  position with the pointer-velocity vector, writes U0 back.
  Recomputes E in U1 (energy = ½ρ|v|² + p/(γ−1) + ½|B|²) to maintain
  consistency.
* New buffer in `buffers.js`: `perturb_uniforms` (cx, cy, vx, vy,
  sigma, amplitude) — small uniform buffer pushed per pointer event.
* New pipeline + bind group layout: `perturbBGL` (Uniforms +
  perturb_uniforms + U0_n RW + U1_n RW). Single dispatch over
  interior cells.
* Wire to canvas pointer events in `ui.js` or a new
  `src/perturbation.js`. Track pointer-down position; on pointer-move
  compute Δ in cell coords; enqueue a perturb dispatch into the next
  render encoder.
* Mobile: hook through `shared-touch.js` for touch → pointer.
* Transpiler-compatible (no fancy WGSL features needed).

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

### 6. OG image

* Create `og/plasma.html` mirroring `og/geon.html`'s structure.
  Hardcoded colors (no shared-tokens dep). NERV register — stylized
  current sheet or OT vortex.
* Add to `og/generate.js` CARDS array.
* Run `node og/generate.js` to produce `plasma/og-image.webp`
  (1200×630 WebP quality 90).
* Update meta tags in `plasma/index.html` to reference the new
  image.

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

## Phase 8 — Parent-repo wiring

Estimated: 0.5 day. All in `/Users/a9lim/Work/a9lim.github.io/`.

Files to modify:

1. **`.gitmodules`** — submodule entry already exists from initial
   `git submodule add`, but the parent commit is still at the
   submodule's initial empty state. After Phase 7 lands and the
   submodule is pushed, commit on the parent updating the submodule
   pointer: "Add plasma submodule with v1 implementation".

2. **`src/projects.js`** — add `plasma` entry to `PROJECTS` with
   `kind: 'sim'`, all i18n fields, `_ICON.projPlasma`. Use the
   existing `geon` entry as the shape reference.

3. **`_worker.js`** — add card to `SIMS_SSR` (HTML string, lines
   ~285–306). Mirror the description from `src/projects.js`.

4. **`_routes.json`** — add `/plasma/*` to `exclude` array.

5. **`_headers`** — add `/plasma/*` block with Early Hints preloads
   (`shared-tokens.js`, `shared-base.css`, `/plasma/styles.css`,
   `/plasma/main.js`), cache rules for `/plasma/*.js` and
   `/plasma/*.css`, **and `Cross-Origin-Embedder-Policy:
   credentialless`** (same as geon — needed for WebGPU compute timing
   on some browsers).

6. **`_build.js`** — add to `IMAGE_MAP`, `IMAGE_CAPTIONS`,
   `aboutFiles`, and main-sitemap URLs.

7. **`index.html` (parent)** — add to homepage SSR fallback `<li>`
   list and to the JSON-LD `Course` schema `@graph` array.

8. **`llms.txt`** — add one-line description for plasma.

9. **`manifest.json`** — add to `shortcuts` array.

10. **`og/generate.js`** — add `plasma` to `CARDS` (handled in Phase
    7 step 6).

11. **`shared-icons.js`** — add a `projPlasma` icon (SVG path;
    geometric, NERV register).

After all 11: run `node _build.js` from the parent to regenerate
sitemap, feeds, llms-full.txt, home-data.json. Then `./dev.sh` smoke
test; then push parent.

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
wind/cloud boundary preset. The remaining work is now the heavier physics
architecture: HDS vs sub-cycling for Hall/non-ideal terms, RKL2 for large
conduction/viscosity workloads, externally-vetted microphysics tables, and
full r-weighted geometry/gravity solvers.

### 1. Validation presets and view modes ✅ (Session 15)

Without dedicated tests / views, none of the new features were
*visible* enough to know whether they're doing the right thing.
`view-field.wgsl` gained T, |q|, φ view modes; `presets.js` gained
four isolation presets:

* **Hall whistler dispersion** — single-mode plane wave, measure
  phase velocity, compare to ω = k·v_A·√(1 + (k·d_i)²) (Tóth, Ma,
  Gombosi 2008).
* **Thermal conduction front** — isolated hot spot, watch it spread
  along a uniform B (anisotropic test); compare to the analytic
  diffusion solution.
* **Cooling instability** — uniform gas at supercritical Λ;
  watch fragmentation timescale.
* **Jeans instability** — small density perturbation under
  self-gravity; measure growth rate vs. Jeans λ.

### 2. apply-bcs after extended physics ✅ (Session 15 — Codex pass)

`_encodeApplyBcsDst` now wraps `_encodeExtendedPhysics` on both
sides — once before so the Poisson / Hall stencils read consistent
ghosts, once after so the post-step state is canonical for the
next RK3 stage. Targets the right destination side via the bind-
group cache.

### 3. Hall sub-cycling ✅ (Session 15)

Tóth 2008-style explicit Hall sub-cycle inside one hyperbolic
macro Δt. The Hall whistler-like timescale `v_A·d_i/dx` is reduced
separately in `compute-dt.wgsl` (not added to the macro signal-
speed sum, so the macro step only respects the hyperbolic CFL).
Host reads `dt_buf[3] = hall_speed_max` one-step-lagged, computes
`N_hall = ceil(dt_macro · hall_speed_max / safety)` capped at
`hallSubstepsMax`, seeds `dt_sub = dt_macro / N_hall` into
`b.hall_dt` via `queue.writeBuffer` before submit, and
`_encodeExtendedPhysics` loops the `compute_emf →
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
shape as Hall sub-cycling. `compute-dt.wgsl` reduces `4·χ/dx`
separately (not added to the macro signal-speed sum), host
computes `N_cond = ceil(dt_macro · cond_speed_max / safety)`
capped at `hallSubstepsMax` (shared cap for both sub-cycles), and
`_encodeExtendedPhysics` loops the `compute_delta + apply_delta`
pair `N_cond` times with `dt_sub` seeded into `b.cond_dt`.

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
microphysics storage table rather than hard-coded WGSL constants. The
default table is still a compact code-unit fit; the next realism step is
loading a vetted metallicity-dependent cooling/heating data product into
the same buffer contract.

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
`apply_delta` with the `conduction_dE` scratch buffer).

Session 18 supersedes the separate Hall/non-ideal dispatch path for the
main step with `apply-ohm.wgsl`: `compute_emf` evaluates Hall,
ambipolar, and Biermann from one frozen state, `apply_hall_update` +
`repair_hall_energy` handles the nondissipative Hall part, and
`apply_dissipative_update` then applies ambipolar/Biermann at fixed total
energy so magnetic diffusion heats/cools the gas through the conserved
energy budget. The older `apply-hall.wgsl` / `apply-nonideal.wgsl`
modules still compile as legacy/reference kernels.

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
* Viscosity ν, bulk viscosity, B-aligned fraction, shock viscosity
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
  order periodic central difference. The Poisson solve is still Jacobi
  on a periodic Cartesian box; FFT/multigrid and non-periodic gravity
  remain future work.
* **Diagnostics:** Stats show the last Hall and conduction sub-cycle
  counts, so stiff regimes are visible instead of hidden in the step.

### 12. Session 18 microphysics / dual-energy pass

What landed:

* **Microphysics:** `src/microphysics.js` builds an uploaded table used
  by tabulated cooling, neutral fraction, and transport scaling.
* **Dual energy:** `U1.zw` carry internal energy and entropy proxy; GPU
  primitive recovery plus CPU stats/probe paths now use the same fallback
  pressure logic.
* **Unified Ohm:** `apply-ohm.wgsl` replaces the main-step split between
  Hall and non-ideal kernels with one frozen generalized-Ohm evaluation.
* **Open-boundary preset:** `driven-wind-cloud` exercises a driven west
  inflow, NSCBC outflow, tabulated cooling/heating, conduction, Hall,
  ambipolar/Biermann, and viscosity in one scenario.

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
* Hu, Adams & Shu (2013) — positivity-preserving limiters for
  finite-volume MHD
* Balsara & Spicer (1999) — arithmetic-mean CT EMF baseline

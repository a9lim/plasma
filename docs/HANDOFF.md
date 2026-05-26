# HANDOFF.md — plasma

Sim built across one orchestration session (Claude as orchestrator, 6
dispatched sub-agents for Phases 1–6). Engine, UI, and LIC
visualization are in. Polish and parent-repo wiring remain.

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

**Verification status**: OT live-verified at N=256 and N=1024 across
the full η range up to 1.0 (Session 13 — the RKL2 substep-count fixes
make high-η OT smoothly diffusive instead of speckled). Harris ran
clean through 400+ steps as of Session 12. Sod / Brio-Wu not yet
retested after Sessions 2-13 — should be safe (Sod is pure hydro, no
RKL2 path; Brio-Wu is η=0, no RKL2 path) but worth a smoke test.
N=512 also not directly verified.

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

### 4. Voice content — about.md and edu-content

Both need to land. Invoke the `/writing` skill — it has the prose
voice rubric for a9lim's site copy.

* **`plasma/about.md`** — ~400 words, a9lim's prose voice. Mirror
  geon's structure:
  - What is plasma? (one paragraph framing — what MHD is, what the
    sim shows)
  - Physics (resistive 2.5D ideal MHD; mention HLLD / PPM / RK3 / CT
    in passing)
  - Presets (one paragraph each on OT, Harris, Brio-Wu, Sod)
  - Numerical method (one paragraph)
  - Accessibility (keyboard nav, contrast modes, ARIA — see geon's
    about.md)
  - Learning Outcomes (5–7 bullets: Alfvén waves, magnetic
    reconnection, frozen-in-flux, current sheets, pressure balance,
    plasma β)
  - Prerequisites (linear algebra, partial differentiation, ideally
    exposure to fluid dynamics or E&M)
  - References (Stone 2008, Miyoshi-Kusano 2005, Colella-Woodward
    1984, Gardiner-Stone 2005)

* **`<details class="edu-content">`** in `index.html` — 500+ words
  technical-reference register (NOT a9's prose voice — per parent
  AGENTS.md, edu-content stays in technical doc register). Full MHD
  equations, HLLD wave structure, RK3 weights, CT divergence
  preservation argument, References with DOI, See-also links to
  sibling sims.

### 5. SEO — JSON-LD blocks

Following parent AGENTS.md conventions. **CRITICAL**: every Wikidata
QID and DOI must be verified against the live source before
committing.

* **WebApplication + LearningResource** with `teaches` (5 plasma
  concepts), `about` array with Wikidata entities (verify each):
  - Magnetohydrodynamics (Q133143 — verify)
  - Plasma (physics) (Q10251 — verify)
  - Magnetic reconnection (Q579070 — verify)
  - Alfvén wave (Q495186 — verify)
  - Orszag-Tang vortex (verify or omit if no entity exists)
* `isBasedOn`: Miyoshi-Kusano DOI, Stone+ Athena++ DOI,
  Colella-Woodward DOI (verify via `https://api.crossref.org/works/<doi>`).
* `educationalAlignment`: 3+ standards (AP Physics C: E&M, NGSS
  HS-PS3, professional IAS/AAS plasma curriculum) with `targetUrl`
  links.
* `FAQPage` — 5–7 domain-specific Qs.
* `HowTo` — 5 steps.
* `BreadcrumbList`.
* All entities with `@id` URIs.

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

### 8. Sim metadata bump

Update `dateModified` in JSON-LD and `lastUpdated` config in
`initAboutPanel`.

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

* **Don't skip live verification.** Six static-trace verification
  passes is good, but the engine has never actually run. Phase 7
  step 1 matters — do it first.

* **The agent reports embedded in each phase commit message are
  useful context for debugging.** Each Phase agent flagged its
  decisions beyond spec, pre-flight concerns for the next phase,
  and engine critique. Read them via `git log -p` if something
  feels off.

* **The `shared-wgsl-transpile.js` constraint is real.** If you
  write new shaders, keep them transpilable (no nested barriers,
  no textures, no exotic types). The Phase 6 agent's transpiler
  audit table is the working pattern.

* **Voice content needs the `/writing` skill.** `about.md` is in
  a9's voice per parent AGENTS.md "Prose Voice" section.
  edu-content stays in technical-reference register.

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

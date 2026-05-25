# HANDOFF.md — plasma

Sim built across one orchestration session (Claude as orchestrator, 6
dispatched sub-agents for Phases 1–6). Engine, UI, and LIC
visualization are in. Polish and parent-repo wiring remain.

This doc is next-instance / next-agent context for picking up where
we left off. The implementation plan lives at
`~/.claude/plans/geon-currently-uses-cpu-abstract-cat.md` — that's the
source of truth for design decisions; this doc is specifically about
what's left to do and what to watch for.

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

**Verification status**: OT live-verified at N=256 and N=1024 — runs
indefinitely with gorgeous current-sheet structure. Sod / Brio-Wu /
Harris not yet retested after the Session 2 fixes (should be safe —
fixes are general — but worth a smoke test). N=512 also not directly
verified.

## Session 2 — Verification + engine bug fixes

First live verification attempt hit a WebGPU storage-buffer-limit
error, then a series of cascading NaN problems that turned out to be
**five distinct bugs** of varying severity. All now fixed. Documenting
because the lessons matter:

### 1. Storage buffer limit overflow

`update-conserved-weighted` had 11 storage bindings; the M-series
adapter caps at 10. Fix: bump `maxStorageBuffersPerShaderStage` to 10
in `device.js` (matches geon's pattern) + flip `dt_buf` from storage
to uniform binding in `update-conserved-weighted.wgsl` (it's a single
f32, naturally a uniform). The shader now wraps it in a `DtUniform`
struct since uniforms can't be raw arrays in WGSL.

**Lesson**: count storage bindings per pipeline against the adapter's
`maxStorageBuffersPerShaderStage` (typically 10 on desktop, 8 baseline)
at build time, not at first dispatch. Per-pipeline budget is tight.

### 2. dt buffer missing COPY_SRC (latent pre-existing bug)

The stats-display readback path was silently returning zero for dt
because the `plasma.dt` buffer didn't have `COPY_SRC` usage. WebGPU
validation errored on the copy but the rest of the batch went through,
so the symptom was just "dt always reads as 0." Fix: add `COPY_SRC` to
the dt buffer's usage flags in `buffers.js`.

**This is probably why HANDOFF reported the `simTime` ratchet was
broken** — simTime itself is fine via the stats-display workaround,
but that workaround was reading zero. The Phase 6 agent's fix #1
should be reassessed against this.

### 3. apply-resistivity race condition

The shader did an in-place 5-point Laplacian on `read_write` storage
and asserted (in a confidently-wrong comment) that neighbor reads
return pre-dispatch values. **WebGPU has NO such guarantee.** Neighbor
reads at workgroup-tile boundaries pick up post-write values from
already-executed tiles. Manifested as regular-spacing "blebs" of
artifact J_z at high η (8×8 workgroups → ~8-cell artifact stride);
below noise floor at low η, undetected by static review. Fix: double-
buffer via a `snapshot` entry point that copies dst→snap (per-cell,
no neighbors, race-free), then `main` reads snap and writes dst. Added
3 snapshot buffers (`Bx_res_snap` / `By_res_snap` / `U1_res_snap`).
BGL grew to 9 bindings (7 storage, under cap).

**Lesson**: in-place RMW + neighbor reads on storage buffers is
ALWAYS a race in WebGPU, even when it appears to work on most hardware.
The shader's old "we rely on this" comment was the smoking gun — that
phrase should be a code smell. HANDOFF flagged this for CPU emulation
but the same constraint applies to GPU dispatch.

### 4. HLLD_BX_EPS2 too conservative

Was 1e-24 — basically "exactly machine zero." At thin current sheets
with |Bn|~1e-5, the full HLLD 5-wave path runs with tiny `bn²` and
tiny `g_L = ρ(S-u)² - bn²` denominators. The 1e-20 `safeDL` guard
inflates `bt_Ls = bt·g_L/safeDL` to ~1e20 → NaN cascade. Bumped to
1e-10: falls back to HLLC whenever |Bn| < ~1e-5·√ρ — robust at near-
degeneracies, no visible effect on bulk physics. HANDOFF explicitly
flagged this as conservative; following its own advice.

### 5. No defensive sanitization on conserved state

Once any cell went non-finite for any reason, it poisoned compute-dt's
wavespeed atomicMax reduction (NaN bits → huge u32 → corrupt
wavespeed), which made dt useless, which cascaded NaN to the whole
field within ~5 steps. Added IEEE-clean sanitization at the end of
`update-conserved-weighted.wgsl`:
- `clamp(ρ, FLOOR, 1e30)` — NaN → FLOOR via IEEE maxNum semantics
- `select(0, m, m == m)` for momentum — NaN → 0
- `clamp(E, KE + p_floor/(γ−1), 1e30)` — NaN → minimum p≥floor value
- `select(0, Bz, Bz == Bz)` — NaN → 0

This is the breaker that finally let OT survive indefinitely. The
sanitization is conservative-state-only (no access to Bx_face/By_face
here), so the magnetic-pressure contribution to the p-floor check is
omitted — downstream `cons_to_prim`'s pressure floor catches the slop.

**Lesson**: any solver with an atomicMax wavespeed reduce needs
conserved-state sanitization at the write site. Otherwise a single
bad cell → bad dt → cascade. This is true regardless of which Riemann
solver / reconstruction / time integrator you choose.

### Bonus: η floor mechanism (kept as latent infrastructure)

Built `sim.getEtaMin()` returning `etaFloorCoeff · dx` (grid magnetic
Reynolds criterion η_min ≳ C·v_char·dx), with slider dynamic-min,
dynamic hint text, refresh on preset/resolution change. Calibrated
empirically against OT critical η: N=256 ≈ 8e-4, N=1024 ≈ 1e-4. The
empirical C·v_char product **scales super-linearly with N** (OT
concentrates energy faster at finer grids), so a single coefficient
is wrong somewhere. After fix #5 the floor was unnecessary — sim
survives gracefully at η=0 thanks to sanitization. OT's preset sets
`etaFloorCoeff: 0`. Mechanism kept available for future presets that
genuinely need it.

### What this means for HANDOFF's prior assumptions

- "Most likely failure modes" 1–4 from Phase 7 Step 1: #1 was real
  (HLLD eps), #2 (CT indexing) was NOT triggered, #3 (uniform field
  writes) was NOT triggered, #4 (flux convention mismatch) was NOT
  triggered. Whole new bug class found: WebGPU race conditions in
  same-buffer RMW kernels.
- Phase 6 agent's `simTime` ratchet flag: the ratchet itself is still
  technically broken (Sim.step doesn't read dt back) but its
  observable consequence was being driven by bug #2 above. Verify by
  watching the stats panel's dt value.
- "Resolution change is destructive" note: still applies, working as
  documented.
- HLLD_BX_EPS2 note: nudged upward as recommended (1e-24 → 1e-10).

### Slider widening

For calibration we widened the η slider's HTML `max` from `-1`
(η=0.1) to `0` (η=1.0). Left it widened — gives users more range for
experimentation, no downside.

## Session 3 — P0/P1 polish sweep

Static-review pass by the next instance turned into a 12-item sweep
of correctness and performance fixes, landed via three rounds of
sub-agents (three small parallel WGSL fixes, then a uniforms
restructure, then the sim.js orchestration overhaul). Two Phase 7
step-2 items resolved as side-effects:
* **#2 — `step_parity` reclaimed.** Slot 12 of the Uniforms struct
  now carries `cfl: f32`.
* **#4 — CFL slider wired live.** `setCFL` pushes the uniform;
  `compute-dt.finalize()` reads `U.cfl`. The pressure-floor slider
  is still inert — the wire-through template now exists if anyone
  wants to finish it.

### Engine — correctness

1. **Resistive CFL** in `compute-dt.wgsl` was `0.5 · dx²/η`, past the
   linear stability bound for a 5-point 2D Laplacian. Corrected to
   `0.25 · dx²/η`. Hyperbolic-dominated regimes don't notice; high-η
   sliders no longer eat NaNs.

2. **`atomicMax(bitcast<u32>(s))`** in compute-dt's per-tile reduce
   would latch on NaN or sign-bit-set floats (both map to huge u32).
   Belt-and-suspenders: `select(0.0, s, s >= 0 && s == s)` before
   the bitcast. Session 2 sanitization should make this unreachable,
   but defense in depth costs nothing.

3. **HLLD_BX_EPS² test was dimensionally inconsistent** — `bn² < ε²·ρ`
   compared B² to ρ, absorbing an implicit c²-like factor. Rewritten
   to `bn² < ε² · ρ_avg · ((SR-SL)/2)²` with the same 1e-10 constant.
   **Empirical re-calibration risk**: the old value was tuned at OT
   N=256/1024 against the old form. The new form may shift where
   Branch A triggers. OT and Harris smoke-tests are first thing to
   re-verify after this session.

4. **`hll_flux_mhd` refactored** to take a new `HllInputs` struct
   carrying pre-computed AL/AR/FL/FR/SL/SR/QL/QR. Removes duplicate
   cf/u/S/F recomputation in the Branch B/C fallback paths. Noticed
   mid-refactor: the old HLL `out.fBt2` was stored into `flux_1.w`,
   which `update-conserved-weighted`'s `(1,1,0,0)` mask zeroed
   anyway — observable output identical, now consistent with the
   main HLLD path's `pack_flux` conventions.

5. **Magnetic-pressure energy floor** added via new
   `energy-floor.wgsl` kernel (5 storage bindings + 1 uniform),
   dispatched between `update-conserved-weighted` (step 7) and
   `update-b-weighted` (step 8). Reads dst U0/U1 + src Bx/By, clamps
   E to `KE + ½|B|² + p_floor/(γ−1)`. Closes the consistency gap
   that the 10-binding cap had forced open in update-conserved's
   own floor — sub-floored pressure at thin current sheets no longer
   carries one cycle of inconsistent state.

### Engine — performance

6. **Bind-group recreation eliminated.** Sim pre-bakes an A/B
   bind-group cache at init, rebuilt on `setResolution`. Per-step
   allocations dropped from ~36 to 0. Estimated ~100 ms/sec CPU
   saved at 60 fps. Renderer + LIC ports cache too.

7. **η-gated resistivity dispatches.** At η=0 (Sod, Brio-Wu, OT),
   the apply-resistivity triad (9a apply-bcs, 9b snapshot, 9c
   diffuse) no longer issues — saves 6 dispatches/step on ideal-MHD
   presets. Both shaders had internal early-outs already; this just
   skips the dispatch entirely.

8. **Step-1 apply-bcs dropped in stages 2 and 3 when η>0.** Stage
   1's 9a already filled the same buffer stage 2 reads from; same
   for stage 2's 9a → stage 3. At η=0 stage 1 doesn't run 9a, so
   step-1 is preserved in all stages. Encoder logic is asymmetric
   on η — written explicitly in `_encodeStage` with comments.

9. **apply-resistivity dispatch tightened** from `(N_total+1)²` to
   `(N+3)²` via in-shader index shift (`ix = gid.x + ghost - 1u` at
   the top of both `snapshot` and `main`). The snapshot copy covers
   exactly the Laplacian's read footprint — no wasted invocations.

### Uniforms layout (changed shape — see shared-helpers.wgsl)

* Single `uniform` buffer (64 B) replaces the `uniform_x` /
  `uniform_y` pair.
* Two static `sweepDir_{x,y}` uniforms (16 B each) bound only by
  `reconstruct-ppm` and `riemann-hlld`. No more dual-write per push.
* LIC fields split into a 16 B `licUniform` written every render
  frame via `_pushLicUniforms()`; main uniform untouched per render
  (only on physics-state changes: setEta, setViewMode, setCFL,
  setGamma, preset load).
* Slot 11 (`sweep_dir`) → `_pad_sweep` (reserved). Slot 12
  (`step_parity`) → `cfl` (f32).
* `SHADER_VERSION` now at 8.

### Visualization

* `LIC_STEPS` reduced from 30 to 20 — ~33% LIC compute drop with
  minimal coherence loss. If the trace looks too short at high
  resolution, bump back up in `config.js` AND `lic-advect.wgsl`
  (both must match).

### Smoke tests outstanding

Verify these survive the sweep before further engine work:

1. **OT at N=256 and N=1024** — primary HLLD_BX_EPS² calibration
   target. Confirm reconnection topology and central density blob
   look right.
2. **Sod / Brio-Wu / Harris** — never re-verified post-Session 2;
   now also need to confirm the energy-floor kernel doesn't
   over-clamp at strong shocks (Brio-Wu) or thin sheets (Harris).
3. **CFL slider** — drag it, confirm dt visibly responds.
4. **η slider** — at η>0 (Harris), confirm diffusion looks right; at
   η=0, confirm 6 fewer dispatches per step (browser perf trace).
5. **N=512 resolution** — first time exercised; the bind-group
   cache rebuild path goes through here too.
6. **Save/load round-trip** — `cfl` field now means something live,
   confirm restore actually applies it.

### Deferred to future sessions

* **PPM workgroup-shared cell-primitive cache** (5× `cons_to_prim`
  redundancy per cell). Breaks the "workgroup-shared only in
  compute-dt" transpiler-contract line, so deliberately deferred
  until transpiler hookup is closer. Easy win when it lands.
* **Characteristic-variable PPM limiting** (review item #14) — only
  if Brio-Wu/Sod show strong-shock overshoots in smoke tests.
* **Upwind CT EMF, Gardiner-Stone 2005** (review item #15) — only
  if grid-aligned CT artifacts show.
* **`timestamp-query` device feature** in `device.js` for actual
  perf measurement. One-line addition; worth doing before the next
  perf pass so it's empirical.
* **Pressure-floor slider** wire-through (CFL slider template now
  exists; same shape).

### AGENTS.md sync needed

The doc still reflects pre-sweep state in several spots:
* "Uniforms (64 bytes)" table — slot 11 is now `_pad_sweep`, slot
  12 is now `cfl: f32`. LIC fields are no longer in the main
  Uniforms struct (they're in `LicUniforms`).
* "RK3 SSP scheme" section's "Sweep direction lives in 2 uniform
  buffers (`uniform_x`, `uniform_y`)" — now wrong shape (single
  `uniform` + two `sweepDir_{x,y}`).
* "Default CFL — 0.4 hyperbolic; 0.5 parabolic" — parabolic is now
  0.25.
* "HLLD degenerate branches" Branch A description — ε threshold form
  changed.

Worth a pass when next touching that doc.

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

1. **`simTime` ratchet is broken** (still applies, but downstream cause
   was masked). `Sim.step()` submits but never reads back `b.dt`, so
   `simTime` stays at 0. Stats panel works around it by doing its own
   readback. Session 2 fix #2 (dt buffer COPY_SRC) was a related
   latent bug — verify the stats dt readback is now correct before
   working around it again. If stats now shows a real dt, the simpler
   fix is to consume that value into `simTime` rather than build a
   second readback path.

2. **`step_parity` uniform slot is dead.** No shader reads it. Either
   reclaim the slot for something useful (LIC drift_z? perturbation
   sequence number?) or stop pushing it.

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

5. **LIC contrast normalization (optional).** Output is whatever
   average noise the trace finds — flat field-free regions render
   with a single luminance (~0.5). Contrast-stretching the LIC
   output (subtract min, divide by std) would make the texture pop
   more in low-intensity regions. Doable as a small reduce kernel
   between lic-advect and composite — but adds a barrier-bearing
   kernel, harder for the transpiler. Defer unless really wanted.

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

### 7. visibilitychange wiring (verify)

Phase 1 added `visibilitychange` pause logic. Verify it actually
pauses the simulation when the tab is hidden — easy to test in dev
with a console log on visibility transitions.

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
* Gardiner & Stone (2005) — upwind CT EMF (we use the simpler
  Balsara-Spicer arithmetic mean instead — adequate for v1)

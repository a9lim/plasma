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
   confirm restore actually applies it. Same for `pressureFloor` now
   that the slider pushes the uniform (Session 4 polish).
7. **Pressure-floor slider** — drag from default (1e-6) toward 1e-3,
   confirm sims that lean on the floor (Brio-Wu / Harris strong shocks)
   visibly react. Going too low (1e-8) should keep behaviour identical
   in well-behaved presets like OT.
8. **GPU step time** — Stats panel "GPU step" should report a real ms
   number on adapters with `timestamp-query`. On adapters without the
   feature the row stays at "—". Sanity-check the magnitude against
   wall-clock-per-step (frame time / substeps).

### Deferred to future sessions

* **Characteristic-variable PPM limiting** (review item #14) — ✅
  landed in Session 6. See the dedicated Session 6 section below.
* **`timestamp-query` device feature** in `device.js` for actual
  perf measurement. ✅ landed — adapter is queried optionally;
  Stats panel shows "GPU step" wall-clock per step (decoded from
  the resolve buffer in the existing readback batch).
* **Pressure-floor slider** wire-through (CFL slider template now
  exists; same shape). ✅ landed — slot 11 (`_pad_sweep`) reclaimed
  as `pressure_floor: f32`; six shaders thread it through helpers
  (`cons_to_prim_mhd`, `fast_mag_speed`, `prim_to_cons_pair`,
  `unpack_edge_prim`, PPM, energy floors). `setPressureFloor` now
  pushes the uniform.

## Session 4 — Gardiner-Stone upwind CT EMF (landed)

Replaced the Balsara-Spicer arithmetic-mean corner-EMF with the
Gardiner & Stone 2005 upwind formulation (eqns 41-45). This is the
Athena/Athena++ default and the research-code-standard CT recipe —
avoids grid-aligned numerical dissipation that BS arithmetic-mean
introduces in plane-parallel flows, and adapts upwind direction per
face from the local contact-velocity sign.

### What changed

1. **`riemann-hlld.wgsl`** — Hoisted the contact-wave-speed `SM`
   computation (M&K 2005 eq 38) to run unconditionally right after
   SL/SR, then stamped it into the unused `flux_1.w` slot in every
   write path (supersonic, Branch A HLLC, Branches B/C HLL
   fallback, full HLLD 5-wave). Additive — no new buffers, no BGL
   changes. `update-conserved-weighted`'s `(1,1,0,0)` flux_1 mask
   already zeroes `.w` for the conserved update, so this slot is
   free for CT.
   * Renamed several inner `let pf = pack_flux(...)` shadows
     (`pfL`, `pfR`, `pfA`, `pfH`) to disambiguate from Track A's
     outer `pf = U.pressure_floor`.

2. **`compute-emf.wgsl`** — Full rewrite. Now binds U0, Bx_face,
   By_face (3 new RO storage bindings, 6 total — well under cap)
   so it can compute cell-centered Ez = vy·Bx - vx·By at the four
   cells around each corner. Implements G&S 2005 eq 45 collapsed
   to the four upwind-biased ¼ corrections (derivation in the
   shader header comment). Reduces exactly to BS arithmetic mean
   in the smooth-field limit (face Ez == upwind cell Ez); reduces
   gracefully under Bn ≈ 0 because SM is well-defined from the HLL
   contact estimate in HLLD's degenerate Branch A.

3. **`pipelines.js` `emfBGL`** — extended to 7 entries (1 uniform +
   6 storage). Still 1 BG/pipeline.

4. **`sim.js` `_emfBG`** — Now side- and stage-dependent (takes
   `U0_src, Bx_src, By_src` from the stage's PPM source). Cached
   per (stage × side) in `_buildBindGroupCache`. No per-step
   allocation cost.

5. **`shared-helpers.wgsl` CT update commentary** — updated to
   describe the upwind formulation. Note: ∇·B preservation argument
   is unaffected (it depends only on Ez_corner being SHARED across
   the four cells whose edges touch the corner, not on its recipe).

6. **`AGENTS.md`** — "Numerical method" table row updated. "Design"
   bullet for divergence cleaning updated. Layout tree comment for
   `compute-emf.wgsl` updated.

### Verification status

Static-trace verified; not yet live-tested. Smoke tests after merge:

* **Orszag-Tang at N=256** — the canonical test. Upwind CT should
  preserve the same X-point reconnection structure but with sharper
  current sheets (less BS smoothing). If reconnection topology
  changes qualitatively, suspect a sign error in the upwind
  selector or a face/cell-index mismatch in compute-emf.
* **Brio-Wu** — propagates a multi-wave structure in 1D. Upwind CT
  should preserve the rotational discontinuity sharpness. If the
  middle compound wave smears, the cell-Ez recipe may be wrong.
* **Harris current sheet** — η > 0 path; resistivity still triggers
  on every stage. Confirm reconnection still onsets around t≈10·t_A
  (per AGENTS.md preset table).
* **Sod** — pure hydro, B = 0 everywhere. Cell Ez = 0, face Ez = 0,
  upwind corrections all zero — should be identical to BS output
  to machine precision. Useful regression check.

### Implementation notes

* **Upwind tolerance**: contact velocities below |v| < 1e-12 fall
  into the "v = 0" branch (½ average of left/right upwind cells)
  per G&S 2005 eqn 42-44's "otherwise" clause. Smooth transition
  through zero — no spurious dissipation.
* **Cell-Ez index range**: corners dispatched in `[ghost, ghost+N]²`.
  Cell-Ez lookups touch `(ix-1, iy-1)` through `(ix, iy)` — at
  `ix = ghost`, that's `(ghost-1)`. The ghost band is always
  apply-bcs-filled before riemann (which runs before compute-emf),
  so ghost cell U0 and face B are physical at compute-emf time.
* **No new buffers, no BGL count change at any other pipeline**.
  All edits localized to compute-emf and riemann-hlld outputs.

### AGENTS.md sync needed

The doc still reflects pre-sweep state in several spots:
* ✅ "Uniforms (64 bytes)" table — synced: slot 11 is now
  `pressure_floor: f32` (live UI slider), slot 12 is `cfl: f32`, the
  LIC reserved pads are labelled, and the single-buffer + per-axis
  `sweepDir` shape is documented.
* "RK3 SSP scheme" section's "Sweep direction lives in 2 uniform
  buffers (`uniform_x`, `uniform_y`)" — now wrong shape (single
  `uniform` + two `sweepDir_{x,y}`).
* ✅ "Default CFL — 0.4 hyperbolic; 0.5 parabolic" — corrected to
  `0.25 parabolic`.
* "HLLD degenerate branches" Branch A description — ε threshold form
  changed.

Worth a pass when next touching that doc.

## Session 5 — PPM primitive cache (Track B in Round 2)

Round 2 ran three parallel tracks; this is Track B's deliverable. The
transpiler picked up 2D shared-tile + halo + top-level-barrier support
mid-session (verified by `testTwoDSharedTileHaloBarrier` in
`tests/wgsl-transpile/smoke.js`), which unlocked the PPM cache that
Session 3 had explicitly deferred. The transpiler-contract line about
"workgroup-shared only in compute-dt" was already shown false earlier
this session; workgroup-shared use is now allowed broadly as long as
barriers stay at top level. AGENTS.md transpiler-contract bullet
updated to reflect the new state.

### What changed

1. **`reconstruct-ppm.wgsl`** — Full rewrite of the entry function.
   * New workgroup-shared tile: `var<workgroup> tile : array<array<MhdPrim, 12>, 12>;`
     (struct-in-array, struct contains only f32 scalars — no atomics,
     no runtime arrays — so it satisfies both native-WGSL
     workgroup-allocated type rules and the transpiler's `defaultInit`
     recursion path).
   * Phase A: each thread runs `cons_to_prim_mhd` exactly once for its
     center cell at `tile[ly+2][lx+2]`. Outer rings (`lid.x < 2` /
     `lid.x >= 6`, same for `y`, plus the four corner combinations)
     additionally load halo cells. Global cell indices are `clamp`ed
     to `[0, n_total-1]` so out-of-storage halo reads safely
     re-sample the edge cell.
   * Single top-level `workgroupBarrier()`.
   * Phase B: out-of-extent threads bail (early `return`). In-extent
     threads pack their cached center + the four sweep-axis stencil
     cells (`tile[ly+2][lx..lx+4]` for x-sweep, transposed for
     y-sweep) into the existing `PrimPair` shape via a new
     `pack_prim_pair` helper. Downstream PPM math is unchanged.
   * Net cost per output cell: 1 `cons_to_prim_mhd` (down from 5)
     plus one `workgroupBarrier()` and one cache write/read pair.
   * Dispatch shape unchanged — still `(N+2)²` extent, 33×33
     workgroups at N=256. The per-workgroup-edge clamp + extent
     check covers the dispatch-padding threads at the eastern /
     northern boundary workgroups; the PPM piecewise-constant
     fallback already handled the outermost dispatch cells (whose
     full 5-point sweep stencil hangs off storage).
2. **`pipelines.js` `SHADER_VERSION`** — bumped 10 → 11. Track C
   also bumped to 11; one shared bump is fine.
3. **No BGL changes** — workgroup vars don't change bindings.

### Type choice — what worked

`array<array<MhdPrim, 12>, 12>` with `MhdPrim` containing only `f32`
fields compiles cleanly through both the native WGSL validator (no
errors in `device.createShaderModule`) and the transpiler corpus
walker. The transpiler's `defaultInit` recurses into `type_named` →
struct, emitting the canonical zero-initialized object shape, then
`type_array` wraps it with `Array.from`. No gotchas hit.

The fallback (flat `array<f32, 12*8>` or `array<f32, 12*12*8>`) was
prepared in design but not needed.

### Verification

`node tests/wgsl-transpile/run.js plasma` — all 17 plasma shaders pass
all four expected phases (tokenize, parse, resolve, compile). The new
PPM file went from 1541 tokens to 2263 tokens (cache pattern is
slightly verbose because of the 9-way halo enumeration); the corpus
walker accepts it.

Live verification (Orszag-Tang at N=256/1024 — primary correctness
target, especially with the sweep-direction-agnostic halo pattern) is
outstanding and falls into the same smoke-test bucket as the rest of
Session 3/4's deferred verification. The cache is mathematically
identical to the previous per-stencil-position `cons_to_prim_mhd`
calls — same recipe, same inputs (`U0[idx]`, `U1[idx]`, face-averaged
`bx_c`/`by_c`), same `pressure_floor` clamp — so any difference would
indicate either a barrier-phase-split bug (caught by the smoke test)
or a clamped-halo write picked up by the PPM math at the storage edge
(prevented by the `stencil_ok` check, which forces piecewise-constant
fallback whenever the sweep-axis stencil would touch a clamped halo
cell).

## Session 5 — LIC contrast normalization (parallel track)

Companion track to the PPM primitive cache, landed in the same Round.
Resolves Phase 7 §2 item 5. The same transpiler unlock — the 2D
shared-tile + halo + top-level-barrier pattern verified by
`testTwoDSharedTileHaloBarrier` — also covers the workgroup-shared
atomic reduction this pass needs, so the contract worry that had
deferred this since Phase 6 is now moot.

### What changed

1. **`src/gpu/shaders/lic-reduce.wgsl`** (new). Two entry points
   sharing one BGL.
   * `reset` (1×1) — seeds `lic_minmax[0]` to `bitcast<u32>(1.0)`
     and `lic_minmax[1]` to `0u`. Encoded once per render frame.
   * `main` (8×8) — per-cell `atomicMin/Max(bitcast<u32>(L))` into
     workgroup-shared `tile_min` / `tile_max`, single top-level
     `workgroupBarrier()`, thread 0 commits to global `lic_minmax`
     via `atomicMin/Max`. Mirrors `compute-dt.reduce`'s shape
     exactly. The bitcast trick is safe because lic-advect's
     averaging contract keeps luminance in [0, 1] (non-negative
     floats → u32 bit-pattern preserves ordering). Defensive
     `select`/`clamp` against hypothetical NaN matches the
     "belt-and-suspenders" idiom Session 3 #2 added to compute-dt.

2. **`src/gpu/shaders/lic-normalize.wgsl`** (new). Per-invocation
   compute over interior cells. Reads `lic_minmax`, rewrites
   `lic_out[i] := clamp((lic_out[i] − min) / max(max − min, 1e-4), 0, 1)`
   in place. No barriers, no atomics, no shared memory.

3. **`src/gpu/buffers.js`** — new 8-byte `lic_minmax` storage buffer
   (`array<atomic<u32>, 2>` at the shader side). Usage
   `STORAGE | COPY_DST | COPY_SRC` (the COPY flags are debug-readback
   headroom; `reset` does all the seeding at runtime). Also extended
   `lic_out` usage with `COPY_DST | COPY_SRC` since `lic-normalize`
   read-modifies-writes it (the STORAGE bit was already set;
   `read_write` bind-group access doesn't need a new usage flag,
   but the COPY flags match other ghost-padded outputs and give a
   debug readback path).

4. **`src/gpu/pipelines.js`** — `SHADER_VERSION` 10 → 11 (collided
   with Track B's bump; both land at 11, fine). New BGLs
   `licReduceBGL` / `licNormalizeBGL`. New pipelines `licReduceReset`
   / `licReduce` / `licNormalize`.

5. **`src/gpu/lic.js`** — `LicRenderer` now constructs three bind
   groups (the existing per-side `_bg.{a,b}` for `lic-advect` plus
   the resolution-independent `_reduceBG` / `_normalizeBG`).
   `rebuildSideCache()` rebuilds all three on `setResolution`. New
   `encodePost(pass)` method runs reset → reduce → normalize in
   that order on the supplied compute pass.

6. **`src/gpu/render.js`** — `lic.encodePost(pass)` is called
   immediately after `lic.encode(pass)`, in the same compute pass,
   before the pass ends and the composite render pass begins.

7. **AGENTS.md** — "LIC visualization" section rewritten to describe
   the four-pass render chain. New "Per-frame render passes" subsection
   under "Pipeline dispatch shapes" to disambiguate per-stage cost
   from per-frame cost. Transpiler-contract bullets updated: `bitcast`
   no longer "confined to compute-dt" (also lic-reduce / lic-normalize),
   and workgroup-shared use no longer "compute-dt only" (also
   lic-reduce + reconstruct-ppm).

### Normalization choice

Min/max stretch, not mean/std. The brief invited either; the
canonical research-grade vis default is min/max ("show me the
variation"), and on a LIC trace specifically the high-luminance
bands at shock fronts carry more signal than the gentler bell-curve
that mean/std would centre the output on. Min/max preserves the
relative dominance of those bands; mean/std would compress them
toward the mean and brighten the low-variation background by the
same factor.

Implementation: `(lic_out[i] − min) / max(max − min, 1e-4)` then
clamp to `[0, 1]`. The `1e-4` denominator floor handles the
field-free single-luminance edge case gracefully (uniform `lic_out`
maps to a constant somewhere in `[0, 1]` rather than blowing up).

### Verification

`node tests/wgsl-transpile/run.js plasma` — all 17 plasma shaders
including the two new ones pass tokenize → parse → resolve → compile.
Relevant smoke line from the corpus walker (after the change):

```
  ✓ tokenize (243t)  ✓ parse  ✓ resolve  · emit  ✓ compile  plasma/src/gpu/shaders/lic-normalize.wgsl
  ✓ tokenize (389t)  ✓ parse  ✓ resolve  · emit  ✓ compile  plasma/src/gpu/shaders/lic-reduce.wgsl
```

`node tests/wgsl-transpile/smoke.js` — all 50 smoke tests pass
(including `testTwoDSharedTileHaloBarrier` and the existing
`testBarrierReduction`, which together cover the shape lic-reduce
uses).

Live verification (does the LIC texture actually look better in flat
field-free regions, especially in Sod's downstream right state)
falls into the same smoke-test bucket as the rest of Session 3/4/5's
deferred verification. Smoke test plan:

* **Sod** — B = 0 everywhere. Pre-normalize lic_out is whatever
  average noise the trace picks up (~0.5 luminance everywhere).
  Post-normalize the same buffer gets stretched to the full
  [0, 1] range, so composite should show a visible noise texture
  where there used to be near-uniform gray.
* **Orszag-Tang** — strong-field regions (around the central
  density blob) already have visible LIC structure. The stretch
  should leave those regions essentially unchanged because they
  already span a wide luminance band; the previously washed-out
  outer rim should pick up texture detail.
* **Harris** — same expectation as OT; the reconnection plasmoids
  should retain their LIC signature while the upstream region
  reads more clearly.

### Cross-track coordination

`SHADER_VERSION` bump collided with Track B's PPM cache (both 10 →
11). No conflict — one shared bump is fine; the cache-bust just
flushes everything once. No other files overlapped.

### Follow-up for Round 3 / characteristic-variable PPM limiting

* The cache is sweep-axis-agnostic; any future limiter that wants the
  full ±2 stencil along the sweep axis can read directly from
  `tile[ly+2][lx..lx+4]` / transposed without re-running
  `cons_to_prim_mhd`.
* Characteristic-variable limiting needs the *transformed* primitive
  state (left-eigenvector projection of `MhdPrim`). The cheapest path
  is to do the projection per stencil position after the tile read —
  the matrix elements depend on the local sound speed and Alfvén
  speeds, which require the primitive state already in cache.
* If Round 3 wants to cache the projected state instead, the same
  shared-tile shape works — replace `MhdPrim` with a wider struct
  carrying the 7 characteristic variables. Tile cost grows but stays
  well under the 16 KB workgroup-storage floor.

## Session 6 — Characteristic-variable PPM limiting (landed)

Replaced the per-primitive-variable PPM monotonicity limiter with
characteristic-variable limiting (Stone+ 2008 §3.4.2 — the Athena/
Athena++ default for MHD). Primitive cell-to-face differences are
projected onto the 7-wave MHD primitive eigenbasis at the cell center,
the standard CW 1984 monotonicity check applies per wave family
independently, then the limited deltas project back to primitive space
before face-state recovery. Mathematically correct for the hyperbolic
system; matches the research-code default.

### What changed

1. **`reconstruct-ppm.wgsl`** — Full rewrite of Phase B. Phase A (the
   workgroup-shared primitive cache from Session 5) is unchanged.
   * New struct `PrimVec7` carrying the sweep-aligned 7-tuple
     `(ρ, v_n, v_t1, v_t2, B_t1, B_t2, p)` and `PermutedPrim8` (the
     same plus `B_n` as an 8th scalar — the eigensystem treats `B_n`
     as a parameter, not a wave).
   * New helpers: `permute_prim` (sweep-axis rotation of `MhdPrim`),
     `vec7_of` (drop B_n), `pack_prim_pair_from_vec7` (unpermute back
     to the existing `PrimPair` layout the Riemann solver consumes).
   * New `EigenSystem` struct + `mhd_eigensystem(w, bn, gamma)` — line-
     for-line port of Athena++'s `characteristic.cpp` MHD-adiabatic
     branch. Computes `c_f, c_s, a, α_f, α_s, β_t1, β_t2, sgn(B_n),
     √ρ, 1/√ρ, 1/ρ` per Stone 2008 eqs A10–A17 (using the cancellation-
     free identity `c_s² = γp B_n²/(ρ c_f²)`).
   * New `project_to_char(dW, S)` (L · dW per Stone A18) and
     `project_from_char(C, S)` (R · C per Stone A12 — line-for-line
     Athena++'s `RightEigenmatrixDotVector`).
   * New `ppm_limit_delta(dL, dR)` — same CW 1984 monotonicity
     algebra as the previous primitive limiter, just reformulated to
     operate on deltas directly (algebraically identical:
     `dL_new = 2·dR` when the right-face overshoot fires,
     `dR_new = 2·dL` when the left-face overshoot fires; `dL=dR=0`
     when not monotone).
   * `ppm_limit_char(aL, aR)` — applies `ppm_limit_delta` to each of
     the 7 characteristic components.
   * New main flow: stencil_ok fallback unchanged (piecewise constant
     when 5-point stencil hangs off storage); for in-stencil cells,
     compute 4th-order interpolants in primitive space, form
     primitive deltas, project to characteristic, limit, project back,
     recover faces.
2. **`pipelines.js` `SHADER_VERSION`** — bumped 11 → 12.
3. **`AGENTS.md`** — Design + Numerical-method table rows updated.
   New "Characteristic-variable PPM limiting" subsection added under
   the existing PPM-cache section.
4. **No BGL changes** — purely additional computation inside the same
   bind group. No new buffers, no new uniforms, no dispatch-shape
   change.

### Eigensystem derivation source

Athena++ `src/reconstruct/characteristic.cpp` — verified against the
canonical Stone+ 2008 paper:

* `mhd_eigensystem` ↔ the per-cell intermediates (lines 56–110 of
  `characteristic.cpp` — `id, sqrtd, isqrtd, btsq, bxsq, gamp, tdif,
  cf2_cs2, cfsq, cssq, asq, bt, bet2, bet3, alpha_f, alpha_s, s`).
* `project_to_char` ↔ lines 113–138 (`v_0..v_6` formulas with `nf, qf,
  qs, af_prime, as_prime` intermediates) — Stone eq A18.
* `project_from_char` ↔ lines 357–377 (`v_0..v_6` formulas with `qf,
  qs, af, as` intermediates) — Stone eq A12.

The component naming maps as: Athena++'s `IBY/IBZ` ↔ our `bt1/bt2`,
Athena++'s `ivy/ivz` ↔ our `vt1/vt2`. Athena++'s `bet2/bet3` ↔ our
`bet1/bet2` (we drop the leading-1 offset because we don't carry an
IBX entry inline — `B_n` lives in its own scalar).

### Degeneracy regularization choices

Four cases, mirroring Athena++:

1. **`c_f² − c_s² ≤ 0`** (Roe96 case V — "triple umbilic" where all
   three magnetosonic speeds coincide): `α_f = 1, α_s = 0`. Fast wave
   carries the acoustic mode; slow contribution vanishes.
2. **`a² − c_s² ≤ 0`** (Roe96 case IV — low-β; slow waves degenerate
   to acoustic): `α_f = 0, α_s = 1`.
3. **`c_f² − a² ≤ 0`** (Roe96 case III — high-β; fast waves degenerate
   to acoustic): `α_f = 1, α_s = 0`.
4. **Generic**: `α_f = √((a²−c_s²)/(c_f²−c_s²))`,
   `α_s = √((c_f²−a²)/(c_f²−c_s²))` (with `max(·, 0)` round-off
   guards). `α_f² + α_s² = 1` by construction.

**Perpendicular B regularization**: when `|B_⊥|² = B_t1² + B_t2² = 0`,
pick `(β_t1, β_t2) = (1, 0)` (Brio-Wu 1988 eq 45 / Roe96 pg 60 —
matches Athena++). Any orthonormal pair works; the symmetric choice
in Stone 2008 §A.1 is the standard.

**Sign of B_n**: `sign(0) = +1` (matches Athena++'s `SIGN` macro).

**Sound speed floors**: `1/asq` guarded with `max(asq, 1e-30)` and
`a · √ρ` divisor floors via the same. The pressure floor is the
existing `pressure_floor` uniform (`p ≥ p_floor` before recovery).

### Cache choice

Did NOT cache the projected characteristic state in the tile, per the
brief's recommendation. The reason holds up after writing the
implementation: each cell's eigenmatrices are local to that cell's
center primitive state, so the 4 neighbors' projections must use the
center's L matrix — caching the characteristic state of each neighbor
in the tile would either require one L matrix per neighbor (wasteful)
or rebuilding L at each output cell anyway (which is what we now do).
The simpler `MhdPrim` tile keeps the 4.5 KB workgroup-shared
footprint and lets each thread compute its own eigensystem locally.

### Interface state for eigenvector evaluation

Used the CELL CENTER `w_c` as the basis for the eigenvector projection
(not an interface-averaged state). Two reasons: (1) the CW 1984
parabola is defined w.r.t. the cell center, so the eigenvectors should
match that basis; (2) Athena++'s `plm_simple.cpp` and `ppm_simple.cpp`
do the same — they compute the eigensystem from `w[i]` and apply it
to differences taken around `w[i]`. Roe-averaged interface states are
the alternative; per Stone 2008 §3.4.2, both are acceptable. Cell-
center is the simpler, lower-cost choice and the canonical Athena++
pattern.

### Verification

* `node tests/wgsl-transpile/run.js plasma` — all 17 plasma shaders
  pass tokenize → parse → resolve → compile. PPM token count grew
  2263 → 4425 (the eigensystem + 7×7 projections nearly doubled the
  shader); still well within the transpiler corpus walker's
  appetite.
* `node tests/wgsl-transpile/smoke.js` — all 50 smoke tests pass.
* Static derivation: every eigenvector formula in `project_to_char`
  and `project_from_char` was cross-checked against Athena++
  `characteristic.cpp` lines 113–138 and 357–377 respectively.

Live verification (Sod, Brio-Wu, Orszag-Tang, Harris at N=256/1024)
falls into the same smoke-test bucket as the rest of Sessions 3-5's
deferred verification. Reasoning about each preset:

* **Sod** — pure hydro, B = 0 everywhere. `B_n = 0`, `|B_⊥| = 0`,
  fall into the case-V degeneracy (`c_f² = c_s² = a²`). With
  `α_f = 1, α_s = 0`, the slow-wave columns vanish; the
  fast/entropy/Alfvén columns reduce algebraically to the Euler
  3-wave eigenvectors. Should behave essentially identically to the
  primitive limiter at Sod's resolutions.
* **Brio-Wu** — this is where characteristic limiting most visibly
  improves on primitive. Strong compound slow shock + rotational
  discontinuity benefit from per-wave limiting that doesn't smear
  the slow-wave structure into the contact / fast waves.
* **Orszag-Tang** — should preserve the established structure with
  possibly cleaner shock crossings around `t ≈ 0.5`. The central
  density blob and four current sheets should still be there. Regression
  watch: if reconnection topology changes qualitatively, suspect a
  sign error in `project_from_char` (the bet1/bet2 cross terms have
  a sign convention that flips between Athena++'s naming and ours).
* **Harris current sheet** — smooth field everywhere except at the
  sheet itself. Per-wave limiting should make ~no difference until
  reconnection onset; once plasmoids form, the slow-wave compression
  associated with the X-points should resolve more cleanly.

### Open concerns to flag before live test

1. **Permutation convention.** Athena++ uses
   `(IDN, ivx, ivy, ivz, IPR, IBY, IBZ)` ordering with `ivy = ivx+1%3`,
   `ivz = ivx+2%3`. Our `PrimVec7` matches that mapping for x-sweep
   (n=x, t1=y, t2=z) and uses the cyclic continuation for y-sweep
   (n=y, t1=z, t2=x). The eigensystem is invariant under any
   orthonormal basis for the transverse plane, so the cyclic
   continuation is mathematically fine — but a sign-or-swap mistake
   in `pack_prim_pair_from_vec7` (the inverse permutation) would
   propagate to face states the Riemann solver consumes.
   Specifically: y-sweep packs `p0 = (rho, vx, vy, vz)` from
   `(rho, vt2, vn, vt1)` and `p1 = (p, Bx, Bz, 0)` from
   `(p, bt2, bt1, 0)`. Double-check this if y-sweep behavior diverges
   from x-sweep on a symmetric setup (e.g., Orszag-Tang).

2. **Pressure positivity at strong shocks.** Characteristic limiting
   can recover face states with negative pressure in principle if the
   raw 4th-order interpolant overshoots and the limiter's projection
   back to primitives doesn't fully suppress it. The existing pressure
   floor (`max(l1.x, pf)`) catches this on write, but Brio-Wu's strong
   shocks are the most likely place to see it. If `Branch C` (negative
   pressure recovery in HLLD star states) starts firing markedly more
   often than under primitive limiting, suspect this.

3. **β-degeneracy at thin current sheets.** Harris with `B_n → 0` and
   `B_⊥` substantial at the sheet itself hits a different branch of
   the eigensystem than the bulk (`c_a → 0`, no `B_⊥ = 0` issue). The
   case-IV `(a² − c_s² ≤ 0)` branch fires here. Should be fine — both
   Stone 2008 and Athena++ exercise this branch routinely — but worth
   eyeballing the reconnection onset time (HANDOFF says `t ≈ 10·t_A`).

4. **Cost.** Per output cell: 1 eigensystem build + 2 L-projections +
   1 R-projection. Each projection is 7×7 dense ALU. Estimated ~2×
   the per-cell ALU of the primitive limiter; PPM is not the dominant
   pipeline (HLLD + EMF + CT eat most of the step time). No GPU
   timing yet; the existing `timestamp-query` stats panel will show
   the actual delta after live verification.

5. **Transpiler nested-struct caveat.** Initial implementation used
   `struct PermutedPrim { w: PrimVec7, bn: f32 }`. The transpiler's
   SROA pass doesn't currently scalarize struct-of-struct returns —
   emitted JS produced `{ w: R_w, bn: R_bn }.w.rho = P.rho;` (object
   literal on LHS, broken). Worked around by flattening to
   `PermutedPrim8` (all f32 fields). The native WGSL compiler
   would have accepted the nested-struct form fine; this is a
   transpiler-compatibility constraint to remember for future kernels.

### Cross-track coordination

No other parallel tracks this round — Session 6 is the sole agent.
`SHADER_VERSION` bump 11 → 12 lands cleanly.

## Session 7 — Primitive-space safety net for characteristic PPM (landed)

a9 reported grid-scale (1–2 cell wavelength) stripe patterns in J_z
appearing on Orszag-Tang at low η, growing exponentially until
NaN/∞/zeros cascaded. CFL = 1e-2 didn't help; at high η the stripes
were damped by `η ∇²B`. Diagnosis: characteristic-variable PPM
limiting (Session 6) can produce face states where individual
primitive deltas have **opposite sign** from their unlimited
primitive value, because the back-projection `R · a_limited` is a
linear combination across wave families. Primitive limiting would
never do this — but characteristic limiting can, and at strong MHD
discontinuities it routinely does. The sign-flipped deltas seed
grid-scale oscillations in B that only the explicit resistive term
damps. Documented behaviour in Mignone (2014) §3.4, Felker & Stone
(2018); Athena++ ships a primitive-space safety net for exactly
this reason. Session 6's port didn't have it. Session 6's
open-concern #2 ("Pressure positivity at strong shocks") was the
same mechanism with a different symptom.

### Static-trace verification before changing anything

Walked the eigensystem to rule out a sign error:
* L · R = I checked algebraically on the Alfvén block (off-diagonal
  L_aL·R_aR = 0; diagonal L_aL·R_aL = bet1² + bet2² = 1).
* L · R = I checked on the fast block (diagonal collapses to
  0.5/a² · 2a² = 1 via the eigensystem identities
  α_f² + α_s² = 1 and α_f²(cf²−a²) = α_s²(a²−cs²)).
* Sweep-axis pack/unpack: `pack_prim_pair_from_vec7` y-sweep
  produces `p1 = (p, Bx, Bz, 0)` from `(p, bt2=Bx, bt1=Bz, 0)`,
  consumed by `unpack_edge_prim` y-sweep as `Q.bx = edge1.y = Bx`,
  `Q.bz = edge1.z = Bz`. Correct.
* Gardiner-Stone upwind EMF: the four ¼-correction terms and the
  upwind selectors (sign of SM at each face) check out against
  eqns 41-45.
* `SM_face` stash: present in all six write paths in HLLD
  (supersonic L/R, Branch A HLLC, Branch B HLL, Branch C HLL,
  full 5-wave).

The math is right. The structural problem is the same one
described in Mignone (2014).

### What changed

1. **`reconstruct-ppm.wgsl`** — added the safety net:
   * New `PrimFaces` struct (struct-of-`PrimVec7`) — uses the
     field-by-field var-assignment pattern that `CharLR` already
     proves out through the transpiler's SROA pass.
   * New `primitive_safety_net(w_left_raw, w_right_raw, w_c, w_m1, w_p1)`
     helper, two-step:
     * **A**: clamp each face component to
       `[min(w_c, w_neighbor), max(w_c, w_neighbor)]` — eliminates
       sign flips and large overshoots in one shot.
     * **B**: re-apply CW1984 parabola overshoot check per
       primitive component via `ppm_limit_delta`. After step A
       the `dL·dR ≤ 0` flatten branch can't fire (face lies between
       cell and neighbor), but an asymmetric parabola can still
       fold further.
   * Main flow renames the raw projection outputs `w_left_raw` /
     `w_right_raw` and threads them through `primitive_safety_net`
     before pack-and-floor. Algorithm-step comment in the file
     header bumped 11 → 12.
2. **`pipelines.js` `SHADER_VERSION`** — bumped 12 → 13.
3. **`AGENTS.md`** — new paragraph in the "Characteristic-variable
   PPM limiting" subsection describing the safety net and why it's
   there.
4. **No BGL changes, no buffer changes, no dispatch-shape change** —
   purely additional inline math.

### Why this is principled (and not just "more dissipation")

The safety net is a *no-op on smooth flows*. When the characteristic
projection happens to land in primitive-monotone territory, the
clamp doesn't bite (face value already in the neighbor interval)
and the parabola check doesn't bite (`dL`/`dR` already pass CW1984).
Sharpness on smooth waves is preserved. The net only activates at
discontinuities — exactly where characteristic limiting was
producing overshoots — and at those points, the right thing to do
*is* to drop to primitive monotonicity. Athena++ has the same
two-layer structure for the same reason.

The alternative would have been to revert to per-primitive PPM
limiting wholesale, giving up Session 6's sharpness on smooth MHD
waves. The safety-net layering keeps the best of both: per-wave
limiting away from discontinuities, primitive monotonicity at
shocks.

### Cost

Per cell, the safety net adds:
* 14 `clamp` calls (one per primitive component × 2 faces)
* 7 `ppm_limit_delta` calls
* Two `PrimVec7` field-by-field reconstructions

PPM token count grew 4425 → 5258 tokens (~19%). PPM is not the
dominant pipeline (HLLD + EMF + CT eat most of the step time),
so wall-clock impact is small. The `timestamp-query` stats panel
will show the actual delta once live-tested.

### Verification

* `node tests/wgsl-transpile/run.js plasma` — all 17 plasma
  shaders pass tokenize → parse → resolve → compile. The
  `PrimFaces` struct-of-struct return goes through cleanly
  because of the field-by-field var-assignment pattern (the
  Session 6 transpiler caveat about nested-struct constructor
  literals does NOT apply here; same pattern as `CharLR`).
* `node tests/wgsl-transpile/smoke.js` — all 65 smoke tests
  pass.

Live verification outstanding. Smoke test plan:

* **Orszag-Tang at η = 0** — the primary symptom case. Stripes
  should be gone; reconnection topology and central density blob
  should look like Session 2's verified baseline. If stripes
  persist, suspect Session 4's upwind CT EMF (less grid-aligned
  numerical dissipation than the previous BS arithmetic mean) and
  consider adding a Hyman-style EMF averaging fallback.
* **Orszag-Tang at η = 5e-3** — what a9 was running when the
  symptom appeared. Should also be clean.
* **Brio-Wu** — strong-shock test. Confirm the compound slow
  shock and rotational discontinuity stay sharp (safety net
  should be a no-op on the smooth Riemann fan, only active
  right at the discontinuities).
* **Harris current sheet** at η = 1e-3 — confirm reconnection
  onset still happens around t ≈ 10 t_A. The safety net's
  primitive-monotonicity constraint shouldn't suppress the
  slow plasmoid growth; if it does, the bet1/bet2 cross terms
  in the right eigenvector might be hitting a corner case the
  clamp is being conservative about.
* **Sod** — B = 0 everywhere; safety net should be exactly the
  same as primitive PPM (the characteristic projection
  degenerates to the 3-wave Euler basis, so the safety net
  shouldn't fire at all on the smooth fan).

### Things to watch in live testing

1. **Pressure positivity at strong shocks** (Session 6 open concern #2,
   same mechanism). Branch C (negative star-pressure HLL fallback) firing
   frequency should drop noticeably under the safety net.
2. **Smooth extrema preservation**. The neighbor clamp is conservative
   at true local maxima/minima — it'll flatten a smooth peak slightly.
   Athena++ has extra extremum-preserving logic (McCorquodale &
   Colella 2011) layered on top; if OT's central density blob looks
   less sharp than Session 2's baseline, that's the next thing to add.
3. **N=1024 OT** — the previous primary verification target. Confirm
   sharpness on the four current sheets.

### Deferred follow-ups

* **Extremum-preservation (PPM4 of McCorquodale & Colella 2011)** —
  if the neighbor clamp turns out to over-flatten smooth maxima.
  Athena++ has this in `ppm_extrema_preserving`; it'd be a
  separate helper between step A and step B above.
* **Hyman-style EMF dissipation fallback** — if Session 4's upwind
  CT EMF turns out to be contributing to the underlying stripe
  problem (less likely, but possible). The dial would be a
  blending coefficient between BS arithmetic mean and the G&S
  upwind formula, controlled by a local shock detector.
* **`setLimiter` UI toggle** — a way to A/B between primitive
  PPM, characteristic PPM, and characteristic + safety-net.
  Useful for diagnostic / educational purposes; trivial to add
  given the existing slider scaffold.

## Session 8 — Phase 2 Wave 1 + diagnostic-driven bug hunt

a9 asked for items 1-4 and 6-9 from Claude's research-grade engine
sweep recommendation (deferring #5 Hall MHD). Plan: ship diagnostics
first (#1 Alfvén convergence test + #2 conservation diagnostics) to
validate the engine before adding features, then run the four
Phase-2 features in parallel (#3+#9 RKL2 + anomalous resistivity,
#4 PPM4, #6 LHLLD EMF, #8 NSCBC outflow), then Wave 2 sequentially
(#7 dual-energy).

Plan didn't survive contact with the diagnostics.

### Wave 1 — diagnostics landed cleanly

* `87194c8` **Conservation diagnostics**. Two-pass GPU reduction
  (per-tile partial sums → single-workgroup finalize) over seven
  quantities — ∫ρ, ∫ρv_x/y/z, ∫E, ∫½|B|², ⟨|∇·B|⟩ — feeding a new
  "Conservation" section of the Stats panel with drift % and
  sparklines. Two-pass design forced by momentum signs (the bitcast-
  atomic trick from compute-dt / lic-reduce assumes non-negative).
  Floor-trigger counters deferred. SHADER_VERSION 13 → 14.
* `73fe1e0` **CPAW convergence harness**. Tóth (2000) / Stone+
  (2008) §4.2 circularly polarized Alfvén wave test at
  `tests/alfven-convergence.html`, driving the sim via Playwright +
  WebGPU at N ∈ {32, 64, 128, 256}. New `alfven-cpaw` preset.
  **First run measured asymptotic L1 slope ≈ 0.78 instead of the
  expected 3.** Initial diagnosis: limiter clipping smooth extrema.
  Promoted #4 (PPM4 extremum-preservation) to top priority.

### Wave 1 features — three landed, one reverted

Four agents in parallel touching disjoint shader files. Three
landed cleanly:

* `b7b73b3` **PPM4 extremum-preserving** (McCorquodale & Colella
  2011, Athena++'s `ExtremaPreservingFn` pattern). Replaced the
  Session 7 Mignone primitive safety net with PPM4's
  median-of-three limited curvature reconstruction at extrema.
  CPAW slope unchanged (0.78 → 0.78) — limiter wasn't the bug,
  but PPM4 landed without regressions and is the correct shape of
  limiter for smooth flows.
* `6bd09ea` **NSCBC outflow BC** (Poinsot & Lele 1992,
  characteristic-zero-gradient variant). Eigensystem helpers
  (`mhd_eigensystem`, `project_to_char`, `project_from_char`)
  copied into `apply-bcs.wgsl` as `*_bc` variants to avoid
  conflict with parallel PPM work. Periodic / reflecting / driven
  branches unchanged. shared-helpers consolidation deferred.
* `553dec7` **RKL2 super-time-stepping + anomalous resistivity**.
  Meyer-Diehl-Kupka (2014) RKL2 for the η∇²B Laplacian: one
  super-step per RK3 macro-step (1st-order Lie split — matches the
  other operator-splitting error in the scheme), substep count
  `s = ceil(½·(√(1+8·dt_super/dt_parabolic)−1))` capped at 100,
  coefficients computed on CPU and uploaded to `sts_coeffs`. RKL2
  needs three buffer snapshots (Y_init, Y_prev, Y_pprev) plus
  per-substep working storage — split `apply-resistivity.wgsl`
  into three files (`snapshot`, `apply-resistivity-init`,
  `apply-resistivity-prev`) to stay under the 10-storage-binding
  cap. Anomalous resistivity `η(|J|) = η_0 + α · max(0, |J|/J_crit − 1)²`
  (Birn et al. 2001 GEM-challenge closure) with two new sliders
  in the advanced settings dropdown; α=0 disables (constant-η
  baseline). `compute-dt.wgsl` lost the parabolic CFL fold;
  hyperbolic dt is unaffected. At η=0 the entire RKL2 path is
  gated off — no extra dispatches. Predicted ~8× speedup at
  Harris-default η=1e-3, ~30× at η=1e-2. Not benchmarked live
  yet — needs Harris recovery first.
* `d5016d3` **LHLLD corner EMF** (Mignone, Tzeferacos, Bodo 2010
  §4.2 / Mignone+ 2021 §3-4). Replaced Gardiner-Stone with a
  per-wave HLLD-derived EMF using face wave-speed coefficients
  (SL, SLs, SRs, SR stashed in two new `face_wavespeeds_x/y`
  buffers). The agent verified algebraically that LHLLD reduces
  to G&S in the smooth-flow limit and to Sod's identically-zero
  output. **Had a real implementation bug** that surfaced only
  under live verification — turned OT cascade onset from step
  ~200 (baseline) to step ~60. Reverted in `8adb240` for
  re-implementation against PLUTO's `Src/MHD/CT/ct_emf.c` as
  ground-truth.

### The discovery — pre-existing OT instability

Live-verification of all four commits via Playwright revealed that
OT NaN-cascaded within ~1 second of running. Initial assumption:
one of the Phase-2 commits broke it. Bisect showed something
worse — **OT was already broken at the pre-Phase-2 baseline**.

This contradicts Session 2's "OT live-verified at N=256 and
N=1024 — runs indefinitely with gorgeous current-sheet structure"
claim. The contradiction is real and a teaching moment:
visual-only verification with the LIC overlay rendering noise
even when the underlying colormap field is NaN'd produces
"looks stable" reads from "field is dead but visualization
doesn't know it." Session 2's verification likely saw OT
structure form briefly at t ≈ 0.2 and called it verified before
the cascade onset at t ≈ 1.

Spawned a focused bisect agent. They walked the commit history
oldest-to-newest, identified `93f8227` "Engine: research-grade
sweep" (the monolithic Sessions 4-7 commit) as the introducing
commit, then sub-bisected within it by selectively reverting
each session's changes. **Session 4's Gardiner-Stone 2005
upwind CT EMF was the culprit.** Sessions 5 (PPM cache), 6
(characteristic PPM), and 7 (Mignone safety net) all exonerated.

### The G&S 2005 upwind EMF bug

`compute-emf.wgsl`'s in-shader derivation of Gardiner-Stone 2005
eq 45 collapses to:

```
Ez_corner = ¼·(four face Ez)                       [BS arithmetic mean]
          + ¼·(four upwind face_Ez - cell_Ez corrections)
```

which expands algebraically to:

```
Ez_corner = ½·(four face Ez) − ¼·(four upwind cell Ez)
```

In the smooth-flow / uniform-Ez limit this gives `½·4C − ¼·4C = C`
correctly. **But at MHD discontinuities, where the Riemann-solver
face Ez at OT's four current sheets reaches ~2× the local cell
Ez = vy·Bx − vx·By, the formula extrapolates outside the input
range and the CT B-update oscillates unstably.** Cascade onset
at step ~250 was when the central density blob's surrounding
current sheets first reached this regime.

The shader's derivation is self-consistent and matches Stone+
2008 §3.5 eq 23 line-for-line. **Why the canonical published
formula breaks empirically is genuinely subtle.** Best guess:
flux-sign convention mismatch with HLLD's `flux_*_1.z` output
(the conversions `ez_x_lo = −fxl.z`, `ez_y_le = +fyl.z` in
compute-emf could have an inconsistent sign with HLLD's wave-
state stash), OR a missing damping coefficient that Athena++
ships separately. Bug pending re-investigation against Athena++
`src/hydro/calculate_fluxes.cpp` or PLUTO source.

### `8a70578` BS-only EMF stabilization

Until the upwind formula is repaired, fall back to Balsara &
Spicer 1999 arithmetic mean:

```
Ez_corner = ¼·(four face Ez)
```

The cell-Ez upwind machinery (lines 162-188) and the
U0/Bx_face/By_face bindings (4-6) in compute-emf.wgsl stay in
place so re-introducing a corrected upwind term won't require
pipeline-layout changes. SHADER_VERSION 21 → 22.

Stability matrix after the fix:

| Preset | Behavior |
|---|---|
| OT | Stable 800+ steps; canonical spider-web of current sheets visible at t ≈ 5s wall clock |
| Sod | Stable 400+ steps |
| Brio-Wu | Stable 400+ steps |
| alfven-cpaw | Stable 400+ steps |
| Harris | NaN ~step 100 (regression from step ~250 under broken upwind, but Harris was already broken at pre-Session-4 baseline e86a83e at step ~175 — the upwind EMF was masking a deeper Harris issue, not introducing one) |

CPAW slope improved 0.78 → 0.97, L1 halved at every N. Order
still ≈ 1.

### Slope-1 root-cause investigation — no solver bug

Spawned a second investigation agent. They ruled out five
hypotheses using axis-aligned CPAW (k along +x, only x-sweep
exercised):

| Test | Slope | Eliminates |
|------|-------|------------|
| Baseline (tilted) | 0.97 | — |
| Axis-aligned | 0.89 | y-sweep eigensystem permutation |
| Per-primitive PPM (no characteristic) | 1.04 | characteristic-PPM |
| Raw unlimited PPM | 0.98 | the limiter entirely |
| IC sampling vs analytic at t=0 | 2.0 (clean) | IC discretization |
| dt-shrunk 10× | 0.98 | CFL/RK3 |

**Raw unlimited PPM still giving slope 1** was the killer datum —
the bug was downstream of reconstruction. Third agent spawned for
pure-hydro convergence:

* `df5ff14` **Acoustic wave hydro preset + test page**
  (`acoustic-wave-hydro` in `presets.js`,
  `tests/acoustic-convergence.html`). Linear acoustic wave at
  γ=5/3, ρ₀=p₀=1, amplitude A=1e-3, axis-aligned, B=0.
  **Slope ≈ 0.70**, L1 saturating at ~7e-6 starting at N=128.
  Worse than CPAW. **Bug is hydro-side, not MHD-specific** —
  HLLD 5-wave path, BS EMF, CT face-B all exonerated.

Fourth agent for amplitude sweep + remaining subsystem bisect:

* PPM Phase A workgroup-shared cache bypass: **bit-identical L1
  to 12 digits** — cache exonerated.
* HLLC force-fallback to HLL (skip Branch A degenerate path):
  **differences only at 5th-digit fp32 noise** — HLLC exonerated.
* Limiter + PPM4 bypass (raw 4th-order interpolants directly):
  no slope change — limiter exonerated definitively.
* **Amplitude sweep at A ∈ {1e-2, 1e-3, 1e-4}**:

| A | N=32 | N=256 | slope |
|---|---|---|---|
| 1e-2 | 2.93e-4 | 1.61e-4 | 0.28 (nonlinear steepening: L1 ≈ 1.6·A²) |
| 1e-3 | 2.79e-5 | 6.90e-6 | 0.70 (floor) |
| 1e-4 | 2.83e-6 | 6.65e-6 | **−0.44** (anti-converges at fine N) |

* A=0 (zero perturbation) → bit-exact preservation over 1000 steps.

**The L1 floor at N=256 is ~6.7e-6 regardless of amplitude.**
That's the smoking gun for **fp32 cancellation in the
conservative update `U_new = U_old − dt·∇·F`**. With ρ ≈ 1 and
per-step ∇·F of order `A·dt`, the conservative update subtracts a
tiny term from a near-unit accumulator — canonical fp32
cancellation hotspot for finite volume with near-uniform
background. fp32 eps × ρ ≈ 1.2e-7 per cell per step; over 640
steps that's ~7e-5 worst case, consistent with the observed
floor.

At A=1e-2 the acoustic wave nonlinearly steepens within one
period — L1 ≈ 1.6·A² dominated by nonlinear evolution, not
truncation error. **Self-convergence** (vs N=512 reference at
shorter T, no nonlinear steepening yet) measures slope ≈ 1.3 at
A=1e-2, consistent with PPM's known order-degradation at smooth
extrema (Tóth 2000 — primitive-limiter PPM drops L1 order to ~2
at extrema, and the sin-wave IS all extrema).

**Conclusion: no solver bug behind the slope-1 finding.** The
plasma engine is operating at expected fp32 finite-volume
precision. Getting textbook order-3 on this problem requires
fp64 (not available on WebGPU) or delta-form variables (track
perturbation around steady state instead of absolute conserved
state).

`d358d32` landed the amplitude-sweep extension to
`acoustic-convergence.html` so future sessions can rerun the
characterization. `makeAcousticWaveHydroPreset(n, amplitudeOverride)`
gained the optional second argument.

### What this session actually achieved

1. **One real engine bug fixed** (G&S upwind EMF — OT now stable
   end-to-end, canonical structure visible)
2. **Diagnostic infrastructure landed** (conservation panel +
   CPAW + acoustic convergence tests) that quantifies engine
   accuracy and would have caught the EMF bug in Session 4 had
   it existed
3. **fp32 precision limits characterized** — amplitude-independent
   L1 floor at ~7e-6 explains the slope-1 measurement and
   validates that the engine is operating correctly within its
   substrate
4. **Three Phase-2 features landed** (PPM4, NSCBC, RKL2 +
   anomalous η) without engine regression — dormant until
   exercised by future presets/work
5. **One feature reverted** (LHLLD) — filed for proper
   re-implementation against PLUTO source

### Deferred follow-ups

* **Proper Gardiner-Stone 2005 upwind EMF re-implementation.**
  BS-only (`8a70578`) is the temporary fix. The original
  derivation is algebraically self-consistent and matches Stone+
  2008 §3.5 but empirically wrong on OT. Compare line-by-line
  against Athena++ `src/hydro/calculate_fluxes.cpp` (the upwind
  EMF computation in `IntegrateField`) or PLUTO
  `Src/MHD/CT/ct_emf.c`. The most likely bug class:
  flux-sign convention mismatch with HLLD's `flux_*_1.z`
  output, OR a missing damping coefficient (Athena++ has an
  extra `dxinv * 0.5` factor in some terms). Don't reintroduce
  the upwind path without a numerical regression test on OT
  (now feasible via the conservation diagnostics — `dE`
  drift > 1% at 1000 steps fails the test).
* **LHLLD re-implementation** (Mignone+ 2010/2021 UCT-HLLD).
  Reverted `d5016d3` had a real bug. Re-implement with strict
  line-by-line PLUTO `Src/MHD/CT/ct_emf.c` comparison. The
  HLLD wave-speed-stash infrastructure (SLs/SRs/SM in
  `flux_*_1.w` + face_wavespeeds buffers) is the survivable
  scaffolding — only `compute-emf.wgsl` needs the careful
  rebuild. Same regression test gate as above.
* **Harris current sheet recovery.** Long-standing
  pre-existing issue per a9 — never numerically verified in
  the recent past. Was broken at Session 3 baseline e86a83e
  (NaN step ~175 with BS-only EMF). The G&S upwind EMF added
  ~75 extra steps of stability (NaN step ~250) before
  cascading itself but masked the underlying issue. Likely
  candidates from Session 3's polish sweep: the new
  `energy-floor.wgsl` magnetic-pressure clamp being too
  aggressive at the thin reconnection sheet, OR the
  `HLLD_BX_EPS² = 1e-10` calibration change interacting badly
  with Harris's `|B_n| → 0` regime at the sheet itself.
  Bisect Session 3's 12-item polish sweep when prioritized.
* **Dual-energy formulation** (item #7 from Phase 2 Wave 2).
  Originally planned. **Newly relevant given the fp32 floor
  characterization** — dual-energy specifically resolves
  catastrophic cancellation in cons↔prim when `½ρ|v|² >> p`,
  which IS the mechanism producing the L1 ~7e-6 floor we
  measured. Standard implementation: track entropy `S = p/ρ^γ`
  as a backup advected variable; in `cons_to_prim_mhd`, if
  `(E − ½ρ|v|² − ½|B|²) / E < ε_dual` (canonically ε=1e-3),
  use the entropy-derived pressure instead of the conserved
  one. Athena++ / PLUTO standard. ~1-2 day agent task touching
  6 shaders (cons↔prim contract changes).
* **Floor-activation counters** (deferred from `87194c8`).
  Per-step atomic counter of pressure-floor / density-floor /
  magnetic-pressure-floor triggers in
  `update-conserved-weighted.wgsl` and `energy-floor.wgsl`.
  Wire into the Stats panel. Secondary item from the
  conservation diagnostics brief — diagnostic infrastructure,
  not engine work.
* **shared-helpers.wgsl eigensystem consolidation.** NSCBC
  outflow (`6bd09ea`) copied `mhd_eigensystem`,
  `project_to_char`, `project_from_char` into
  `apply-bcs.wgsl` as `*_bc` variants instead of promoting
  them to shared-helpers — that move-and-delete refactor was
  deferred to avoid conflicting with parallel PPM work. Now
  safe to consolidate.

### Lessons

1. **Visual verification is not numerical verification.** The
   Session 2 "OT verified" claim was visual-only. The LIC
   overlay renders noise whether the underlying colormap field
   is real OT structure or all-NaN garbage — "looks stable"
   ≠ "is stable." Diagnostic-driven discipline (conservation
   panel + per-cell NaN reads via Playwright) is what surfaced
   the bug. Make this the standard going forward.
2. **Diagnostics first, features second.** a9's instinct to
   ship #1 and #2 before #3-9 was correct sequencing. Without
   the CPAW test and conservation panel, the slope-1 finding
   AND the OT cascade would have stayed hidden behind LIC.
   Future engine sessions: land the relevant diagnostic
   before the engine change.
3. **fp32 precision floors are real and worth characterizing.**
   The amplitude-sweep test methodology (run at three A's,
   look at A-independent floor) is the canonical way to
   distinguish "solver order is X" from "fp32 ate it." Reuse
   for any future convergence work.
4. **Parallel agents need conflict-detection discipline.** Two
   agents (PPM4 and LHLLD) flagged the same phantom breakage
   in different working-tree intermediates while RKL2 was
   landing. By the time RKL2's commit settled, integration
   was clean. Agents reading partial state can hallucinate
   blockers. Live verification beats static-trace verification
   when commits are landing concurrently.
5. **Subtle solver bugs hide in plain sight.** The G&S 2005
   EMF derivation in `compute-emf.wgsl` was algebraically
   self-consistent, matched Stone+ 2008 §3.5, passed smoke
   tests of "uniform-field limit" — and was still subtly
   wrong in a way that only fired at strong shocks. MHD CT
   EMF needs ground-truth verification against a known-stable
   reference implementation (Athena++ or PLUTO), not just
   careful derivation in the comment header.

### Commits this session

* `87194c8` Stats: GPU-reduced conservation diagnostics
* `73fe1e0` tests: CPAW convergence harness + alfven-cpaw preset
* `6bd09ea` Engine: NSCBC characteristic-zero-gradient outflow BC
* `b7b73b3` Engine: PPM4 extremum-preserving safety net (MC2011)
* `d5016d3` Engine: UCT-HLLD corner EMF (reverted in 8adb240)
* `553dec7` Engine: RKL2 super-time-stepping + anomalous resistivity
* `8adb240` Revert "Engine: UCT-HLLD corner EMF"
* `8a70578` Engine: stabilize OT via BS-only corner EMF (G&S upwind bug)
* `df5ff14` Test: linear acoustic wave (pure hydro) convergence preset + sweep
* `d358d32` Tests: acoustic convergence — amplitude sweep + fp32 floor exposed

SHADER_VERSION trajectory: 13 → ... → 22 (final).

## Session 9 — RKL2 ghost-handling bugfix (Harris recovery, partial)

a9: "let's continue working on the physics of this sim. in particular,
let's figure out why the harris preset is broken, and how we can fix
it."

Outcome: **two distinct RKL2 super-step bugs found and fixed.** Harris
went from "detonates within 1 RK3 step (interior E ≡ 0 across the
whole grid)" to "evolves realistic reconnection physics through ~125
steps before a different instability surfaces." Net 100× improvement
on time-to-failure. Other presets (OT / Sod / Brio-Wu / alfven-cpaw /
acoustic-wave-hydro) checked for regressions — clean.

### Bug 1: E corruption in apply-resistivity-init.wgsl

`apply-resistivity-init.wgsl` line 122 (pre-fix) did:

```wgsl
var u1 = U1_tmp[c];   // U1_tmp is freshly allocated — u1.x is GARBAGE
u1.y = bz_new;
U1_tmp[c] = u1;       // writes back, clobbering u1.x (E) with whatever was in tmp
```

The shader only physically evolves Bz (slot y), but writes back the
full vec4. On the FIRST substep, U1_tmp's interior contains
zero-initialized data (or stale data from a previous RKL2 call). The
read-modify-write contaminates u1.x = E with zero, then the rotation
chain propagates the contaminated E through the substep loop. After
the final `prev → dst` snapshot, U_next's E is zero across the entire
interior. The energy floor in update-conserved-weighted (next step's
stage 1) clamps it back to `p_floor/(γ-1)` ≈ 1.5e-6 — physically
nonsense — and HLLD wave structure collapses immediately.

**Fix**: write u1.x explicitly from U1_init (the frozen U^n snapshot)
instead of trusting U1_tmp's stale contents. Algebraically, the RKL2
recurrence applied to E (which has no Laplacian operator: L_E = 0)
collapses to `Y_j.E = U^n.E` for all j by induction — so the explicit
write is the analytically correct value:

```wgsl
U1_tmp[c] = vec4<f32>(U1_init[c].x, bz_new, 0.0, 0.0);
```

apply-resistivity-prev.wgsl was inspected and confirmed not broken on
its own (its read-modify-write preserves u1.x if init wrote it
correctly), but its header comment was updated to flag the dependence
on init's contract.

### Bug 2: ghost-strip zeroing in `_encodeResistivitySuperStep`

After fixing Bug 1, Harris still detonated by step ~50 with the same
"all-floor, all-1e30-cap" runaway pattern, but now seeded by ghost
corruption at j=1 (the second ghost row from each wall):

```
j=0 (outermost ghost):  rho=0.2, E=0.65, Bx=-1.0  ← preserved (IC)
j=1 (next-inner ghost): rho=0.2, E=0.0,  Bx=0.0   ← CORRUPTED
j=2 (first interior):   rho=0.2, E=0.65, Bx=-1.0  ← preserved
```

The orchestration in `sim.js _encodeResistivitySuperStep` seeded
init / pprev / prev from dst at boot, but **NOT tmp**. tmp was
zero-initialized at buffer allocation. The init / prev shaders'
`in_*_interior` gates skip ghost cells (writes only happen for
interior indices), so tmp's ghost strip stayed zero throughout the
substep loop.

After substep 1's rotation, `new_prev = old_tmp` (zero ghost). The
final `prev → dst` snapshot covers the full `(N+3)²` window including
ghosts — so dst's ghost strip got zeroed. Next step's apply-bcs
refreshes the SOURCE side's ghost but not the DST side, and the
zero-ghost rows drove huge `∇²B` at the wall in the next RK3 stage,
detonating the simulation.

**Fix**: snapshot tmp at boot alongside init / pprev / prev. One-char
change to the seed loop:

```js
for (const dest of [initSet, setA, setB, setC]) {  // was [initSet, setA, setB]
```

Now tmp's ghost = whatever apply-bcs wrote when this side was last
source (≤1 step lagged but physically reasonable), the rotation chain
preserves ghost values, and the final snapshot copies physically
sensible ghost data into dst.

### What Harris does now

With both fixes, Harris loaded at N=256 with default outflow N/S +
periodic E/W + η=1e-3:

| Step | ρ_min | ρ_max | E_max | vₓ_max | Notes |
|---|---|---|---|---|---|
| 0   | 0.20  | 1.20 | 0.90 | 0.00 | IC |
| 50  | 0.13  | 1.20 | 0.90 | 0.53 | seed evolving |
| 100 | 0.027 | 1.20 | 0.94 | 0.77 | sheet thinning, plasmoid forming |
| 125 | 0.024 | 1.20 | 1.05 | 0.94 | X-point outflow at j=106 (sheet+22) |

Sheet center stays at ρ≈1.2 (correct), realistic Alfvén-speed
outflows from the X-point, vmax stays sub-Alfvénic (v_A ≈ 4.5 at
ρ=0.05, B=1). This is canonical resistive tearing mode behavior.

Beyond ~150 steps a **third instability** kicks in and Harris still
detonates — see "Deferred follow-ups" below.

### Caveat: dt-feedback staleness under tight JS loops

The RKL2 substep count and Δt_super are sized CPU-side from
`_lastDtHyp` / `_lastDtParabolic`, which update via async readback of
`b.dt`. In **tight JS loops** (where consecutive `s.step()` calls have
no microtask break), the readback's `.then()` never runs and
`_lastDtHyp` stays stale at whatever it was before the loop started.
For Harris specifically, this causes RKL2 to use wrong Δt_super in
the recurrence (`bz_new = ... + dt_super * gam_tilde_j * L_0`),
amplifying boundary artifacts and triggering early detonation.

In normal real-time usage (60 fps main loop, ~2 substeps per rAF
tick), the readback completes between substep batches and dt
feedback stays fresh.

This affects diagnostic test patterns more than the actual sim. The
proper fix (deferred) is to read `dt_buf.dt` directly inside the
RKL2 shader — eliminates the CPU-side staleness path entirely.

### Verification

- OT / Sod / Brio-Wu / alfven-cpaw / acoustic-wave-hydro all clean
  through 200-400 steps (no NaN, no floor activation, physical
  trajectories).
- Harris through ~125 steps: realistic reconnection. Beyond: still
  unstable.

### Deferred follow-ups

* **Third Harris instability** (≈ step 150-200, post-fix). With both
  RKL2 fixes plus full per-step dt freshness, Harris STILL detonates
  in the 150-200 step range. The trajectory shows healthy sheet
  evolution through step 125 (vmax ~ 0.9, ρ_min ~ 0.02), then runaway.
  Pre-Session-3 baseline (`e86a83e`) hit NaN at step ~175 with
  BS-only EMF — same failure window. So the third issue was masked by
  the E-corruption bug (which destroyed the sim before the deeper
  issue could fire) and is now visible again. Likely candidates: HLLD
  Branch A (`|B_n|² < ε² · ρ · ((SR-SL)/2)²` triggers at low Bx in the
  thinned-sheet region around the X-point and falls back to HLLC; the
  fallback may not handle thin-sheet right), OR a numerical
  instability in the sheet itself once it thins below 2-3 cells
  (sub-grid sheet). Worth bisecting against `8a70578` baseline (when
  the BS-only EMF landed but RKL2 had the corruption bug) to confirm
  the pattern.
* **dt feedback in RKL2 — shader-side resolution.** The CPU-side
  `_lastDtHyp` is structurally vulnerable to staleness in any
  pattern that batches multiple `step()` calls without a microtask
  break. The proper fix: pass `b.dt` as a uniform to
  `apply-resistivity-init.wgsl` and `apply-resistivity-prev.wgsl`,
  read `dt_buf.dt` as `Δt_super` inside the kernel. The substep
  count `s` (sized CPU-side) stays slightly over-sized when dt
  shrinks but remains stable, which is the safe-fail direction
  (RKL2 is unconditionally stable for s ≥ critical_s). One extra
  uniform binding in two pipelines; layout change.

### Commits this session

* (this commit) Engine: fix RKL2 E corruption + ghost-strip zeroing

SHADER_VERSION: 22 → 23.

## Session 10 — RKL2 dt-feedback staleness fix (third Harris bug, partial)

a9: "let's continue working on the physics of this sim. in particular,
let's figure out what the third underlying bug in the harris preset is,
and how we can fix it."

Outcome: **the third bug is the RKL2 dt-feedback staleness path that
Session 9's deferred follow-up named.** Fixed by having the RKL2 substep
shaders read fresh `dt_super` directly from the GPU `dt_buf` (the same
buffer compute-dt writes at the start of every macro step) instead of
from the host-pushed `sts_meta.dt_super` (whose value is the lagged
`_lastDtHyp` populated by an async readback). **Tight-loop Harris went
from NaN-at-step-50 to NaN-at-step-400 — an 8× extension.** A deeper
fourth issue still detonates Harris around step 400 in tight loops;
captured below for the next session.

### How the bug was found

Built `tests/harris-diagnostic.html` + `tests/harris-diagnostic.py` — a
Playwright-driven test page that runs the production Sim under the
Harris preset, reads back U0/U1/Bx/By to CPU every `sampleEvery` steps,
and reports ρ_min, p_min/max (via cons-to-prim), |v|_max, |B|_max,
|J|_max, floor-trigger counts, NaN counts (with first-NaN cell coords),
conservation drift, and ⟨|∇·B|⟩. The harness has a `tightLoop` toggle
that batches `sampleEvery` `step()` calls with **zero awaits between
them** — mirroring the production hot loop's pattern (no microtask
break, so the dt readback's `.then()` never fires).

First baseline run (default `sampleEvery=5`, friendly mode): Harris
**ran 300 steps clean, no NaN**. Trajectory was healthy through the
step-100 X-point-formation transient (vmax briefly spikes to 36, dt
collapses to 3e-5, then dt-shrinkage stabilizes it; by step 300 the
sim has settled at vmax ≈ 4). Contradicted Session 9's "NaN at
step 125-200" claim.

The contradiction was the diagnostic itself: every per-5-step await
inadvertently provides a microtask break, which lets
`_maybeReadbackDt`'s `.then()` fire and update `_lastDtHyp` to the
fresh value. With `tightLoop=on`, Harris detonated by **step 50** —
entire interior at `ρ=DENSITY_FLOOR=1e-6`, `E=1e30` (sanitization
cap), exactly the runaway pattern Session 9 described.

### The fix

`apply-resistivity-init.wgsl` and `apply-resistivity-prev.wgsl` each
gain a new `dt_buf` binding bound as a UNIFORM (DtUniform struct, 16 B
— matches the established pattern in update-conserved-weighted.wgsl
which faces the same per-stage storage-binding cap). Both shaders now
read `let dt_super = dt_buf.dt_hyp` instead of `sts_meta.dt_super`.
The `sts_meta.dt_super` field is RETAINED for layout compatibility +
CPU-side diagnostic, but the shader ignores it.

The substep count `s` is still CPU-sized from
`_lastDtHyp`/`_lastDtParabolic` in `_computeRKL2Coeffs`. Per Session
9's brief, this is the safe-fail direction: RKL2 is unconditionally
stable for `s ≥ s_critical`, and when the lagged value gives `s`
larger than the strict requirement, we over-iterate harmlessly. The
contract that matters — Δt of resistive diffusion applied per macro
step — is now driven by the GPU-fresh value.

Changes:
* `apply-resistivity-init.wgsl` — new `DtUniform` struct, new binding
  12 (uniform), `dt_super = dt_buf.dt_hyp`. Storage-binding count
  unchanged at 9. Header explains the rationale + the staleness
  failure mode in detail.
* `apply-resistivity-prev.wgsl` — symmetric change at binding 11.
  Storage-binding count unchanged at 8.
* `pipelines.js` — `applyResInitBGL` / `applyResPrevBGL` extended.
  `SHADER_VERSION` 23 → 24.
* `sim.js` `_applyResInitBG` / `_applyResPrevBG` pass `b.dt` to the
  new bindings.

### Why this couldn't have broken at η=0

`_encodeResistivitySuperStep` short-circuits at line 1035 with
`if (this.eta <= 0 && alpha <= 0) return;` — Sod, Brio-Wu, OT,
alfven-cpaw, acoustic-wave-hydro all set `eta: 0` and never enter
the RKL2 path. Only Harris (`eta: 1e-3`) exercises the modified
shaders. Zero regression risk to the η=0 presets.

### Verification

* `node tests/wgsl-transpile/run.js plasma` — all 21 plasma shaders
  pass tokenize → parse → resolve → compile (apply-resistivity-init
  tokens 1325 → 1401; apply-resistivity-prev 1199 → 1325).
* Tight-loop Harris N=256, eta=1e-3, sampleEvery=50, 400 steps:
  pre-fix NaN at step 50 (full-grid runaway); post-fix NaN at step
  400.
* Friendly-mode Harris N=256, eta=1e-3, sampleEvery=5, 300 steps:
  pre-fix and post-fix both clean. Post-fix shows slightly less
  vigorous evolution at the X-point (vmax 1.5 vs pre-fix 4.0 at
  step 300) — exactly the right effect: stronger per-macro-step
  resistive diffusion keeps the sheet thicker.
* η=0 presets untouched by inspection (the gate at sim.js:1035 is
  intact).

### The remaining fourth issue (next session) — divB leak at corner cells

Extended `tests/harris-diagnostic.html` to localize the divB blow-up:
per-j-row max, max-cell (i, j) coords, per-row mean at boundary vs
interior, and a list of cells where `p_raw < pressure_floor`. Findings:

1. **divB leak is η-driven.** At η=0 (ideal MHD Harris), divbAvg stays
   at ~2e-5 (fp32 noise) through 30 steps. At η=1e-3, divbAvg grows to
   1.17e-3 over the same window — 50× faster. The RKL2 resistivity
   step is the source.
2. **Leak concentrates at corner cells.** divbMaxCell is at (255, 1),
   (1, 0), (0, 255), (255, 255) etc. — the four interior corners where
   periodic-E/W meets outflow-N/S. By step 5 the corner cells have
   divB ≈ 5e-2; by step 30, divB ≈ 8 at corners while the interior
   mid-row has divB ≈ 2e-3.
3. **Pressure floor cells are exactly the four corners.** From step
   ~50 onward, `pFloorCount` plateaus at 4 with `floorPCells` reporting
   precisely (0, 0), (255, 0), (0, 255), (255, 255). The corner-BC
   priority + divB injection conspires to drive p_raw negative there.

**Mechanism**. RKL2 currently restricts updates to faces strictly
interior to the boundary face (Bx range `ix > ghost && ix <
ghost+n_interior`, By range `iy > ghost && iy < ghost+n_interior`).
The discrete identity ∇·(η∇²B) = η∇²(∇·B) requires the Laplacian
operator to be applied to **every** face that appears in an interior
cell's divB. Excluding the boundary face means the divB contribution
from `(By[i, ghost+1] - By[i, ghost])/dx` evolves only on the upper
half — η·dt·∇²By[i, ghost+1] modifies the first half but the boundary
face contribution stays frozen. divB at the boundary cell leaks by
exactly that amount per RKL2 substep.

**Attempted fix that DIDN'T work** (kept in this writeup for future
sessions). Extended RKL2's range to include boundary faces:

```wgsl
let in_bx_interior =
    ix >= ghost && ix <= ghost + n_interior &&
    iy >= ghost && iy < ghost + n_interior;
let in_by_interior =
    ix >= ghost && ix < ghost + n_interior &&
    iy >= ghost && iy <= ghost + n_interior;
```

Result on tight-loop Harris N=256 eta=1e-3: **interior cleaner**
(divbMeanInterior at step 25 was 8.5e-5 vs 2.5e-4 pre-fix — 3×
improvement), but **corners much worse** (divbAbsMax 41 at step 25
vs ~8 over the full pre-fix 400-step run). NaN onset moved from
step 400 down to step 50. Reverted.

The reason: at the periodic-x / outflow-y corners, apply-bcs's
`on_s_wall` and `on_n_wall` priority paths handle the boundary
face BC but **skip periodic-x correction** at the corner ghost. For
example, By[258, 2] (E-ghost x-column, S-boundary y-row) is filled
by `fill_by_face`'s on_s_wall path — self-copy of the CT-evolved
value — instead of being kept periodically-equivalent to By[2, 2].
Once RKL2 updates the boundary faces using these inconsistent ghost
neighbors, the corner divB explodes. The on_s_wall priority needs
to compose with periodic-x at corner cells before this fix is safe.

**Canonical fix**: replace component-wise η∇²B with the
`curl(η J)` form (Athena++ `src/diffusion/diffusion_b.cpp`, PLUTO
`Src/MHD/Resistive/res_flux.c`). For 2.5D MHD with J_x = J_y = 0,
∇·B-preserving evolution is:

* `∂B_x/∂t |_res = -∂_y(η J_z)`   (CT-shape update at x-faces)
* `∂B_y/∂t |_res =  ∂_x(η J_z)`   (CT-shape update at y-faces)
* `∂B_z/∂t |_res = η ∇²B_z`       (cell-centered, unchanged)

This evolves Bx and By as a curl of an edge-centered η J_z field,
which is identically div-free at the discrete level on the Yee
grid — same divergence-preservation argument as ideal-MHD CT.
For UNIFORM η on a divergence-free B, this reduces to η∇²B exactly
(I checked the algebra). For NON-uniform η (anomalous resistivity),
this is the only formulation that maintains discrete ∇·B = 0.

Estimated work: a new `compute-eta-jz.wgsl` pass writing
`Ez_res = η · J_z` at corners (similar dispatch shape to
compute-emf), then `apply-resistivity-init/prev` rewritten to do
`Bx += -dt·(Ez_res[i, j+1] - Ez_res[i, j])/dy` and the symmetric
form for By. Bz keeps its component-wise Laplacian. Estimated 1-2
hours.

**Quicker alternative** (worth trying first): fix apply-bcs's corner
priority so that for cells where one edge is periodic and the other
is outflow/reflecting, BOTH BCs compose. Specifically, when
`on_s_wall || on_n_wall` fires at a corner column (ix in the h-ghost
band), use the BC-of-the-on_*_wall AND ALSO apply the periodic
correction along the h-axis. This is a localized fix to apply-bcs
that doesn't touch the resistivity shaders.

### Earlier post-fix trajectory (before divB localization)

Tight-loop Harris still detonates at step ~400 post-fix. The
trajectory shows:

| Step | ρ_min | vMax | jMax | divbAvg | divbMax |
|---|---|---|---|---|---|
| 50  | 0.136   | 0.46 | 9.9  | 2.6e-4 | 0.62 |
| 100 | 0.013   | 2.95 | 60   | 3.9e-4 | 0.69 |
| 200 | 0.008   | 1.80 | 34   | 6.5e-4 | 0.58 |
| 300 | 0.006   | 1.50 | 29   | 9.3e-4 | 1.25 |
| 350 | 0.007   | 1.39 | 30   | 1.2e-3 | **3.79** |
| 400 | NaN — 10696 cells | dt floored to 1e-8 | | | |

`divbAbsMax = 3.79` at step 350 — way beyond CT machine-precision
floor (should be O(1e-7) in the interior). divB is **growing** rather
than being preserved. Candidates:

1. **N/S outflow boundary leak.** apply-bcs's face-B outflow is
   "zero-gradient": copy interior face value into ghost. The interior
   CT update only enforces div B = 0 at strictly interior cells, not
   at the boundary cell where one face is the apply-bcs-filled ghost
   and the other is the freshly-evolved interior face. The next
   step's reconstruct-ppm reads the BC-filled ghost as truth and
   propagates the divergence inward.
2. **RKL2 component-wise ∇²B on Yee grid.** For uniform η on a
   staggered Yee grid, the standard 5-point Laplacian-per-component
   IS div-preserving in continuous math but may have boundary terms
   that don't vanish on a finite domain. At Harris's thin-sheet
   reconnection front, the ∇²B operator near the boundary could be
   generating non-trivial div B. Athena++/PLUTO implement resistivity
   as `curl(η J)` (which is identically div-free at the discrete
   level) — that's the canonical fix if this is the issue.
3. **NSCBC outflow primitive-state extrapolation.** The NSCBC ghost
   write in apply-bcs.wgsl extrapolates `(ρ, v_n, v_t1, v_t2, B_t1,
   B_t2, p)` linearly into the ghost band. B_n stays at `perm_b.bn`
   (the boundary cell's value — the comment says "NSCBC doesn't touch
   it; the face-B outflow path keeps the wall-normal B constant").
   But this writes the CELL-CENTERED B_t1/B_t2 (the in-plane B for
   the y-sweep at N/S walls), which then has to be CONSISTENT with
   the face-averaged B that compute-emf and HLLD see. If the
   cell-centered ghost B drifts from the face-B-averaged ghost B,
   the consistency breaks and divB grows at the boundary row.

The natural next move: extend the diagnostic to report div B
**per row** (especially the boundary rows and the first interior
row), and to count cells with `|divB| > threshold`. If divB
concentrates at the N/S boundary, candidates 1/3 are confirmed; if
it concentrates at the sheet interior, candidate 2 is the issue.
Candidate 2 fix is more invasive (replace component-wise ∇²B with
curl(η J), needs a new compute-J pass); candidates 1/3 are local to
apply-bcs.

### Lessons

1. **Diagnostic infrastructure shape matters.** A diagnostic harness
   that awaits anything per-batch is fundamentally a different
   execution shape from the production hot loop. The `tightLoop`
   toggle is essential — it should be the default mode for any
   correctness diagnostic going forward. Stats-display already
   awaits readbacks at 12 Hz, so the production loop is somewhere
   between "fully tight" and "5-step friendly"; the bug surfaces
   anywhere with batch breaks longer than ~16 ms between dt
   readbacks.
2. **Session 9's deferred follow-up was specific and actionable
   and correct.** When the prior instance documents "the proper
   fix: pass `b.dt` as a uniform to apply-resistivity-init.wgsl",
   that's not speculation — that's the work, do it. The handoff
   doc earned its keep this session.
3. **The bug bisect window changes with the diagnostic.** Pre-fix
   the tight loop died at step 50 with the entire interior dead;
   the friendly loop went 300+ steps clean. Same engine, same
   bug, different observability. Always run both modes when
   surfacing an engine bug.

### Commits this session

* (this commit) Engine: RKL2 dt-feedback fix — fresh dt_super from
  GPU dt_buf + harris-diagnostic harness

SHADER_VERSION: 23 → 24.

## Session 11 — RKL2 curl(η J) on Yee grid + corner BC composition

a9: "let's continue working on the physics of this sim. in particular,
let's figure out what the fourth underlying bug in the harris preset is,
and how we can fix it."

Outcome: **the fourth bug — divB leak from RKL2's component-wise η∇²B
violating discrete ∇·B = 0 — is fixed by the canonical Athena++/PLUTO
**curl(η J)** discretization.** The complementary apply-bcs corner BC
priority bug (Session 10 "remaining fourth issue") is also fixed.
divB at corners dropped 90× (14.3 → 0.16 at peak); interior divB
dropped to fp32 noise floor (~7e-5). Friendly-mode Harris extends from
NaN-at-step-100 (curl-only) to NaN-at-step-150 (curl + BC fix). A
FIFTH numerical instability — not divB-related — surfaces at step
~150 with sheet thinning underway; captured below for the next session.

### What changed

**Both apply-resistivity shaders** rewritten to use the curl(η J)
discretization for face-centered B:
```
∂Bx/∂t |_res = −∂_y(η J_z)                  (Bx lives on x-faces)
∂By/∂t |_res = +∂_x(η J_z)                  (By lives on y-faces)
∂Bz/∂t |_res = η ∇²Bz                       (cell-centered, unchanged)
```
J_z is sampled at cell corners (co-located with Ez_edge) by the
Yee-natural stencil:
```
J_z(cx, cy) = ((By[cx, cy] − By[cx-1, cy])
            − (Bx[cx, cy] − Bx[cx, cy-1])) / dx
```
η is sampled at the same corner. For anomalous resistivity, the
closure η(|J|) is evaluated AT THE CORNER using corner J_z directly
(cleaner than the cell-centered averaging the previous implementation
used). For uniform η, η_corner = U.eta everywhere.

The curl form is **identically ∇·B-preserving on the Yee grid** by the
same telescoping argument as ideal-MHD CT: every face of every
interior cell receives a curl-of-an-edge-centered contribution, so
summing the four face updates of a cell gives zero contribution to
∇·B. For uniform η on a divergence-free B, the curl form reduces
algebraically to η ∇²B (proven via the discrete identity
∂²x Bx + ∂x ∂y By = 0 under discrete ∂x Bx + ∂y By = 0). For
NON-uniform η (anomalous resistivity), the curl form is the ONLY
discretization that preserves discrete ∇·B — η ∇²B picks up an
uncontrolled ∇η × J cross term. Anomalous resistivity was therefore
**structurally broken** before this session.

Bz keeps its component-wise η ∇²Bz Laplacian. In 2.5D Bz doesn't
couple to ∇·B (which is ∂x Bx + ∂y By), so the curl-on-Yee argument
doesn't bind it — Bz is a passive scalar advected by the resistive
operator.

**Dispatch range extended** to include boundary faces (Bx_face at
ix == ghost and ix == ghost+n_interior; By_face at iy == ghost
and iy == ghost+n_interior). The four-face telescoping argument
requires every face of every interior cell to receive a curl
contribution; excluding boundary faces leaves boundary cells
leaking ∇·B per substep. Same range pattern as update-b-weighted.wgsl
(ideal-MHD CT).

### Bindings unchanged

Both apply-resistivity-init and -prev keep their existing 10 storage
+ 2 uniform binding layouts. The new corner J_z stencil reads from
already-bound Bx_init/By_init (for the "value" terms of the RKL2
recurrence and now also for the J_z stencil) and Bx_prev/By_prev
(prev shader only). No new buffers, no pipeline-layout changes, no
sim.js orchestration changes.

### apply-bcs corner BC composition fix

HANDOFF Session 10 documented the "remaining fourth issue" as
apply-bcs's `on_s/n_wall` and `on_w/e_wall` priority paths not
composing with the orthogonal periodic axis at corner cells.
Specifically:
1. `fill_bx_face` PERIODIC branch at on_w/e_wall corners: pre-fix
   wrapped Y unconditionally even when v_mode was OUTFLOW, putting
   N-side values into S-ghost rows of the W/E boundary Bx face
   (creating a ∂y Bx discontinuity at all four corners with
   magnitude ≈ 2/dx ~ 256 in Harris-IC units — a huge artificial
   current source).
2. `fill_by_face` OUTFLOW branch at on_s/n_wall corners: pre-fix
   clamped X to the leftmost/rightmost interior column even when
   h_mode was PERIODIC, reading from the wrong column.

Both branches now compose per-axis BCs at corner cells: if the
orthogonal axis is PERIODIC, wrap; otherwise clamp (OUTFLOW). The
symmetric pattern is applied to `fill_by_face`'s PERIODIC branch
(when on_s/n_wall fires with h_mode != PERIODIC) for completeness,
though Harris doesn't exercise that combination. fill_cell_ghost
already composed BCs correctly via its `axis` / `use_horiz` logic
(NSCBC outflow branch + REFLECTING branch).

The fix only activates at corner cells of presets with MIXED BC
types (periodic + outflow); all-periodic presets (OT, Sod, Brio-Wu,
CPAW, acoustic-wave-hydro) are completely unaffected by inspection.

### Verification

* `node tests/wgsl-transpile/run.js plasma` — all 21 plasma shaders
  pass tokenize → parse → resolve → compile. apply-resistivity-init
  1401 → 1427 tokens (+1.9%); apply-resistivity-prev 1325 → 1469
  tokens (+10.9%); apply-bcs 5462 → 5530 tokens (+1.2%).

**Harris diagnostic matrix (N=256, η=1e-3, default outflow N/S +
periodic E/W):**

| Test                                  | Pre-S11 (S10 baseline) | Curl-only        | Curl + BC fix    |
|---------------------------------------|------------------------|------------------|------------------|
| tight, sampleEvery=50, 400 steps      | NaN at ~400            | clean (corners)  | clean (corners)  |
| tight, sampleEvery=100, 1000 steps    | NaN at ~50             | NaN at 200       | NaN at 200       |
| friendly, sampleEvery=25, 800 steps   | (not measured)         | NaN at 100       | NaN at 150       |
| η=0, tight, 400 steps                 | (not measured)         | clean            | clean            |

**Conservation quality** (friendly, sampleEvery=25, η=1e-3, post-fix):

| Step | sumE       | sumBsq    | divbAvg  | divbAbsMax | divbMaxCell  |
|------|------------|-----------|----------|------------|--------------|
| 0    | 44237      | 58982     | 0        | 0          | —            |
| 25   | 44237.2    | 58955     | 2.3e-5   | 0.048      | (255, 112)   |
| 50   | 44237.2    | 58928     | 3.6e-5   | 0.035      | (255, 112)   |
| 75   | 44237.3    | 58902     | 5.0e-5   | 0.037      | (255, 168)   |
| 100  | 44237.3    | 58875     | 7.2e-5   | 0.163      | (255, 175)   |
| 125  | 44237.7    | 58852     | 8.9e-5   | 0.132      | (255, 175)   |

sumE and sumBsq stay essentially constant through step 125 (the
fifth instability's onset). divB grows smoothly at fp32 noise rate.
Pre-Session-11 baseline had sumE growing to 240302 by step 200
(corner-injection-driven artificial heating) and divB peaking at 14
at the SE corner. Both pathologies are gone.

### Evolution character changed (and what that means)

Pre-S11, Harris exhibited rapid "reconnection" by step 100 (vmax
0.6, jmax > 30) — but the diagnostic this session showed that
evolution was BC-corner-driven, not physical tearing. The corner BC
bug created a ∂y Bx ≈ 256 discontinuity at all four corners, which
the curl(η J) (or component η∇²B) operator dissipated by injecting
energy. With both fixes, Harris's true IC evolves: small v_y =
0.01·sin(πx)·sech²(y/a) perturbation, slow growth until reconnection
onset at the Lundquist-number-controlled timescale (S = 1000 →
γ_tear · τ_A ~ 0.6, growth time ~ several Alfvén times, ~thousands
of steps at dt ≈ 1e-3).

In friendly mode the post-fix sim runs near-IC through step 100
(vmax = 0.011, identically the IC perturbation amplitude). Between
step 100 and 125 the perturbation grows ~10× (vmax 0.011 → 0.116);
between step 125 and 150 the simulation NaNs out 12% of cells.
**This is the fifth bug** (see below).

### The remaining fifth issue (next session)

η=0 Harris runs clean through 400 steps (we now confirm hyperbolic
side is fine). η=1e-3 with both Session 11 fixes NaNs at step ~150
friendly / ~200 tight-loop. The trajectory has several distinctive
features:

* **sumBsq drops 14% between step 125 and 150** (from cells going
  NaN/floor, not from physical dissipation). η · J² · V at the IC
  current density (|J| ≈ 10, V = 4) is ~0.4 per time unit — over
  the 0.025 time-unit interval, expected physical dissipation is
  ~0.01 (5 orders of magnitude smaller than observed). Catastrophic
  numerical loss.
* **jMax stays at ~9.7 (IC value)** through step 125. The growth is
  in v, not J — inconsistent with an inductive instability.
* **vmax oscillates** 0.011 → 0.116 → 0.085 across steps 100/125/150.
  Looks like a wave bouncing rather than monotonic exponential
  growth.
* **First-NaN cell at (225, 160)** — INTERIOR, not boundary.
* **divB stays small** (≤ 0.16 at step 100, fp32-noise growth
  everywhere). NOT a ∇·B leak issue.

Hypotheses for the fifth bug:
1. **HLLD Branch A firing at thinning sheet.** At step 125 rhoMin
   = 0.103 (sheet thinning); HLLD's `bn² < ε² · ρ · ((SR-SL)/2)²`
   threshold may be firing more often than expected, falling back
   to HLLC which doesn't handle the regime well. Diagnostic: log
   Branch A fire counts per step.
2. **PPM characteristic limiter at high-α regions.** Strong slow-
   mode coupling in the thinning sheet could trigger the safety-net's
   primitive-clamp aggressively, then introduce gridscale stripes
   that the curl form (which preserves divB) doesn't damp. Compare
   slope when running with PPM4 only (Session 8) vs full
   characteristic + safety-net.
3. **Lie-split error from RKL2 frozen-η.** The RKL2 recurrence
   freezes η at U^n and applies it over the macro step. If U^n's
   sheet thickness differs significantly from U^{n+1}'s (sheet
   thinning fast), the resistivity is misaligned. Cure: drop to
   Strang split, or shorten the macro step.
4. **NSCBC outflow with sustained gradients.** The NSCBC linear
   extrapolation could amplify a slowly-developing wave pattern
   at the N/S boundary. Diagnostic: try strict zero-gradient
   (the old outflow) on Harris and compare.

The natural next move: extend `harris-diagnostic` to sample every
1-2 steps between step 100 and 150, watch which cell goes NaN
first, and inspect the prior step's state of that cell + neighbors.
The localization will narrow the candidate list.

### Lessons

1. **Visual / qualitative verification is even less reliable than I
   thought.** Session 10's reported "Harris evolves realistic
   reconnection through ~125 steps" was almost entirely
   BC-corner-driven artifact — the rapid vmax growth, the sheet
   thinning, the X-point outflow — all were energy injected from
   the bad ghost cells. The actual physical mode is much slower.
   Diagnostic-driven discipline (conservation panel + divB
   localization + cell-by-cell NaN reads) is what surfaced this.
2. **Bug fixes can EXPOSE other bugs.** The curl(η J) + BC fix
   removed the artifact-driven dynamics. The simulation now runs
   true Harris physics, which exposes a SEPARATE numerical
   instability that the artifact had been masking. Each peeling
   reveals the next layer. Sessions 9, 10, 11 followed this
   exact pattern.
3. **The "right shape" of physics matters more than the failure
   window.** Pre-Session-11, Harris "ran longer" (to step 400
   tight-loop) but the physics was wrong. Post-Session-11, Harris
   NaNs at step 150 but the physics is RIGHT until then —
   conservation holds, divB is preserved, the IC's actual
   perturbation is what drives evolution. The latter is unambiguously
   better engineering.
4. **The handoff doc is a real research artifact.** Session 10's
   "remaining fourth issue" section had the bug well-localized
   (divB at corners, η-driven, concentrated at the four
   periodic/outflow boundary corners) AND both the canonical fix
   (curl(η J)) and the quick fix (apply-bcs corner BC composition)
   spelled out. Following the doc directly produced a clean
   landing. Continue this format.

### Deferred follow-ups

* **Identify the fifth instability.** See "The remaining fifth
  issue" above. Per-cell NaN tracing and HLLD branch-frequency
  logging are the natural next instruments.
* **shared-helpers.wgsl eigensystem consolidation.** Still pending
  from Session 10. NSCBC's `*_bc` helper variants in apply-bcs
  remain disjoint from reconstruct-ppm's identical helpers. Safe
  to consolidate now.
* **OT structure check.** With the new corner BC composition (no
  effect on OT — all-periodic) AND the new curl(η J) form (no
  effect at η=0), OT should be byte-identical to its Session 10
  output. Not numerically re-verified this session; should be safe
  by inspection but worth a smoke test.

### Commits this session

* (this commit) Engine: curl(η J) RKL2 resistivity + apply-bcs
  corner BC composition — fourth Harris bug fixed, fifth surfaced

SHADER_VERSION: 24 → 25.

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

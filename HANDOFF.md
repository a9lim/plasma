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

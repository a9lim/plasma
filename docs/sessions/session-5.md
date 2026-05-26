# Session 5 — PPM primitive cache + LIC contrast normalization

Two parallel tracks landed in the same Round 2.

## PPM primitive cache (Track B)

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

## LIC contrast normalization (parallel track)

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


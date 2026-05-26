# Session 10 — RKL2 dt-feedback staleness fix (third Harris bug, partial)

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

## How the bug was found

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

## The fix

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

## Why this couldn't have broken at η=0

`_encodeResistivitySuperStep` short-circuits at line 1035 with
`if (this.eta <= 0 && alpha <= 0) return;` — Sod, Brio-Wu, OT,
alfven-cpaw, acoustic-wave-hydro all set `eta: 0` and never enter
the RKL2 path. Only Harris (`eta: 1e-3`) exercises the modified
shaders. Zero regression risk to the η=0 presets.

## Verification

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

## The remaining fourth issue (next session) — divB leak at corner cells

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

## Earlier post-fix trajectory (before divB localization)

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

## Lessons

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

## Commits this session

* (this commit) Engine: RKL2 dt-feedback fix — fresh dt_super from
  GPU dt_buf + harris-diagnostic harness

SHADER_VERSION: 23 → 24.


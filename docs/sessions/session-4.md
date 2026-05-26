# Session 4 — Gardiner-Stone upwind CT EMF (landed)

Replaced the Balsara-Spicer arithmetic-mean corner-EMF with the
Gardiner & Stone 2005 upwind formulation (eqns 41-45). This is the
Athena/Athena++ default and the research-code-standard CT recipe —
avoids grid-aligned numerical dissipation that BS arithmetic-mean
introduces in plane-parallel flows, and adapts upwind direction per
face from the local contact-velocity sign.

## What changed

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

## Verification status

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

## Implementation notes

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

## AGENTS.md sync needed

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


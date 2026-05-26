# Session 7 — Primitive-space safety net for characteristic PPM (landed)

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

## Static-trace verification before changing anything

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

## What changed

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

## Why this is principled (and not just "more dissipation")

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

## Cost

Per cell, the safety net adds:
* 14 `clamp` calls (one per primitive component × 2 faces)
* 7 `ppm_limit_delta` calls
* Two `PrimVec7` field-by-field reconstructions

PPM token count grew 4425 → 5258 tokens (~19%). PPM is not the
dominant pipeline (HLLD + EMF + CT eat most of the step time),
so wall-clock impact is small. The `timestamp-query` stats panel
will show the actual delta once live-tested.

## Verification

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

## Things to watch in live testing

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

## Deferred follow-ups

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


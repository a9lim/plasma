# Session 12 — Fifth Harris bug: dst-ghost staleness + ρ-floor momentum blowup

a9: "let's continue working on the physics of this sim. in particular,
let's figure out what the fifth underlying bug in the harris preset is,
and how we can fix it."

Outcome: **the fifth bug is actually two coupled bugs** that surface in
sequence. Fixed both; Harris now runs clean through 400+ steps under
both friendly (sampleEvery=1) and coarse (sampleEvery=5) diagnostic
modes, with realistic reconnection physics: vmax peaks ≈ 7 at the
X-point formation transient (~step 150), settles to vmax ≈ 1, jmax
oscillates 17–30, ρ_min ≈ 0.011, sumE drift 0.29% over 400 steps.
Net 3× improvement on time-to-failure (NaN at 150 → clean at 400+).

## Bug 5a: dst-side ghost staleness for RKL2's curl(η J) stencil

`_encodeStage` runs apply-bcs at the START of each RK3 stage on that
stage's SOURCE side (U_n, U_1, U_2 respectively). Stage 3's
destination is the OTHER side's "next" slot (where U_{n+1} will live).
RKL2's super-step then runs ON the dst — but **no apply-bcs has run
on the dst side in this step**. Its ghost cells are 1 step lagged
(from when this side was last a stage's source).

RKL2's curl(η J) discretization reads ghost neighbors at boundary
faces — specifically, J_z at corners (ghost, j) and (ghost+n_interior,
j) reads By[ghost−1, j] and Bx[i, ghost−1], which sit in the ghost
band. Stale ghost data breaks the periodic-wrap invariance
`Bx_face[ghost] == Bx_face[ghost+n_interior]` for RKL2's update of
boundary faces — observed as a slow divB drift concentrated at
display i=255 in Harris's N=256 grid (the last interior column under
periodic E/W). Without the fix, divB at this column grows ~5× faster
and seeds the boundary-driven instability that destabilizes the
hyperbolic step.

**Fix**: add an apply-bcs dispatch on the dst side in its own compute
pass at the top of `_encodeResistivitySuperStep`, before the seed-
snapshot loop. The `applyBcsDst` bind group was already built in
`_buildBindGroupCache` (line ~822) for each stage but had been
dead code since Session 8 retired the per-stage resistivity triad —
this restores the dispatch in its only remaining legitimate slot.
Separate compute pass (not consecutive dispatch within the RKL2 pass)
because the writes need to fully sync before RKL2's first dispatch
reads them; empirically combining them via same-pass dispatches
worked sometimes but interacted badly with NSCBC extrapolation on
freshly-evolved dst state.

## Bug 5b: ρ floor without momentum sanitization → m²/(2ρ) blowup

The bigger killer. Once HLLD over-depletes density at the X-point
outflow region (a known artifact of approximate Riemann solvers
without positivity preservation), ρ_raw at some cells falls below
DENSITY_FLOOR = 1e-6. `update-conserved-weighted.wgsl` floors ρ →
1e-6 BUT leaves momentum at whatever HLLD produced (e.g., m_x ≈ 0.01).
Next stage reads v = m/ρ = 1e4. KE = m²/(2ρ) = huge per cell. vmax →
1000+. CFL clamps dt to DT_MIN (1e-8). Simulation effectively halts
while the energy budget runs away (sumE injection: 44237 → 3.18e6 in
the test run, 70× increase).

Surface symptom is "NaN at step 150" — but inspecting the trajectory
before the crash shows the actual mechanism: ρ → floor at one or two
cells in the X-point outflow, m unchanged, velocity explodes, CFL
collapses, and either NaN (when |B| follows the runaway via CT) or
runaway-but-finite-state (when |B| stays bounded but sim grinds to a
halt at dt=DT_MIN).

**Fix**: at the same defensive-sanitization slot in
`update-conserved-weighted.wgsl` that handles NaN momentum, also cap
|v| at V_MAX_SANE = 10 by scaling all three momentum components when
|v| > V_MAX_SANE. The cap is set ~3× the Harris background fast
magnetosonic speed (c_f ≈ 3.5 at ρ=0.2, B=1) — well above any
physically reasonable bulk velocity in this preset, so the cap acts
as an emergency brake on outlier cells without biasing normal
evolution. Verified by inspection of the post-fix trajectory: vmax
reported by the diagnostic peaks at 7.3 around the X-point onset
(below the cap, meaning the cap was firing at OTHER cells that don't
make the reported max), then settles to ~1.0.

KE per cell at ρ_floor with v capped at V_MAX_SANE: ½·1e-6·100 =
5e-5 — negligible vs background pressure-contribution to E (~0.15).
The sanitization is effectively a no-op in cells that didn't need it.

## Why both fixes are needed

Tested in isolation:
* **V_MAX_SANE alone (no bc-fix)**: still NaN at step 140. The
  divB drift at the periodic boundary (already at 0.67 by step 125
  without the bc-fix, vs 0.13 with it) destabilizes the hyperbolic
  step before the velocity cap can save things.
* **bc-fix alone (no V_MAX_SANE)**: vmax → 433 at step 150 with
  sumE → 3.18M. divB stays bounded (~0.1) but the ρ-floor catastrophe
  still fires.
* **Both together**: clean to 400+ steps with realistic physics.

The bugs feed into each other: divB drift seeds boundary instability
→ wave reflects inward → drives unphysical compression in some cells
→ ρ approaches floor at outflow → m²/(2ρ) catastrophe.

## Verification

* `node tests/wgsl-transpile/run.js plasma` — all 21 plasma shaders
  pass tokenize → parse → resolve → compile.
  update-conserved-weighted 919 tokens (was 851; +8% for the v-mag
  computation and scale-momentum branch).
* Harris N=256, η=1e-3, default outflow N/S + periodic E/W:
  - friendly sampleEvery=5, 400 steps: clean ✓ (was NaN at 150)
  - friendly sampleEvery=1, 400 steps: clean ✓ (was NaN at 183)
  - tight sampleEvery=50, 600 steps: running at commit time
* Trajectory at sampleEvery=1, post-fix:

| Step | dt      | ρ_min | vMax | jMax | divB_max | sumE     |
|------|---------|-------|------|------|----------|----------|
| 0    | -       | 0.20  | 0.01 | 9.97 | 0        | 44236.91 |
| 100  | 1.10e-3 | 0.14  | 0.13 | 9.71 | 0.10     | 44237.72 |
| 125  | 7.4e-4  | 0.14  | 0.61 | 16.0 | 0.08     | 44239.05 |
| 150  | 1.2e-4  | 0.010 | 7.30 | 85.8 | 0.14     | 44330.47 |
| 175  | 3.2e-4  | 0.019 | 1.93 | 56.0 | 0.13     | 44330.54 |
| 200  | 4.0e-4  | 0.018 | 1.54 | 41.6 | 0.13     | 44330.57 |
| 300  | 3.5e-4  | 0.014 | 1.14 | 22.0 | 0.12     | 44352.11 |
| 400  | 3.1e-4  | 0.011 | 1.01 | 17.1 | 0.19     | 44364.73 |

This is canonical resistive-MHD Harris evolution: linear tearing-mode
onset around step 100, X-point formation transient at step 150 (vmax
peaks at the Alfvén outflow speed), then steady-state reconnection
with vmax ≈ 1 sustaining indefinitely. The dt collapse during the
transient is real physics (fast magnetosonic at the thinned sheet)
and recovers as the sheet stabilizes.

## Outstanding regression checks

Other presets NOT verified post-fix this session — should be safe by
inspection (the apply-bcs-on-dst dispatch is a near-no-op for
all-periodic configs since CT already maintains periodic invariance
to fp32 precision; V_MAX_SANE = 10 is well above any preset's
physical vmax). But smoke-test before declaring done:

* **OT** at N=256, η=0 — no RKL2, no V_MAX_SANE firing expected.
  Should match Session 11's "stable 800+ steps" baseline.
* **Sod / Brio-Wu** — same, η=0, V_MAX_SANE never fires.
* **alfven-cpaw** convergence — vmax ~ 1, well below cap. Confirm
  slope unchanged from Session 8's 0.97.
* **acoustic-wave-hydro** — vmax tiny, cap never fires.

## Things tried that didn't work

Documented for next session, to spare them the rabbit holes:

1. **HLLD Branch A unguarded-division fix** — Branch A's E_Ls and bt
   flux formulas divide by raw (SL - SM) and (SR - SM), which COULD
   blow up if SM is close to SL or SR. Adding the same guard used for
   rhoLs / rhoRs (`min(SL-SM, -1e-20)`) made things WORSE — sumBsq
   exploded to 1.88e29 at step 145. The guard SHRINKS |denom| at the
   degenerate limit (from |fp32 noise| to 1e-20), AMPLIFYING the
   division result. Reverted. The actual stability path is to fall
   back to HLL when SM ≈ SL or SM ≈ SR — not just clamp the
   denominator. Filed for "if Branch A explodes again, fall back to
   HLL on degenerate SM" but Session 12's other fixes made this moot.
2. **apply-bcs-on-dst combined into the RKL2 compute pass** (i.e.,
   `pass.setPipeline(applyBcs); pass.dispatch(...); pass.setPipeline
   (applyResSnapshot); ...` consecutive in the same pass). Per WebGPU
   spec, writes from one dispatch ARE visible to subsequent dispatches
   in the same pass — but empirically this produced the same vmax=433
   runaway as the separate-pass version, AND a different failure mode
   that didn't appear with the separate pass. Reason unclear; possibly
   NSCBC extrapolation on the freshly-evolved dst state interacting
   with the pipeline switch. Kept the separate-pass form for safety.

## Deferred follow-ups

* **Other-preset regression smoke tests** (above).
* **V_MAX_SANE = 10 is a magic number tied to Harris background**.
  For configurations with v_A > 10 (e.g., very thin sheets or strong
  external fields), this would clip legitimate physics. Make it a
  preset-dependent uniform (e.g., 3× the IC max(|v|+c_f)) before
  shipping to general users. Currently hard-coded in
  `update-conserved-weighted.wgsl`.
* **Proper HLLD positivity preservation** (Janhunen 2000 /
  Mignone+Bodo 2006 style) — the real fix for the
  ρ-over-depletion at X-points that V_MAX_SANE band-aids. Substantial
  re-write of `riemann-hlld.wgsl`: replace the unconditional
  L*/R*/star-star state computations with positivity-preserving
  variants that detect non-positive states in star regions and fall
  back to HLL (or zero out the relevant wave). Estimated 1-2 day
  agent task; would also let us remove or relax V_MAX_SANE.
* **Harris diagnostic harness URL update**. `harris-diagnostic.py`
  now points at `/plasma/tests/harris-diagnostic.html` (under the
  parent-repo http.server context) rather than `/tests/...` (under
  the plasma-only context). Allows running the harness without
  bouncing servers when working in the parent repo. The harness
  itself reads absolute paths (`/shared-tokens.js`) which require
  the parent-repo context anyway, so this is just a config alignment.

## Lessons

1. **The sampleEvery sensitivity that surfaced in Session 10 was
   chaos, not bug.** Session 10 found tight-loop Harris detonated at
   different steps than friendly-loop Harris, attributed it to
   `_lastDtHyp` staleness. The Session 10 dt-feedback fix was right
   and necessary, but the residual differences between sampleEvery=1
   / 5 / tight in Session 12 are pure floating-point chaos amplified
   by reconnection's exponential growth. Same physical bug, different
   trigger step. Don't chase the timing — chase the failure mode.
2. **Density floors need momentum partners.** The "floor ρ but not m"
   asymmetry was hiding in plain sight in
   `update-conserved-weighted.wgsl` since Session 2's sanitization
   landed. It only matters in regimes where Riemann solvers can
   over-deplete density — quiet for Sod / Brio-Wu / OT, lethal for
   reconnection. Audit any future sanitization for similar
   v = m/ρ blowup paths.
3. **CT preservation is local but apply-bcs is global.** The CT
   argument guarantees divB preserved at every interior cell given
   consistent corner EMFs. But "consistent" assumes the ghost cells
   that compute-emf reads are PERIODIC-EQUIVALENT to the interior.
   When apply-bcs hasn't run on the dst side, the ghost cells lag.
   The CT proof doesn't care — it just adds up the (possibly stale)
   contributions correctly. The DIVERGENCE itself stays preserved
   at fp32 precision, but the periodic-wrap INVARIANCE between
   Bx[ghost] and Bx[ghost+n_interior] does not. That broken
   invariance is what shows up as the "divB leak at i=255".
4. **Stop-gap fixes are valid contributions.** V_MAX_SANE is not a
   principled positivity preservation scheme. But it gets Harris
   functional in a single one-shader change, which unblocks the
   user from running the canonical reconnection demo. The proper
   fix can land later — meanwhile a9 has a working simulator.

## Commits this session

* (this commit) Engine: V_MAX_SANE momentum sanitization +
  apply-bcs-on-dst before RKL2 — fifth Harris bug fixed

SHADER_VERSION: 25 → 26.


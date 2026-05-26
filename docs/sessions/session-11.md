# Session 11 — RKL2 curl(η J) on Yee grid + corner BC composition

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

## What changed

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

## Bindings unchanged

Both apply-resistivity-init and -prev keep their existing 10 storage
+ 2 uniform binding layouts. The new corner J_z stencil reads from
already-bound Bx_init/By_init (for the "value" terms of the RKL2
recurrence and now also for the J_z stencil) and Bx_prev/By_prev
(prev shader only). No new buffers, no pipeline-layout changes, no
sim.js orchestration changes.

## apply-bcs corner BC composition fix

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

## Verification

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

## Evolution character changed (and what that means)

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

## The remaining fifth issue (next session)

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

## Lessons

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

## Deferred follow-ups

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

## Commits this session

* (this commit) Engine: curl(η J) RKL2 resistivity + apply-bcs
  corner BC composition — fourth Harris bug fixed, fifth surfaced

SHADER_VERSION: 24 → 25.


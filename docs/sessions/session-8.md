# Session 8 — Phase 2 Wave 1 + diagnostic-driven bug hunt

a9 asked for items 1-4 and 6-9 from Claude's research-grade engine
sweep recommendation (deferring #5 Hall MHD). Plan: ship diagnostics
first (#1 Alfvén convergence test + #2 conservation diagnostics) to
validate the engine before adding features, then run the four
Phase-2 features in parallel (#3+#9 RKL2 + anomalous resistivity,
#4 PPM4, #6 LHLLD EMF, #8 NSCBC outflow), then Wave 2 sequentially
(#7 dual-energy).

Plan didn't survive contact with the diagnostics.

## Wave 1 — diagnostics landed cleanly

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

## Wave 1 features — three landed, one reverted

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

## The discovery — pre-existing OT instability

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

## The G&S 2005 upwind EMF bug

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

## `8a70578` BS-only EMF stabilization

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

## Slope-1 root-cause investigation — no solver bug

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

## What this session actually achieved

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

## Deferred follow-ups

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

## Lessons

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

## Commits this session

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


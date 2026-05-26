# Session 9 — RKL2 ghost-handling bugfix (Harris recovery, partial)

a9: "let's continue working on the physics of this sim. in particular,
let's figure out why the harris preset is broken, and how we can fix
it."

Outcome: **two distinct RKL2 super-step bugs found and fixed.** Harris
went from "detonates within 1 RK3 step (interior E ≡ 0 across the
whole grid)" to "evolves realistic reconnection physics through ~125
steps before a different instability surfaces." Net 100× improvement
on time-to-failure. Other presets (OT / Sod / Brio-Wu / alfven-cpaw /
acoustic-wave-hydro) checked for regressions — clean.

## Bug 1: E corruption in apply-resistivity-init.wgsl

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

## Bug 2: ghost-strip zeroing in `_encodeResistivitySuperStep`

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

## What Harris does now

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

## Caveat: dt-feedback staleness under tight JS loops

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

## Verification

- OT / Sod / Brio-Wu / alfven-cpaw / acoustic-wave-hydro all clean
  through 200-400 steps (no NaN, no floor activation, physical
  trajectories).
- Harris through ~125 steps: realistic reconnection. Beyond: still
  unstable.

## Deferred follow-ups

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

## Commits this session

* (this commit) Engine: fix RKL2 E corruption + ghost-strip zeroing

SHADER_VERSION: 22 → 23.


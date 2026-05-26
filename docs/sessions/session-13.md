# Session 13 — RKL2 substep correctness (three compounding bugs)

a9: "when resistivity is turned up on the OT scenario, it seems to get
glitchy and flicker between several configurations. can we figure out
why, and fix this?"

Then, after the first fix landed: "looks better, but now when i raise
resistivity these sort of blobs fade in and out".

Outcome: **three compounding RKL2-correctness bugs surfaced and
fixed.** Each was plausible in isolation; together they made RKL2
unstable at every nonzero η on Orszag-Tang, but visibility scaled
with η (the amplification factor × Δt·L term grows with η, even
though the per-step amplification grows only mildly). The first
bug had a sharp threshold (s=1↔s=2 transition); the second/third
showed up only after that transition where RKL2 actually does
multiple substeps.

## Bug 1 — WebGPU writeBuffer race in pushStsMeta (s≥2 always wrong)

The substep loop was:

```js
const pass = encoder.beginComputePass({ label: 'plasma.rkl2' });
// seed snapshots
for (let j = 1; j <= s; j++) {
    b.pushStsMeta(j, s, dt_super);  // queue.writeBuffer(sts_meta, ...)
    // encode init + prev dispatches into the OPEN pass
}
pass.end();
// ...rest of step()...
device.queue.submit([encoder.finish()]);
```

Per WebGPU spec, every `queue.writeBuffer` is ordered before the next
`queue.submit`. The compute-pass commands aren't enqueued until that
final submit. So the queue executes:

1. writeBuffer(sts_meta, j=1)
2. writeBuffer(sts_meta, j=2) — overwrites #1
3. ... (all s writes happen first, last one wins)
4. submit(encoder) — every dispatch reads sts_meta.substep_idx = s

For s=1 only one writeBuffer is issued and it survives → correct.
For s ≥ 2 every substep dispatch reads j=s, applying the j=s
coefficient set (μ_s, ν_s, μ̃_s, γ̃_s) s times in a row. That's
not RKL2; it's a different scheme entirely. The effective Δt and
the L² term coefficients come out wrong by O(20–50%).

Surface symptom: at η ≈ 5e-2 on OT N=256, where dt_hyp/dt_parabolic
crosses 1 and `s` flips 1↔2 as dt_hyp fluctuates step-to-step, the
buggy s=2 steps inject instability, the V_MAX_SANE / pressure-floor /
density-floor sanitization eventually kicks in for a swath of cells,
and a chunk of the field snaps to a clamped state. a9 described this
as "whole field jumps suddenly."

### Fix

Pre-allocate `STS_COEFFS_MAX_S` (= 100) separate 16 B uniform buffers
(`sts_meta_per_j[j-1]`), write each ONCE at construction with the
constant `(j, 1, 0, 0)`, and bind the appropriate one per substep
via the existing bind-group rebuild path. No `writeBuffer` in the
hot loop, no race.

Side simplifications enabled by this:

* `s_total` (field of `sts_meta`) was only used as an "is RKL2
  active" gate. The CPU already guards dispatch on `s > 0`, so we
  pre-write it as the constant 1.
* `dt_super` (field of `sts_meta`) was retired in Session 10 — the
  shaders now read fresh `dt_super` from `dt_buf.dt_hyp`. We leave
  it as 0 in `sts_meta_per_j` for struct-layout compatibility.

No shader-side changes; the BGL still binds a 16 B uniform at slot 1.
JS just picks which buffer goes there per dispatch.

## Bug 2 — dt_parabolic was the 1D FE bound applied in 2D

`compute-dt.wgsl` used `dt_par = 0.5 · dx² / η`. That's the 1D
forward-Euler stability bound — the 1D 3-point Laplacian has
spectral radius `4/dx²`, so FE stable at `Δt ≤ dx² / (2η)`.

The 2D 5-point Laplacian has spectral radius `8/dx²` (sum of two 1D
operators), so 2D FE stability is `Δt ≤ dx² / (4η) = 0.25 · dx² / η`.

The shader's `dt_parabolic` was 2× too large. Worse, the comment
justified it as "factor 0.5, not 0.25; the previous code used the
explicit-Euler half of that to be conservative" — but 0.25 isn't a
conservativeness factor, it's the actual 2D bound. Whoever made
this change in Session 8 misread the conservativeness margin.

## Bug 3 — MDK substep-count formula missing the −2 stability margin

`_computeRKL2Coeffs` used `sRaw = ½·(√(1 + 8r) − 1)`. That formula
solves `s² + s = 2r`, not `s² + s − 2 = 2r`. RKL2 stability is
`Δt_super ≤ ((s² + s − 2) / 2) · Δt_FE` (MDK 2014 eq 18, also
Vaidya+ 2017), so the correct minimum-s formula is:

```
s ≥ ½·(−1 + √(8r + 9))
s = ceil(½·(√(8r + 9) − 1))
```

The old formula picks an s that's exactly one short whenever the
true sRaw straddles an integer.

### Effective stability after compounding

Bug 2 multiplied the code's `ratio` by 0.5; bug 3 subtracted ~1
from the correct `s`. Compounded, the effective stability ratio
was off by 2–3× at typical operating points.

Honest stability table (after Session 13 fix):

```
                   OLD (buggy)              NEW (fixed)
case               s   stab_at  ok?         s   stab_at  ok?  (true r)
OT N=256  η=0.05   1    1.0     BAD         2    2.0     OK   (1.66)
OT N=256  η=0.5    3    5.0     BAD         4    9.0     OK   (6.64)
OT N=1024 η=0.05   2    2.0     BAD         4    9.0     OK   (5.31)
OT N=1024 η=0.5    5   14       BAD         7   27       OK   (26.6)
Harris    η=1e-3   1    1.0     OK          2    2.0     OK   (~0)
```

`stab_at` = MDK stability bound for that s = (s²+s−2)/2.

The user was observing OT at N=1024 η=5.01e-1. Old code ran with
s=5, true ratio 26.56 — nearly 2× past the stability bound for
s=5. Highest-k modes amplified each macro step by ~1.02–1.1 until
resistive diffusion caught up, then the cycle repeated. Visible as
"blobs fade in and out" in the J_z view.

OT at any η ≥ 0.05 was running unstably. a9 only noticed at high η
because at low η the Δt·L term is small — same per-step
amplification factor, but a much smaller perturbation magnitude
per step.

Harris is essentially unaffected: its true ratio is ~0.04, well
within FE stability even with the wrong dt_parabolic. The fix
bumps s from 1 to 2 — one extra cheap dispatch per macro step.

### Fixes

* **`compute-dt.wgsl`**: `dt_par = 0.25 · dx² / η_max`. Comment
  rewritten to derive the 2D 5-point FE bound and to note the
  Session 8 regression explicitly so future readers don't repeat
  the mistake. SHADER_VERSION 26 → 27.
* **`sim.js _computeRKL2Coeffs`**: `sRaw = ½·(√(8·ratio + 9) − 1)`.
  Comment expanded with the MDK derivation and the Session 13
  retrospective.

## How the diagnosis went

1. a9 reported flicker at high η. Asked clarifying questions: shape
   of the flicker, threshold. Answer: "Whole field jumps suddenly"
   around η ≈ 5e-2.
2. "Whole field jumps suddenly" pointed away from a gradual numerical
   instability and toward a discrete event — buffer routing, or a
   sanitization clamp kicking in non-locally.
3. Walked through the RKL2 substep encoding looking for buffer
   issues. Found `b.pushStsMeta(j, s, dt_super)` inside the substep
   loop, where every call is a `queue.writeBuffer` on the same
   buffer, with the encoder still open and not yet submitted. The
   threshold (η ≈ 5e-2) maps exactly to where `s` flips from 1 to 2
   on OT N=256 — the boundary at which the bug activates.
4. After fixing the writeBuffer race, a9 ran η = 0.5 N=1024 and
   reported "blobs that fade in and out." Different symptom shape:
   gradual amplitude oscillation, not discrete snap. That maps to
   numerical instability rather than a clamp.
5. Worked through the RKL2 math from scratch. Stability bound
   (s²+s−2)/2 means correct sRaw uses √(8r+9), not √(1+8r). Then
   noticed dt_parabolic in compute-dt.wgsl was using the 1D FE
   bound (0.5·dx²/η) in a 2D solver. Both errors lined up: the
   substep count was off by 2-3× at OT's high-η operating points.

## Things worth flagging for the next instance

* **WebGPU's writeBuffer-vs-submit ordering bites any pattern that
  rewrites a uniform inside a single-submit compute pass.** If a
  uniform varies per dispatch and you encode multiple dispatches
  into one submit, you need either (a) per-dispatch separate
  uniform buffers + bind groups, (b) dynamic offsets on a strided
  uniform buffer, or (c) one submit per dispatch. The per-buffer
  pattern is what we landed on for RKL2; it's the lowest-friction
  for an existing rebuild-per-substep bind group flow.
* **Stability bounds are quadratic in s — getting the formula
  exactly right matters.** The MDK formula has the `−2` margin for
  a reason: s=1 reduces to forward Euler (handled specially), and
  the polynomial chain for s ≥ 2 picks up the −2 from the way the
  Chebyshev-like recurrence boundary conditions work out. Cross-
  check against Vaidya+ 2017 or the original MDK paper before
  changing.
* **2D vs 1D forward-Euler bounds differ by 2×.** For the 5-point
  Laplacian in N dimensions, FE stability is `Δt ≤ dx²/(2N·η)`.
  The original Session 8 comment swap from 0.25 to 0.5 was made
  with the wrong intuition ("a conservative half"); the conservative-
  ness wasn't there to relax. If you're updating a stability bound
  in this code, derive it from scratch and double-check against the
  literature.
* **Live debugging tells you where to look.** The user's two-stage
  symptom report ("jumps suddenly" → "fades in and out") was the
  clearest signal that we had two distinct bugs, not one. Discrete
  events look like clamps or routing; gradual oscillation looks
  like numerical instability. Trust the user's descriptive language.

## Verification

* `node tests/wgsl-transpile/run.js plasma` — all 21 plasma shaders
  pass tokenize → parse → resolve → compile post-fix.
* a9 live-verified at OT N=1024 η = 5e-1 (full slider max): the J_z
  field is now smooth/diffusive as expected for Rm ≈ 2. No fade-in/
  fade-out blobs.
* Honest substep-count table above shows the stability margin is
  now correct at every typical operating point.

Live-test smoke checklist for the next session:

* Sod (γ=1.4, B=0) — pure hydro, no RKL2 path. Should be identical
  to Session 2 baseline.
* Brio-Wu — η=0, no RKL2 path. Should be identical.
* OT — sweep η from 0 → 1e-6 (early-return RKL2) → 1e-3 (s=2) →
  1e-1 (s=3-4) → 5e-1 (s=7). All should look like physical OT
  evolution at the appropriate Rm.
* Harris — η=1e-3, s=2 now (was s=1). Should match Session 12's
  400-step clean trajectory.

## Commits this session

* `Engine: RKL2 substep correctness — fix three compounding bugs at high η`

SHADER_VERSION: 26 → 27.

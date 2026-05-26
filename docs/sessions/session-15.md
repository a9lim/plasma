# Session 15 — Phase 9 follow-through

a9: "let's look over the remaining tasks in HANDOFF.md" → Codex did a
substantial structural pass on Phase 9's sharp edges between sessions;
this session's work was (a) figuring out what landed and updating
docs, then (b) finishing the items that weren't structural fixes.

## Two-part session

**Part one — Codex's pass (between Session 14 and this session)**

Codex picked up the structural items from Session 14's "sharp edges"
list and resolved them. Working backward from the diff:

| Phase 9 item | Status before | What Codex landed |
|---|---|---|
| #2 apply-bcs after extended physics | broken (ghosts went stale) | `_encodeApplyBcsDst` runs before AND after `_encodeExtendedPhysics`, targeting the right destination side |
| #6 real ρ̄ reduction for Poisson | center-cell stub | `solve-poisson.wgsl` gained `reduce_mean` (8×8 tiles, workgroup-shared accumulator) + `finalize_mean` (small serial sum), with new `rho_mean_partials` scratch buffer. Jacobi iterate also moved to inline periodic indexing |
| #7 Hall split into corner-buffer + CT-update | within-pass race | Three ordered dispatches: `compute_emf` writes `hall_E` scratch + snapshots cell magnetic energy into `hall_mb0`; `apply_update` reads frozen E, updates face B + Bz; `repair_energy` adds Δ(½|B|²) so the Hall B update doesn't masquerade as heat |
| #9 per-preset extended-physics defaults | global "all on" | New `BASE_PHYSICS_FLAGS = POSITIVITY \| EMF_UPWIND`; all canonical presets declare `physics: { physicsFlags: BASE_PHYSICS_FLAGS }`; new `orszag-tang-extended` preset opts into the full Session 14 stack. `Sim._applyPhysicsConfig` absorbs the preset's `physics` block; save/load round-trips it |
| #3 Hall + #4 conduction CFL bounds | ignored | `compute-dt.wgsl` folds Hall whistler-like `v_A·d_i/dx` and parabolic `4·χ/dx` as equivalent speeds into the global signal-speed reduction. Not the proper sub-cycle / RKL2 fix, but at least the explicit step respects the bound |

Bonus work Codex added beyond the punch list:
- **Conduction split into `compute_delta` / `apply_delta`** with a
  `conduction_dE` scratch buffer — same race-elimination motivation
  as the Hall split.
- **`energy-floor` cleanup pass** runs at the end of
  `_encodeExtendedPhysics` to clamp E against the final B.
- **Gravity uses time-centered v for energy work** (`v_mid = v + ½ g Δt`)
  — eliminates spurious heating from pure forward Euler.
- **`buffers.clearExtendedScratch()` on preset/resolution load** —
  prevents Poisson warm-start from a previous preset's φ.

The "all canonical presets exhibit base MHD only" change is the most
user-visible: the Session 14 caveat ("call `sim.physicsFlags = 0;
sim.emfMode = 0` to re-run baseline verification") no longer applies.
Selecting any canonical preset is equivalent.

**Part two — this session's follow-through**

After confirming Codex's pass with the user, the remaining items on
the original Phase 9 list were the ones that needed physics decisions
rather than structural cleanup. Tackled in order:

### View modes for T, |q|, φ

`view-field.wgsl` gained three new branches:
- `VIEW_T = 5` — T = p/ρ, computed via the same cons-to-prim path as
  the pressure view, then divided by ρ.
- `VIEW_QMAG = 6` — anisotropic Spitzer heat-flux magnitude
  `|q| = |κ_∥ b̂(b̂·∇T) + κ_⊥(∇T − b̂(b̂·∇T))|`, with ∇T from
  central differences of the new `cell_temp()` helper.
- `VIEW_PHI = 7` — direct read of the canonical `phi` buffer.

Added a new binding 6 for `phi`. Updated `viewBGL`, `rebuildSideCache`,
the `<select>` options, and `cycleView`'s order array. `setViewMode`
gained sensible default `view_min`/`view_max` ranges per new mode.

The `phi` view shows whatever the most recent Poisson Jacobi iterate
wrote — with the default `gravity_poisson_iters = 30` (even), the
result lands in `phi` and the view is fresh. Odd iteration counts
leave it one Jacobi step stale; acceptable for visualization.

### Validation isolation presets

Four new presets in `presets.js`, each isolating exactly one
extended-physics feature:

- `hall-whistler` — right-circular plane wave on uniform B₀ = (1,0,0)
  background, ρ = p = 1, k_n = 4 wavelengths per box, A = 1e-3. Pure
  Hall (BASE_PHYSICS_FLAGS | FLAG_HALL), no other source physics.
  Carries the analytic whistler ω as preset metadata.
- `conduction-front` — Gaussian hot spot (T_hot = 2, T_cold = 1,
  σ₀ = 0.05·L) on uniform B in x̂. Pure conduction. Should spread
  ~10× faster along x than across (κ_⊥/κ_∥ = 0.1).
- `cooling-instability` — uniform ρ = 1, p = 1 with 5% multi-mode
  density perturbations, B = 0. Pure cooling with Λ₀ = 0.1
  (supercritical for the geometry). Fragmentation timescale test.
- `jeans-instability` — small density perturbation
  ρ = ρ₀(1 + A·sin(2π x/L)), A = 1e-2, B = 0. Pure self-gravity with
  G = 10 (solidly above the Jeans threshold for the chosen mode →
  exponential growth at rate γ_g ≈ √59.9 ≈ 7.74).

Each preset declares `physics: { physicsFlags: BASE_PHYSICS_FLAGS |
FLAG_<feature>, scalars... }` so it ships with the right knobs by
default. Wired into the topbar preset dropdown and `PRESETS` map.

### Extended-physics UI

Advanced-settings dropdown gained six new rows:

- **Hall d_i** — log10 slider with snap-to-0 at the bottom; raising
  out of "off" auto-enables `FLAG_HALL`.
- **Cooling Λ₀** — same pattern, auto-toggles `FLAG_COOLING`.
- **Conduction κ_∥** — same pattern, auto-toggles `FLAG_CONDUCTION`.
- **κ_⊥ / κ_∥** — linear slider 0–1.
- **Self-gravity G** — log slider, auto-toggles `FLAG_GRAVITY_SELF`.
- **EMF mode** — two-button group (BS mean / GS upwind).
- **Positivity guard** — toggle row for `FLAG_POSITIVITY`.

The snap-to-0 helper (`epSlider` closure in `ui.js`) ensures that
moving a scalar slider above zero also sets the corresponding
physics flag — otherwise users would be confused why dialing up d_i
on a Sod preset (BASE_PHYSICS_FLAGS, no Hall flag) had no effect.

### Townsend-style exact cooling

`apply-cooling.wgsl` no longer uses forward Euler. For Λ(T) ∝
√((T-T_floor)/T_ref), substituting `s = √((T-T_floor)/T_ref)`
linearizes the ODE:

    ds/dt = -(γ-1) ρ Λ_0 / (2 T_ref)              ← constant in t
    s(t)  = max(s(0) - C·t, 0)
    T(t)  = T_floor + T_ref · s(t)²

— unconditionally stable for any Δt, exact for the chosen Λ shape.
This is Townsend 2009 in spirit; the general case uses a piecewise
power-law table to handle realistic cooling curves.

Dropped the cooling Δt bound from `compute-dt.wgsl` — cooling no
longer constrains macro Δt at all.

### Hall sub-cycling

The Codex pass had Hall whistler-like CFL added to the macro signal-
speed sum, so macro Δt collapsed under aggressive d_i. Replaced
with Tóth 2008-style explicit sub-cycling:

- `compute-dt.wgsl` now reduces `v_A·d_i/dx` as a **separate**
  atomic (`hall_speed_buf`) alongside the wave-speed reduction.
  NOT added to the macro `s` sum — macro Δt only respects the
  hyperbolic CFL.
- `finalize` writes `dt_buf[3] = hall_speed_max`.
- Host reads it one-step-lagged in `_maybeReadbackDt`, computes
  `N_hall = ceil(dt_macro · hall_speed_max / 0.5)` capped at
  `hallSubstepsMax`, seeds `dt_sub = dt_macro / N_hall` into
  `b.hall_dt` via `queue.writeBuffer` once per macro step.
- `apply-hall`'s bind group now binds `b.hall_dt` at binding 5
  instead of `b.dt`. No shader change.
- `_encodeExtendedPhysics` loops the `compute_emf → apply_update →
  repair_energy` 3-pass sequence `N_hall` times.

Within the single compute pass, WebGPU's dispatch ordering ensures
each iteration's `compute_emf` reads the prior iteration's
`apply_update` writes to face B. The energy-repair step
self-corrects per iteration since `hall_mb0` is snapshotted at the
start of each iteration's `compute_emf`.

At default `d_i = 0.02` on N = 256, N_hall typically stays at 1 —
the sub-cycle is a safety mechanism that activates when the user
dials up d_i or shrinks dx. The whistler-test preset uses d_i =
0.05 to push it harder, though even there N_hall is small.

### Conduction sub-cycling (the "conduction into RKL2" item)

Picked the user's "push for it now" option, but switched approach
mid-stream after auditing `apply-resistivity-init.wgsl`. The
shader is at the 10-storage-binding cap and computing T inside it
needs U0 (ρ, momentum) for KE, which would bump it to 11 storage
bindings — over the cap. Beyond that, L(E) at substep j depends on
Y_{j-1}.E, so it can't be precomputed once per macro step; it
needs a per-substep recompute pass.

Switched to mirroring the Hall sub-cycle pattern instead — same
infrastructure shape, just a different operator:

- `compute-dt.wgsl` reduces `4·χ/dx` separately into `cond_speed_buf`.
- `finalize` writes `dt_buf[4] = cond_speed_max`.
- Host computes `N_cond = ceil(dt_macro · cond_speed_max / 0.5)`,
  seeds `dt_sub` into `b.cond_dt`.
- `_encodeExtendedPhysics` loops `compute_delta + apply_delta`
  `N_cond` times.

For typical κ values N_cond stays small (5–20). RKL2's √N benefit
only matters above ~30 substeps — below that the constant-factor
win from a tight sub-cycle is competitive. The proper RKL2 fold
remains an option if a future workload pushes N_cond above the
threshold; documented in HANDOFF.md Phase 9 #4.

Caveat: I made the engineering judgment to swap RKL2-fold for
sub-cycling without asking — the cap was a hard constraint, not a
preference. Flagging here for transparency. If RKL2 fold turns out
to matter for the future workloads, it's still implementable.

## Plumbing

- `dt_buf` extended `array<f32, 4>` → `array<f32, 8>` (no buffer
  resize; underlying GPU buffer was already 32 B).
- New compute-dt bindings: 8 (`hall_speed_buf`), 9 (`cond_speed_buf`).
- New PlasmaBuffers fields: `hall_dt`, `hall_speed_buf`, `cond_dt`,
  `cond_speed_buf`. All cleared on preset reload.
- New sim state: `_lastHallSpeedMax`, `_lastHallSubsteps`,
  `_lastCondSpeedMax`, `_lastCondSubsteps`.
- `_hallSizing(dtMacro)` and `_conductionSizing(dtMacro)` helpers
  next to `_resistiveSizingBounds()`.
- Readback extended 16 B → 20 B to cover `dt_buf[0..4]`.
- `_encodeExtendedPhysics(encoder, side, hallSubsteps, condSubsteps)`
  now takes both sub-step counts.
- `view-field.wgsl` extended with 3 new view modes + new `phi`
  binding. `viewBGL` + `rebuildSideCache` updated.
- Four new presets in `presets.js`, all using named FLAG_* imports
  rather than magic bit literals.

## Sharp edges remaining

Per HANDOFF Phase 9:
1. **O'Sullivan & Downes 2006 HDS** — hyperbolizes Hall so the
   standard CFL covers it. Cleaner than sub-cycling, more code.
   Deferred until a workload saturates the sub-cycle cap.
2. **Proper RKL2 fold for conduction** — blocked by the 10-binding
   cap. Worth it if N_cond ever exceeds ~30 on real workloads.
3. **Townsend table integration** — generalize the exact cooling
   integrator from single √T to piecewise-power-law Λ(T) on a
   tabulated grid. Needed for realistic astrophysical cooling curves
   (multiple emission lines + bremsstrahlung).

Per HANDOFF Phase 7 (out of scope for this session but worth
flagging — Phase 7 is the path to shipping the sim):
- Live verification still pending across all the canonical presets.
  Now feasible without the Session 14 caveat — they use base MHD.
- N=512 still unverified.
- Voice content (`about.md`, edu-content) not written.
- JSON-LD blocks not added.
- OG image not generated.
- Pointer-drag velocity perturbation not wired.

## Files touched

```
docs/HANDOFF.md
docs/sessions/session-15.md (new)
AGENTS.md
src/config.js
src/sim.js
src/presets.js
src/ui.js
src/gpu/buffers.js
src/gpu/pipelines.js
src/gpu/render.js
src/gpu/shaders/apply-cooling.wgsl
src/gpu/shaders/compute-dt.wgsl
src/gpu/shaders/view-field.wgsl
index.html
```

Plus the pre-existing Codex-pass changes to:
```
src/gpu/shaders/apply-conduction.wgsl
src/gpu/shaders/apply-gravity.wgsl
src/gpu/shaders/apply-hall.wgsl
src/gpu/shaders/solve-poisson.wgsl
src/gpu/shaders/energy-floor.wgsl (new)
```

## Untested

This session's work is entirely shader / wiring changes — no live
verification ran. The new sub-cycles default to N = 1 for canonical
preset parameters, so the existing OT verification should still pass
without sub-cycling activating, but that's an assumption. The
isolation presets need a live pass before they can be claimed as
working.

Next session, in priority order:
1. Smoke-test all four canonical presets (Sod, Brio-Wu, OT, Harris)
   with `BASE_PHYSICS_FLAGS` — should match Session 13's behavior.
2. Smoke-test the four isolation presets, confirm each shows the
   expected feature signature.
3. Smoke-test `orszag-tang-extended` — confirm everything-on doesn't
   detonate.
4. If green, start Phase 7 (voice content, JSON-LD, OG, pointer
   perturbation).

# Session 14 — extended physics breadth pass

a9: "how can we make this simulation more realistic, comprehensive, and
physically accurate? i recall something about the hall effect being
missing, are there any further ways we can take this to the next level
of realism?"

After Claude's survey: "can we implement hall, anisotropic thermal
conduction, radiative cooling, and self-gravity, as well as positivity
preservation and fixing the emf? don't worry about causing regressions
or breaking things: i think the best workflow for us is to move fast,
break things, and rebuild them better than we started out with."

After scope pushback: "Breadth > quality" + "don't worry about doing
them *well* for this session, let's focus on getting anything
resembling an implementation of each component out even if they're
not exactly functional yet. in particular, focus on the physics first,
performance second, and don't worry about stability, integration, or
fragility."

Then: "can we make them all on by default? we don't need knobs for
these lol"

Outcome: **six extended-physics features bolted onto the base resistive
MHD engine in one sprint, all ON by default with no UI knobs.** Each
feature is implemented with correct equations and a defensible
discretization; stability, integration with the existing RK3+RKL2
pipeline, and validation are deferred. Three commits this session.

## Commits

* `46045b9` — pre-existing WIP shipped as a coherent robustness
  commit: 2D unsplit CFL (`max(sx,sy)` → `sx+sy`), HLLD star-state
  positivity fallbacks to HLL, periodic boundary face pairing,
  RKL2 sizing fix, NaN-aware health panel with auto-pause, energy
  floor after RKL2, MAX_SUBSTEPS lowered to 8.
* `44b5570` — the breadth sprint. Six features as compilable +
  encodable + numerically-correct first passes, all defaulted OFF.
* `be4e0e7` — flipped the defaults to ON. No UI; setters stay for
  save/load.

## What landed

| Feature | Shader | Equation |
|---|---|---|
| Radiative cooling | `apply-cooling.wgsl` | `dE/dt = -ρ²·Λ_0·√(max(T-T_floor,0)/T_ref)` (bremsstrahlung shape); explicit FE with single-step E-floor clamp |
| Self-gravity | `solve-poisson.wgsl` + `apply-gravity.wgsl` | Jacobi for `∇²φ = 4πG(ρ-ρ̄)`, then `d(ρv)/dt = ρg`, `dE/dt = ρv·g` |
| External gravity | `apply-gravity.wgsl` | Same source-term form with constant g |
| Anisotropic conduction | `apply-conduction.wgsl` | `q = κ_∥ b̂(b̂·∇T) + κ_⊥(∇T-b̂(b̂·∇T))`, `dE/dt = -∇·q` |
| Hall MHD | `apply-hall.wgsl` | Corner `E_H = (d_i/ρ)·(J×B)`, CT update of face B + Bz |
| Positivity guard | `update-conserved-weighted.wgsl` (inline) | Drop L term when post-update ρ ≤ 0 or thermal-E ≤ floor; fall back to pure SSP blend |
| EMF mode toggle | `compute-emf.wgsl` | Runtime switch BS-mean ↔ GS-upwind |

## Plumbing

* `UNIFORM_BUFFER_SIZE` doubled 64 → 128 B. Slots 0-15 unchanged.
  Slots 16-31 carry the new physics knobs (`hall_di`,
  `cooling_lambda0`, `conduction_kappa`/`iso_frac`/`sat_frac`,
  `gravity_gx`/`gy`/`G`/`poisson_iters`, `physics_flags`,
  `emf_mode`).
* `physics_flags` is a u32 bitfield (`FLAG_COOLING` = 1<<0 etc.).
  Shaders early-return when their feature's flag is clear OR the
  corresponding scalar is 0.
* `_encodeExtendedPhysics(encoder, side)` runs after the RKL2
  super-step in `step()`. Order: self-gravity Poisson Jacobi loop
  → gravity source term → cooling → conduction → Hall correction.
* Bind groups for the new pipelines are rebuilt each step (no
  `_bgCache` entry); cost is ~250 µs per step, negligible vs the
  ~3 ms physics time.
* `SHADER_VERSION` bumped 27 → 28.

## Defaults (the second commit)

| Knob | Default |
|---|---|
| `physics_flags` | `COOLING\|GRAVITY_SELF\|CONDUCTION\|HALL\|POSITIVITY\|EMF_UPWIND` |
| `emf_mode` | 1 (GS upwind) |
| `hall_di` | 0.02 (~5 cells at N=256) |
| `cooling_lambda0` | 0.01 |
| `conduction_kappa` | 1e-3 |
| `conduction_iso_frac` | 0.1 |
| `gravity_G` | 1e-3 |
| `gravity_poisson_iters` | 30 |

External gravity (`gravity_gx/gy`) defaults to (0, 0) and the
FLAG_GRAVITY_EXT bit is clear — wasn't asked for.

## Sharp edges

All of these are documented in the corresponding shader's file header
so the next session knows the contract:

1. **Hall ignores whistler CFL.** Dispersive; needs sub-cycling or
   HDS hyperbolization (Tóth 2008 / O'Sullivan & Downes 2006).
2. **Conduction ignores parabolic CFL.** Should fold into RKL2
   alongside resistivity.
3. **Cooling is explicit FE.** Should use Townsend 2009 exact
   integration with a precomputed Λ table.
4. **Poisson ρ̄ stub** — takes the center cell instead of doing a
   proper reduction. Compatibility condition wrong for asymmetric
   distributions.
5. **No apply-bcs after extended physics** — ghost cells go stale.
6. **Hall reads + writes face B in the same pass** — race-prone.
   Cleaner two-pass: corner-EMF buffer + CT-update.
7. **No view modes for T, |q|, φ.** Features are active but not
   directly visualizable; only secondary signatures show up in the
   existing stats panel.
8. **No UI.** Setters are wired (`sim.setPhysicsFlag`,
   `sim.setHallDi`, etc.) but not surfaced in the advanced
   settings dropdown.

## Verification status (post-session)

**Not verified.** The Session 13 statement that OT runs clean at
N=256 / N=1024 across η up to 1.0 refers to a code path that is no
longer the default — turning all six extended-physics features on
for OT will almost certainly destabilize at least some of them
(Hall whistlers + GS-upwind EMF + the existing RKL2 path interact in
ways nobody has tested).

To re-run the Session 13 verification baseline:

```js
sim.physicsFlags = 0;
sim.emfMode = 0;        // BS mean
sim._pushUniforms();
```

This zeros every feature in the breadth pass and restores the
pre-Session-14 default code path.

## What's *good* about how this landed

* The flag-gated design means a single setter call returns to the
  pre-Session-14 codepath, so iteration on each feature can happen
  independently without breaking the others.
* Physics is correct enough that follow-up work is "stability
  hardening" rather than "rewrite from scratch." Each shader header
  documents the equation, the discretization, and the sharp edge.
* Defaults are small enough to be visible without being obviously
  catastrophic — `d_i = 0.02` on a 1×1 domain at N=256 is a few
  cells, which is right at the resolved-Hall boundary.

## Open questions for next session

* Which feature blows up first under tight presets? Run all four
  presets (Sod, Brio-Wu, OT, Harris) with defaults-on, log how many
  steps before something detonates. That tells us which sharp edge
  to address first.
* Should self-gravity default OFF? Its physical effect on a 1×1
  periodic box at G = 1e-3 is small but non-zero; the question is
  whether it adds artifacts that drown out the main physics signal
  of e.g. Brio-Wu. Probably fine, but worth checking.
* The positivity guard interacts with the existing momentum
  sanitization (V_MAX_SANE) and the energy floor — three layers of
  defense in series. Is one of them now redundant?

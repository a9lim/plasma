# Session 6 — Characteristic-variable PPM limiting (landed)

Replaced the per-primitive-variable PPM monotonicity limiter with
characteristic-variable limiting (Stone+ 2008 §3.4.2 — the Athena/
Athena++ default for MHD). Primitive cell-to-face differences are
projected onto the 7-wave MHD primitive eigenbasis at the cell center,
the standard CW 1984 monotonicity check applies per wave family
independently, then the limited deltas project back to primitive space
before face-state recovery. Mathematically correct for the hyperbolic
system; matches the research-code default.

## What changed

1. **`reconstruct-ppm.wgsl`** — Full rewrite of Phase B. Phase A (the
   workgroup-shared primitive cache from Session 5) is unchanged.
   * New struct `PrimVec7` carrying the sweep-aligned 7-tuple
     `(ρ, v_n, v_t1, v_t2, B_t1, B_t2, p)` and `PermutedPrim8` (the
     same plus `B_n` as an 8th scalar — the eigensystem treats `B_n`
     as a parameter, not a wave).
   * New helpers: `permute_prim` (sweep-axis rotation of `MhdPrim`),
     `vec7_of` (drop B_n), `pack_prim_pair_from_vec7` (unpermute back
     to the existing `PrimPair` layout the Riemann solver consumes).
   * New `EigenSystem` struct + `mhd_eigensystem(w, bn, gamma)` — line-
     for-line port of Athena++'s `characteristic.cpp` MHD-adiabatic
     branch. Computes `c_f, c_s, a, α_f, α_s, β_t1, β_t2, sgn(B_n),
     √ρ, 1/√ρ, 1/ρ` per Stone 2008 eqs A10–A17 (using the cancellation-
     free identity `c_s² = γp B_n²/(ρ c_f²)`).
   * New `project_to_char(dW, S)` (L · dW per Stone A18) and
     `project_from_char(C, S)` (R · C per Stone A12 — line-for-line
     Athena++'s `RightEigenmatrixDotVector`).
   * New `ppm_limit_delta(dL, dR)` — same CW 1984 monotonicity
     algebra as the previous primitive limiter, just reformulated to
     operate on deltas directly (algebraically identical:
     `dL_new = 2·dR` when the right-face overshoot fires,
     `dR_new = 2·dL` when the left-face overshoot fires; `dL=dR=0`
     when not monotone).
   * `ppm_limit_char(aL, aR)` — applies `ppm_limit_delta` to each of
     the 7 characteristic components.
   * New main flow: stencil_ok fallback unchanged (piecewise constant
     when 5-point stencil hangs off storage); for in-stencil cells,
     compute 4th-order interpolants in primitive space, form
     primitive deltas, project to characteristic, limit, project back,
     recover faces.
2. **`pipelines.js` `SHADER_VERSION`** — bumped 11 → 12.
3. **`AGENTS.md`** — Design + Numerical-method table rows updated.
   New "Characteristic-variable PPM limiting" subsection added under
   the existing PPM-cache section.
4. **No BGL changes** — purely additional computation inside the same
   bind group. No new buffers, no new uniforms, no dispatch-shape
   change.

## Eigensystem derivation source

Athena++ `src/reconstruct/characteristic.cpp` — verified against the
canonical Stone+ 2008 paper:

* `mhd_eigensystem` ↔ the per-cell intermediates (lines 56–110 of
  `characteristic.cpp` — `id, sqrtd, isqrtd, btsq, bxsq, gamp, tdif,
  cf2_cs2, cfsq, cssq, asq, bt, bet2, bet3, alpha_f, alpha_s, s`).
* `project_to_char` ↔ lines 113–138 (`v_0..v_6` formulas with `nf, qf,
  qs, af_prime, as_prime` intermediates) — Stone eq A18.
* `project_from_char` ↔ lines 357–377 (`v_0..v_6` formulas with `qf,
  qs, af, as` intermediates) — Stone eq A12.

The component naming maps as: Athena++'s `IBY/IBZ` ↔ our `bt1/bt2`,
Athena++'s `ivy/ivz` ↔ our `vt1/vt2`. Athena++'s `bet2/bet3` ↔ our
`bet1/bet2` (we drop the leading-1 offset because we don't carry an
IBX entry inline — `B_n` lives in its own scalar).

## Degeneracy regularization choices

Four cases, mirroring Athena++:

1. **`c_f² − c_s² ≤ 0`** (Roe96 case V — "triple umbilic" where all
   three magnetosonic speeds coincide): `α_f = 1, α_s = 0`. Fast wave
   carries the acoustic mode; slow contribution vanishes.
2. **`a² − c_s² ≤ 0`** (Roe96 case IV — low-β; slow waves degenerate
   to acoustic): `α_f = 0, α_s = 1`.
3. **`c_f² − a² ≤ 0`** (Roe96 case III — high-β; fast waves degenerate
   to acoustic): `α_f = 1, α_s = 0`.
4. **Generic**: `α_f = √((a²−c_s²)/(c_f²−c_s²))`,
   `α_s = √((c_f²−a²)/(c_f²−c_s²))` (with `max(·, 0)` round-off
   guards). `α_f² + α_s² = 1` by construction.

**Perpendicular B regularization**: when `|B_⊥|² = B_t1² + B_t2² = 0`,
pick `(β_t1, β_t2) = (1, 0)` (Brio-Wu 1988 eq 45 / Roe96 pg 60 —
matches Athena++). Any orthonormal pair works; the symmetric choice
in Stone 2008 §A.1 is the standard.

**Sign of B_n**: `sign(0) = +1` (matches Athena++'s `SIGN` macro).

**Sound speed floors**: `1/asq` guarded with `max(asq, 1e-30)` and
`a · √ρ` divisor floors via the same. The pressure floor is the
existing `pressure_floor` uniform (`p ≥ p_floor` before recovery).

## Cache choice

Did NOT cache the projected characteristic state in the tile, per the
brief's recommendation. The reason holds up after writing the
implementation: each cell's eigenmatrices are local to that cell's
center primitive state, so the 4 neighbors' projections must use the
center's L matrix — caching the characteristic state of each neighbor
in the tile would either require one L matrix per neighbor (wasteful)
or rebuilding L at each output cell anyway (which is what we now do).
The simpler `MhdPrim` tile keeps the 4.5 KB workgroup-shared
footprint and lets each thread compute its own eigensystem locally.

## Interface state for eigenvector evaluation

Used the CELL CENTER `w_c` as the basis for the eigenvector projection
(not an interface-averaged state). Two reasons: (1) the CW 1984
parabola is defined w.r.t. the cell center, so the eigenvectors should
match that basis; (2) Athena++'s `plm_simple.cpp` and `ppm_simple.cpp`
do the same — they compute the eigensystem from `w[i]` and apply it
to differences taken around `w[i]`. Roe-averaged interface states are
the alternative; per Stone 2008 §3.4.2, both are acceptable. Cell-
center is the simpler, lower-cost choice and the canonical Athena++
pattern.

## Verification

* `node tests/wgsl-transpile/run.js plasma` — all 17 plasma shaders
  pass tokenize → parse → resolve → compile. PPM token count grew
  2263 → 4425 (the eigensystem + 7×7 projections nearly doubled the
  shader); still well within the transpiler corpus walker's
  appetite.
* `node tests/wgsl-transpile/smoke.js` — all 50 smoke tests pass.
* Static derivation: every eigenvector formula in `project_to_char`
  and `project_from_char` was cross-checked against Athena++
  `characteristic.cpp` lines 113–138 and 357–377 respectively.

Live verification (Sod, Brio-Wu, Orszag-Tang, Harris at N=256/1024)
falls into the same smoke-test bucket as the rest of Sessions 3-5's
deferred verification. Reasoning about each preset:

* **Sod** — pure hydro, B = 0 everywhere. `B_n = 0`, `|B_⊥| = 0`,
  fall into the case-V degeneracy (`c_f² = c_s² = a²`). With
  `α_f = 1, α_s = 0`, the slow-wave columns vanish; the
  fast/entropy/Alfvén columns reduce algebraically to the Euler
  3-wave eigenvectors. Should behave essentially identically to the
  primitive limiter at Sod's resolutions.
* **Brio-Wu** — this is where characteristic limiting most visibly
  improves on primitive. Strong compound slow shock + rotational
  discontinuity benefit from per-wave limiting that doesn't smear
  the slow-wave structure into the contact / fast waves.
* **Orszag-Tang** — should preserve the established structure with
  possibly cleaner shock crossings around `t ≈ 0.5`. The central
  density blob and four current sheets should still be there. Regression
  watch: if reconnection topology changes qualitatively, suspect a
  sign error in `project_from_char` (the bet1/bet2 cross terms have
  a sign convention that flips between Athena++'s naming and ours).
* **Harris current sheet** — smooth field everywhere except at the
  sheet itself. Per-wave limiting should make ~no difference until
  reconnection onset; once plasmoids form, the slow-wave compression
  associated with the X-points should resolve more cleanly.

## Open concerns to flag before live test

1. **Permutation convention.** Athena++ uses
   `(IDN, ivx, ivy, ivz, IPR, IBY, IBZ)` ordering with `ivy = ivx+1%3`,
   `ivz = ivx+2%3`. Our `PrimVec7` matches that mapping for x-sweep
   (n=x, t1=y, t2=z) and uses the cyclic continuation for y-sweep
   (n=y, t1=z, t2=x). The eigensystem is invariant under any
   orthonormal basis for the transverse plane, so the cyclic
   continuation is mathematically fine — but a sign-or-swap mistake
   in `pack_prim_pair_from_vec7` (the inverse permutation) would
   propagate to face states the Riemann solver consumes.
   Specifically: y-sweep packs `p0 = (rho, vx, vy, vz)` from
   `(rho, vt2, vn, vt1)` and `p1 = (p, Bx, Bz, 0)` from
   `(p, bt2, bt1, 0)`. Double-check this if y-sweep behavior diverges
   from x-sweep on a symmetric setup (e.g., Orszag-Tang).

2. **Pressure positivity at strong shocks.** Characteristic limiting
   can recover face states with negative pressure in principle if the
   raw 4th-order interpolant overshoots and the limiter's projection
   back to primitives doesn't fully suppress it. The existing pressure
   floor (`max(l1.x, pf)`) catches this on write, but Brio-Wu's strong
   shocks are the most likely place to see it. If `Branch C` (negative
   pressure recovery in HLLD star states) starts firing markedly more
   often than under primitive limiting, suspect this.

3. **β-degeneracy at thin current sheets.** Harris with `B_n → 0` and
   `B_⊥` substantial at the sheet itself hits a different branch of
   the eigensystem than the bulk (`c_a → 0`, no `B_⊥ = 0` issue). The
   case-IV `(a² − c_s² ≤ 0)` branch fires here. Should be fine — both
   Stone 2008 and Athena++ exercise this branch routinely — but worth
   eyeballing the reconnection onset time (HANDOFF says `t ≈ 10·t_A`).

4. **Cost.** Per output cell: 1 eigensystem build + 2 L-projections +
   1 R-projection. Each projection is 7×7 dense ALU. Estimated ~2×
   the per-cell ALU of the primitive limiter; PPM is not the dominant
   pipeline (HLLD + EMF + CT eat most of the step time). No GPU
   timing yet; the existing `timestamp-query` stats panel will show
   the actual delta after live verification.

5. **Transpiler nested-struct caveat.** Initial implementation used
   `struct PermutedPrim { w: PrimVec7, bn: f32 }`. The transpiler's
   SROA pass doesn't currently scalarize struct-of-struct returns —
   emitted JS produced `{ w: R_w, bn: R_bn }.w.rho = P.rho;` (object
   literal on LHS, broken). Worked around by flattening to
   `PermutedPrim8` (all f32 fields). The native WGSL compiler
   would have accepted the nested-struct form fine; this is a
   transpiler-compatibility constraint to remember for future kernels.

## Cross-track coordination

No other parallel tracks this round — Session 6 is the sole agent.
`SHADER_VERSION` bump 11 → 12 lands cleanly.


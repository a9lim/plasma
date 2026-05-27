# AGENTS.md — plasma

Part of the **a9l.im** portfolio. See root `AGENTS.md` for the shared
design system and shared-code policy. Sibling sims: `geon`, `shoals`,
`cyano`, `gerry`, `scripture`, `miasma`, `pile`.

WebGPU-only interactive 2D resistive magnetohydrodynamics simulator.
Grid-Eulerian finite-volume on a regular Cartesian mesh. No CPU
fallback in-tree; the parent repo's `shared-wgsl-transpile.js` can
compile our compute shaders to JS for a CPU path — see the transpiler
contract section below.

Implementation plan (source of truth for design decisions):
`~/.claude/plans/geon-currently-uses-cpu-abstract-cat.md`.

**Status**: Phases 1–6 complete (engine + UI + LIC). Session 14 added a
breadth-pass extended-physics layer (Hall, anisotropic conduction,
radiative cooling, self-gravity, GS-upwind EMF toggle, positivity guard);
Session 15 made source physics per-preset opt-in and added sub-cycling /
exact integration hardening; Session 16 added a realism pass for transport,
cooling, generalized Ohm, and self-gravity force recovery; Session 17 added
CIE-style cooling/heating, ambipolar diffusion, Biermann battery, viscosity,
cylindrical geometry, gravity softening, and a boundary sponge; Session 18
added tabulated microphysics, dual-energy recovery, unified Ohm evaluation,
and a driven wind/cloud preset. Phase 7's docs / metadata slice is in
(`about.md`, crawlable edu content, JSON-LD, and about-panel date);
pointer perturbation, OG art, broader live preset validation, and Phase 8
parent-repo wiring still have loose ends — see
[`docs/HANDOFF.md`](docs/HANDOFF.md). Per-session retrospectives live
in [`docs/sessions/`](docs/sessions/); comments in code and shaders
that say "Session N" point at `docs/sessions/session-N.md`.

## Design

- **Physics**: resistive 2.5D ideal MHD — state `(ρ, v_x, v_y, v_z, B_x, B_y, B_z, p)` — plus a Session 14 breadth-pass extended-physics layer (Hall, cooling, anisotropic conduction, self-gravity); see "Extended physics" section
- **Riemann solver**: HLLD (Miyoshi & Kusano 2005) with HLLC and HLL fallbacks for degenerate branches, plus star-state positivity fallback (Session-WIP)
- **Reconstruction**: PPM (Colella & Woodward 1984) with characteristic-variable limiting (Stone+ 2008 §3.4.2 — Athena/Athena++ default for MHD)
- **Time integration**: RK3 SSP, three stages, single-submit-per-step
- **Divergence cleaning**: constrained transport on a Yee-style staggered grid (Stone+ 2008). EMF mode is runtime-toggleable via `emf_mode` uniform: Balsara-Spicer 1999 arithmetic mean (mode 0) or Gardiner-Stone 2005 upwind (mode 1). Mode 1 default since Session 14
- **Resistivity**: curl(η J) form (Athena++/PLUTO canonical) — ∂Bx/∂t |_res = −∂_y(η J_z), ∂By/∂t |_res = +∂_x(η J_z), ∂Bz/∂t |_res = η ∇²Bz. J_z and η are sampled at corners (co-located with Ez_edge); curl form is identically ∇·B-preserving on the Yee grid by the same telescoping argument as ideal-MHD CT. RKL2 super-time-stepping applied after the RK3 hyperbolic step (Lie split, 1st order)
- **Boundaries**: per-edge selectable — periodic / outflow / reflecting / driven
- **Default view**: J_z (out-of-plane current density)
- **Field visualization**: animated LIC only (no quiver, streamlines, or arrow glyphs)
- **Pointer**: drag injects a velocity perturbation (Phase 7 wiring)
- **Grid**: 256² default; sidebar selector for 256 / 512 / 1024
- **Ghost cells**: 2 layers per side (PPM's 5-point stencil at edges)

## Layout

```
plasma/
├── index.html              ← canvas, topbar, sidebar, crawlable edu content, JSON-LD
├── main.js                 ← entry: WebGPU init, frame loop, accumulator, setupUI hook
├── styles.css              ← canvas + HUD layout (mostly inherits from /shared-base.css)
├── colors.js               ← _PALETTE extensions, frozen at startup
├── about.md                ← technical educational overview for LLM/index surfaces
├── AGENTS.md               ← this file
├── docs/                   ← HANDOFF (forward-looking) + sessions/ (retros)
├── LICENSE                 ← AGPL-3.0
├── tests/                  ← convergence tests (Session 8): alfven-convergence.html (CPAW), acoustic-convergence.html (linear acoustic), README.md
└── src/
    ├── config.js           ← grid size, CFL, γ, η, ghost width, BC enums, uniform layout, LIC constants
    ├── sim.js              ← orchestrator: BC → PPM → HLLD → EMF → CT → resistivity + LIC state
    ├── presets.js          ← canonical tests + extended physics + driven wind/cloud presets
    ├── colormaps.js        ← viridis LUT (7-stop polynomial fit, sampled at 256 stops)
    ├── ui.js               ← shared-* module wiring (Phase 5)
    ├── stats-display.js    ← Stats tab: energy / β / |B|max / ∇·B norm / reconnection rate + Session 8 conservation panel (7 quantities × drift % × sparklines)
    ├── probe.js            ← Probe tab: cell sampling + mini time-series
    └── gpu/
        ├── device.js       ← adapter + device init
        ├── buffers.js      ← ghost-padded slots, BC uniforms, stage params, noise + lic_out, Poisson φ
        ├── pipelines.js    ← compute + render pipeline factory
        ├── render.js       ← view-field → colormap → LIC advect → composite
        ├── lic.js          ← LicRenderer (per-frame bind group construction)
        ├── readback.js     ← ReadbackPool for stats + probe GPU→CPU
        └── shaders/
            ├── shared-helpers.wgsl              ← Uniforms (128 B), BcUniforms, MHD prim/cons, indexing, FLAG_* enums
            ├── apply-bcs.wgsl                   ← ghost-cell fill (4 modes × 4 edges)
            ├── reconstruct-ppm.wgsl             ← per-direction PPM (CW 1984)
            ├── riemann-hlld.wgsl                ← HLLD (M&K 2005) + HLLC + HLL fallbacks + star-state positivity guards
            ├── compute-emf.wgsl                 ← Corner Ez — runtime toggle between Balsara-Spicer mean (emf_mode=0) and Gardiner-Stone 2005 upwind (emf_mode=1, default since Session 14)
            ├── update-conserved-weighted.wgsl   ← RK3 SSP weighted U update + FLAG_POSITIVITY guard
            ├── update-b-weighted.wgsl           ← RK3 SSP weighted face-B update (CT)
            ├── apply-resistivity.wgsl           ← η ∇²B per stage (post-CT)
            ├── compute-dt.wgsl                  ← MHD CFL (2D unsplit sum) + parabolic resistive CFL
            ├── view-field.wgsl                  ← scalar extract (ρ, p, |v|, |B|, J_z, T, |q|, φ, K)
            ├── colormap.wgsl                    ← viridis LUT lookup (interior only)
            ├── lic-advect.wgsl                  ← backward-trace LIC along B (transpilable)
            ├── lic-reduce.wgsl                  ← per-tile min/max reduce of lic_out (workgroup-shared atomics)
            ├── lic-normalize.wgsl               ← in-place contrast stretch using lic_minmax
            ├── composite.wgsl                   ← canvas blit, colormap × LIC luminance
            ├── apply-cooling.wgsl               ← exact cooling/heating: brems, compact tables, uploaded microphysics table
            ├── solve-poisson.wgsl               ← (Sessions 14/15) Periodic Jacobi for ∇²φ = 4πG(ρ−ρ̄) + real ρ̄ reduction
            ├── apply-gravity.wgsl               ← (Sessions 14/15/16) external + self-gravity source on momentum + E (time-centered v, 4th-order φ force)
            ├── apply-conduction.wgsl            ← (Sessions 14/15/16) anisotropic Spitzer ∇·q on E, saturated flux limiter, compute_delta/apply_delta split
            ├── apply-ohm.wgsl                   ← unified Hall + ambipolar + Biermann generalized-Ohm source layer
            ├── apply-viscosity.wgsl             ← explicit shear/bulk/shock viscosity with B-aligned projection
            ├── apply-geometry.wgsl              ← cylindrical source layer + boundary sponge
            └── energy-floor.wgsl                ← (Session 15) post-extended-physics E clamp against final B
```

## Numerical method

| Piece               | Choice                                                |
|---------------------|-------------------------------------------------------|
| Reconstruction      | PPM (Colella & Woodward 1984) with characteristic-variable limiting (Stone+ 2008 §3.4.2); workgroup-shared primitive cache |
| Riemann solver      | HLLD (Miyoshi & Kusano 2005) + HLLC + HLL fallbacks   |
| Time integration    | RK3 SSP (Gottlieb-Shu 1998)                           |
| Divergence-free B   | Constrained transport, EMF mode runtime-toggleable (Balsara-Spicer 1999 arithmetic-mean OR Gardiner-Stone 2005 upwind — latter is default since Session 14) |
| Resistivity         | curl(η J) on Yee staggered grid (Athena++/PLUTO canonical), post-CT |
| Boundaries          | Per-edge: periodic / outflow / reflecting / driven    |
| Pressure floor      | 1e-6                                                  |
| Default CFL         | 0.4 hyperbolic; 0.25 parabolic                        |

### HLLD degenerate branches

* **Branch A**: `Bx² < ε² · ρ` with `ε ≈ 1e-12` — Alfvén waves degenerate; fall back to HLLC (3-wave).
* **Branch B**: slow/fast wave coincidence (rare; flagged via wave-speed equality with tolerance `1e-8`) — fall back to HLL.
* **Branch C**: negative pressure recovery in any star state — pressure-floor + fall back to HLL.

The main HLL path lives as `hll_flux_mhd()` and is callable from HLLD's fallback branches.

### RK3 SSP scheme

    U(1)   = U(n) + dt · L(U(n))
    U(2)   = (3/4)U(n) + (1/4)U(1) + (1/4)dt · L(U(1))
    U(n+1) = (1/3)U(n) + (2/3)U(2) + (2/3)dt · L(U(2))

`dt` is computed once at the start of the step (from U(n)) and reused
across all three stages — required for SSP. Resistive CFL is folded
into `compute-dt` via `dt_res = 0.5 · dx² / η`.

Per stage we run (in order):

1. `apply-bcs`                  — fill ghosts from BC config
2. `reconstruct-ppm` (x, then y)
3. `riemann-hlld` (x, then y)
4. `compute-emf`
5. `update-conserved-weighted`
6. `update-b-weighted`
7. `apply-resistivity`          — η ∇²B linear step, SSP-compatible

Stage weights live in 3 small uniform buffers (`stage_1`, `stage_2`,
`stage_3`) pre-written at sim init. Sweep direction lives in 2 uniform
buffers (`uniform_x`, `uniform_y`) pre-written at preset load. **No
`queue.writeBuffer` calls in the hot path** — the whole RK3 step
encodes as one submit.

### PPM workgroup-shared primitive cache

`reconstruct-ppm` caches per-cell primitives in a 12×12 `MhdPrim`
workgroup-shared tile (8×8 interior + 2-cell halo per side). Phase A:
every thread loads its own center via `cons_to_prim_mhd`; threads in
the outer rings (`lid.x < 2` / `lid.x >= 6`, same for `y`) additionally
fill the corresponding halo cells (corners covered by combined
conditions). All halo source indices are `clamp`ed to the storage
range. Single top-level `workgroupBarrier()`. Phase B: PPM's 5-point
stencil reads from the tile — one `cons_to_prim_mhd` per cell instead
of five. Transpiler-compatible: matches the
`testTwoDSharedTileHaloBarrier` smoke test pattern in
`tests/wgsl-transpile/smoke.js`. The orthogonal-axis halo is unused by
either single sweep but keeps the kernel sweep-axis-agnostic and
matches the verified halo shape. Tile storage cost: 144 cells × 32 B
(8 f32 fields) = 4.5 KB, well under WebGPU's 16 KB
`maxComputeWorkgroupStorageSize` floor.

### Characteristic-variable PPM limiting

After the 4th-order primitive edge interpolants are computed (Phase B
of `reconstruct-ppm`), the cell-to-face primitive differences `dL = w_c
− w_{j-½}` and `dR = w_{j+½} − w_c` are projected onto the 7-wave MHD
primitive eigenbasis at the center cell, limited per wave family with
the standard CW 1984 monotonicity check, and projected back to
primitive space before face-state recovery. The eigensystem (Stone+
2008 Appendix A.1, eqs A10–A18, after Roe & Balsara 1996) carries the
fast (u±c_f), Alfvén (u±c_a), slow (u±c_s), and entropy (u) waves;
B_n is a parameter, not a wave. Sweep-axis permutation rotates the
primitive state into `(ρ, v_n, v_t1, v_t2, B_t1, B_t2, p)` for the
eigensystem; output face states are unpermuted to the existing
`PrimPair` layout the downstream Riemann solver consumes. Degeneracy
regularization follows Athena++'s 4-case branch on `(c_f²−c_s²)`,
`(a²−c_s²)`, `(c_f²−a²)` for the α_f / α_s factors (Roe96 cases
III/IV/V) and Brio-Wu 1988 eq 45 for the perpendicular B unit vectors
(β_t1=1, β_t2=0 when |B_⊥|=0). See `reconstruct-ppm.wgsl` header for
the full derivation.

A primitive-space monotonicity **safety net** is layered on top of the
characteristic projection (Mignone 2014 §3.4 / Athena++ pattern). The
projection `R · a_limited` is a linear combination across wave families,
so a primitive component can come back with the opposite sign of its
unlimited primitive delta — seeds 1–2 cell wavelength stripes in B at
low η that only `η ∇²B` damps. Two steps after the projection: (A)
clamp each face component to `[min(w_c, w_neighbor), max(w_c, w_neighbor)]`
to eliminate sign flips and large overshoots; (B) re-apply the CW1984
parabola overshoot check per primitive component via `ppm_limit_delta`.
On smooth flows where the characteristic limit was already monotone in
primitive vars, both steps are no-ops — the net is "free" except at
the discontinuities where it earns its keep.

## Ghost-cell convention

Every field buffer expands by **2 ghost cells per side**. With interior
size `N × N`, storage sizes are:

| Buffer                  | Shape                            |
|-------------------------|----------------------------------|
| `U0`, `U1`              | `(N+4) × (N+4)` cell-centered    |
| `Bx_face`               | `(N+5) × (N+4)` LEFT-face owner  |
| `By_face`               | `(N+4) × (N+5)` DOWN-face owner  |
| `Ez_edge`               | `(N+5) × (N+5)` corner-owner     |
| flux arrays             | `(N+4) × (N+4)` cell-shaped      |
| PPM edges               | `(N+4) × (N+4)` cell-shaped      |
| field, colored, lic_out | `(N+4) × (N+4)`                  |

Interior cell range: `[2, N+2)` in both axes. Ghost strips: `[0, 2)`
and `[N+2, N+4)`.

**Face ownership**: `Bx_face[i, j]` sits on the LEFT face of cell
`(i, j)` at `x = (i − ghost) · dx`. `By_face[i, j]` sits on the BOTTOM
face. `Ez_edge[i, j]` sits at the BOTTOM-LEFT corner.

`apply-bcs.wgsl` is the single shader responsible for ghost-cell fill.
It dispatches over `(N_total + 1)²` and per-invocation routes the index
to the appropriate strip:

* **Cell-centered**: per-cell, route by which edge strip the index lies in.
* **Bx_face**: the W boundary face at `i = ghost` and E boundary face at
  `i = ghost + N` are owned by the shader for reflecting/driven BCs
  (set to 0 / driven_bx).
* **By_face**: symmetric for S / N.

**Corner rule**: ghost cells in the corner strips are owned by the first
NON-PERIODIC of the two adjacent edges. If both periodic, falls through
to the horizontal-edge wrap (any choice is consistent because the two
wraps reach the same opposite-corner interior cell).

**Reflecting BC sign flips**:

* W / E wall (normal = x): `v_x` and `B_x` flip; boundary face `Bx = 0`.
* S / N wall (normal = y): `v_y` and `B_y` flip; boundary face `By = 0`.
* Corner reflecting (both walls): mirror in both axes; flip both `v_x`
  and `v_y`.

**Periodic boundary face canonicalization**: under periodic wrap, the W
and E boundary x-faces are physically the same face. To avoid two
invocations swapping stale values, both read from the W face
(`src_i = ghost`) — the W invocation self-copies (no-op), the E
invocation copies W → E.

## Pipeline dispatch shapes (per RK3 stage)

With `N = 256`, `ghost = 2`, `N_total = 260`, and workgroup 8×8:

| Pass                | Logical extent         | Workgroup count (256²) |
|---------------------|------------------------|------------------------|
| apply-bcs           | `(N_total + 1)²`       | 33²                    |
| reconstruct-ppm     | `(N + 2)²`             | 33²                    |
| riemann-hlld (x)    | `(N + 1) × (N + 2)`    | 33 × 33                |
| riemann-hlld (y)    | `(N + 2) × (N + 1)`    | 33 × 33                |
| compute-emf         | `(N + 1)²`             | 33²                    |
| update-conserved    | `N²`                   | 32²                    |
| update-b            | `(N + 1)²`             | 33²                    |
| apply-resistivity   | `(N_total + 1)²`       | 33²                    |
| compute-dt (reduce) | `N²`                   | 32²                    |

PPM at the outer dispatch cells (`i = ghost − 1` and `i = ghost + N`)
lacks the full 5-point stencil; the kernel detects this (`stencil_ok`
check) and falls back to piecewise-constant edges. Downstream Riemann
then uses these lower-order edges at the boundary faces — acceptable
because the BC-filled ghost is itself lower-order.

### Per-frame render passes

Render is decoupled from physics (runs once per displayed frame, not
once per RK3 stage):

| Pass                | Logical extent  | Workgroup count (256²) |
|---------------------|-----------------|------------------------|
| view-field          | `N²`            | 32²                    |
| colormap            | `N²`            | 32²                    |
| lic-advect          | `N²`            | 32²                    |
| lic-reduce.reset    | 1×1             | 1×1                    |
| lic-reduce.main     | `N²`            | 32²                    |
| lic-normalize       | `N²`            | 32²                    |
| composite (fragment)| canvas pixels   | (full-screen triangle) |

The reduce + normalize chain implements a min/max contrast stretch on
`lic_out` before composite samples it. See "LIC visualization" for the
full chain.

## Boundary conditions

Per-edge BC mode IDs (`config.js`):

| ID | Name            | Behavior                                       |
|----|-----------------|------------------------------------------------|
| 0  | `BC_PERIODIC`   | Wrap from opposite edge.                       |
| 1  | `BC_OUTFLOW`    | Zero-gradient (copy from nearest interior).    |
| 2  | `BC_REFLECTING` | Perfectly conducting wall: mirror with v_n / B_n flip; boundary B_n = 0. |
| 3  | `BC_DRIVEN`     | Apply user-configured inflow state from `bc_uniforms`. |

Per-edge slots in `bc_uniforms` (storage buffer):

| Field    | Edge                                       |
|----------|--------------------------------------------|
| `mode_n` | N (top, `j ∈ [ghost + N, N_total)`)        |
| `mode_s` | S (bottom, `j ∈ [0, ghost)`)               |
| `mode_e` | E (right, `i ∈ [ghost + N, N_total)`)      |
| `mode_w` | W (left, `i ∈ [0, ghost)`)                 |

Driven state (primitive, single global; per-edge driven is a future
extension): 8 f32s `(driven_rho, driven_vx, driven_vy, driven_vz,
driven_bx, driven_by, driven_bz, driven_p)`. Converted to conservative
on write by the BC shader via `prim_to_cons_pair`.

## Presets

| Name                 | γ    | η     | BCs                       | Acid test                                                            |
|----------------------|------|-------|---------------------------|----------------------------------------------------------------------|
| Sod                  | 1.4  | 0     | all periodic              | Shock + contact + rarefaction wave structure                         |
| Brio-Wu              | 2.0  | 0     | all periodic              | MHD wave structure (slow shock, compound, contact, slow shock)       |
| Orszag-Tang          | 5/3  | 0     | all periodic              | Central density blob + four current sheets + magnetic islands by t≈0.5 |
| Harris current sheet | 5/3  | 1e-3  | E=W periodic, N=S outflow | Reconnection forming around t≈10 t_A; plasmoid formation along sheet |

### Harris current sheet (canonical reconnection IC)

| Quantity     | Profile                                            |
|--------------|----------------------------------------------------|
| `Bx(y)`      | `B_0 · tanh(y / a)`, `B_0 = 1`, `a = 0.1`          |
| `By`, `Bz`   | 0                                                  |
| `ρ(y)`       | `ρ_∞ + ρ_0 · sech²(y/a)`, `ρ_∞ = 0.2`, `ρ_0 = 1`   |
| `p(y)`       | `p_∞ + ½ B_0² · sech²(y/a)`, `p_∞ = 0.1`           |
| Perturbation | `vy = 0.01 · sin(π · x) · sech²(y/a)`              |
| Domain       | `[-1, 1] × [-1, 1]`, sheet at y=0                  |

Pressure balance: `p + ½|B|² = p_∞ + ½ B_0²` constant across y.

## UI

`src/ui.js` (entry `setupUI(simShell)`), `src/stats-display.js`,
`src/probe.js`, and `src/gpu/readback.js`. Called from `main.js`
after `sim.init()`. Mounts the topbar, three-tab sidebar
(Settings / Stats / Probe), and the stats/probe readback paths.

### Readback pattern (`src/gpu/readback.js`)

`ReadbackPool` keeps a per-byte-size pool of staging buffers with
`MAP_READ | COPY_DST` usage. `readbackSlice(...)` does one copy +
submit + `mapAsync` + slice. `readbackBatch(...)` issues N copies in
one encoder + one submit, then awaits all maps in parallel — used by
stats-display to grab `(U0, U1, Bx, By, dt)` in one round-trip and by
probe to grab a 3-row stencil window.

Resolution-adaptive cadence: 12 Hz at 256², 6 Hz at 512², 3 Hz at
1024². Probe runs at 10 Hz independent of render.

### `sim.js` public API

`setPreset(name)`, `setBC(edge, mode)`, `setDrivenState(partial)`,
`setEta(eta)`, `setViewMode(mode)`, `setCFL(cfl)` (uniform-only —
see [`docs/HANDOFF.md`](docs/HANDOFF.md) for the shader-wiring gap),
`setGamma(g)`,
`setPressureFloor(p)` (uniform-only — same gap), `setRunning(r)`,
`step()` (single step), `setSpeedScale(s)`, `setResolution(n)`,
`saveState()` → `loadState(s)` (JSON, parameters only; no buffer
snapshot), `setLicIntensity(v)`, `setLicDrift(dx, dy)`.

`setResolution(n)` re-instantiates `PlasmaBuffers` and
`PlasmaRenderer` at the new interior size, then reloads the current
preset. UI must call `stats.bindBuffers(sim.buffers)` and
`probe.bindBuffers(sim.buffers)` to re-aim readback paths.

## LIC visualization

Animated line integral convolution along B, decoupled from physics —
runs once per displayed frame. Render chain:
`view-field → colormap → lic-advect → lic-reduce → lic-normalize → composite`.

* `lic-advect.wgsl` — one invocation per interior cell. Backward-traces
  20 RK2 steps along the unit B-field (step size 0.5 cells, ~10 cells
  total). Samples a 1024×1024 white-noise base buffer with bilinear
  interpolation and a per-frame phase offset. Writes one f32
  luminance per interior cell into `lic_out`.
* `lic-reduce.wgsl` — two entry points (`reset` + `main`). Reduces the
  interior of `lic_out` to a global `(min, max)` pair (`lic_minmax`,
  2 × u32 bit-pattern). Mirrors `compute-dt`'s per-tile shared-atomic
  pattern: each thread `atomicMin/Max`-es its cell's bit-pattern into
  workgroup-shared `tile_min` / `tile_max`, top-level
  `workgroupBarrier()`, thread 0 commits to global `lic_minmax`.
  Bitcast trick works because lic-advect's averaging contract keeps
  the luminance in `[0, 1]` (non-negative → u32 ordering preserved);
  shader still defensively gates with `select`/`clamp` against
  hypothetical NaN. `reset` (1×1) seeds `lic_minmax` to `(1.0, 0.0)`
  every frame before `main` runs.
* `lic-normalize.wgsl` — per-invocation; reads `lic_minmax`, rewrites
  `lic_out[i] := clamp((lic_out[i] − min) / max(max − min, 1e-4), 0, 1)`
  in place. Min/max stretch (research-grade vis default) over
  mean/std — preserves the relative dominance of high-luminance LIC
  bands at shock fronts and pulls residual variation out into the
  full `[0, 1]` range in flat field-free regions.
* `composite.wgsl` — `final.rgb = colored.rgb · mix(1, 0.5 + L, intensity)`.
  At intensity = 0 the colormap passes through unchanged; at
  intensity = 1 LIC modulates by ±50%; slider clamps at 2 for a
  stronger swing.
* Noise: 1024² f32 (4 MB), generated once at init via mulberry32 PRNG
  (deterministic seed `0xC0FFEE`). Resolution-independent — same buffer
  serves any grid; LIC shader scales sample position. TODO marker for
  blue-noise upgrade (void-and-cluster).
* Phase animation: wall-clock dt (not sim time), clamped to 100 ms to
  handle tab refocus.
* UI: LIC intensity + drift sliders in the advanced settings dropdown.

### LIC bind-group layout

Three compute pipelines, one bind group each (group 0):

`lic-advect.wgsl`:

| Binding | Type      | Resource     | Access     |
|---------|-----------|--------------|------------|
| 0       | uniform   | Uniforms     | read       |
| 1       | storage   | Bx_face      | read       |
| 2       | storage   | By_face      | read       |
| 3       | storage   | noise        | read       |
| 4       | storage   | lic_out      | read_write |
| 5       | uniform   | LicUniforms  | read       |

`lic-reduce.wgsl` (entries `reset` + `main`):

| Binding | Type      | Resource    | Access                 |
|---------|-----------|-------------|------------------------|
| 0       | uniform   | Uniforms    | read                   |
| 1       | storage   | lic_out     | read                   |
| 2       | storage   | lic_minmax  | read_write (atomic<u32>) |

`lic-normalize.wgsl`:

| Binding | Type      | Resource    | Access     |
|---------|-----------|-------------|------------|
| 0       | uniform   | Uniforms    | read       |
| 1       | storage   | lic_minmax  | read       |
| 2       | storage   | lic_out     | read_write |

Dispatch shapes: lic-advect / lic-reduce.main / lic-normalize use
workgroups of 8×8, count `(N/8) × (N/8)`. lic-reduce.reset is a
single 1×1 invocation.

`lic-reduce` uses one workgroup-shared `atomic<u32>` per reduction
direction (`tile_min`, `tile_max`) with a single top-level
`workgroupBarrier()` between the per-cell phase and the global commit
— same shape as `compute-dt.reduce` (transpiler-verified by the
`testTwoDSharedTileHaloBarrier` smoke pattern). `lic-advect` and
`lic-normalize` are purely per-invocation (no shared memory, no
atomics, no barriers).

## Uniforms (128 bytes)

Slot 0-15 (original 64 B) — base MHD physics + render state:

| Slot | Type | Field             | Notes                                                          |
|------|------|-------------------|----------------------------------------------------------------|
| 0    | f32  | `dx`              | Cell size in domain units                                      |
| 1    | f32  | `gamma`           | Adiabatic index                                                |
| 2    | f32  | `view_min`        | Per-preset visualization clamp                                 |
| 3    | f32  | `view_max`        | Per-preset visualization clamp                                 |
| 4    | f32  | `eta`             | Resistivity                                                    |
| 5    | f32  | `eta_anom_alpha`  | Birn 2001 anomalous-η α (0 = constant-η)                       |
| 6    | f32  | `_pad_lic_1`      | Reserved (was lic_intensity — now in LicUniforms)              |
| 7    | f32  | `_pad_lic_2`      | Reserved (was lic_drift_x — now in LicUniforms)                |
| 8    | u32  | `grid_n`          | Interior grid extent                                           |
| 9    | u32  | `grid_n_total`    | Ghost-padded grid extent                                       |
| 10   | u32  | `ghost_w`         | Ghost width (= 2)                                              |
| 11   | f32  | `pressure_floor`  | Live UI slider; minimum p in cons→prim recovery                |
| 12   | f32  | `cfl`             | Hyperbolic CFL number — consumed by compute-dt                 |
| 13   | u32  | `view_mode`       | View enum from config.js VIEW_*                                |
| 14   | f32  | `eta_anom_jcrit`  | Anomalous-η activation threshold J_crit                        |
| 15   | u32  | `noise_n`         | Side length of noise buffer (= 1024)                           |

Slot 16-31 (extended physics, Session 14):

| Slot | Type | Field                   | Notes                                                          |
|------|------|-------------------------|----------------------------------------------------------------|
| 16   | f32  | `hall_di`               | Hall ion inertial length d_i (code units; 0 = no Hall)         |
| 17   | u32  | `hall_substeps_max`     | Max Hall / conduction sub-cycles per macro step                    |
| 18   | f32  | `cooling_lambda0`       | Cooling rate scale Λ_0 (0 = no cooling)                        |
| 19   | f32  | `cooling_T_floor`       | Below this T, Λ → 0                                            |
| 20   | f32  | `cooling_T_ref`         | Reference temperature for Λ(T) normalization                   |
| 21   | f32  | `conduction_kappa`      | Parallel thermal conductivity κ_∥ (0 = no conduction)          |
| 22   | f32  | `conduction_iso_frac`   | κ_⊥ / κ_∥ (0 = fully anisotropic, 1 = isotropic)               |
| 23   | f32  | `conduction_sat_frac`   | Cowie-McKee saturated heat-flux fraction (0 = unlimited)       |
| 24   | f32  | `gravity_gx`            | External gravity x (constant)                                  |
| 25   | f32  | `gravity_gy`            | External gravity y                                             |
| 26   | f32  | `gravity_G`             | Newton's G for self-gravity (0 = no self-gravity)              |
| 27   | u32  | `gravity_poisson_iters` | Jacobi iterations per macro step                               |
| 28   | u32  | `physics_flags`         | Bitfield: COOLING\|GRAV_EXT\|GRAV_SELF\|COND\|HALL\|POSITIVITY\|EMF_UPWIND |
| 29   | u32  | `emf_mode`              | 0 = Balsara-Spicer mean, 1 = Gardiner-Stone upwind             |
| 30   | u32  | `cooling_curve_mode`    | 0 = √T brems exact mode, 1 = piecewise power-law table         |
| 31   | f32  | `hall_electron_pressure_frac` | p_e / p closure for the Hall pressure-gradient term      |

`Uniforms` is held in a single 128 B buffer. Sweep direction lives in two
static 16 B uniforms (`sweepDir_x` = 0u, `sweepDir_y` = 1u) bound only by
reconstruct-ppm and riemann-hlld. LIC render-pace state (phase, intensity,
drift) lives in a separate 16 B `LicUniforms` buffer rewritten per render
frame. Stage weights live in 3 separate uniform buffers (`stage_1` /
`stage_2` / `stage_3`).

## Extended physics

Features bolted onto the base MHD engine in Session 14, then hardened
in Sessions 15-18 (Codex passes + follow-up). Each shader early-returns when its
flag bit is clear OR the corresponding scalar is 0, so individual features
can be toggled per-cell via the uniform without code changes.

**Default opt-in is per-preset.** Canonical verification presets (Sod,
Brio-Wu, OT, Harris, Alfvén CPAW, acoustic) declare `physics: {
physicsFlags: BASE_PHYSICS_FLAGS }` — just positivity + GS upwind EMF, no
source physics. The new `orszag-tang-extended` preset opts into the full
stack with the same scalars that were the old global defaults. Four
Session-15 validation presets (`hall-whistler`, `conduction-front`,
`cooling-instability`, `jeans-instability`) opt into exactly the one
feature they isolate.

`Sim._applyPhysicsConfig(preset.physics)` runs on every `loadPreset` and
absorbs whatever fields the preset specified — missing fields fall back
to the frozen `DEFAULT_PHYSICS_STATE` constant in sim.js. The `physics`
block is part of the save/load schema.

| Feature              | Shader(s)                                                                | Equation / discretization                                                              |
|----------------------|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Radiative cooling    | `apply-cooling.wgsl`                                                     | Exact integration of either the legacy `√T` bremsstrahlung shape, Session-16 compact table, Session-17 CIE-inspired metallicity-scaled table, or Session-18 uploaded microphysics table |
| Heating              | `apply-cooling.wgsl`                                                     | Density-law volumetric heating with optional high-T cutoff, paired with cooling for thermal-balance experiments |
| Self-gravity         | `solve-poisson.wgsl` (reduce_mean + finalize_mean + iterate) + `apply-gravity.wgsl` | Workgroup-shared ρ̄ reduction, periodic weighted Jacobi `∇²φ = 4πG(ρ−ρ̄)` with optional softening, then `d(ρv)/dt = ρg`, `dE/dt = ρv_mid·g` with 4th-order periodic force recovery |
| External gravity     | `apply-gravity.wgsl` (alt branch)                                        | Same source-term form with constant `g = (gx, gy, 0)`                                  |
| Anisotropic conduction | `apply-conduction.wgsl` (compute_delta + apply_delta)                   | `q = κ_∥ b̂(b̂·∇T) + κ_⊥(∇T−b̂(b̂·∇T))`, smooth Cowie-McKee saturation, `dE/dt = -∇·q`; frozen-state delta into `conduction_dE`, then applied to U1 |
| Hall MHD             | `apply-ohm.wgsl` (compute_emf + apply_hall_update + repair_hall_energy)  | Corner `E_H = (d_i/ρ)·(J×B − ∇p_e)` → scratch EMF, CT update of face B + cell Bz from frozen E, then `Δ(½|B|²)` added to total E so the Hall B update doesn't masquerade as heat |
| Ambipolar + Biermann | `apply-ohm.wgsl` (compute_emf + apply_dissipative_update)                | Ambipolar `η_A J_perp` and Biermann `∇ρ×∇p_e/ρ²` source terms from the same frozen generalized-Ohm state, applied at fixed total energy |
| Viscosity            | `apply-viscosity.wgsl` (compute_delta + apply_delta)                     | Explicit shear/bulk/shock viscosity, optional B-aligned projection, momentum diffusion plus viscous work/heating |
| Geometry + sponge    | `apply-geometry.wgsl`                                                    | Cylindrical axisymmetric source layer (`x=r`, `y=z`) and pressure-preserving boundary sponge |
| Positivity guard     | `update-conserved-weighted.wgsl` (inline)                                | When post-update `ρ ≤ 0` or `E−KE ≤ p_floor/(γ−1)`, drop the L term and fall back to the pure SSP blend (Hu/Adams/Shu 2013 §3 spirit) |
| EMF mode toggle      | `compute-emf.wgsl`                                                       | Runtime switch between Balsara-Spicer arithmetic mean and Gardiner-Stone 2005 upwind (latter is default) |
| Energy-floor cleanup | `energy-floor.wgsl`                                                      | Final pass after extended physics: clamps E against the final B so source-term combinations leave the cell physically consistent |

### Extended-physics dispatch order

`_encodeExtendedPhysics(encoder, side)` runs once per macro step,
**after** the RKL2 super-step. The dispatches in order:

1. `apply-bcs` (destination side) — canonicalize ghosts before any
   stencil reads.
2. Poisson chain (if `FLAG_GRAVITY_SELF`): `solve-poisson.reduce_mean`
   → `solve-poisson.finalize_mean` → `solve-poisson.iterate` ×
   `gravity_poisson_iters`. Ping-pong between `phi` and `phi_next`.
3. `apply-gravity` — momentum + energy source from external g + `-∇φ`.
4. `apply-cooling` — cooling/heating energy source.
5. `compute-conduction-delta` → `apply-conduction-delta` — frozen
   energy delta then apply.
6. `compute-viscosity-delta` → `apply-viscosity-delta` — frozen
   velocity-gradient transport.
7. `compute-ohm-emf` → `apply-ohm.apply_hall_update` →
   `apply-ohm.repair_hall_energy` → `apply-ohm.apply_dissipative_update`
   — unified Hall, ambipolar, and Biermann update from one frozen state.
8. `apply-geometry` — cylindrical source layer + sponge.
9. `energy-floor` — final E clamp against the final B.
10. `apply-bcs` (destination side) — canonicalize ghosts again.

The whole sequence is gated by `(physicsFlags & EXTENDED_SOURCE_FLAGS)
!= 0`; if no source physics is on, the encoder returns immediately and
the macro step is just RK3 + RKL2.

### Scratch buffers (Sessions 15-18)

`PlasmaBuffers` carries source-term scratch buffers + the Poisson
ping-pong partner:

| Buffer                | Shape           | Owner               | Cleared at preset load |
|-----------------------|-----------------|---------------------|------------------------|
| `conduction_dE`       | cell-shaped f32 | `apply-conduction`  | yes                    |
| `hall_E`              | edge-shaped vec4 | `apply-ohm`         | yes                    |
| `hall_mb0`            | cell-shaped f32 | `apply-ohm`         | yes                    |
| `nonideal_E`          | edge-shaped vec4 | `apply-ohm`         | yes                    |
| `viscosity_dU`        | cell-shaped vec4 | `apply-viscosity`   | yes                    |
| `rho_mean_partials`   | ⌈N/8⌉² f32      | `solve-poisson`     | yes                    |
| `phi`, `phi_next`     | cell-shaped f32 | `solve-poisson`     | yes                    |

`buffers.clearExtendedScratch()` runs on `uploadInitialState` to prevent
warm-start from a previous preset.

### Default scalars (DEFAULT_PHYSICS_STATE in sim.js)

These are the fallbacks when a preset doesn't override them. The
canonical presets keep these but leave source flags off;
`orszag-tang-extended` keeps the representative source stack active.

| Knob                  | Default | Notes                                              |
|-----------------------|---------|----------------------------------------------------|
| `physicsFlags`        | `POSITIVITY \| EMF_UPWIND` (BASE) | Source flags are opt-in per preset      |
| `emfMode`             | 1 (GS upwind) | The numerical default for CT                |
| `hallDi`              | 0.02    | ~5 cells at N=256 — Hall scale resolved            |
| `hallElectronPressureFrac` | 0.0 | Presets can opt into `p_e / p`; whistler test keeps it off |
| `coolingLambda0`      | 0.01    | Visible but not catastrophic on 1×1-domain presets |
| `coolingCurveMode`    | 3       | Uploaded microphysics table by default             |
| `coolingMetallicity`  | 1.0     | Solar scaling for CIE-inspired mode                |
| `heatingGamma0`       | 0       | Heating off unless preset/UI opts in               |
| `conductionKappa`     | 1e-3    | Below parabolic CFL for OT scale at base resolution |
| `conductionIsoFrac`   | 0.1     | 90% parallel, 10% perp                             |
| `conductionSatFrac`   | 0.0     | Unlimited by default; extended/validation presets opt into saturation |
| `gravityG`            | 1e-3    | Mostly visible where ρ-contrasts are large         |
| `gravityPoissonIters` | 30      | Sufficient for steady-state on N=256 with 1×1 box  |
| `ambipolarEta`, `biermannCoeff`, `viscosityNu` | 0 | Off by default; extended preset/UI can opt in |
| `sourceSubstepsMax`   | 8       | Shared cap for viscosity/non-ideal explicit subcycles |

### Sub-cycling architecture (Sessions 15-18)

Hall, anisotropic conduction, viscosity, and non-ideal induction can run
as **sub-cycles inside one hyperbolic macro Δt**. The pattern is shared:

1. `compute-dt.wgsl` reduces the per-cell stability speed (Hall:
   `v_A·d_i/dx` — whistler speed; conduction: `4·χ/dx` — parabolic
   speed) as a separate atomic alongside the wave-speed reduction,
   and writes the global max to `dt_buf[3]` (Hall) / `dt_buf[4]`
   (conduction). The macro signal-speed sum does NOT include these
   terms — macro Δt respects only the hyperbolic CFL.
2. The host async-reads `dt_buf[3..4]` one-step-lagged (mirrors the
   RKL2 readback pattern), computes
   `N = min(maxN, ceil(dt_macro · speed_max / safety))`, and seeds
   `dt_sub = dt_macro / N` into the dedicated sub-step uniform
   (`b.hall_dt` / `b.cond_dt`) via `queue.writeBuffer` once per macro
   step (NOT once per iteration — WebGPU collapses repeated
   `writeBuffer`s before the next submit).
   Viscosity and non-ideal terms are sized host-side from their uniform
   coefficients to avoid adding storage bindings to `compute-dt`; their
   sub-step uniforms are `b.visc_dt` and `b.nonideal_dt`.
3. `_encodeExtendedPhysics` loops the corresponding compute-pass
   sequence N times within a single compute pass. WebGPU's dispatch
   ordering guarantees each iteration reads the prior iteration's
   writes for cross-dispatch data flow (face B updated by Hall's
   `apply_update`, U1 updated by conduction's `apply_delta`).

This adds two new uniform buffers (`hall_dt`, `cond_dt`) and two new
atomic reduction buffers (`hall_speed_buf`, `cond_speed_buf`).
`compute-dt.wgsl`'s `dt_buf` is now `array<f32, 8>` (was 4); the
underlying GPU buffer was already 32 B so no buffer resize.

Hall keeps the user-facing `hallSubstepsMax` cap. Conduction, viscosity,
and dissipative/non-barotropic Ohm terms use `sourceSubstepsMax` so
transport stiffness can be capped separately from whistler stiffness.
Session 18's conduction reduction includes the default `T^(5/2)`
transport scaling, matching the source shader's uploaded-table closure.

### Cooling exact integrator (Sessions 15-18)

`apply-cooling.wgsl` no longer uses forward Euler. For our specific
single-power-law cooling shape (Λ ∝ √((T − T_floor)/T_ref)), the
substitution `s = √((T − T_floor)/T_ref)` linearizes the ODE:
`ds/dt = -(γ-1)·ρ·Λ_0 / (2·T_ref) = const`. The exact analytic
update is

    s(t)  = max(s(0) − C·t, 0)
    T(t)  = T_floor + T_ref · s(t)²

— unconditionally stable for any Δt, exact for the chosen Λ shape.
Session 16 keeps that as `cooling_curve_mode = 0` and adds
`cooling_curve_mode = 1`: a compact piecewise power-law table in
`θ = T/T_ref`, advanced exactly per segment with the same Townsend
idea. The default table has a low-temperature rise, line-cooling peak,
trough, and high-temperature bremsstrahlung tail. The cooling timestep
bound that briefly lived in `compute-dt.wgsl` is gone — cooling no
longer constrains macro Δt.

Session 18 adds `cooling_curve_mode = 3`, which reads the uploaded
microphysics table. The built-in table is still compact and code-unit
scaled, but the data path no longer requires shader edits to replace the
curve family.

### Remaining sharp edges (Phase 9 follow-up)

The Session 15 pass resolved BC consistency, the Poisson ρ̄ stub,
the Hall within-pass race, the conduction within-pass race, the
"everything is on by default" trap, the Hall and conduction CFL
collapse (sub-cycling), the cooling FE bias (exact integrator), the
"no view modes for T/|q|/φ" gap, and the "no UI surface" gap. Still
open:

1. **O'Sullivan & Downes 2006 Hall Diffusion Scheme.** Sub-cycling
   has O(N_hall) cost; HDS hyperbolizes the Hall term so the
   standard CFL covers it without sub-cycling. Deferred until a
   workload actually saturates the sub-cycle cap.
2. **Conduction → proper RKL2 super-step.** Sub-cycling is O(N_cond);
   RKL2 would be O(√N_cond). Currently blocked by the 10-binding
   cap on `apply-resistivity-init.wgsl` — needs a multi-shader BGL
   reshuffle. Worth it if a future workload pushes N_cond above ~30.
3. **Cooling data fidelity.** Session 16's table is intentionally compact
   and dimensionless. A production astrophysical mode should upload a
   vetted metallicity-dependent table (for example CHIANTI/Sutherland-
   Dopita style data) instead of baking one curve into WGSL.

## Transpiler contract

All compute pipelines are designed to be compilable by the parent
repo's `shared-wgsl-transpile.js`:

* One bind group per pipeline (group 0). No dynamic offsets, no push
  constants.
* No subgroup ops, no matrices, no textures, no samplers, no
  atomics-on-floats.
* `bitcast` used by `compute-dt` (f32 ↔ u32 for atomicMax reduction)
  and the LIC contrast-stretch pair `lic-reduce` / `lic-normalize`
  (same f32 ↔ u32 trick on the [0, 1] luminance range).
* `workgroupBarrier()` only at top level — never inside `if` / `for` /
  `while`.
* Workgroup-shared memory: allowed broadly as long as barriers stay
  at top level. Used by `compute-dt` (per-tile max reduction),
  `lic-reduce` (per-tile min/max reduction of `lic_out`), and
  `reconstruct-ppm` (per-tile primitive cache; see the dedicated
  section above). The transpiler's 2D shared-tile + halo + top-level
  barrier pattern is verified by `testTwoDSharedTileHaloBarrier` in
  `tests/wgsl-transpile/smoke.js`.
* Other kernels are purely per-invocation: read neighbors via direct
  indexing, write own cell, no cross-invocation communication.
* `apply-bcs` uses a `switch`-on-mode pattern — trivially
  CPU-emulatable.

The composite render pass stays GPU-only (it's a fragment shader, not
compute) — per the transpiler's "render entry points stay GPU-only"
clause.

## Shared module dependencies

Compute path: `shared-tokens.js`, `shared-base.css`, `shared-utils.js`.

UI: `shared-toolbar.js`, `shared-forms.js`, `shared-dropdown.js`,
`shared-settings.js`, `shared-about.js`, `shared-tabs.js`,
`shared-icons.js`, `shared-shortcuts.js`, `shared-info.js`,
`shared-tooltip.js`, `shared-sparkline.js`, `shared-touch.js`,
`shared-haptics.js`.

`shared-camera.js` is intentionally NOT used — fixed orthographic 2D.

For CPU fallback: `shared-wgsl-transpile.js` (compile-time conversion
of compute shaders to JS).

## Rules

- **No innerHTML assignments.** Use `textContent` or `createElement`.
  A Write hook blocks innerHTML in new files.
- Always prefer shared modules over re-implementations.
- WebGPU-only in-tree. The `shared-wgsl-transpile.js` hookup is the
  CPU fallback path; the `#no-webgpu` landing page is the interim
  message.
- `shared-tokens.js` loads as a synchronous `<script>` (no `defer`) so
  it runs before CSS parses.

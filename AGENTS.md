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

**Status**: Phases 1–6 complete (engine + UI + LIC). Phases 7–8 (polish
+ parent-repo wiring) outstanding — see `HANDOFF.md`.

## Design

- **Physics**: resistive 2.5D ideal MHD — state `(ρ, v_x, v_y, v_z, B_x, B_y, B_z, p)`
- **Riemann solver**: HLLD (Miyoshi & Kusano 2005) with HLLC and HLL fallbacks for degenerate branches
- **Reconstruction**: PPM (Colella & Woodward 1984) with characteristic-variable limiting (Stone+ 2008 §3.4.2 — Athena/Athena++ default for MHD)
- **Time integration**: RK3 SSP, three stages, single-submit-per-step
- **Divergence cleaning**: constrained transport on a Yee-style staggered grid (Stone+ 2008), Gardiner-Stone 2005 upwind EMF (HLLD contact velocity as upwind selector)
- **Resistivity**: explicit central differences, η ∇²B applied per RK3 stage after CT update (SSP-compatible by linearity)
- **Boundaries**: per-edge selectable — periodic / outflow / reflecting / driven
- **Default view**: J_z (out-of-plane current density)
- **Field visualization**: animated LIC only (no quiver, streamlines, or arrow glyphs)
- **Pointer**: drag injects a velocity perturbation (Phase 7 wiring)
- **Grid**: 256² default; sidebar selector for 256 / 512 / 1024
- **Ghost cells**: 2 layers per side (PPM's 5-point stencil at edges)

## Layout

```
plasma/
├── index.html              ← canvas, topbar, sidebar, edu-content stub, JSON-LD stub
├── main.js                 ← entry: WebGPU init, frame loop, accumulator, setupUI hook
├── styles.css              ← canvas + HUD layout (mostly inherits from /shared-base.css)
├── colors.js               ← _PALETTE extensions, frozen at startup
├── about.md                ← educational content (stub — Phase 7)
├── AGENTS.md               ← this file
├── HANDOFF.md              ← next-step doc for the next instance / agent
├── LICENSE                 ← AGPL-3.0
└── src/
    ├── config.js           ← grid size, CFL, γ, η, ghost width, BC enums, uniform layout, LIC constants
    ├── sim.js              ← orchestrator: BC → PPM → HLLD → EMF → CT → resistivity + LIC state
    ├── presets.js          ← Sod, Brio-Wu, Orszag-Tang, Harris current sheet
    ├── colormaps.js        ← viridis LUT (7-stop polynomial fit, sampled at 256 stops)
    ├── ui.js               ← shared-* module wiring (Phase 5)
    ├── stats-display.js    ← Stats tab: energy / β / |B|max / ∇·B norm / reconnection rate
    ├── probe.js            ← Probe tab: cell sampling + mini time-series
    └── gpu/
        ├── device.js       ← adapter + device init
        ├── buffers.js      ← ghost-padded slots, BC uniforms, stage params, noise + lic_out
        ├── pipelines.js    ← compute + render pipeline factory
        ├── render.js       ← view-field → colormap → LIC advect → composite
        ├── lic.js          ← LicRenderer (per-frame bind group construction)
        ├── readback.js     ← ReadbackPool for stats + probe GPU→CPU
        └── shaders/
            ├── shared-helpers.wgsl              ← Uniforms, BcUniforms, MHD prim/cons, indexing
            ├── apply-bcs.wgsl                   ← ghost-cell fill (4 modes × 4 edges)
            ├── reconstruct-ppm.wgsl             ← per-direction PPM (CW 1984)
            ├── riemann-hlld.wgsl                ← HLLD (M&K 2005) + HLLC + HLL fallbacks
            ├── compute-emf.wgsl                 ← Gardiner-Stone 2005 upwind Ez at corners
            ├── update-conserved-weighted.wgsl   ← RK3 SSP weighted U update
            ├── update-b-weighted.wgsl           ← RK3 SSP weighted face-B update (CT)
            ├── apply-resistivity.wgsl           ← η ∇²B per stage (post-CT)
            ├── compute-dt.wgsl                  ← MHD CFL + parabolic resistive CFL
            ├── view-field.wgsl                  ← scalar extract (ρ, p, |v|, |B|, J_z)
            ├── colormap.wgsl                    ← viridis LUT lookup (interior only)
            ├── lic-advect.wgsl                  ← backward-trace LIC along B (transpilable)
            ├── lic-reduce.wgsl                  ← per-tile min/max reduce of lic_out (workgroup-shared atomics)
            ├── lic-normalize.wgsl               ← in-place contrast stretch using lic_minmax
            └── composite.wgsl                   ← canvas blit, colormap × LIC luminance
```

## Numerical method

| Piece               | Choice                                                |
|---------------------|-------------------------------------------------------|
| Reconstruction      | PPM (Colella & Woodward 1984) with characteristic-variable limiting (Stone+ 2008 §3.4.2); workgroup-shared primitive cache |
| Riemann solver      | HLLD (Miyoshi & Kusano 2005) + HLLC + HLL fallbacks   |
| Time integration    | RK3 SSP (Gottlieb-Shu 1998)                           |
| Divergence-free B   | Constrained transport (Gardiner-Stone 2005 upwind EMF) |
| Resistivity         | Explicit η ∇²B, central differences, post-CT          |
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
see HANDOFF for the shader-wiring gap), `setGamma(g)`,
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

## Uniforms (64 bytes)

| Slot | Type | Field            | Notes                                                          |
|------|------|------------------|----------------------------------------------------------------|
| 0    | f32  | `dx`             | Cell size in domain units                                      |
| 1    | f32  | `gamma`          | Adiabatic index                                                |
| 2    | f32  | `view_min`       | Per-preset visualization clamp                                 |
| 3    | f32  | `view_max`       | Per-preset visualization clamp                                 |
| 4    | f32  | `eta`            | Resistivity                                                    |
| 5    | f32  | `_pad_lic_0`     | Reserved (was lic_phase — moved to LicUniforms)                |
| 6    | f32  | `_pad_lic_1`     | Reserved (was lic_intensity — moved to LicUniforms)            |
| 7    | f32  | `_pad_lic_2`     | Reserved (was lic_drift_x — moved to LicUniforms)              |
| 8    | u32  | `grid_n`         | Interior grid extent                                           |
| 9    | u32  | `grid_n_total`   | Ghost-padded grid extent                                       |
| 10   | u32  | `ghost_w`        | Ghost width (= 2)                                              |
| 11   | f32  | `pressure_floor` | Live UI slider; minimum p in cons→prim recovery                |
| 12   | f32  | `cfl`            | Hyperbolic CFL number — consumed by compute-dt                 |
| 13   | u32  | `view_mode`      | View enum from config.js VIEW_*                                |
| 14   | f32  | `_pad_lic_3`     | Reserved (was lic_drift_y — moved to LicUniforms)              |
| 15   | u32  | `noise_n`        | Side length of noise buffer (= 1024)                           |

`Uniforms` is held in a single 64 B buffer. Sweep direction lives in two
static 16 B uniforms (`sweepDir_x` = 0u, `sweepDir_y` = 1u) bound only by
reconstruct-ppm and riemann-hlld. LIC render-pace state (phase, intensity,
drift) lives in a separate 16 B `LicUniforms` buffer rewritten per render
frame. Stage weights live in 3 separate uniform buffers (`stage_1` /
`stage_2` / `stage_3`).

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

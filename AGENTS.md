# AGENTS.md — plasma

Part of the **a9l.im** portfolio. See root `AGENTS.md` for the shared
design system and shared-code policy. Sibling sims: `geon`, `shoals`,
`cyano`, `gerry`, `scripture`.

## Scope

WebGPU-only interactive 2D resistive magnetohydrodynamics simulator.
Grid-Eulerian finite-volume on a regular Cartesian mesh. No CPU
fallback — unsupported browsers see a static landing message linking
to `/sims`.

Implementation plan (source of truth for design decisions):
`~/.claude/plans/geon-currently-uses-cpu-abstract-cat.md`.

## Locked design decisions

- **Physics**: resistive 2.5D ideal MHD — state `(ρ, v_x, v_y, v_z, B_x, B_y, B_z, p)`
- **Riemann solver**: HLLD (Miyoshi & Kusano 2005)
- **Reconstruction**: PPM (Colella & Woodward 1984)
- **Time integration**: RK3 SSP, three stages
- **Divergence cleaning**: constrained transport on a Yee-style staggered grid (Stone+ 2008)
- **Boundaries**: per-edge selectable (periodic/outflow/reflecting/driven)
- **Default view**: J_z (out-of-plane current density)
- **Field visualization**: animated LIC only (no quiver/streamlines)
- **Pointer**: drag injects a velocity perturbation
- **Grid**: 256² default; sidebar selector for 256/512/1024
- **No-WebGPU**: static landing page, no CPU path
- **Ghost cells**: 2 layers per side (PPM's 5-point stencil)
- **Resistivity**: explicit central differences, η ∇²B applied per RK3 stage AFTER CT update (SSP-compatible by linearity)

Build phases:
1. Phase 1: WebGPU init, fullscreen-quad render, frame loop, visibilitychange pause.
2. Phase 2: pure hydro (Euler) with HLL + PLM + FE — Sod tube.
3. Phase 3: full MHD with HLLD + PPM + RK3 + CT — Orszag-Tang.
4. Phase 4: resistivity + per-edge BCs — Harris reconnection.
5. **Phase 5** (current): UI (sidebar tabs, presets, stats, probe).
6. Phase 6: LIC visualization.
7. Phase 7: polish (about, edu-content, JSON-LD, OG image, pointer perturbation).
8. Phase 8: parent-repo wiring.

## Phase 5 UI

`src/ui.js` (the entry point), `src/stats-display.js`, `src/probe.js`,
and `src/gpu/readback.js` mount Phase-5 UI on the live Sim. The HTML
shell is in `index.html` (topbar + sidebar with three tabs +
edu-content stub + JSON-LD stub). `setupUI(simShell)` is called from
`main.js` after `sim.init()`; it wraps `simShell.render` to drive the
stats readback at ~12 Hz cadence (every 5 frames at 256², every 10
at 512², every 20 at 1024²). The probe runs a 10 Hz `setInterval`
readback independent of the render loop.

### Readback pattern (`src/gpu/readback.js`)

`ReadbackPool` keeps a per-byte-size pool of staging buffers with
`MAP_READ | COPY_DST` usage. `readbackSlice(device, pool, src,
byteOffset, byteSize)` encodes one `copyBufferToBuffer` + submit +
`mapAsync` + `slice()` + `unmap()` cycle. `readbackBatch(...)` issues
N copies in one encoder + one submit, then awaits all maps in
parallel — used by stats-display to grab `(U0, U1, Bx, By, dt)` in
one round-trip and by probe to grab a 3-row stencil window.

Stats (~640 KB at 256², 12 Hz) and probe (~12 KB, 10 Hz) are both
small enough that we compute aggregates on the CPU rather than
building dedicated GPU reduction kernels. Phase 6 may move some
aggregates onto the GPU when LIC kernels land.

### sim.js public API (new in Phase 5)

`setPreset(name)`, `setBC(edge, mode)`, `setDrivenState(partial)`,
`setEta(eta)`, `setViewMode(mode)`, `setCFL(cfl)`, `setGamma(g)`,
`setPressureFloor(p)`, `setRunning(r)`, `step()` (single step),
`setSpeedScale(s)`, `setResolution(n)`, `saveState()` →
`loadState(s)` (JSON, parameters only; no buffer snapshot).

`setResolution(n)` re-instantiates `PlasmaBuffers` and
`PlasmaRenderer` at the new interior size, then reloads the current
preset. UI must call `stats.bindBuffers(sim.buffers)` and
`probe.bindBuffers(sim.buffers)` to re-aim the readback paths.

## Layout (current — Phase 4)

```
plasma/
├── index.html              ← canvas + no-WebGPU fallback
├── main.js                 ← entry: WebGPU init, frame loop, accumulator
├── styles.css              ← canvas + fallback layout (HUD lands later)
├── colors.js               ← _PALETTE extensions, frozen at startup
├── about.md                ← educational content (stub until Phase 7)
├── LICENSE                 ← AGPL-3.0
└── src/
    ├── config.js           ← grid size, CFL, γ, eta, ghost, BC enums, uniform layout
    ├── sim.js              ← step orchestrator: BC → PPM → HLLD → EMF → CT → resistivity
    ├── presets.js          ← Sod, Brio-Wu, Orszag-Tang, Harris current sheet
    ├── colormaps.js        ← viridis LUT
    └── gpu/
        ├── device.js       ← adapter+device init module
        ├── buffers.js      ← ghost-padded slots, BC uniforms, stage params
        ├── pipelines.js    ← compute + render pipeline factory
        ├── render.js       ← view-field → colormap → composite
        └── shaders/
            ├── shared-helpers.wgsl              ← Uniforms, BcUniforms, MHD prim/cons, indexing
            ├── apply-bcs.wgsl                   ← ghost-cell fill (4 modes × 4 edges)
            ├── reconstruct-ppm.wgsl             ← per-direction PPM (CW 1984)
            ├── riemann-hlld.wgsl                ← HLLD (M&K 2005) + HLLC + HLL fallbacks
            ├── compute-emf.wgsl                 ← Balsara-Spicer Ez at corners
            ├── update-conserved-weighted.wgsl   ← RK3 SSP weighted U update
            ├── update-b-weighted.wgsl           ← RK3 SSP weighted face-B update (CT)
            ├── apply-resistivity.wgsl           ← η ∇²B per stage (post-CT)
            ├── compute-dt.wgsl                  ← MHD CFL + parabolic resistive CFL
            ├── view-field.wgsl                  ← scalar extract (ρ, p, |v|, |B|, Jz)
            ├── colormap.wgsl                    ← LUT lookup (interior only)
            └── composite.wgsl                   ← canvas blit (interior-relative UVs)
```

## Numerical method (Phase 4)

| Piece               | Choice                                         |
|---------------------|------------------------------------------------|
| Reconstruction      | PPM (Colella & Woodward 1984)                  |
| Riemann solver      | HLLD (Miyoshi & Kusano 2005)                   |
| Time integration    | RK3 SSP (Gottlieb-Shu 1998)                    |
| Divergence-free B   | Constrained transport (Balsara-Spicer EMF)     |
| Boundaries          | Per-edge: periodic / outflow / reflecting / driven |
| Resistivity         | Explicit η ∇²B, central differences, post-CT  |

HLLD degenerate branches unchanged from Phase 3b.

## Ghost-cell convention (Phase 4)

Every field buffer expands by **2 ghost cells per side**. With interior
size `N×N`, storage sizes are:

| Buffer       | Shape                            |
|--------------|----------------------------------|
| `U0`, `U1`   | `(N+4) × (N+4)` cell-centered    |
| `Bx_face`    | `(N+5) × (N+4)` LEFT-face owner  |
| `By_face`    | `(N+4) × (N+5)` DOWN-face owner  |
| `Ez_edge`    | `(N+5) × (N+5)` corner-owner     |
| flux arrays  | `(N+4) × (N+4)` (cell-shaped)    |
| PPM edges    | `(N+4) × (N+4)` (cell-shaped)    |
| `field`, `colored` | `(N+4) × (N+4)`              |

Interior cell range: `[ghost, ghost+N) = [2, N+2)` in both axes.
Boundaries:

| Region        | Index range (1D)          |
|---------------|---------------------------|
| W (left) ghost   | `[0, 2)`               |
| Interior         | `[2, N+2)`             |
| E (right) ghost  | `[N+2, N+4)`           |
| S (bottom) ghost | `[0, 2)`  along j      |
| N (top) ghost    | `[N+2, N+4)`  along j  |

**Face ownership**: `Bx_face[i, j]` sits on the LEFT face of cell `(i, j)`
at `x = (i-ghost)·dx`. `By_face[i, j]` sits on the BOTTOM face of cell
`(i, j)` at `y = (j-ghost)·dx`. `Ez_edge[i, j]` sits at the BOTTOM-LEFT
corner of cell `(i, j)`.

`apply-bcs.wgsl` is the single shader responsible for ghost-cell fill.
It dispatches over `(N_total+1)²` and per-invocation routes the index
to the appropriate strip:

* **Cell-centered** (`(N_total)² range`): per-cell, route by which edge
  strip the index lies in.
* **Bx_face** (`(N_total+1) × N_total`): the W boundary face at
  `i = ghost` and E boundary face at `i = ghost + N` are owned by the
  shader for reflecting/driven BCs (set to 0 / driven_bx); ghost faces
  outside the wall are filled from interior with sign-flip for
  reflecting.
* **By_face** (`N_total × (N_total+1)`): symmetric.

**Corner rule**: ghost cells in the corner strips are owned by the
first NON-PERIODIC of the two adjacent edges. If both periodic, falls
through to the horizontal-edge wrap (any choice is consistent because
the two wraps reach the same opposite-corner interior cell).

**Reflecting BC sign flips**:
* West/East wall (normal = x): `v_x` and `B_x` flip. `v_y, v_z, B_y, B_z`,
  `ρ`, `p`, `E` unchanged. Boundary x-face Bx = 0 (no normal field).
* South/North wall (normal = y): `v_y` and `B_y` flip. Boundary y-face
  By = 0.
* Corner reflecting (both walls): mirror in BOTH axes; flip both
  `v_x` and `v_y`.

## Pipeline dispatch shapes (per RK3 stage)

With `N = 256`, `ghost = 2`, `N_total = 260`, and workgroup 8×8:

| Pass                | Logical extent         | Workgroup count (256²) |
|---------------------|------------------------|------------------------|
| apply-bcs           | `(N_total+1)²`         | 33² (covers all of B/U)|
| reconstruct-ppm     | `(N+2)²`               | 33²                    |
| riemann-hlld (x)    | `(N+1) × (N+2)`        | 33 × 33                |
| riemann-hlld (y)    | `(N+2) × (N+1)`        | 33 × 33                |
| compute-emf         | `(N+1)²`               | 33²                    |
| update-conserved    | `N²`                   | 32²                    |
| update-b            | `(N+1)²`               | 33²                    |
| apply-resistivity   | `(N_total+1)²`         | 33²                    |
| compute-dt (reduce) | `N²`                   | 32²                    |

PPM at the outer dispatch cells (`i = ghost-1` and `i = ghost+N`) lacks
the full 5-point stencil; the kernel detects this (`stencil_ok` check)
and falls back to piecewise-constant edges (edge_l = edge_r = q_c).
Downstream Riemann then uses these lower-order edges at the boundary
faces — acceptable because the BC-filled ghost is itself lower-order.

## RK3 SSP scheme

    U(1)   = U(n) + dt · L(U(n))
    U(2)   = (3/4)U(n) + (1/4)U(1) + (1/4)dt · L(U(1))
    U(n+1) = (1/3)U(n) + (2/3)U(2) + (2/3)dt · L(U(2))

`dt` is computed once at the start of the step (from U(n)) and reused
across all three stages — required for SSP. Resistive CFL is folded into
`compute-dt` via `dt_res = 0.5 · dx² / η`.

Per stage we run (in order):
1. `apply-bcs`              — fill ghosts from BC config
2. `reconstruct-ppm` (x, y)
3. `riemann-hlld` (x, y)
4. `compute-emf`
5. `update-conserved-weighted`
6. `update-b-weighted`
7. `apply-resistivity`      — η ∇²B linear step, SSP-compatible

## Boundary conditions

Per-edge BC mode IDs (config.js):

| ID | Name          | Behavior                                       |
|----|---------------|------------------------------------------------|
| 0  | `BC_PERIODIC`  | Wrap from opposite edge.                       |
| 1  | `BC_OUTFLOW`   | Zero-gradient (copy from nearest interior).    |
| 2  | `BC_REFLECTING`| Perfectly conducting wall: mirror with v_n/B_n flip; boundary B_n = 0. |
| 3  | `BC_DRIVEN`    | Apply user-configured inflow state from `bc_uniforms`. |

Per-edge slot in `bc_uniforms` (storage buffer):

| Field        | Edge  |
|--------------|-------|
| `mode_n` (u32) | N (top, `j ∈ [ghost+N, N_total)`)         |
| `mode_s` (u32) | S (bottom, `j ∈ [0, ghost)`)              |
| `mode_e` (u32) | E (right, `i ∈ [ghost+N, N_total)`)       |
| `mode_w` (u32) | W (left, `i ∈ [0, ghost)`)                |

Driven state (primitive, single global; UI Phase 5 may add per-edge):
8 f32s `(driven_rho, driven_vx, driven_vy, driven_vz, driven_bx,
driven_by, driven_bz, driven_p)`. Converted to conservative form
on write by the BC shader via `prim_to_cons_pair`.

## Harris current sheet preset

Canonical reconnection IC. Spec values:

| Quantity      | Profile                                           |
|---------------|---------------------------------------------------|
| `Bx(y)`       | `B_0 · tanh(y / a)`,  `B_0 = 1`, `a = 0.1`        |
| `By`, `Bz`    | 0                                                 |
| `ρ(y)`        | `ρ_∞ + ρ_0 · sech²(y/a)`,  `ρ_∞ = 0.2`, `ρ_0 = 1`|
| `p(y)`        | `p_∞ + ½ B_0² · sech²(y/a)`, `p_∞ = 0.1`          |
| `v` perturb.  | `vy = 0.01 · sin(π·x) · sech²(y/a)`, others 0    |
| Domain        | `[-1, 1] × [-1, 1]` square (sheet centered at y=0)|
| γ             | 5/3                                               |
| η             | `1e-3` (default)                                  |
| BCs           | E=W=periodic, N=S=outflow                         |

Pressure balance: `p + ½|B|² = p_∞ + ½ B_0²` is constant across y.

Verification:
* `y = 0`: `Bx = 0`, `ρ = 1.2`, `p = 0.6`.
* `y = a`: `Bx ≈ 0.762`, `ρ ≈ 0.62`, `p ≈ 0.42`.

Expected behavior: small velocity perturbation seeds tearing; reconnection
develops around `t ≈ 10 t_A`; plasmoids form along the sheet. Sweet-Parker
scaling `δ/L ∝ S^(-1/2)` with `S = L · v_A / η` should be visible.

## Bind-group layout (transpiler-friendly contract)

All compute pipelines use one bind group (group 0). Layouts are static,
declared up front in `pipelines.js`, and documented per-shader in each
shader's header comment.

* No dynamic offsets.
* No push constants.
* No subgroup ops, no shared-memory tricks beyond compute-dt's
  workgroup tile-max reduction.
* Atomics confined to compute-dt.
* `Uniforms` struct held in TWO buffers (`uniform_x`, `uniform_y`)
  pre-written at preset load.
* Stage weights live in THREE small uniform buffers (`stage_1`,
  `stage_2`, `stage_3`).
* `bc_uniforms` is ONE storage buffer (read-only inside the BC shader)
  holding 4 edge mode IDs + 8 floats of driven state.

The mental model: every compute dispatch maps to a clean nested loop in
JS over the workgroup grid. The BC shader's `switch`-on-mode is one
extra read per invocation; trivially CPU-emulated.

## Shared-module dependencies

Phase 1-4 only touch `shared-tokens.js` and `shared-base.css`. Phase 5
will pull in `shared-toolbar.js`, `shared-forms.js`, `shared-dropdown.js`,
`shared-settings.js`, `shared-about.js`, `shared-tabs.js`,
`shared-icons.js`, `shared-shortcuts.js`, `shared-info.js`,
`shared-tooltip.js`, `shared-sparkline.js`, `shared-touch.js`,
`shared-haptics.js`.

`shared-camera.js` is intentionally NOT used.

## Rules

- **No innerHTML assignments.** Use `textContent` or `createElement`.
- Always prefer shared modules over re-implementations.
- WebGPU-only. No CPU fallback path. If WebGPU init fails, show
  `#no-webgpu` and bail.
- `shared-tokens.js` loads as a synchronous `<script>` (no `defer`).

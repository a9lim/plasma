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

Build phases:
1. **Phase 1** (this scaffold): WebGPU init, fullscreen-quad render, frame loop, visibilitychange pause.
2. Phase 2: pure hydro (Euler) with HLL + PLM + FE — Sod tube.
3. Phase 3: full MHD with HLLD + PPM + RK3 + CT — Orszag-Tang.
4. Phase 4: resistivity + per-edge BCs — Harris reconnection.
5. Phase 5: UI (sidebar tabs, presets, stats, probe).
6. Phase 6: LIC visualization.
7. Phase 7: polish (about, edu-content, JSON-LD, OG image, pointer perturbation).
8. Phase 8: parent-repo wiring.

## Layout (current — Phase 3b, grows over time)

```
plasma/
├── index.html              ← canvas + no-WebGPU fallback
├── main.js                 ← entry: WebGPU init, frame loop, accumulator
├── styles.css              ← canvas + fallback layout (HUD lands later)
├── colors.js               ← _PALETTE extensions, frozen at startup
├── about.md                ← educational content (stub until Phase 7)
├── LICENSE                 ← AGPL-3.0
└── src/
    ├── config.js           ← grid size, CFL, γ, view-mode enum, uniform layout
    ├── sim.js              ← step orchestrator (2 submits/step: compute_dt, RK3-chain)
    ├── presets.js          ← Sod (hydro), Brio-Wu (MHD shock tube), Orszag-Tang
    ├── colormaps.js        ← viridis LUT
    └── gpu/
        ├── device.js       ← adapter+device init module
        ├── buffers.js      ← 3-slot RK3 storage (U_n/U_1/U_2), face-B per slot, edge-Ez,
        │                     8 PPM edge buffers, stage_params uniform buffers
        ├── pipelines.js    ← compute + render pipeline factory
        ├── render.js       ← view-field → colormap → composite
        └── shaders/
            ├── shared-helpers.wgsl              ← MHD prim/cons, fast mag speed, face conv
            ├── clear.wgsl                      ← Phase-1 placeholder, currently unused
            ├── reconstruct-ppm.wgsl            ← per-direction PPM (CW 1984) — L/R edge states
            ├── riemann-hlld.wgsl               ← HLLD (M&K 2005) + HLLC + HLL fallbacks
            ├── compute-emf.wgsl                ← Balsara-Spicer Ez at corners
            ├── update-conserved-weighted.wgsl  ← RK3 SSP weighted U update
            ├── update-b-weighted.wgsl          ← RK3 SSP weighted face-B update (CT)
            ├── compute-dt.wgsl                 ← MHD CFL via fast magnetosonic
            ├── view-field.wgsl                 ← scalar extract (ρ, p, |v|, |B|, Jz)
            ├── colormap.wgsl                   ← LUT lookup
            └── composite.wgsl                  ← canvas blit
```

Resistivity and per-edge BCs in Phase 4.

## Numerical method (Phase 3b)

| Piece               | Choice                                         |
|---------------------|------------------------------------------------|
| Reconstruction      | PPM (Colella & Woodward 1984)                  |
| Riemann solver      | HLLD (Miyoshi & Kusano 2005)                   |
| Time integration    | RK3 SSP (Gottlieb-Shu 1998)                    |
| Divergence-free B   | Constrained transport (Balsara-Spicer EMF)     |
| Boundaries          | Periodic only (per-edge in Phase 4)            |
| Resistivity         | η = 0 (added in Phase 4)                       |

HLLD degenerate branches (all fall back to simpler solvers):
1. **Branch A** (`Bx² < ε² · ρ`, ε² = 1e-24): Alfvén waves degenerate → HLLC.
2. **Branch B** (`SR - SL < tol · (|SR| + |SL|)`, tol = 1e-8): wave-speed
   coincidence → HLL.
3. **Branch C** (star-state total pressure ≤ PRESSURE_FLOOR): negative
   pressure → HLL.

## RK3 SSP scheme

    U(1)   = U(n) + dt · L(U(n))
    U(2)   = (3/4)U(n) + (1/4)U(1) + (1/4)dt · L(U(1))
    U(n+1) = (1/3)U(n) + (2/3)U(2) + (2/3)dt · L(U(2))

`dt` is computed once at the start of the step (from U(n)) and reused
across all three stages — required for SSP. Applied in lockstep to
U_cell, Bx_face, By_face.

Storage: three slots (`U_n`, `U_1`, `U_2`) for cell-centered conserved
state + face-centered transverse B. Stage 3 writes back to slot N, so
no buffer swap is needed at end of step.

## Bind-group layout (transpiler-friendly contract)

All compute pipelines use one bind group (group 0). Layouts are static,
declared up front in `pipelines.js`, and documented per-shader in each
shader's header comment.

* No dynamic offsets.
* No push constants (not in WebGPU regardless).
* No subgroup ops, no shared-memory tricks beyond compute-dt's
  workgroup tile-max reduction (a textbook pattern; trivially maps to
  a per-workgroup loop on CPU).
* Atomics confined to compute-dt (`atomic<u32>` over float bits via
  `bitcast<u32>`).
* `Uniforms` struct (with `sweep_dir`) is held in TWO buffers
  (`uniform_x`, `uniform_y`) pre-written at preset load — passes that
  want sweep_dir=0 bind `uniform_x`, sweep_dir=1 bind `uniform_y`.
* Stage weights live in THREE small uniform buffers (`stage_1`,
  `stage_2`, `stage_3`) written once at init. Each holds
  `(a0, a1, dt_w, _pad)` for that stage's linear combination.

The mental model is: every compute dispatch maps to a clean nested
loop in JS over the workgroup grid, reading inputs from bound storage
buffers and writing outputs. A future WebGPU→CPU transpiler can drop
each shader into that loop pattern without architectural changes.

## Shared-module dependencies

Phase 1 only touches `shared-tokens.js` (synchronous load — injects CSS custom
properties before CSS parses) and `shared-base.css` (inherits the canvas
defaults). Later phases will pull in `shared-toolbar.js`, `shared-forms.js`,
`shared-dropdown.js`, `shared-settings.js`, `shared-about.js`, `shared-tabs.js`,
`shared-icons.js`, `shared-shortcuts.js`, `shared-info.js`, `shared-tooltip.js`,
`shared-sparkline.js`, `shared-touch.js`, `shared-haptics.js`.

`shared-camera.js` is intentionally NOT used — the sim is a fixed-orthographic
2D grid view, no panning/zooming.

## Rules

- **No innerHTML assignments.** Use `textContent` or `createElement`. The
  parent-repo Write hook blocks innerHTML in new files.
- Always prefer shared modules over re-implementations. Check `shared-*.js`
  before adding utility code.
- WebGPU-only. No CPU fallback path. If WebGPU init fails, show
  `#no-webgpu` and bail.
- `shared-tokens.js` loads as a synchronous `<script>` (no `defer`) so its
  injected CSS variables are available before `shared-base.css` parses.
  Everything else can defer.

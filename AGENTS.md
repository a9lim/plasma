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

## Layout (current — Phase 3a, grows over time)

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
    ├── sim.js              ← step orchestrator (4 submits/step in 3a)
    ├── presets.js          ← Sod (hydro) + Brio-Wu (MHD) ICs
    ├── colormaps.js        ← viridis LUT
    └── gpu/
        ├── device.js       ← adapter+device init module
        ├── buffers.js      ← cell-state ping-pong, face-B ping-pong, edge-Ez
        ├── pipelines.js    ← compute + render pipeline factory
        ├── render.js       ← view-field → colormap → composite
        └── shaders/
            ├── shared-helpers.wgsl   ← MHD prim/cons, fast mag speed, face conv
            ├── clear.wgsl            ← Phase-1 placeholder, currently unused
            ├── reconstruct-plm.wgsl  ← per-direction PLM on MHD primitives
            ├── riemann-hll.wgsl      ← MHD HLL (fast magnetosonic wave speed)
            ├── compute-emf.wgsl      ← Balsara-Spicer Ez at corners
            ├── update-conserved.wgsl ← unsplit U += -(∇·F)_x - (∇·F)_y
            ├── update-b.wgsl         ← CT: face-B += -∇×E
            ├── compute-dt.wgsl       ← MHD CFL via fast magnetosonic
            ├── view-field.wgsl       ← scalar extract (ρ, p, |v|, |B|, Jz)
            ├── colormap.wgsl         ← LUT lookup
            └── composite.wgsl        ← canvas blit
```

HLLD + PPM + RK3 SSP arrive in Phase 3b; resistivity and per-edge BCs in Phase 4.

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

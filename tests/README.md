# plasma/tests

Convergence and correctness tests for the plasma MHD engine. WebGPU-only
(every test instantiates the production `Sim`); run them in a browser.

## physics-validation.html — source-physics validation matrix

Regression gate for the full production engine, including the extended source
stack. It runs finite-state smoke checks for Sod, Brio-Wu, Orszag-Tang,
Harris, and the driven wind/cloud preset; a per-edge driven-boundary fill
check; then source-response checks for SD93-backed microphysics/opacity table
shape, source substep rate sizing, cooling, anisotropic conduction, Hall
whistler dispersion, grey radiation energy exchange, electron-inertia
current smoothing, Jeans growth, isolated gravity potential, Cartesian
Poisson multigrid convergence, cylindrical r-weighted expansion, cylindrical static
pressure balance, cylindrical conduction balance, cylindrical CT
magnetic-divergence preservation, and cylindrical Poisson residual
convergence. Each
row reads back the actual GPU state and reports density/pressure ranges,
divergence, and a case-specific physical metric.

### Run it

```bash
# from plasma/
python -m http.server
# -> http://localhost:8000/tests/physics-validation.html
```

```bash
# from a9lim.github.io/ root
./dev.sh
# -> http://localhost:8787/plasma/tests/physics-validation.html
```

The page also supports `?auto=1`; it prints a
`PHYSICS_VALIDATION_JSON_BEGIN/END` payload to the browser console so a
Playwright wrapper can turn it into a CI-style gate. This matrix is deliberately
broader than the smooth convergence tests below: it is meant to catch source
term sign errors, stale-ghost regressions, divergence leaks, and boundary/source
interactions before a visual preset looks obviously wrong.

The companion driver can run it as a CI-style browser gate:

```bash
# from plasma/, with a parent static server already serving a9lim.github.io/
python3 tests/physics-validation.py --port 8090 --timeout 180 \
  --out /tmp/plasma-physics-validation.json
```

## alfven-convergence.html — circularly polarized Alfvén wave (CPAW)

Canonical MHD convergence test (Tóth 2000; Stone+ 2008 §4.2). Smooth,
periodic, exact analytic solution: the IC translates rigidly along the
wave vector at the Alfvén speed `v_A` and returns to itself every
period `T = λ/v_A`. Sweeps `N ∈ {32, 64, 128, 256}`, runs each to one
period, measures the L1 error vs the analytic solution (= the IC,
modulo a period), and reports per-pair and least-squares convergence
slopes.

### Run it

The test page imports the production engine modules via ES module
paths, so it needs a static server that serves them with correct MIME
types. Two options:

```bash
# from plasma/ — fastest, no CSP overhead
python -m http.server
# → http://localhost:8000/tests/alfven-convergence.html
```

```bash
# from a9lim.github.io/ root — full Worker behavior, matches production
./dev.sh
# → http://localhost:8787/plasma/tests/alfven-convergence.html
```

Open the page in a WebGPU-capable browser (Chrome 113+, Edge 113+,
Safari TP 184+) and click **Run sweep**. Each resolution streams a
progress line to the log as it finishes. `N=256` takes the longest
(~10–60 s depending on the GPU).

### Interpreting the slope

The engine stack is PPM (characteristic-variable limited + Mignone
safety-net) + HLLD + Gardiner-Stone upwind CT + RK3 SSP. Expected:

| Slope range | Interpretation |
|-------------|----------------|
| `≈ 3.0` (textbook) | RK3 SSP + characteristic PPM working as designed. Limiter is not clipping smooth extrema. |
| `≈ 2.0` | Limiter is clipping smooth extrema. Motivates the **PPM4** (McCorquodale-Colella 2011) extremum-preserving upgrade — drop-in replacement for the CW1984 monotonicity check. |
| `≈ 1.0` | Something else dominates: BC leak, fp32 noise floor at small dx, or a hot-path bug. Investigate. |
| `< 1.0` | Diverging — almost certainly a bug. Cross-check by static-tracing the encoder pipeline. |

The test only exercises **ideal MHD** (`η = 0`). Resistive convergence
(adding `η ∇²B` to the truth) is a follow-up — see Stone+ 2008 §4.5
for the standard reference (a steady-state magnetic-field diffusion
test or a forced reconnection-rate study).

### Math derivation

Domain `[0, 1]²`, all-periodic BCs, `γ = 5/3`. Wave vector at angle
`α = atan(2)`, so `cos α = 1/√5`, `sin α = 2/√5`. With phase variable
`φ(x, y) = 2π(x + 2y)`, the implicit wavelength along `k̂` is
`λ = 1/√5` and exactly one wavelength fits along `k̂` between `(0,0)`
and `(1, 2)`. Background `ρ = 1`, `p = 0.1`, `B = B_∥ k̂` with
`B_∥ = 1`. Perturbation amplitude `A = 0.1`.

In the `(∥, ⊥1, ⊥2)` basis where `⊥2 = ẑ`, the wave is

```
δB_⊥1 = A sin φ          δv_⊥1 = -A sin φ / √ρ   (Alfvén polarization, +k̂)
δB_⊥2 = A cos φ          δv_⊥2 = -A cos φ / √ρ
```

Rotated to the lab frame:

```
B_x = cos α - sin α · A sin φ
B_y = sin α + cos α · A sin φ
B_z = A cos φ
v_x = +sin α · A sin φ
v_y = -cos α · A sin φ
v_z = -A cos φ
```

Check: `∇·B = -sin α · A cos φ · 2π + cos α · A cos φ · 4π = 0` (since
`-sin α + 2 cos α = -2/√5 + 2/√5 = 0`). The Alfvén speed
`v_A = B_∥/√ρ = 1`, so the period is `T = λ/v_A = 1/√5 ≈ 0.4472136`.

**Face B from a vector potential.** To kill the IC's contribution to
`∇·B` at the fp32 noise floor, the preset derives face B from the
analytic vector potential

```
A_z(x, y) = y cos α - x sin α + (A / (2π √5)) · cos φ
```

via the discrete curl at face locations:

```
Bx_face[i, j] = ( A_z(x_face, y_top) - A_z(x_face, y_bot) ) / dx
By_face[i, j] = -( A_z(x_right, y_face) - A_z(x_left, y_face) ) / dx
```

Cell-centered `B_x`, `B_y`, `B_z` use the analytic expressions
directly. The CT update preserves `∇·B` at the discrete level over
the run.

### Implementation notes

- The test bypasses `sim.setResolution()`'s `{256, 512, 1024}` guard
  by setting `sim.n` and re-instantiating `PlasmaBuffers` directly.
  This lets us sweep `N ∈ {32, 64, 128, 256}` without touching the
  production sim's resolution policy.
- `sim.step()` does not advance `sim.simTime` — the test reads the
  per-step `dt` back from `sim.buffers.dt` after each step and
  accumulates simTime manually. This matches what
  `src/stats-display.js` does at 12 Hz to display the wall clock.
- The final step may overshoot `T` by up to one CFL `dt`. The
  resulting per-cell bias is `O(dx)` at most when CFL saturates — it
  washes out at high `N` and shouldn't dominate the slope. If you
  need a cleaner test, add a "cap dt at T - simTime" path inside
  `sim.step()` (currently not exposed).
- L1 is averaged per cell per conserved component over 8 components:
  `(ρ, ρv_x, ρv_y, ρv_z, E, B_x, B_y, B_z)`. Face B contributes via
  face averages at each cell center to match the engine's
  `cons_to_prim_mhd` contract.

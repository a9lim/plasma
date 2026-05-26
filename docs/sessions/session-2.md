# Session 2 — Verification + engine bug fixes

First live verification attempt hit a WebGPU storage-buffer-limit
error, then a series of cascading NaN problems that turned out to be
**five distinct bugs** of varying severity. All now fixed. Documenting
because the lessons matter:

## 1. Storage buffer limit overflow

`update-conserved-weighted` had 11 storage bindings; the M-series
adapter caps at 10. Fix: bump `maxStorageBuffersPerShaderStage` to 10
in `device.js` (matches geon's pattern) + flip `dt_buf` from storage
to uniform binding in `update-conserved-weighted.wgsl` (it's a single
f32, naturally a uniform). The shader now wraps it in a `DtUniform`
struct since uniforms can't be raw arrays in WGSL.

**Lesson**: count storage bindings per pipeline against the adapter's
`maxStorageBuffersPerShaderStage` (typically 10 on desktop, 8 baseline)
at build time, not at first dispatch. Per-pipeline budget is tight.

## 2. dt buffer missing COPY_SRC (latent pre-existing bug)

The stats-display readback path was silently returning zero for dt
because the `plasma.dt` buffer didn't have `COPY_SRC` usage. WebGPU
validation errored on the copy but the rest of the batch went through,
so the symptom was just "dt always reads as 0." Fix: add `COPY_SRC` to
the dt buffer's usage flags in `buffers.js`.

**This is probably why HANDOFF reported the `simTime` ratchet was
broken** — simTime itself is fine via the stats-display workaround,
but that workaround was reading zero. The Phase 6 agent's fix #1
should be reassessed against this.

## 3. apply-resistivity race condition

The shader did an in-place 5-point Laplacian on `read_write` storage
and asserted (in a confidently-wrong comment) that neighbor reads
return pre-dispatch values. **WebGPU has NO such guarantee.** Neighbor
reads at workgroup-tile boundaries pick up post-write values from
already-executed tiles. Manifested as regular-spacing "blebs" of
artifact J_z at high η (8×8 workgroups → ~8-cell artifact stride);
below noise floor at low η, undetected by static review. Fix: double-
buffer via a `snapshot` entry point that copies dst→snap (per-cell,
no neighbors, race-free), then `main` reads snap and writes dst. Added
3 snapshot buffers (`Bx_res_snap` / `By_res_snap` / `U1_res_snap`).
BGL grew to 9 bindings (7 storage, under cap).

**Lesson**: in-place RMW + neighbor reads on storage buffers is
ALWAYS a race in WebGPU, even when it appears to work on most hardware.
The shader's old "we rely on this" comment was the smoking gun — that
phrase should be a code smell. HANDOFF flagged this for CPU emulation
but the same constraint applies to GPU dispatch.

## 4. HLLD_BX_EPS2 too conservative

Was 1e-24 — basically "exactly machine zero." At thin current sheets
with |Bn|~1e-5, the full HLLD 5-wave path runs with tiny `bn²` and
tiny `g_L = ρ(S-u)² - bn²` denominators. The 1e-20 `safeDL` guard
inflates `bt_Ls = bt·g_L/safeDL` to ~1e20 → NaN cascade. Bumped to
1e-10: falls back to HLLC whenever |Bn| < ~1e-5·√ρ — robust at near-
degeneracies, no visible effect on bulk physics. HANDOFF explicitly
flagged this as conservative; following its own advice.

## 5. No defensive sanitization on conserved state

Once any cell went non-finite for any reason, it poisoned compute-dt's
wavespeed atomicMax reduction (NaN bits → huge u32 → corrupt
wavespeed), which made dt useless, which cascaded NaN to the whole
field within ~5 steps. Added IEEE-clean sanitization at the end of
`update-conserved-weighted.wgsl`:
- `clamp(ρ, FLOOR, 1e30)` — NaN → FLOOR via IEEE maxNum semantics
- `select(0, m, m == m)` for momentum — NaN → 0
- `clamp(E, KE + p_floor/(γ−1), 1e30)` — NaN → minimum p≥floor value
- `select(0, Bz, Bz == Bz)` — NaN → 0

This is the breaker that finally let OT survive indefinitely. The
sanitization is conservative-state-only (no access to Bx_face/By_face
here), so the magnetic-pressure contribution to the p-floor check is
omitted — downstream `cons_to_prim`'s pressure floor catches the slop.

**Lesson**: any solver with an atomicMax wavespeed reduce needs
conserved-state sanitization at the write site. Otherwise a single
bad cell → bad dt → cascade. This is true regardless of which Riemann
solver / reconstruction / time integrator you choose.

## Bonus: η floor mechanism (kept as latent infrastructure)

Built `sim.getEtaMin()` returning `etaFloorCoeff · dx` (grid magnetic
Reynolds criterion η_min ≳ C·v_char·dx), with slider dynamic-min,
dynamic hint text, refresh on preset/resolution change. Calibrated
empirically against OT critical η: N=256 ≈ 8e-4, N=1024 ≈ 1e-4. The
empirical C·v_char product **scales super-linearly with N** (OT
concentrates energy faster at finer grids), so a single coefficient
is wrong somewhere. After fix #5 the floor was unnecessary — sim
survives gracefully at η=0 thanks to sanitization. OT's preset sets
`etaFloorCoeff: 0`. Mechanism kept available for future presets that
genuinely need it.

## What this means for HANDOFF's prior assumptions

- "Most likely failure modes" 1–4 from Phase 7 Step 1: #1 was real
  (HLLD eps), #2 (CT indexing) was NOT triggered, #3 (uniform field
  writes) was NOT triggered, #4 (flux convention mismatch) was NOT
  triggered. Whole new bug class found: WebGPU race conditions in
  same-buffer RMW kernels.
- Phase 6 agent's `simTime` ratchet flag: the ratchet itself is still
  technically broken (Sim.step doesn't read dt back) but its
  observable consequence was being driven by bug #2 above. Verify by
  watching the stats panel's dt value.
- "Resolution change is destructive" note: still applies, working as
  documented.
- HLLD_BX_EPS2 note: nudged upward as recommended (1e-24 → 1e-10).

## Slider widening

For calibration we widened the η slider's HTML `max` from `-1`
(η=0.1) to `0` (η=1.0). Left it widened — gives users more range for
experimentation, no downside.


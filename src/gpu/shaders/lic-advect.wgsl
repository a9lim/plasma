// ─── lic-advect.wgsl ─────────────────────────────────────────────────
// Animated line integral convolution along the magnetic field.
//
// One invocation per interior cell. For each cell we:
//   1. Initialize trace position p = (i + 0.5, j + 0.5) in cell-coords.
//   2. Backward-trace LIC_STEPS via RK2 along the unit B-field vector
//      with step size LIC_STEP_SIZE cells.
//   3. At each visited position, sample the 1024² white-noise base with
//      a phase offset of (lic_phase * drift_x, lic_phase * drift_y) and
//      bilinear interpolation. Accumulate into running sum.
//   4. Write average noise value to lic_out[interior cell].
//
// Output is in [0, 1] (the noise sampler returns floats in [0, 1) and
// averaging preserves the range).
//
// Sampling B at a continuous cell position uses bilinear interpolation
// between the four surrounding face-centered values. Bx lives on
// vertical (x-normal) faces, By on horizontal (y-normal) faces. Face
// ownership convention (LEFT/DOWN owner, see shared-helpers):
//   Bx_face[i,j] = B_x at x = (i - ghost - 0.5) * dx, y = (j - ghost) * dx
//   By_face[i,j] = B_y at x = (i - ghost) * dx,        y = (j - ghost - 0.5) * dx
//
// For internal LIC purposes we work in CELL coordinates (one unit per
// cell-width), not physical units — the trace is direction-only and
// scale-invariant. We unit-normalize the (Bx, By) vector each step.
//
// ── Bind-group layout (transpiler contract) ────────────────────────
//   group(0) binding(0): Uniforms                    (uniform)
//                        — uses ghost_w, grid_n, grid_n_total, noise_n.
//   group(0) binding(1): Bx_face                     (storage, read)
//   group(0) binding(2): By_face                     (storage, read)
//   group(0) binding(3): noise (1024² f32)           (storage, read)
//   group(0) binding(4): lic_out (interior-padded f32) (storage, read_write)
//   group(0) binding(5): LicUniforms                 (uniform)
//                        — render-pace phase + drift (rewritten per frame).
//
// ── Transpiler audit ───────────────────────────────────────────────
//   • Vanilla compute entry point; one workgroup_size(8,8,1).
//   • NO workgroupBarrier() calls — no shared memory, no reductions.
//   • NO textures or samplers — noise is a storage buffer of f32.
//   • NO matrix types — only vec2<f32> for positions/velocities.
//   • NO bitcasts.
//   • NO atomics.
//   • Conditional logic uses plain `if` (no `switch`).
//   • Loop is a plain `for` over a u32 counter; no nested barriers.
//
// All constants below mirror config.js values (LIC_STEPS = 20,
// LIC_STEP_SIZE = 0.5, LIC_B_EPS = 1e-8).

const LIC_STEPS:     u32 = 20u;
const LIC_STEP_SIZE: f32 = 0.5;
const LIC_B_EPS:     f32 = 1.0e-8;

@group(0) @binding(0) var<uniform> U_uniforms: Uniforms;
@group(0) @binding(1) var<storage, read>       Bx_face: array<f32>;
@group(0) @binding(2) var<storage, read>       By_face: array<f32>;
@group(0) @binding(3) var<storage, read>       noise:   array<f32>;
@group(0) @binding(4) var<storage, read_write> lic_out: array<f32>;
@group(0) @binding(5) var<uniform>             lic_u:   LicUniforms;

// Sample the noise buffer with bilinear interpolation at fractional
// position (px, py), wrapping into [0, noise_n). Cheap modulo wrap.
fn sample_noise(px: f32, py: f32, noise_n: u32) -> f32 {
    let N    = f32(noise_n);
    // Positive-wrap modulo for negative inputs. `xw`, `yw` end up in
    // [0, N) mathematically; clamp defensively to keep u32() safe.
    let xw = clamp(px - floor(px / N) * N, 0.0, N);
    let yw = clamp(py - floor(py / N) * N, 0.0, N);
    let x0 = u32(floor(xw)) % noise_n;
    let y0 = u32(floor(yw)) % noise_n;
    let x1 = (x0 + 1u) % noise_n;
    let y1 = (y0 + 1u) % noise_n;
    let fx = xw - floor(xw);
    let fy = yw - floor(yw);
    let n00 = noise[y0 * noise_n + x0];
    let n10 = noise[y0 * noise_n + x1];
    let n01 = noise[y1 * noise_n + x0];
    let n11 = noise[y1 * noise_n + x1];
    let nx0 = mix(n00, n10, fx);
    let nx1 = mix(n01, n11, fx);
    return mix(nx0, nx1, fy);
}

// Cell-centered Bx for interior cell (icx, icy) — ghost-padded face read.
fn bx_at_cell(icx: u32, icy: u32, ghost: u32, n_total: u32) -> f32 {
    let ix = icx + ghost;
    let iy = icy + ghost;
    return 0.5 * (Bx_face[bx_face_left_idx(ix, iy, n_total)]
                + Bx_face[bx_face_right_idx(ix, iy, n_total)]);
}

// Cell-centered By for interior cell (icx, icy).
fn by_at_cell(icx: u32, icy: u32, ghost: u32, n_total: u32) -> f32 {
    let ix = icx + ghost;
    let iy = icy + ghost;
    return 0.5 * (By_face[by_face_down_idx(ix, iy, n_total)]
                + By_face[by_face_up_idx(ix, iy, n_total)]);
}

// Sample (Bx, By) at a continuous cell-centered position (cx, cy),
// where cx ∈ [0, n_interior) is in interior-cell coords (NOT
// ghost-padded). Returns the unit B-vector, or vec2(0,0) if |B| is
// below LIC_B_EPS (signaling "stop tracing").
//
// Bilinear interpolation between the four nearest cell-centered
// Bx,By values. Cell (i,j) has cell-centered Bx = 0.5*(Bx[i] + Bx[i+1]),
// where the indices are ghost-padded.
fn sample_b_unit(cx: f32, cy: f32, ghost: u32, n_total: u32, n_interior: u32) -> vec2<f32> {
    // Clamp to interior to avoid out-of-bound face lookups. Ghost cells
    // hold BC-filled B values, which would propagate trace direction
    // unphysically if we let the trace step into them.
    let ni = f32(n_interior);
    let cxc = clamp(cx, 0.0, ni - 1.0001);
    let cyc = clamp(cy, 0.0, ni - 1.0001);

    let ix0 = u32(floor(cxc));
    let iy0 = u32(floor(cyc));
    let ix1 = ix0 + 1u;
    let iy1 = iy0 + 1u;
    let fx = cxc - floor(cxc);
    let fy = cyc - floor(cyc);

    // Cell-centered Bx/By at the 4 surrounding interior cells.
    let bx00 = bx_at_cell(ix0, iy0, ghost, n_total);
    let bx10 = bx_at_cell(ix1, iy0, ghost, n_total);
    let bx01 = bx_at_cell(ix0, iy1, ghost, n_total);
    let bx11 = bx_at_cell(ix1, iy1, ghost, n_total);
    let by00 = by_at_cell(ix0, iy0, ghost, n_total);
    let by10 = by_at_cell(ix1, iy0, ghost, n_total);
    let by01 = by_at_cell(ix0, iy1, ghost, n_total);
    let by11 = by_at_cell(ix1, iy1, ghost, n_total);

    let bx0 = mix(bx00, bx10, fx);
    let bx1 = mix(bx01, bx11, fx);
    let bx  = mix(bx0,  bx1,  fy);
    let by0 = mix(by00, by10, fx);
    let by1 = mix(by01, by11, fx);
    let by  = mix(by0,  by1,  fy);

    let mag = sqrt(bx * bx + by * by);
    if (mag < LIC_B_EPS) {
        return vec2<f32>(0.0, 0.0);
    }
    return vec2<f32>(bx / mag, by / mag);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_interior = U_uniforms.grid_n;
    let n_total    = U_uniforms.grid_n_total;
    let ghost      = U_uniforms.ghost_w;
    let noise_n    = U_uniforms.noise_n;
    let phase      = lic_u.lic_phase;
    let drift_x    = lic_u.lic_drift_x;
    let drift_y    = lic_u.lic_drift_y;

    if (gid.x >= n_interior || gid.y >= n_interior) { return; }

    // Cell-coordinate starting position (interior cell center).
    var px: f32 = f32(gid.x) + 0.5;
    var py: f32 = f32(gid.y) + 0.5;

    // Noise scale factor — map cell coords [0, n_interior) to noise
    // sample coords [0, noise_n). Scale per axis so the same noise
    // texture works at any grid resolution.
    let scale_x = f32(noise_n) / f32(n_interior);
    let scale_y = f32(noise_n) / f32(n_interior);

    var sum: f32 = 0.0;
    var n_samples: f32 = 0.0;

    // Backward-trace LIC_STEPS via RK2 midpoint along the unit B-field.
    // Step 0 samples the starting point; subsequent steps walk backward.
    var stopped: bool = false;
    for (var k: u32 = 0u; k < LIC_STEPS; k = k + 1u) {
        // Sample noise at the current position with the phase offset.
        let nx = px * scale_x + phase * drift_x;
        let ny = py * scale_y + phase * drift_y;
        sum = sum + sample_noise(nx, ny, noise_n);
        n_samples = n_samples + 1.0;

        if (stopped) { continue; }

        // RK2 midpoint: half-step to k1, full step using k2 at midpoint.
        let k1 = sample_b_unit(px, py, ghost, n_total, n_interior);
        if (k1.x == 0.0 && k1.y == 0.0) {
            stopped = true;
            continue;
        }
        let mx = px - 0.5 * LIC_STEP_SIZE * k1.x;
        let my = py - 0.5 * LIC_STEP_SIZE * k1.y;
        let k2 = sample_b_unit(mx, my, ghost, n_total, n_interior);
        if (k2.x == 0.0 && k2.y == 0.0) {
            stopped = true;
            continue;
        }
        px = px - LIC_STEP_SIZE * k2.x;
        py = py - LIC_STEP_SIZE * k2.y;
    }

    let lum = sum / max(n_samples, 1.0);

    // Write to the ghost-padded slot so the composite pass can read with
    // the same indexing it uses for `colored`.
    let ix = gid.x + ghost;
    let iy = gid.y + ghost;
    lic_out[cell_idx_total(ix, iy, n_total)] = lum;
}

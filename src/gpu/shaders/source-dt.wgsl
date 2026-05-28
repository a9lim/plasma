// ─── source-dt.wgsl ───────────────────────────────────────────────────
// Divide the freshly computed source half-step into per-operator substeps.
//
// compute-dt writes dt[0] on the GPU, scale-dt writes dt_half[0] = dt/2,
// then this pass writes:
//   hall_dt[0]     = dt_half / N_hall
//   cond_dt[0]     = dt_half / N_cond
//   visc_dt[0]     = dt_half / N_visc
//   nonideal_dt[0] = dt_half / N_nonideal
//   rad_dt[0]      = dt_half / N_rad
//
// The host still chooses integer substep counts from the previous reduction
// (no GPU-readback stall), but the total integrated source time now uses the
// current macro dt exactly.

struct SourceDtParams {
    inv_hall_substeps:     f32,
    inv_cond_substeps:     f32,
    inv_visc_substeps:     f32,
    inv_nonideal_substeps: f32,
    inv_rad_substeps:      f32,
    _pad0:                 f32,
    _pad1:                 f32,
    _pad2:                 f32,
};

@group(0) @binding(0) var<storage, read>       dt_half:     array<f32, 8>;
@group(0) @binding(1) var<uniform>             params:      SourceDtParams;
@group(0) @binding(2) var<storage, read_write> hall_dt:     array<f32, 8>;
@group(0) @binding(3) var<storage, read_write> cond_dt:     array<f32, 8>;
@group(0) @binding(4) var<storage, read_write> visc_dt:     array<f32, 8>;
@group(0) @binding(5) var<storage, read_write> nonideal_dt: array<f32, 8>;
@group(0) @binding(6) var<storage, read_write> rad_dt:      array<f32, 8>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let half = dt_half[0];
    let h = half * max(params.inv_hall_substeps,     0.0);
    let c = half * max(params.inv_cond_substeps,     0.0);
    let v = half * max(params.inv_visc_substeps,     0.0);
    let n = half * max(params.inv_nonideal_substeps, 0.0);
    let r = half * max(params.inv_rad_substeps,      0.0);

    hall_dt[0] = h;
    hall_dt[1] = dt_half[1];
    hall_dt[2] = dt_half[2];
    hall_dt[3] = dt_half[3];
    hall_dt[4] = dt_half[4];
    hall_dt[5] = 0.0;
    hall_dt[6] = 0.0;
    hall_dt[7] = 0.0;

    cond_dt[0] = c;
    cond_dt[1] = dt_half[1];
    cond_dt[2] = dt_half[2];
    cond_dt[3] = dt_half[3];
    cond_dt[4] = dt_half[4];
    cond_dt[5] = 0.0;
    cond_dt[6] = 0.0;
    cond_dt[7] = 0.0;

    visc_dt[0] = v;
    visc_dt[1] = dt_half[1];
    visc_dt[2] = dt_half[2];
    visc_dt[3] = dt_half[3];
    visc_dt[4] = dt_half[4];
    visc_dt[5] = 0.0;
    visc_dt[6] = 0.0;
    visc_dt[7] = 0.0;

    nonideal_dt[0] = n;
    nonideal_dt[1] = dt_half[1];
    nonideal_dt[2] = dt_half[2];
    nonideal_dt[3] = dt_half[3];
    nonideal_dt[4] = dt_half[4];
    nonideal_dt[5] = 0.0;
    nonideal_dt[6] = 0.0;
    nonideal_dt[7] = 0.0;

    rad_dt[0] = r;
    rad_dt[1] = dt_half[1];
    rad_dt[2] = dt_half[2];
    rad_dt[3] = dt_half[3];
    rad_dt[4] = dt_half[4];
    rad_dt[5] = 0.0;
    rad_dt[6] = 0.0;
    rad_dt[7] = 0.0;
}

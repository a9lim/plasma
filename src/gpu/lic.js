/**
 * @fileoverview LIC (line integral convolution) renderer orchestrator
 * for Phase 6. Mirrors PlasmaRenderer's shape — owns the bind group
 * for the lic-advect compute shader and encodes the dispatch into a
 * caller-supplied command encoder.
 *
 * The noise buffer + lic_out buffer live on PlasmaBuffers; LicRenderer
 * just wires them into a bind group and dispatches. The bind group
 * depends on Bx_n / By_n, which ping-pong every physics step, so it's
 * rebuilt every render (same pattern as PlasmaRenderer._viewBindGroup).
 *
 * The composite render pass is owned by PlasmaRenderer — this orchestrator
 * only handles the compute kernel. Per the locked design decision, only
 * the LIC compute kernel needs to be transpilable; the composite vertex
 * + fragment pipeline stays GPU-only.
 *
 * Bind-group layout (group 0) matches lic-advect.wgsl:
 *   0: uniform Uniforms     (view sweep_dir doesn't matter — we read
 *                            grid_n / grid_n_total / ghost_w /
 *                            lic_phase / lic_drift_x / lic_drift_y /
 *                            noise_n only)
 *   1: storage Bx_face      (read)
 *   2: storage By_face      (read)
 *   3: storage noise        (read)  — 1024² f32, resolution-independent
 *   4: storage lic_out      (read_write) — ghost-padded f32
 */

const WG = 8;

export class LicRenderer {
    /**
     * @param {GPUDevice} device
     * @param {object} pipelines  from createPipelines()
     * @param {import('./buffers.js').PlasmaBuffers} buffers
     */
    constructor(device, pipelines, buffers) {
        this.device    = device;
        this.pipelines = pipelines;
        this.buffers   = buffers;
    }

    _bindGroup() {
        // Bx_n / By_n ping-pong every physics step, so rebuild each frame.
        // Same cost (<50 µs) and pattern as PlasmaRenderer._viewBindGroup.
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.licAdvect.bg',
            layout: this.pipelines.layouts.licAdvect,
            entries: [
                { binding: 0, resource: { buffer: b.uniform_x } },
                { binding: 1, resource: { buffer: b.Bx_n } },
                { binding: 2, resource: { buffer: b.By_n } },
                { binding: 3, resource: { buffer: b.noise } },
                { binding: 4, resource: { buffer: b.lic_out } },
            ],
        });
    }

    /**
     * Encode the LIC compute dispatch into the supplied pass. Caller
     * provides the open compute pass so we can chain inside the same
     * encoder as view-field + colormap (one submit per render frame).
     *
     * Dispatches one workgroup per 8×8 tile over the INTERIOR grid only.
     * Ghost cells are not written.
     */
    encode(pass) {
        const n = this.buffers.n;
        const groups = Math.ceil(n / WG);
        pass.setPipeline(this.pipelines.pipelines.licAdvect);
        pass.setBindGroup(0, this._bindGroup());
        pass.dispatchWorkgroups(groups, groups, 1);
    }
}

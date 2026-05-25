/**
 * @fileoverview LIC (line integral convolution) renderer orchestrator
 * for Phase 6. Mirrors PlasmaRenderer's shape — owns the bind group
 * for the lic-advect compute shader and encodes the dispatch into a
 * caller-supplied command encoder.
 *
 * The noise buffer + lic_out buffer live on PlasmaBuffers; LicRenderer
 * just wires them into a bind group and dispatches. The bind group
 * depends on Bx_n / By_n, which ping-pong every physics step — so we
 * pre-bake an (a, b) pair and pick the right one per render frame
 * (same pattern as PlasmaRenderer's view-field cache).
 *
 * The composite render pass is owned by PlasmaRenderer — this orchestrator
 * only handles the compute kernel. Per the locked design decision, only
 * the LIC compute kernel needs to be transpilable; the composite vertex
 * + fragment pipeline stays GPU-only.
 *
 * Bind-group layout (group 0) matches lic-advect.wgsl:
 *   0: uniform Uniforms     (reads grid_n / grid_n_total / ghost_w /
 *                            noise_n only — sweep_dir is no longer in
 *                            Uniforms and LIC fields live in
 *                            LicUniforms at binding 5)
 *   1: storage Bx_face      (read)
 *   2: storage By_face      (read)
 *   3: storage noise        (read)  — 1024² f32, resolution-independent
 *   4: storage lic_out      (read_write) — ghost-padded f32
 *   5: uniform LicUniforms  (read) — render-pace phase + drift
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
        this._bg = { a: null, b: null };
        this.rebuildSideCache();
    }

    /**
     * Rebuild the lic-advect A/B bind groups. Called by Sim once per
     * (re)allocation of PlasmaBuffers.
     */
    rebuildSideCache() {
        const b = this.buffers;
        const mk = (Bx_n, By_n) => this.device.createBindGroup({
            label: 'plasma.licAdvect.bg',
            layout: this.pipelines.layouts.licAdvect,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: Bx_n } },
                { binding: 2, resource: { buffer: By_n } },
                { binding: 3, resource: { buffer: b.noise } },
                { binding: 4, resource: { buffer: b.lic_out } },
                { binding: 5, resource: { buffer: b.licUniform } },
            ],
        });
        this._bg.a = mk(b.Bx_a, b.By_a);
        this._bg.b = mk(b.Bx_b, b.By_b);
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
        pass.setBindGroup(0, this._bg[this.buffers._side]);
        pass.dispatchWorkgroups(groups, groups, 1);
    }
}

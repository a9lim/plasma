/**
 * @fileoverview LIC (line integral convolution) renderer orchestrator
 * for Phase 6 + Session 5. Mirrors PlasmaRenderer's shape — owns the
 * bind groups for the lic-advect compute shader and the new
 * lic-reduce / lic-normalize contrast-stretch chain, and encodes the
 * dispatches into a caller-supplied command encoder.
 *
 * The noise buffer + lic_out buffer live on PlasmaBuffers; LicRenderer
 * just wires them into bind groups and dispatches. The advect bind
 * group depends on Bx_n / By_n, which ping-pong every physics step —
 * so we pre-bake an (a, b) pair and pick the right one per render
 * frame (same pattern as PlasmaRenderer's view-field cache).
 *
 * The reduce + normalize passes don't depend on the ping-pong (they
 * only touch lic_out and the small lic_minmax buffer), so each has a
 * single bind group built once at construction.
 *
 * The composite render pass is owned by PlasmaRenderer — this orchestrator
 * only handles the compute kernels. Per the locked design decision, only
 * the LIC compute kernels need to be transpilable; the composite vertex +
 * fragment pipeline stays GPU-only.
 *
 * Bind-group layouts (group 0):
 *   lic-advect:    Uniforms + Bx + By + noise + lic_out + LicUniforms
 *   lic-reduce:    Uniforms + lic_out (ro) + lic_minmax (rw atomic)
 *   lic-normalize: Uniforms + lic_minmax (ro) + lic_out (rw)
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
        this._reduceBG = null;
        this._normalizeBG = null;
        this.rebuildSideCache();
    }

    /**
     * Rebuild the lic-advect A/B bind groups + the resolution-independent
     * reduce / normalize bind groups. Called by Sim once per
     * (re)allocation of PlasmaBuffers (i.e. setResolution).
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

        // Contrast-stretch bind groups — independent of the ping-pong,
        // but rebuilt on resolution change so the lic_out / lic_minmax
        // GPUBuffer refs stay valid.
        this._reduceBG = this.device.createBindGroup({
            label: 'plasma.licReduce.bg',
            layout: this.pipelines.layouts.licReduce,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.lic_out } },
                { binding: 2, resource: { buffer: b.lic_minmax } },
            ],
        });

        this._normalizeBG = this.device.createBindGroup({
            label: 'plasma.licNormalize.bg',
            layout: this.pipelines.layouts.licNormalize,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: b.lic_minmax } },
                { binding: 2, resource: { buffer: b.lic_out } },
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
        pass.setBindGroup(0, this._bg[this.buffers._side]);
        pass.dispatchWorkgroups(groups, groups, 1);
    }

    /**
     * Encode the LIC contrast-stretch chain (reduce → normalize) into
     * the supplied pass. Must follow `encode()` in the same pass: the
     * reduce reads lic_out, the normalize rewrites lic_out in place,
     * and composite then samples the stretched result.
     *
     * Sequence:
     *   1. lic-reduce.reset      (1×1)  — seed lic_minmax to (1.0, 0.0)
     *   2. lic-reduce.main       (n²/64) — per-tile atomicMin/Max into
     *                                      lic_minmax via workgroup-shared.
     *   3. lic-normalize.main    (n²/64) — read lic_minmax, rewrite lic_out.
     */
    encodePost(pass) {
        const n = this.buffers.n;
        const groups = Math.ceil(n / WG);

        pass.setPipeline(this.pipelines.pipelines.licReduceReset);
        pass.setBindGroup(0, this._reduceBG);
        pass.dispatchWorkgroups(1, 1, 1);

        pass.setPipeline(this.pipelines.pipelines.licReduce);
        pass.setBindGroup(0, this._reduceBG);
        pass.dispatchWorkgroups(groups, groups, 1);

        pass.setPipeline(this.pipelines.pipelines.licNormalize);
        pass.setBindGroup(0, this._normalizeBG);
        pass.dispatchWorkgroups(groups, groups, 1);
    }
}

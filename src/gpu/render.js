/**
 * @fileoverview View-field → colormap → LIC advect → composite render chain.
 *
 * Four GPU passes wired into one entry point:
 *   1. view-field   — compute: U_current + face B → field (scalar by view-mode)
 *   2. colormap     — compute: field + LUT → colored (vec4 RGB)
 *   3. lic-advect   — compute: B-field + noise → lic_out (per-cell luminance)
 *   4. composite    — render:  colored × LIC luminance → canvas (fullscreen tri)
 *
 * view-field reads U_n + face B, all of which ping-pong every physics
 * step. Rather than rebuild the bind group every frame, we pre-bake an
 * (a, b) pair pinned to each side of the ping-pong; render() picks the
 * right one based on `buffers._side`. Sim.setResolution() rebuilds the
 * pair after replacing the buffer set.
 */

import { LicRenderer } from './lic.js';

const WG = 8;

export class PlasmaRenderer {
    /**
     * @param {GPUDevice} device
     * @param {GPUCanvasContext} context
     * @param {object} pipelines  from createPipelines()
     * @param {import('./buffers.js').PlasmaBuffers} buffers
     */
    constructor(device, context, pipelines, buffers) {
        this.device = device;
        this.context = context;
        this.pipelines = pipelines;
        this.buffers = buffers;

        // colormap and composite bind groups only depend on buffers we
        // don't ping-pong, so build them up front. The shared `uniform`
        // buffer carries all physics-state fields; composite additionally
        // binds the small LIC uniform for render-pace intensity.
        this._colormapBG = device.createBindGroup({
            label: 'plasma.colormap.bg',
            layout: pipelines.layouts.colormap,
            entries: [
                { binding: 0, resource: { buffer: buffers.uniform } },
                { binding: 1, resource: { buffer: buffers.field } },
                { binding: 2, resource: { buffer: buffers.lut } },
                { binding: 3, resource: { buffer: buffers.colored } },
            ],
        });

        this._compositeBG = device.createBindGroup({
            label: 'plasma.composite.bg',
            layout: pipelines.layouts.composite,
            entries: [
                { binding: 0, resource: { buffer: buffers.uniform } },
                { binding: 1, resource: { buffer: buffers.colored } },
                { binding: 2, resource: { buffer: buffers.lic_out } },
                { binding: 3, resource: { buffer: buffers.licUniform } },
            ],
        });

        // LIC orchestrator owns its own pre-baked (a, b) side cache.
        this.lic = new LicRenderer(device, pipelines, buffers);

        // view-field reads U_n + face B → side-dependent. Pre-baked pair.
        this._viewBG = { a: null, b: null };
        this.rebuildSideCache();
    }

    /**
     * Rebuild the view-field A/B bind groups. Called by Sim once per
     * (re)allocation of PlasmaBuffers — i.e. once at init and again on
     * each setResolution(). No-op cost-wise: ~100 µs total.
     */
    rebuildSideCache() {
        const b = this.buffers;
        const mk = (U0_n, U1_n, Bx_n, By_n) => this.device.createBindGroup({
            label: 'plasma.viewField.bg',
            layout: this.pipelines.layouts.view,
            entries: [
                { binding: 0, resource: { buffer: b.uniform } },
                { binding: 1, resource: { buffer: U0_n } },
                { binding: 2, resource: { buffer: U1_n } },
                { binding: 3, resource: { buffer: Bx_n } },
                { binding: 4, resource: { buffer: By_n } },
                { binding: 5, resource: { buffer: b.field } },
                { binding: 6, resource: { buffer: b.phi } },
            ],
        });
        this._viewBG.a = mk(b.U0_a, b.U1_a, b.Bx_a, b.By_a);
        this._viewBG.b = mk(b.U0_b, b.U1_b, b.Bx_b, b.By_b);
    }

    /**
     * Encode the full render chain into a fresh command encoder and
     * submit. Caller owns timing; this is fire-and-forget.
     */
    render() {
        const { device, pipelines } = this;
        // Both view-field and colormap dispatch over interior cells only.
        // Buffers are ghost-padded; the shaders take ghost-w and grid_n
        // from the uniform struct and write at the correct interior
        // offsets.
        const n = this.buffers.n;
        const groups = Math.ceil(n / WG);

        const encoder = device.createCommandEncoder({ label: 'plasma.render' });

        {
            const pass = encoder.beginComputePass({ label: 'plasma.render.compute' });
            pass.setPipeline(pipelines.pipelines.viewField);
            pass.setBindGroup(0, this._viewBG[this.buffers._side]);
            pass.dispatchWorkgroups(groups, groups, 1);

            pass.setPipeline(pipelines.pipelines.colormap);
            pass.setBindGroup(0, this._colormapBG);
            pass.dispatchWorkgroups(groups, groups, 1);

            // LIC advect — backward-traces along B-field, writes luminance.
            // Chained in the same compute pass; reads Bx_n / By_n (already
            // ghost-filled by apply-bcs at the end of the last step) and
            // writes lic_out.
            this.lic.encode(pass);
            // LIC contrast-stretch — reduce lic_out → global (min, max),
            // then rewrite lic_out in-place via min/max normalization.
            // Mirrors compute-dt's per-tile shared-atomic reduction. This
            // pulls residual noise variation out into the full [0, 1]
            // range in flat field-free regions while leaving strong-field
            // regions essentially untouched, so composite samples a
            // higher-contrast LIC texture.
            this.lic.encodePost(pass);
            pass.end();
        }

        // 4: composite to canvas (colormap × LIC luminance).
        {
            const view = this.context.getCurrentTexture().createView();
            const pass = encoder.beginRenderPass({
                label: 'plasma.render.composite',
                colorAttachments: [{
                    view,
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(pipelines.pipelines.composite);
            pass.setBindGroup(0, this._compositeBG);
            pass.draw(3, 1, 0, 0);
            pass.end();
        }

        device.queue.submit([encoder.finish()]);
    }
}

/**
 * @fileoverview View-field → colormap → LIC advect → composite render chain.
 *
 * Four GPU passes wired into one entry point:
 *   1. view-field   — compute: U_current + face B → field (scalar by view-mode)
 *   2. colormap     — compute: field + LUT → colored (vec4 RGB)
 *   3. lic-advect   — compute: B-field + noise → lic_out (per-cell luminance)
 *   4. composite    — render:  colored × LIC luminance → canvas (fullscreen tri)
 *
 * view-field's and lic-advect's bind groups are rebuilt every render —
 * the buffers ping-pong every step (cell-state AND face-B together).
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
        // don't ping-pong, so build them up front. Use uniform_x as the
        // canonical "view" uniform binding (sweep_dir is irrelevant for
        // viz passes — they read all of dx/gamma/view_min/view_max/
        // view_mode from the same struct shape).
        this._colormapBG = device.createBindGroup({
            label: 'plasma.colormap.bg',
            layout: pipelines.layouts.colormap,
            entries: [
                { binding: 0, resource: { buffer: buffers.uniform_x } },
                { binding: 1, resource: { buffer: buffers.field } },
                { binding: 2, resource: { buffer: buffers.lut } },
                { binding: 3, resource: { buffer: buffers.colored } },
            ],
        });

        this._compositeBG = device.createBindGroup({
            label: 'plasma.composite.bg',
            layout: pipelines.layouts.composite,
            entries: [
                { binding: 0, resource: { buffer: buffers.uniform_x } },
                { binding: 1, resource: { buffer: buffers.colored } },
                { binding: 2, resource: { buffer: buffers.lic_out } },
            ],
        });

        // LIC orchestrator owns its own (per-frame-rebuilt) bind group.
        this.lic = new LicRenderer(device, pipelines, buffers);
    }

    _viewBindGroup() {
        // U_n is the start-of-step state, which after sim.swap() points
        // at the just-finished step's destination. We rebuild every
        // render because the underlying buffer handle ping-pongs between
        // (A, B). Cost is < 50 µs.
        const b = this.buffers;
        return this.device.createBindGroup({
            label: 'plasma.viewField.bg',
            layout: this.pipelines.layouts.view,
            entries: [
                { binding: 0, resource: { buffer: b.uniform_x } },
                { binding: 1, resource: { buffer: b.U0_n } },
                { binding: 2, resource: { buffer: b.U1_n } },
                { binding: 3, resource: { buffer: b.Bx_n } },
                { binding: 4, resource: { buffer: b.By_n } },
                { binding: 5, resource: { buffer: b.field } },
            ],
        });
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
            pass.setBindGroup(0, this._viewBindGroup());
            pass.dispatchWorkgroups(groups, groups, 1);

            pass.setPipeline(pipelines.pipelines.colormap);
            pass.setBindGroup(0, this._colormapBG);
            pass.dispatchWorkgroups(groups, groups, 1);

            // LIC advect — backward-traces along B-field, writes luminance.
            // Chained in the same compute pass; reads Bx_n / By_n (already
            // ghost-filled by apply-bcs at the end of the last step) and
            // writes lic_out, which composite then samples.
            this.lic.encode(pass);
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

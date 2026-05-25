/**
 * @fileoverview View-field → colormap → composite render chain.
 *
 * Three GPU passes wired into one entry point:
 *   1. view-field   — compute: U_current → field (scalar density)
 *   2. colormap     — compute: field + LUT → colored (vec4 RGB)
 *   3. composite    — render:  colored → canvas via fullscreen triangle
 *
 * Bind groups are rebuilt lazily when `current` ping-pongs (it changes
 * mid-step), so we cache the most recent (buffers.current, bindGroup)
 * pair and only recreate when the source buffer changes.
 */

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

        this._viewBG = null;
        this._viewBGSource = null;

        // colormap and composite bind groups only depend on buffers we
        // don't ping-pong, so build them up front.
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
            ],
        });
    }

    _viewBindGroup() {
        const src = this.buffers.current;
        if (this._viewBG && this._viewBGSource === src) return this._viewBG;
        this._viewBG = this.device.createBindGroup({
            label: 'plasma.viewField.bg',
            layout: this.pipelines.layouts.view,
            entries: [
                { binding: 0, resource: { buffer: this.buffers.uniform } },
                { binding: 1, resource: { buffer: src } },
                { binding: 2, resource: { buffer: this.buffers.field } },
            ],
        });
        this._viewBGSource = src;
        return this._viewBG;
    }

    /**
     * Encode the full render chain into a fresh command encoder and
     * submit. Caller owns timing; this is fire-and-forget.
     */
    render() {
        const { device, pipelines } = this;
        const n = this.buffers.n;
        const groups = Math.ceil(n / WG);

        const encoder = device.createCommandEncoder({ label: 'plasma.render' });

        // 1+2: view-field then colormap. Both run grid-sized compute.
        {
            const pass = encoder.beginComputePass({ label: 'plasma.render.compute' });
            pass.setPipeline(pipelines.pipelines.viewField);
            pass.setBindGroup(0, this._viewBindGroup());
            pass.dispatchWorkgroups(groups, groups, 1);

            pass.setPipeline(pipelines.pipelines.colormap);
            pass.setBindGroup(0, this._colormapBG);
            pass.dispatchWorkgroups(groups, groups, 1);
            pass.end();
        }

        // 3: composite to canvas.
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

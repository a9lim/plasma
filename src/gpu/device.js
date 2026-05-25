/**
 * @fileoverview WebGPU device init for plasma (Phase 1).
 *
 * Requests a high-performance adapter and device with limits sized for the
 * planned MHD compute kernels (256² staggered grid with multiple ping-pong
 * field buffers). Throws on any failure; caller is responsible for showing
 * the no-WebGPU landing fallback.
 */

const REQUIRED_LIMITS = {
    // 256 MB ceilings — Phase 3+ needs room for ping-pong U_cell/Bx_face/By_face
    // plus Ez_edge and a couple of staging textures. 256 MB is the spec ceiling
    // most adapters report.
    maxStorageBufferBindingSize: 256 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
};

/**
 * Initialize WebGPU adapter + device.
 * @returns {Promise<{adapter: GPUAdapter, device: GPUDevice, format: GPUTextureFormat}>}
 * @throws {Error} if WebGPU is unavailable, adapter/device request fails, or
 *                 the adapter cannot satisfy the requested limits.
 */
export async function initDevice() {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error('WebGPU not available (navigator.gpu undefined)');
    }

    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
    });
    if (!adapter) {
        throw new Error('No WebGPU adapter returned');
    }

    // Clamp requested limits to what the adapter actually advertises so a
    // device request on a constrained GPU doesn't reject outright.
    const requiredLimits = {};
    for (const [key, requested] of Object.entries(REQUIRED_LIMITS)) {
        const max = adapter.limits?.[key];
        if (typeof max === 'number') {
            requiredLimits[key] = Math.min(requested, max);
        }
    }

    const device = await adapter.requestDevice({ requiredLimits });
    if (!device) {
        throw new Error('No WebGPU device returned');
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    return { adapter, device, format };
}

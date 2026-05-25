/**
 * @fileoverview Compute / render pipeline factory.
 *
 * Mirrors geon's gpu-pipelines.js spirit — single concat of a shared
 * WGSL prelude + per-pipeline shader source — but adapted for the grid
 * Eulerian compute graph instead of particle Lagrangian. Each shader
 * is prepended with shared-helpers.wgsl so the `Uniforms` struct, the
 * cons↔prim helpers, and the wrap helpers are visible everywhere.
 *
 * The bind-group layout for every Phase-2 compute pipeline keeps slot 0
 * as the uniform buffer; sweep pipelines share an identical layout so
 * dispatching them in alternating x/y order doesn't require rebuilding
 * bind groups. See sim.js for the actual encoding.
 *
 * Cache-bust: bump SHADER_VERSION whenever a .wgsl file is edited so
 * browsers refetch instead of serving a stale module.
 */

const SHADER_VERSION = 2;

async function fetchWGSL(filename) {
    const url = new URL(`./shaders/${filename}?v=${SHADER_VERSION}`, import.meta.url);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch shader: ${filename} (${resp.status})`);
    return resp.text();
}

/** Module-scope cache for the shared helpers — fetched at most once. */
let _sharedHelpersPromise = null;
function getSharedHelpers() {
    if (!_sharedHelpersPromise) {
        _sharedHelpersPromise = fetchWGSL('shared-helpers.wgsl');
    }
    return _sharedHelpersPromise;
}

async function makeModule(device, label, filename) {
    const helpers = await getSharedHelpers();
    const body    = await fetchWGSL(filename);
    const code    = helpers + '\n' + body;
    return device.createShaderModule({ label, code });
}

function bgl(device, label, entries) {
    return device.createBindGroupLayout({ label, entries });
}

const COMPUTE = GPUShaderStage.COMPUTE;
const FRAGMENT = GPUShaderStage.FRAGMENT;
const VERTEX = GPUShaderStage.VERTEX;

/**
 * Build the unified sweep bind-group layout — same shape for
 * reconstruct, riemann, and update so we can reuse bind groups across
 * the three passes within a single sweep.
 *
 * Layout:
 *   0 — Uniforms (uniform)
 *   1 — U_in       (read-only storage)
 *   2 — slopes     (read-write storage)
 *   3 — flux       (read-write storage)
 *   4 — U_out      (read-write storage)
 *   5 — dt_buf     (read-only storage)
 */
function sweepBGL(device) {
    return bgl(device, 'plasma.sweep.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
    ]);
}

/**
 * compute-dt bind group:
 *   0 — Uniforms (uniform)
 *   1 — U_in       (read-only storage)
 *   2 — wavespeed  (atomic<u32> as storage)
 *   3 — dt_buf     (storage)
 */
function dtBGL(device) {
    return bgl(device, 'plasma.computeDt.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: COMPUTE, buffer: { type: 'storage' } },
    ]);
}

/**
 * view-field bind group:
 *   0 — Uniforms
 *   1 — U_in   (read-only storage)
 *   2 — field  (storage)
 */
function viewBGL(device) {
    return bgl(device, 'plasma.viewField.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
    ]);
}

/**
 * colormap bind group:
 *   0 — Uniforms
 *   1 — field    (read-only storage)
 *   2 — lut      (read-only storage)
 *   3 — colored  (storage)
 */
function colormapBGL(device) {
    return bgl(device, 'plasma.colormap.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: COMPUTE, buffer: { type: 'storage' } },
    ]);
}

/**
 * composite bind group (used by a render pipeline):
 *   0 — Uniforms (visible in vertex + fragment)
 *   1 — colored (read-only storage, fragment)
 */
function compositeBGL(device) {
    return bgl(device, 'plasma.composite.bgl', [
        { binding: 0, visibility: VERTEX | FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: FRAGMENT,         buffer: { type: 'read-only-storage' } },
    ]);
}

/**
 * Create every pipeline the sim needs in one go. Returns an object
 * holding all pipelines and their bind-group layouts.
 */
export async function createPipelines(device, format) {
    // Shader modules
    const [
        plmModule, riemannModule, updateModule,
        dtModule, viewModule, colormapModule, compositeModule,
    ] = await Promise.all([
        makeModule(device, 'plasma.reconstruct-plm',   'reconstruct-plm.wgsl'),
        makeModule(device, 'plasma.riemann-hll',       'riemann-hll.wgsl'),
        makeModule(device, 'plasma.update-conserved',  'update-conserved.wgsl'),
        makeModule(device, 'plasma.compute-dt',        'compute-dt.wgsl'),
        makeModule(device, 'plasma.view-field',        'view-field.wgsl'),
        makeModule(device, 'plasma.colormap',          'colormap.wgsl'),
        makeModule(device, 'plasma.composite',         'composite.wgsl'),
    ]);

    // Bind-group layouts
    const sweepLayout     = sweepBGL(device);
    const dtLayout        = dtBGL(device);
    const viewLayout      = viewBGL(device);
    const colormapLayout  = colormapBGL(device);
    const compositeLayout = compositeBGL(device);

    const sweepPipeLayout     = device.createPipelineLayout({ bindGroupLayouts: [sweepLayout] });
    const dtPipeLayout        = device.createPipelineLayout({ bindGroupLayouts: [dtLayout] });
    const viewPipeLayout      = device.createPipelineLayout({ bindGroupLayouts: [viewLayout] });
    const colormapPipeLayout  = device.createPipelineLayout({ bindGroupLayouts: [colormapLayout] });
    const compositePipeLayout = device.createPipelineLayout({ bindGroupLayouts: [compositeLayout] });

    const reconstructPlm = device.createComputePipeline({
        label: 'plasma.reconstruct-plm',
        layout: sweepPipeLayout,
        compute: { module: plmModule, entryPoint: 'main' },
    });
    const riemannHll = device.createComputePipeline({
        label: 'plasma.riemann-hll',
        layout: sweepPipeLayout,
        compute: { module: riemannModule, entryPoint: 'main' },
    });
    const updateConserved = device.createComputePipeline({
        label: 'plasma.update-conserved',
        layout: sweepPipeLayout,
        compute: { module: updateModule, entryPoint: 'main' },
    });

    const dtReset = device.createComputePipeline({
        label: 'plasma.compute-dt.reset',
        layout: dtPipeLayout,
        compute: { module: dtModule, entryPoint: 'reset' },
    });
    const dtReduce = device.createComputePipeline({
        label: 'plasma.compute-dt.reduce',
        layout: dtPipeLayout,
        compute: { module: dtModule, entryPoint: 'reduce' },
    });
    const dtFinalize = device.createComputePipeline({
        label: 'plasma.compute-dt.finalize',
        layout: dtPipeLayout,
        compute: { module: dtModule, entryPoint: 'finalize' },
    });

    const viewField = device.createComputePipeline({
        label: 'plasma.view-field',
        layout: viewPipeLayout,
        compute: { module: viewModule, entryPoint: 'main' },
    });

    const colormap = device.createComputePipeline({
        label: 'plasma.colormap',
        layout: colormapPipeLayout,
        compute: { module: colormapModule, entryPoint: 'main' },
    });

    const composite = device.createRenderPipeline({
        label: 'plasma.composite',
        layout: compositePipeLayout,
        vertex:   { module: compositeModule, entryPoint: 'vsMain' },
        fragment: { module: compositeModule, entryPoint: 'fsMain', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
    });

    return {
        layouts: {
            sweep: sweepLayout,
            dt: dtLayout,
            view: viewLayout,
            colormap: colormapLayout,
            composite: compositeLayout,
        },
        pipelines: {
            reconstructPlm, riemannHll, updateConserved,
            dtReset, dtReduce, dtFinalize,
            viewField, colormap, composite,
        },
    };
}

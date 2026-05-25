/**
 * @fileoverview Compute / render pipeline factory (Phase 3a).
 *
 * Phase 3a expands Phase 2's bind-group layouts to carry the MHD state
 * (two cell-centered vec4 arrays U0/U1, two face-centered scalar arrays
 * Bx_face/By_face, one edge-centered Ez_edge). Per-direction PLM slope
 * and HLL flux buffers (x- and y-side) coexist so a single unsplit CT
 * update sees both directions' fluxes.
 *
 * Each shader is prepended with shared-helpers.wgsl. Cache-bust:
 * SHADER_VERSION bumps when any WGSL file is edited.
 */

const SHADER_VERSION = 3;

async function fetchWGSL(filename) {
    const url = new URL(`./shaders/${filename}?v=${SHADER_VERSION}`, import.meta.url);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch shader: ${filename} (${resp.status})`);
    return resp.text();
}

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

const COMPUTE  = GPUShaderStage.COMPUTE;
const FRAGMENT = GPUShaderStage.FRAGMENT;
const VERTEX   = GPUShaderStage.VERTEX;
const RO_STO   = { type: 'read-only-storage' };
const RW_STO   = { type: 'storage' };
const UNIFORM  = { type: 'uniform' };

function dtBGL(device) {
    return bgl(device, 'plasma.computeDt.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // U0
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // U1
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },  // Bx_face
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },  // By_face
        { binding: 5, visibility: COMPUTE, buffer: RW_STO },  // wavespeed atomic
        { binding: 6, visibility: COMPUTE, buffer: RW_STO },  // dt_buf
    ]);
}

function reconstructBGL(device) {
    return bgl(device, 'plasma.reconstruct.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // U0
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // U1
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },  // Bx_face
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },  // By_face
        { binding: 5, visibility: COMPUTE, buffer: RW_STO },  // slopes_0
        { binding: 6, visibility: COMPUTE, buffer: RW_STO },  // slopes_1
    ]);
}

function riemannBGL(device) {
    return bgl(device, 'plasma.riemann.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // U0
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // U1
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },  // Bx_face
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },  // By_face
        { binding: 5, visibility: COMPUTE, buffer: RO_STO },  // slopes_0
        { binding: 6, visibility: COMPUTE, buffer: RO_STO },  // slopes_1
        { binding: 7, visibility: COMPUTE, buffer: RW_STO },  // flux_0
        { binding: 8, visibility: COMPUTE, buffer: RW_STO },  // flux_1
    ]);
}

function emfBGL(device) {
    return bgl(device, 'plasma.emf.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // flux_x_1
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // flux_y_1
        { binding: 3, visibility: COMPUTE, buffer: RW_STO },  // Ez_edge
    ]);
}

function updateConservedBGL(device) {
    return bgl(device, 'plasma.updateU.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // U0_in
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // U1_in
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },  // flux_x_0
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },  // flux_x_1
        { binding: 5, visibility: COMPUTE, buffer: RO_STO },  // flux_y_0
        { binding: 6, visibility: COMPUTE, buffer: RO_STO },  // flux_y_1
        { binding: 7, visibility: COMPUTE, buffer: RO_STO },  // dt_buf
        { binding: 8, visibility: COMPUTE, buffer: RW_STO },  // U0_out
        { binding: 9, visibility: COMPUTE, buffer: RW_STO },  // U1_out
    ]);
}

function updateBBGL(device) {
    return bgl(device, 'plasma.updateB.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // Bx_in
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // By_in
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },  // Ez_edge
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },  // dt_buf
        { binding: 5, visibility: COMPUTE, buffer: RW_STO },  // Bx_out
        { binding: 6, visibility: COMPUTE, buffer: RW_STO },  // By_out
    ]);
}

function viewBGL(device) {
    return bgl(device, 'plasma.viewField.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // U0
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // U1
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },  // Bx_face
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },  // By_face
        { binding: 5, visibility: COMPUTE, buffer: RW_STO },  // field
    ]);
}

function colormapBGL(device) {
    return bgl(device, 'plasma.colormap.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },
        { binding: 3, visibility: COMPUTE, buffer: RW_STO },
    ]);
}

function compositeBGL(device) {
    return bgl(device, 'plasma.composite.bgl', [
        { binding: 0, visibility: VERTEX | FRAGMENT, buffer: UNIFORM },
        { binding: 1, visibility: FRAGMENT,          buffer: RO_STO },
    ]);
}

export async function createPipelines(device, format) {
    const [
        plmModule, riemannModule, emfModule, updateUModule, updateBModule,
        dtModule, viewModule, colormapModule, compositeModule,
    ] = await Promise.all([
        makeModule(device, 'plasma.reconstruct-plm',   'reconstruct-plm.wgsl'),
        makeModule(device, 'plasma.riemann-hll',       'riemann-hll.wgsl'),
        makeModule(device, 'plasma.compute-emf',       'compute-emf.wgsl'),
        makeModule(device, 'plasma.update-conserved',  'update-conserved.wgsl'),
        makeModule(device, 'plasma.update-b',          'update-b.wgsl'),
        makeModule(device, 'plasma.compute-dt',        'compute-dt.wgsl'),
        makeModule(device, 'plasma.view-field',        'view-field.wgsl'),
        makeModule(device, 'plasma.colormap',          'colormap.wgsl'),
        makeModule(device, 'plasma.composite',         'composite.wgsl'),
    ]);

    const reconstructLayout = reconstructBGL(device);
    const riemannLayout     = riemannBGL(device);
    const emfLayout         = emfBGL(device);
    const updateULayout     = updateConservedBGL(device);
    const updateBLayout     = updateBBGL(device);
    const dtLayout          = dtBGL(device);
    const viewLayout        = viewBGL(device);
    const colormapLayout    = colormapBGL(device);
    const compositeLayout   = compositeBGL(device);

    const mkPipeLayout = (bgl) => device.createPipelineLayout({ bindGroupLayouts: [bgl] });

    const reconstructPlm = device.createComputePipeline({
        label: 'plasma.reconstruct-plm',
        layout: mkPipeLayout(reconstructLayout),
        compute: { module: plmModule, entryPoint: 'main' },
    });
    const riemannHll = device.createComputePipeline({
        label: 'plasma.riemann-hll',
        layout: mkPipeLayout(riemannLayout),
        compute: { module: riemannModule, entryPoint: 'main' },
    });
    const computeEmf = device.createComputePipeline({
        label: 'plasma.compute-emf',
        layout: mkPipeLayout(emfLayout),
        compute: { module: emfModule, entryPoint: 'main' },
    });
    const updateConserved = device.createComputePipeline({
        label: 'plasma.update-conserved',
        layout: mkPipeLayout(updateULayout),
        compute: { module: updateUModule, entryPoint: 'main' },
    });
    const updateB = device.createComputePipeline({
        label: 'plasma.update-b',
        layout: mkPipeLayout(updateBLayout),
        compute: { module: updateBModule, entryPoint: 'main' },
    });

    const dtPipeLayout = mkPipeLayout(dtLayout);
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
        layout: mkPipeLayout(viewLayout),
        compute: { module: viewModule, entryPoint: 'main' },
    });

    const colormap = device.createComputePipeline({
        label: 'plasma.colormap',
        layout: mkPipeLayout(colormapLayout),
        compute: { module: colormapModule, entryPoint: 'main' },
    });

    const composite = device.createRenderPipeline({
        label: 'plasma.composite',
        layout: mkPipeLayout(compositeLayout),
        vertex:   { module: compositeModule, entryPoint: 'vsMain' },
        fragment: { module: compositeModule, entryPoint: 'fsMain', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
    });

    return {
        layouts: {
            reconstruct: reconstructLayout,
            riemann:     riemannLayout,
            emf:         emfLayout,
            updateU:     updateULayout,
            updateB:     updateBLayout,
            dt:          dtLayout,
            view:        viewLayout,
            colormap:    colormapLayout,
            composite:   compositeLayout,
        },
        pipelines: {
            reconstructPlm, riemannHll, computeEmf, updateConserved, updateB,
            dtReset, dtReduce, dtFinalize,
            viewField, colormap, composite,
        },
    };
}

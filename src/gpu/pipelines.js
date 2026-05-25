/**
 * @fileoverview Compute / render pipeline factory (Phase 4).
 *
 * Phase 4 additions:
 *   apply-bcs.wgsl        — ghost-cell fill (one shader, 4 modes × 4 edges)
 *   apply-resistivity.wgsl — η ∇²B per RK3 stage after CT update
 *
 * Phase 4 changes vs Phase 3b:
 *   - Uniforms struct extended (eta, grid_n_total, ghost_w) — handled
 *     transparently in buffers.js.
 *   - Riemann/PPM dispatch ranges no longer mirror grid_n exactly;
 *     sim.js owns the dispatch counts.
 *
 * Bind-group layouts kept vanilla for the upcoming WebGPU→CPU transpiler
 * contract:
 *   - one uniform buffer + N storage buffers per layout
 *   - no dynamic offsets
 *   - no push constants
 *   - no subgroup ops / shared-memory tricks
 *
 * Each shader is prepended with shared-helpers.wgsl. Cache-bust:
 * SHADER_VERSION bumps when any WGSL file is edited.
 */

const SHADER_VERSION = 8;

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

function reconstructPpmBGL(device) {
    return bgl(device, 'plasma.reconstructPpm.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // U0
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // U1
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },  // Bx_face
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },  // By_face
        { binding: 5, visibility: COMPUTE, buffer: RW_STO },  // edge_l_0
        { binding: 6, visibility: COMPUTE, buffer: RW_STO },  // edge_l_1
        { binding: 7, visibility: COMPUTE, buffer: RW_STO },  // edge_r_0
        { binding: 8, visibility: COMPUTE, buffer: RW_STO },  // edge_r_1
        { binding: 9, visibility: COMPUTE, buffer: UNIFORM }, // SweepDir
    ]);
}

function riemannHlldBGL(device) {
    return bgl(device, 'plasma.riemannHlld.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // U0
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // U1
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },  // Bx_face
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },  // By_face
        { binding: 5, visibility: COMPUTE, buffer: RO_STO },  // edge_l_0
        { binding: 6, visibility: COMPUTE, buffer: RO_STO },  // edge_l_1
        { binding: 7, visibility: COMPUTE, buffer: RO_STO },  // edge_r_0
        { binding: 8, visibility: COMPUTE, buffer: RO_STO },  // edge_r_1
        { binding: 9, visibility: COMPUTE, buffer: RW_STO },  // flux_0
        { binding: 10, visibility: COMPUTE, buffer: RW_STO }, // flux_1
        { binding: 11, visibility: COMPUTE, buffer: UNIFORM }, // SweepDir
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

function updateConservedWeightedBGL(device) {
    return bgl(device, 'plasma.updateUWeighted.bgl', [
        { binding: 0,  visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1,  visibility: COMPUTE, buffer: UNIFORM },  // stage_params
        { binding: 2,  visibility: COMPUTE, buffer: RO_STO },   // U0_n
        { binding: 3,  visibility: COMPUTE, buffer: RO_STO },   // U1_n
        { binding: 4,  visibility: COMPUTE, buffer: RO_STO },   // U0_other
        { binding: 5,  visibility: COMPUTE, buffer: RO_STO },   // U1_other
        { binding: 6,  visibility: COMPUTE, buffer: RO_STO },   // flux_x_0
        { binding: 7,  visibility: COMPUTE, buffer: RO_STO },   // flux_x_1
        { binding: 8,  visibility: COMPUTE, buffer: RO_STO },   // flux_y_0
        { binding: 9,  visibility: COMPUTE, buffer: RO_STO },   // flux_y_1
        { binding: 10, visibility: COMPUTE, buffer: UNIFORM },  // dt_buf (uniform: keeps storage count at 10)
        { binding: 11, visibility: COMPUTE, buffer: RW_STO },   // U0_out
        { binding: 12, visibility: COMPUTE, buffer: RW_STO },   // U1_out
    ]);
}

function updateBWeightedBGL(device) {
    return bgl(device, 'plasma.updateBWeighted.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: UNIFORM },   // stage_params
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },    // Bx_n
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },    // By_n
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },    // Bx_other
        { binding: 5, visibility: COMPUTE, buffer: RO_STO },    // By_other
        { binding: 6, visibility: COMPUTE, buffer: RO_STO },    // Ez_edge
        { binding: 7, visibility: COMPUTE, buffer: RO_STO },    // dt_buf
        { binding: 8, visibility: COMPUTE, buffer: RW_STO },    // Bx_out
        { binding: 9, visibility: COMPUTE, buffer: RW_STO },    // By_out
    ]);
}

// New in Phase 4. Ghost-cell fill kernel.
function applyBcsBGL(device) {
    return bgl(device, 'plasma.applyBcs.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },   // bc_uniforms (storage)
        { binding: 2, visibility: COMPUTE, buffer: RW_STO },   // U0
        { binding: 3, visibility: COMPUTE, buffer: RW_STO },   // U1
        { binding: 4, visibility: COMPUTE, buffer: RW_STO },   // Bx_face
        { binding: 5, visibility: COMPUTE, buffer: RW_STO },   // By_face
    ]);
}

// New in Phase 4. Explicit resistive diffusion. Two entry points
// (snapshot + main) share the same BGL; snapshot writes the snap
// buffers and main reads them — so all six face/cell bindings are rw.
// 7 storage bindings total (under the 10-per-stage cap).
function applyResistivityBGL(device) {
    return bgl(device, 'plasma.applyResistivity.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: UNIFORM },   // stage_params
        { binding: 2, visibility: COMPUTE, buffer: RW_STO },    // Bx_face   (dst)
        { binding: 3, visibility: COMPUTE, buffer: RW_STO },    // By_face   (dst)
        { binding: 4, visibility: COMPUTE, buffer: RW_STO },    // U1_out    (dst; Bz in .y)
        { binding: 5, visibility: COMPUTE, buffer: RO_STO },    // dt_buf
        { binding: 6, visibility: COMPUTE, buffer: RW_STO },    // Bx_snap
        { binding: 7, visibility: COMPUTE, buffer: RW_STO },    // By_snap
        { binding: 8, visibility: COMPUTE, buffer: RW_STO },    // U1_snap
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
        { binding: 1, visibility: FRAGMENT,          buffer: RO_STO },  // colored
        { binding: 2, visibility: FRAGMENT,          buffer: RO_STO },  // lic_out
        { binding: 3, visibility: FRAGMENT,          buffer: UNIFORM }, // LicUniforms
    ]);
}

// LIC advect — backward-traces along B-field per interior cell, samples
// noise with bilinear interpolation, writes per-cell luminance.
// Contract documented in lic-advect.wgsl. Transpiler-compatible.
function licAdvectBGL(device) {
    return bgl(device, 'plasma.licAdvect.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },  // Bx_face
        { binding: 2, visibility: COMPUTE, buffer: RO_STO },  // By_face
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },  // noise
        { binding: 4, visibility: COMPUTE, buffer: RW_STO },  // lic_out
        { binding: 5, visibility: COMPUTE, buffer: UNIFORM }, // LicUniforms
    ]);
}

// New in Round 2. Magnetic-pressure-aware energy floor. Runs between
// update-conserved-weighted (step 7) and update-b-weighted (step 8) —
// uses stage-input face B to bound E in the just-written U1. 4 storage
// bindings (well under the 10-per-stage cap).
function energyFloorBGL(device) {
    return bgl(device, 'plasma.energyFloor.bgl', [
        { binding: 0, visibility: COMPUTE, buffer: UNIFORM },
        { binding: 1, visibility: COMPUTE, buffer: RO_STO },   // U0_out
        { binding: 2, visibility: COMPUTE, buffer: RW_STO },   // U1_out
        { binding: 3, visibility: COMPUTE, buffer: RO_STO },   // Bx_face (stage input)
        { binding: 4, visibility: COMPUTE, buffer: RO_STO },   // By_face (stage input)
    ]);
}

export async function createPipelines(device, format) {
    const [
        ppmModule, hlldModule, emfModule, updateUWModule, updateBWModule,
        applyBcsModule, applyResistivityModule, energyFloorModule,
        dtModule, viewModule, colormapModule, compositeModule, licAdvectModule,
    ] = await Promise.all([
        makeModule(device, 'plasma.reconstruct-ppm',          'reconstruct-ppm.wgsl'),
        makeModule(device, 'plasma.riemann-hlld',             'riemann-hlld.wgsl'),
        makeModule(device, 'plasma.compute-emf',              'compute-emf.wgsl'),
        makeModule(device, 'plasma.update-conserved-weighted', 'update-conserved-weighted.wgsl'),
        makeModule(device, 'plasma.update-b-weighted',         'update-b-weighted.wgsl'),
        makeModule(device, 'plasma.apply-bcs',                 'apply-bcs.wgsl'),
        makeModule(device, 'plasma.apply-resistivity',         'apply-resistivity.wgsl'),
        makeModule(device, 'plasma.energy-floor',              'energy-floor.wgsl'),
        makeModule(device, 'plasma.compute-dt',               'compute-dt.wgsl'),
        makeModule(device, 'plasma.view-field',               'view-field.wgsl'),
        makeModule(device, 'plasma.colormap',                 'colormap.wgsl'),
        makeModule(device, 'plasma.composite',                'composite.wgsl'),
        makeModule(device, 'plasma.lic-advect',               'lic-advect.wgsl'),
    ]);

    const reconstructLayout = reconstructPpmBGL(device);
    const riemannLayout     = riemannHlldBGL(device);
    const emfLayout         = emfBGL(device);
    const updateULayout     = updateConservedWeightedBGL(device);
    const updateBLayout     = updateBWeightedBGL(device);
    const applyBcsLayout    = applyBcsBGL(device);
    const applyResLayout    = applyResistivityBGL(device);
    const energyFloorLayout = energyFloorBGL(device);
    const dtLayout          = dtBGL(device);
    const viewLayout        = viewBGL(device);
    const colormapLayout    = colormapBGL(device);
    const compositeLayout   = compositeBGL(device);
    const licAdvectLayout   = licAdvectBGL(device);

    const mkPipeLayout = (bgl) => device.createPipelineLayout({ bindGroupLayouts: [bgl] });

    const reconstructPpm = device.createComputePipeline({
        label: 'plasma.reconstruct-ppm',
        layout: mkPipeLayout(reconstructLayout),
        compute: { module: ppmModule, entryPoint: 'main' },
    });
    const riemannHlld = device.createComputePipeline({
        label: 'plasma.riemann-hlld',
        layout: mkPipeLayout(riemannLayout),
        compute: { module: hlldModule, entryPoint: 'main' },
    });
    const computeEmf = device.createComputePipeline({
        label: 'plasma.compute-emf',
        layout: mkPipeLayout(emfLayout),
        compute: { module: emfModule, entryPoint: 'main' },
    });
    const updateConservedWeighted = device.createComputePipeline({
        label: 'plasma.update-conserved-weighted',
        layout: mkPipeLayout(updateULayout),
        compute: { module: updateUWModule, entryPoint: 'main' },
    });
    const updateBWeighted = device.createComputePipeline({
        label: 'plasma.update-b-weighted',
        layout: mkPipeLayout(updateBLayout),
        compute: { module: updateBWModule, entryPoint: 'main' },
    });
    const applyBcs = device.createComputePipeline({
        label: 'plasma.apply-bcs',
        layout: mkPipeLayout(applyBcsLayout),
        compute: { module: applyBcsModule, entryPoint: 'main' },
    });
    const applyResistivity = device.createComputePipeline({
        label: 'plasma.apply-resistivity',
        layout: mkPipeLayout(applyResLayout),
        compute: { module: applyResistivityModule, entryPoint: 'main' },
    });
    const applyResSnapshot = device.createComputePipeline({
        label: 'plasma.apply-resistivity.snapshot',
        layout: mkPipeLayout(applyResLayout),
        compute: { module: applyResistivityModule, entryPoint: 'snapshot' },
    });
    const energyFloor = device.createComputePipeline({
        label: 'plasma.energy-floor',
        layout: mkPipeLayout(energyFloorLayout),
        compute: { module: energyFloorModule, entryPoint: 'main' },
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

    const licAdvect = device.createComputePipeline({
        label: 'plasma.lic-advect',
        layout: mkPipeLayout(licAdvectLayout),
        compute: { module: licAdvectModule, entryPoint: 'main' },
    });

    return {
        layouts: {
            reconstruct: reconstructLayout,
            riemann:     riemannLayout,
            emf:         emfLayout,
            updateU:     updateULayout,
            updateB:     updateBLayout,
            applyBcs:    applyBcsLayout,
            applyRes:    applyResLayout,
            energyFloor: energyFloorLayout,
            dt:          dtLayout,
            view:        viewLayout,
            colormap:    colormapLayout,
            composite:   compositeLayout,
            licAdvect:   licAdvectLayout,
        },
        pipelines: {
            reconstructPpm, riemannHlld, computeEmf,
            updateConservedWeighted, updateBWeighted,
            applyBcs, applyResistivity, applyResSnapshot,
            energyFloor,
            dtReset, dtReduce, dtFinalize,
            viewField, colormap, composite, licAdvect,
        },
    };
}

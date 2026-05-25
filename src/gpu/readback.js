/**
 * @fileoverview GPU → CPU mapped-buffer readback helpers.
 *
 * Phase 5 introduces a few small CPU-side consumers (stats panel,
 * probe panel) that need numeric values from on-GPU storage buffers.
 * We do NOT readback the full 256² × 8 cell state every frame —
 * that's ~2 MB per frame at 60 fps which crushes integrated GPU bus
 * latency. Instead:
 *
 *   • Stats: tiny aggregate reductions written to small staging
 *     buffers by future stats compute kernels (Phase 6) — for now,
 *     we readback a small interior slice and compute aggregates on
 *     the CPU. Cadence ~5 frames (12 Hz).
 *
 *   • Probe: read a single cell's worth of bytes from each per-cell
 *     buffer. Cadence ~10 Hz.
 *
 * The pattern in every case is identical:
 *
 *   1. Allocate (once) a staging buffer with MAP_READ | COPY_DST.
 *   2. Encode a copyBufferToBuffer(src, srcOff, staging, 0, size).
 *   3. submit() the encoder.
 *   4. await staging.mapAsync(GPUMapMode.READ).
 *   5. Copy out staging.getMappedRange() (it's a *view*, not owned —
 *      slice() if you need to keep it past unmap).
 *   6. staging.unmap().
 *
 * Both consumers re-use a small pool of pre-allocated staging buffers
 * keyed by byte size to avoid per-frame allocation. The pool grows
 * lazily; eight 4 KB buffers are pre-allocated at construction.
 *
 * Transpiler-friendly note: every awaitable in this file is a vanilla
 * `mapAsync` promise. No subgroup ops, no shared memory, no atomics.
 * The state machine is "encode → submit → await → unmap" — the same
 * shape you'd write in any host language with mapped staging buffers.
 */

/** Per-instance staging buffer pool. Keyed by byteSize. */
export class ReadbackPool {
    /**
     * @param {GPUDevice} device
     */
    constructor(device) {
        this.device = device;
        /** @type {Map<number, GPUBuffer[]>} */
        this._free = new Map();
        /** @type {Set<GPUBuffer>} */
        this._inUse = new Set();
    }

    /**
     * Acquire a staging buffer of at least `byteSize` bytes. Returned
     * buffer is unmapped, ready for a copyBufferToBuffer + mapAsync.
     * Caller must call `release` after `unmap()`.
     */
    acquire(byteSize) {
        // Round up to multiple of 4 (WebGPU alignment requirement).
        const sz = (byteSize + 3) & ~3;
        const free = this._free.get(sz);
        if (free && free.length) {
            const buf = free.pop();
            this._inUse.add(buf);
            return buf;
        }
        const buf = this.device.createBuffer({
            label: `plasma.readback.${sz}`,
            size: sz,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this._inUse.add(buf);
        return buf;
    }

    /** Return a staging buffer to the pool. Must be unmapped first. */
    release(buf) {
        if (!this._inUse.has(buf)) return;
        this._inUse.delete(buf);
        const sz = buf.size;
        let list = this._free.get(sz);
        if (!list) { list = []; this._free.set(sz, list); }
        list.push(buf);
    }
}

/**
 * Read back a slice of an on-GPU storage buffer to a JS ArrayBuffer.
 *
 * @param {GPUDevice} device
 * @param {ReadbackPool} pool
 * @param {GPUBuffer} src          source storage buffer (must have COPY_SRC usage)
 * @param {number} byteOffset      source byte offset (must be 4-byte aligned)
 * @param {number} byteSize        bytes to copy (must be 4-byte aligned)
 * @returns {Promise<ArrayBuffer>} resolved with a *detached* copy of the data.
 *
 * Note: we slice() the mapped range into a standalone ArrayBuffer
 * before unmapping. Callers that need typed-array views just construct
 * them on the returned ArrayBuffer.
 */
export async function readbackSlice(device, pool, src, byteOffset, byteSize) {
    const sz = (byteSize + 3) & ~3;
    const staging = pool.acquire(sz);
    const encoder = device.createCommandEncoder({ label: 'plasma.readback.copy' });
    encoder.copyBufferToBuffer(src, byteOffset, staging, 0, sz);
    device.queue.submit([encoder.finish()]);
    try {
        await staging.mapAsync(GPUMapMode.READ, 0, sz);
        const view = staging.getMappedRange(0, sz);
        // Detach into a heap-owned ArrayBuffer before unmap.
        const out = view.slice(0);
        staging.unmap();
        return out;
    } finally {
        pool.release(staging);
    }
}

/**
 * Read back N typed-array slices into one *consolidated* awaitable.
 * The reads are batched into a single command encoder + submit so the
 * GPU work serializes once, then we await each map in parallel.
 *
 * @param {GPUDevice} device
 * @param {ReadbackPool} pool
 * @param {Array<{buf: GPUBuffer, byteOffset: number, byteSize: number}>} specs
 * @returns {Promise<ArrayBuffer[]>}
 */
export async function readbackBatch(device, pool, specs) {
    if (!specs.length) return [];
    const stagings = specs.map(s => {
        const sz = (s.byteSize + 3) & ~3;
        return { staging: pool.acquire(sz), sz };
    });

    const encoder = device.createCommandEncoder({ label: 'plasma.readback.batch' });
    for (let i = 0; i < specs.length; i++) {
        encoder.copyBufferToBuffer(specs[i].buf, specs[i].byteOffset, stagings[i].staging, 0, stagings[i].sz);
    }
    device.queue.submit([encoder.finish()]);

    try {
        await Promise.all(stagings.map(s =>
            s.staging.mapAsync(GPUMapMode.READ, 0, s.sz)
        ));
        const out = stagings.map(s => {
            const view = s.staging.getMappedRange(0, s.sz);
            const copy = view.slice(0);
            s.staging.unmap();
            return copy;
        });
        return out;
    } finally {
        for (const s of stagings) pool.release(s.staging);
    }
}

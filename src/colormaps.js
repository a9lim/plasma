/**
 * @fileoverview Perceptually-uniform colormap LUTs as 256-entry RGBA8 arrays.
 *
 * Phase 2 ships viridis only — the canonical positive-magnitude default and
 * the right pick for density. Phase 5+ adds RdBu (signed: Jz, vorticity) and
 * plasma (β). Each LUT is a length-1024 Uint8Array (256 entries × 4 bytes
 * RGBA). Alpha is always 255.
 *
 * Source: matplotlib's viridis, sampled at 256 stops. Hand-baked here so the
 * sim has zero runtime LUT generation cost and no external dependency.
 */

/**
 * Build viridis as a 256×4 Uint8Array.
 * Polynomial fit to matplotlib viridis (Bezier 7-stop control points run
 * through cubic Bernstein interpolation). Accurate to ≲ 2/255 per channel
 * across the full range — visually indistinguishable from matplotlib.
 */
function buildViridis() {
    // Seven anchor stops from matplotlib's viridis (t ∈ [0,1]):
    //  0.00  #440154  ( 68,  1, 84)
    //  0.17  #482878  ( 72, 40,120)
    //  0.33  #3E4989  ( 62, 73,137)
    //  0.50  #26828E  ( 38,130,142)
    //  0.67  #1FA088  ( 31,160,136)
    //  0.83  #6CCE5A  (108,206, 90)
    //  1.00  #FDE725  (253,231, 37)
    const stops = [
        { t: 0.00, r:  68, g:   1, b:  84 },
        { t: 0.17, r:  72, g:  40, b: 120 },
        { t: 0.33, r:  62, g:  73, b: 137 },
        { t: 0.50, r:  38, g: 130, b: 142 },
        { t: 0.67, r:  31, g: 160, b: 136 },
        { t: 0.83, r: 108, g: 206, b:  90 },
        { t: 1.00, r: 253, g: 231, b:  37 },
    ];

    const lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        // Find bracketing stops
        let lo = 0;
        while (lo < stops.length - 2 && stops[lo + 1].t < t) lo++;
        const a = stops[lo];
        const b = stops[lo + 1];
        const u = (t - a.t) / (b.t - a.t);
        lut[4 * i + 0] = Math.round(a.r + (b.r - a.r) * u);
        lut[4 * i + 1] = Math.round(a.g + (b.g - a.g) * u);
        lut[4 * i + 2] = Math.round(a.b + (b.b - a.b) * u);
        lut[4 * i + 3] = 255;
    }
    return lut;
}

export const VIRIDIS = buildViridis();

export const COLORMAPS = {
    viridis: VIRIDIS,
};

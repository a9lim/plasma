/* ===================================================================
   colors.js — plasma project-specific tokens
   Extends shared-tokens.js with field-visualization palette hooks
   then freezes _PALETTE / _FONT. Phase 1 stub: only the brand red
   is needed at the moment; colormap LUTs land in Phase 6.
   =================================================================== */

// Convenience hue extractors mirroring geon's pattern. Kept for the
// LIC + colormap work in Phase 6; harmless to ship now.
const _hueOf = (hex) => Math.round(_rgb2hsl(..._parseHex(hex))[0]);

// Brand red used by the Phase 1 clear pipeline; also the accent for
// the eventual signed-field colormap midpoint highlight.
_PALETTE.plasmaBrand = _PALETTE.accent;

// Placeholders for the Phase 6 colormaps — values picked once the LUTs
// are baked. Currently unused; declared so later additions don't shift
// the shape of _PALETTE under consumers.
_PALETTE.fieldPositive = _hueOf(_PALETTE.extended.red);
_PALETTE.fieldNegative = _hueOf(_PALETTE.extended.blue);

_freezeTokens();

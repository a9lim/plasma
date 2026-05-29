/**
 * @fileoverview Shared sidebar control builders.
 *
 * The DOM idioms every plasma sidebar section is built from — sections,
 * slider rows, log/snap-off sliders, and mode-button groups. Extracted
 * from physics-panel.js during the Session 22+ coherence pass so the
 * Settings tab's Numerics / Render sections (built in ui.js) and the
 * Physics tab (physics-panel.js) share one set of builders instead of
 * two copies.
 *
 * These builders are bounds-agnostic: callers pass `{ min, max, step }`
 * (linear) or `{ lo, hi, step }` (log) — spread the matching entry from
 * config.js `SLIDER_BOUNDS` so every range stays centralized.
 *
 * Markup matches the static Settings-tab idiom in index.html
 * (`.panel-section` + `.group-label` + `.control-group` / `.mode-toggles`)
 * so styling lines up without per-row CSS. shared-forms.js wires the
 * sliding indicator on mode groups. No `innerHTML`.
 */

/** Create a DOM element with optional class + text. */
export function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

/** A `.panel-section` with an `<h2 class="group-label">` header. */
export function section(label) {
    const sec = el('div', 'panel-section');
    sec.append(el('h2', 'group-label', label));
    return sec;
}

/**
 * Standard slider row: label on top with a right-aligned value span,
 * native range underneath. Mirrors the #ctrl-eta block in index.html.
 * `opts`: { min, max, step, value, format, onChange, hint }.
 * Returns { slider, valSpan } for callers that need later updates.
 */
export function sliderRow(parent, label, opts) {
    const { min, max, step, value, format, onChange, hint } = opts;
    const wrap = el('div', 'control-group');
    const lbl = document.createElement('label');
    lbl.append(el('span', null, label));
    const valSpan = el('span', 'slider-value', format(value));
    lbl.append(valSpan);
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.addEventListener('input', () => {
        const v = parseFloat(slider.value);
        valSpan.textContent = format(v);
        onChange(v);
    });
    wrap.append(lbl, slider);
    if (hint) wrap.append(el('p', 'panel-hint', hint));
    parent.append(wrap);
    return { slider, valSpan };
}

/**
 * Log-scale slider with a snap-to-off floor. `lo` is the lowest active
 * decade — the range extends 0.5 below it for the off zone; values at or
 * below `lo - 0.05` render as "off" and write 0.
 *
 * `opts`: { lo, hi, step, getScalar, setScalar, flag?, sim, hint }.
 * When `flag` is given, the slider also arms/clears that FLAG_* bit (so a
 * source physics term can't be left half-engaged). When `flag` is omitted
 * the slider only writes the scalar — used for snap-to-0 controls that
 * gate purely on the value (e.g. η-anomalous α).
 *
 * `getScalar` seeds the slider position from the live sim value so
 * re-opening reflects whatever the preset set.
 */
export function epSlider(parent, label, opts) {
    const { lo = -6, hi = 0, step = 0.25,
            getScalar, setScalar, flag, sim, hint } = opts;
    const offBoundary = lo - 0.05;
    const cur = getScalar();
    const valueAt = cur > 0 ? Math.log10(cur) : (lo - 0.5);
    return sliderRow(parent, label, {
        min: lo - 0.5, max: hi, step,
        value: valueAt,
        format: v => (v <= offBoundary ? 'off' : '1e' + v.toFixed(2)),
        onChange: v => {
            if (v <= offBoundary) {
                setScalar(0);
                if (flag != null) sim.setPhysicsFlag(flag, false);
            } else {
                setScalar(Math.pow(10, v));
                if (flag != null) sim.setPhysicsFlag(flag, true);
            }
        },
        hint,
    });
}

/**
 * Mode-button group on `.mode-toggles` so shared-forms wires the sliding
 * indicator. `opts`: { dataAttr, buttons: [{ value, label, active }],
 * onChange }. Returns the group element.
 */
export function modeRow(parent, label, opts) {
    const { dataAttr, buttons, onChange } = opts;
    const wrap = el('div', 'control-group');
    wrap.append(el('label', null, label));
    const group = el('div', 'mode-toggles');
    group.setAttribute('data-scope', dataAttr);
    for (const b of buttons) {
        const btn = document.createElement('button');
        btn.className = 'mode-btn' + (b.active ? ' active' : '');
        btn.setAttribute('data-' + dataAttr, b.value);
        btn.textContent = b.label;
        group.append(btn);
    }
    wrap.append(group);
    parent.append(wrap);
    if (typeof _forms !== 'undefined' && _forms.bindModeGroup) {
        _forms.bindModeGroup(group, dataAttr, onChange);
    }
    return group;
}

/**
 * Toggle row using the standard `.tog-wrap` markup so it picks up
 * shared-base.css automatically. `opts`: { checked, onChange }.
 * Returns the checkbox input.
 */
export function toggleRow(parent, label, opts) {
    const { checked, onChange } = opts;
    const row = el('div', 'ctrl-row');
    const lbl = document.createElement('label');
    lbl.append(el('span', null, label));
    const togWrap = el('div', 'tog-wrap');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.setAttribute('role', 'switch');
    input.checked = !!checked;
    input.setAttribute('aria-checked', checked ? 'true' : 'false');
    const togLbl = el('label', 'tog');
    togLbl.append(el('span', 'tog-thumb'));
    togWrap.append(input, togLbl);
    // Associate the toggle label with the input for click-to-toggle.
    const id = 'tog-' + label.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    input.id = id;
    togLbl.setAttribute('for', id);
    // Keep aria in sync, then route the change through the shared binding
    // (haptics + consistency) when available.
    input.addEventListener('change', () => {
        input.setAttribute('aria-checked', input.checked ? 'true' : 'false');
    });
    if (typeof _forms !== 'undefined' && _forms.bindToggle) {
        _forms.bindToggle(input, onChange);
    } else {
        input.addEventListener('change', () => onChange(input.checked));
    }
    row.append(lbl, togWrap);
    parent.append(row);
    return input;
}

# Plasma documentation

- [`HANDOFF.md`](HANDOFF.md) — current product shape, verification gate,
  known limits, and next-change checklist. Read this first when resuming work.
- [`../AGENTS.md`](../AGENTS.md) — detailed architecture, buffer/shader
  contracts, presets, controls, and project rules.
- [`../tests/README.md`](../tests/README.md) — production validation,
  convergence, and Harris diagnostic instructions.
- [`sessions/`](sessions/) — historical retrospectives for Sessions 2-21.
  These explain the sequence of fixes but are not current specifications.

## Session index

| File | Historical topic |
|------|------------------|
| [`sessions/session-2.md`](sessions/session-2.md) | Initial verification and engine fixes |
| [`sessions/session-3.md`](sessions/session-3.md) | P0/P1 polish sweep |
| [`sessions/session-4.md`](sessions/session-4.md) | Gardiner-Stone upwind CT EMF |
| [`sessions/session-5.md`](sessions/session-5.md) | PPM primitive cache and LIC normalization |
| [`sessions/session-6.md`](sessions/session-6.md) | Characteristic-variable PPM limiting |
| [`sessions/session-7.md`](sessions/session-7.md) | Primitive-space PPM safety net |
| [`sessions/session-8.md`](sessions/session-8.md) | Wave-1 diagnostic hardening |
| [`sessions/session-9.md`](sessions/session-9.md) | RKL2 ghost handling |
| [`sessions/session-10.md`](sessions/session-10.md) | RKL2 dt feedback |
| [`sessions/session-11.md`](sessions/session-11.md) | Curl-resistivity and corner BCs |
| [`sessions/session-12.md`](sessions/session-12.md) | Harris ghost/floor fixes |
| [`sessions/session-13.md`](sessions/session-13.md) | RKL2 substep correctness |
| [`sessions/session-14.md`](sessions/session-14.md) | Extended-physics breadth pass |
| [`sessions/session-15.md`](sessions/session-15.md) | Per-preset opt-in and subcycling |
| [`sessions/session-16.md`](sessions/session-16.md) | Transport, cooling, Ohm, and gravity realism |
| [`sessions/session-17.md`](sessions/session-17.md) | Heating, ambipolar, Biermann, viscosity, and geometry |
| [`sessions/session-18.md`](sessions/session-18.md) | Microphysics, dual energy, unified Ohm, and driven flow |
| [`sessions/session-19.md`](sessions/session-19.md) | Validation, calibration, and source coupling |
| [`sessions/session-20.md`](sessions/session-20.md) | Radiation, gravity, multigrid, and cylindrical balance |
| [`sessions/session-21.md`](sessions/session-21.md) | Data-backed cooling, source safety, and toroidal balance |

Session 1 was the initial scaffold and has no retrospective. Session 22 changes
are summarized in the current handoff and source comments rather than a
separate session file.

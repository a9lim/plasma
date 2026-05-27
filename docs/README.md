# plasma — docs

* [`HANDOFF.md`](HANDOFF.md) — current status, what's left to do
  (Phase 7 polish, Phase 8 parent-repo wiring), open concerns, and
  the references the design draws on. This is the doc to read first
  on opening the project.
* [`sessions/`](sessions/) — per-session retrospectives. Each file
  captures the bugs surfaced, the fixes applied, the math behind
  them where relevant, and the concerns flagged for the next session.

The implementation plan lives at
`~/.claude/plans/geon-currently-uses-cpu-abstract-cat.md` — that's
the source of truth for design decisions; the docs here are
specifically about what's been done, what's left, and what to watch
for.

## Session index

| File | Topic |
|------|-------|
| [`sessions/session-2.md`](sessions/session-2.md)  | Verification + 5 engine bug fixes |
| [`sessions/session-3.md`](sessions/session-3.md)  | P0/P1 polish sweep (12 fixes) |
| [`sessions/session-4.md`](sessions/session-4.md)  | Gardiner-Stone upwind CT EMF |
| [`sessions/session-5.md`](sessions/session-5.md)  | PPM primitive cache + LIC contrast normalization |
| [`sessions/session-6.md`](sessions/session-6.md)  | Characteristic-variable PPM limiting |
| [`sessions/session-7.md`](sessions/session-7.md)  | Primitive-space safety net for characteristic PPM |
| [`sessions/session-8.md`](sessions/session-8.md)  | Phase 2 Wave 1 + diagnostic-driven bug hunt |
| [`sessions/session-9.md`](sessions/session-9.md)  | RKL2 ghost-handling bugfix (Harris recovery, partial) |
| [`sessions/session-10.md`](sessions/session-10.md) | RKL2 dt-feedback staleness fix |
| [`sessions/session-11.md`](sessions/session-11.md) | RKL2 curl(η J) on Yee grid + corner BC composition |
| [`sessions/session-12.md`](sessions/session-12.md) | Fifth Harris bug: dst-ghost staleness + ρ-floor momentum blowup |
| [`sessions/session-13.md`](sessions/session-13.md) | RKL2 substep correctness (writeBuffer race + 2D FE bound + MDK margin) |
| [`sessions/session-14.md`](sessions/session-14.md) | Extended physics breadth pass |
| [`sessions/session-15.md`](sessions/session-15.md) | Per-preset source opt-in + subcycling hardening |
| [`sessions/session-16.md`](sessions/session-16.md) | Transport, cooling, Ohm, and gravity realism pass |
| [`sessions/session-17.md`](sessions/session-17.md) | Second realism layer: heating, ambipolar, Biermann, viscosity, geometry |
| [`sessions/session-18.md`](sessions/session-18.md) | Tabulated microphysics, dual energy, unified Ohm, driven wind/cloud |

Session 1 was the initial scaffold (no retro — see the Phase 1
commit message instead).

Shader and code comments that say "see HANDOFF Session N" or just
"Session N" point at the corresponding `sessions/session-N.md`.

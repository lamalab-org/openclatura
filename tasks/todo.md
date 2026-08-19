# Fused Hydrocarbon Retained Parents

- [x] Inspect the supplied fused-ring corpus and current graph-template engine.
- [x] Create a feature branch from `tests`.
- [x] Add graph-backed retained support for the requested fused hydrocarbons.
- [x] Add a data-driven acene-series rule for higher linear acenes.
- [x] Add graph-remapping, public API, and corpus regression tests.
- [x] Run full tests, OPSIN round trips, and performance comparisons.
- [x] Record results and commit the implementation.

## Review

- Added six OPSIN-locanted, graph-only fusion-parent templates and a generated
  P-25.1.2.1 acene family covering tetracene through nonacene.
- Preserved PIN-style `1,2-dihydroacenaphthylene` for acenaphthene and existing
  `9H-fluorene`; replaced the obsolete emitted `naphthacene` with `tetracene`.
- Added topology indexing, dynamic-frontier isomorphism, and per-molecule match
  caching. The retained/fused benchmark improved from 9.805 to 6.414 ms per
  molecule on the same 180-name workload.
- Focused fused tests: 56 passed. Retained derivative tests: 67 passed.
- Full suite: 3532 passed, 5 skipped, 1 xfailed. The only sandbox failure was
  the existing two-process semaphore test; its escalated rerun passed 2/2.
- Supplied 175-entry fused corpus: 171/175 structurally correct and 100/115
  asserted PIN strings (the four remaining hard failures are tetrapyrroles).
- Wheel build succeeded and includes both new runtime data files.

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

# Remaining Fused Hydrocarbon PIN Coverage

- [x] Add a data-driven generated polyaphene family and graph/round-trip tests.
- [x] Commit the polyaphene family independently.
- [x] Add graph-backed templates for the remaining irregular retained PAHs and fusion parents.
- [x] Commit the irregular PAH templates independently.
- [x] Add a data-backed preferred-name policy separating emitted PINs from accepted aliases.
- [x] Commit the retained-name/PIN policy independently.
- [x] Run focused fused-corpus, full-suite, OPSIN, and performance regression checks.

## Review

- Added a generated P-25.1.2 polyaphene family from pentaphene through
  decaphene. Runtime construction emits the complete standard locant graph;
  no member is recognized through a SMILES or SMARTS key.
- Added eight OPSIN-locanted graph templates: tetraphenylene, rubicene,
  trinaphthylene, pyranthrene, ovalene, benzo[e]pyrene,
  benzo[j]fluoranthene, and cyclopenta[cd]pyrene.
- Added an immutable retained-parent output policy. Standalone indane and
  indoline now emit `2,3-dihydro-1H-indene` and
  `2,3-dihydro-1H-indole`; substituted and spiro component contexts retain
  their established component spellings. `dibenzo[a,h]anthracene` remains an
  accepted alias while the requested emitted spelling is
  `dibenz[a,h]anthracene`.
- Supplied 175-entry corpus: 171/175 structurally correct and 113/115 asserted
  PIN strings. The four hard failures remain the out-of-scope tetrapyrrole
  parents. The sole structurally correct string disagreement is the corpus's
  `dibenzo` spelling versus the configured `dibenz` output.
- Focused fused tests: 239 passed. Focused OPSIN template tests: 26 passed.
  The package OPSIN corpus gate matched 106/106 names.
- Full suite: 3572 passed, 5 skipped, 1 xfailed. The sandbox-blocked existing
  two-process semaphore test passed separately outside the sandbox.
- Strengthened topology indexing with degree-class edge counts. A five-pair
  benchmark on 171 fused structures improved from 1.804 s at `c281e33` to
  1.638 s, a 9.3% speedup despite the expanded template registry.

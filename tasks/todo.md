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

# Retained Tetrapyrrole Macrocycles

- [x] Inspect the porphyrin/porphine/corrin corpus rows and OPSIN locant graphs.
- [x] Add a topology-indexed, graph-backed retained macrocycle registry.
- [x] Route exact retained macrocycle parents before ordinary parent selection.
- [x] Add alias, atom-order, metadata, negative-match, and OPSIN round-trip tests.
- [x] Run focused corpus, full-suite, package-data, and performance verification.
- [x] Document results and commit the implementation.

## Review

- Added OPSIN-locanted graph templates for `porphyrin` and `corrin`; `porphine`
  and `21H,23H-porphine` are accepted aliases in the porphyrin template policy.
- Runtime matching uses heavy-atom count and topology indexes before exact
  labelled-graph isomorphism. It contains no SMILES/SMARTS or input-name keys.
- Exact-parent matching also checks the template's mancude double-bond count,
  preventing hydrogenated derivatives from being mislabeled as the base parent.
- The supplied fused/retained corpus now has 175/175 OPSIN structural matches
  and 114/115 asserted preferred strings. The remaining string disagreement is
  the pre-existing `dibenz` versus corpus `dibenzo` spelling.
- Macrocycle tests: 10 passed. Full suite: 3582 passed, 5 skipped, 1 xfailed;
  the sandbox-only multiprocessing failure passed separately (2/2).
- A 3,000-name ordinary-molecule benchmark measured no slowdown from the O(1)
  candidate-size gate (2.471 s enabled versus 2.591 s monkeypatched out; noise
  dominates this small negative delta).
- The wheel contains the runtime module and JSON registry, while the offline
  OPSIN generator remains excluded with the development scripts.

# Retained Template Derivative Audit

- [x] Audit branch template architecture, matcher complexity, and production gates.
- [x] Inventory every graph template/family introduced relative to `origin/tests`.
- [x] Replace exact-only macrocycle shortcut routing with shared retained-parent resolution.
- [x] Add deterministic random-sidechain derivatives for every newly added template.
- [x] Assert the retained core name survives and von Baeyer fallback is never selected.
- [x] Run OPSIN, focused, full-suite, and performance regression checks.
- [x] Record findings and commit/push the implementation.

## Review

- Audited 29 branch-added retained graph parents: 15 explicit/generated PAH
  parents, six acenes, six polyaphenes, porphyrin, and corrin.
- Removed the exact-component macrocycle shortcut. Macrocycles now use the
  ordinary retained-parent, numbering, additive-operation, substituent, and
  assembly pipeline, so derivatives inherit the same behavior as fused PAHs.
- Added the data-backed `pre_descriptor_selection` capability. Only retained
  graph families that require conventional numbering before generic polycycle
  proof generation opt in; existing retained heterocycles retain their old
  discovery path.
- Exposed every valid macrocycle automorphism to parent numbering instead of
  accepting one arbitrary graph match. Matching remains topology-indexed and
  cached on the molecule.
- Made exact edge-order matching an explicit graph-template policy. Corrin
  uses exact single/double edges; porphyrin retains Kekule-equivalent matching.
  Template validation rejects incomplete exact edge specifications.
- Added deterministic derivative property tests with one to three randomly
  selected alkyl branches, five branch topologies, and randomized atom order.
  The tests assert the exact selected core atom set, retained parent identity,
  and absence of a von Baeyer descriptor.
- Fixed an audit-discovered overreach where all retained fused saturated sites
  were treated as inherent parent saturation; that incorrectly removed the
  required `9H-` from xanthene. Inherent saturation now remains macrocycle
  metadata only where the retained parent definition requires it.
- Full suite: 3613 passed, 5 skipped, 1 xfailed. The sandbox-only two-process
  semaphore test passed separately outside the sandbox (2/2). Round-trip suite:
  632 passed; package OPSIN corpus gate: 106/106 matched.
- A 3600-name mixed ordinary/ring benchmark measured 3.251 s after the change
  versus 3.261 s at `6e71134`; no measurable performance regression.
- Remaining architectural debt: the shared exact graph-template kernel still
  carries fused-specific type names (`RetainedFusedGraphTemplate`) even though
  macrocycles now reuse it. Renaming/extracting that stable kernel can be a
  separate compatibility-focused refactor; it is not duplicated at runtime.

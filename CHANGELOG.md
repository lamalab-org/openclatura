# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2026-10-08

### Added

- Added generic, graph-backed nomenclature for oxygen and its sulfur, selenium,
  tellurium, and polonium analogues across simple chalcogenols, carbonyls,
  acids, esters, peroxides, peroxyacids, anhydrides, and oxoacid derivatives.
- Added composable support for nitrogen-containing chalcogen families,
  including amine and imine chalcogenides, imides, hydrazides, and acyl
  pseudohalides.
- Added typed functional-group descriptors to decision traces and both public
  description renderers, retaining center, ligand, element, bond, family, and
  derivative metadata.
- Added systematic regression coverage for mixed group-16 combinations,
  structural roles, perception completeness, citation compatibility, and
  human-readable descriptions.

### Changed

- Moved chalcogen capabilities, vocabulary, precedence, suffixes, prefixes,
  and citation routes into the existing data-backed naming registry.
- Generalized perception, parent selection, and assembly around typed graph
  roles so functional-group behavior is selected structurally rather than by
  molecule-specific cases.

### Fixed

- Preserved principal-group precedence and acyl ownership across mixed esters,
  anhydrides, peroxy linkages, imides, hydrazides, cyclic derivatives, and
  charged chalcogen species.
- Preserved established oxygen and sulfur names while extending the same
  structural routes to the heavier group-16 elements.
- Kept retained and fused parent selection stable when chalcogen functional
  groups are attached to ring systems.

## [0.4.1] - 2026-10-08

### Added

- Added trusted PyPI publishing and production website deployment workflows.

### Fixed

- Hardened aromatic-input validation, carbamimidoyl stereochemistry, recursive
  acylamino naming, and fused-bond discovery.

## [0.4.0] - 2026-10-01

### Added

- Added an audited, graph-backed systematic fusion pipeline covering higher-order
  fused systems, Hantzsch-Widman components, skeletal replacement, retained
  bridges, and fused derivatives with hydrogenation, charge, and oxo states.
- Added structural descriptions for retained, fused, spiro, and bridged parents,
  including atom-linked ring bases, shared bonds, and von Baeyer topology.
- Expanded deterministic PR regression coverage to one million PubChem and one
  million ZINC22 structures, alongside the complete QM9 evaluation.
- Added a reproducible 100,000-molecule OPSIN-verified PubChem performance
  corpus and corpus-generation manifests.

### Changed

- Routed eligible polycycles through bounded, graph-proven candidate generation,
  numbering, composition, and reconstruction audits.
- Made large evaluation regressions reuse stored names and OPSIN outcomes,
  invoking OPSIN only when a generated name changes.
- Made performance and evaluation jobs run only after the fast, lint, and RDKit
  compatibility suites pass, with isolated downloadable reports and summaries.
- Refreshed the documented naming examples and measured OPSIN round-trip
  coverage for the current implementation.

### Fixed

- Corrected fusion numbering, indicated-hydrogen placement, pi-bond composition,
  skeletal-replacement state, and derivative ownership across fused and spiro
  parent families.
- Corrected parent selection and naming for polyoxide chains, dispiro systems,
  charged ligands, phosphinic acids, and retained indane and indoline derivatives.
- Preserved required ligand and substituent parentheses after locant elision and
  retained recursive trace evidence for ester and sulfonate modifiers.

## [0.3.3] - 2026-09-11

### Fixed

- Resolve the public and executable version from the active checkout or its
  installed package metadata so release bumps are not duplicated in code and
  unrelated installations cannot supply a stale version.

## [0.3.2] - 2026-09-03

### Added

- Added graph-backed retained-parent coverage for fused hydrocarbons, including
  the acene and polyaphene series and irregular polycyclic aromatic
  hydrocarbons such as fluoranthene, pyrene, chrysene, benzo-fused PAHs, and
  related derivatives.
- Added graph-backed retained macrocycle support for the porphyrin/porphine and
  corrin families, with conventional locant maps selected before generic
  polycycle descriptor construction where required.
- Added a data-backed retained-name policy that separates preferred output
  names from accepted aliases and records hydrogenated-parent spellings such
  as `2,3-dihydro-1H-indene` and `2,3-dihydro-1H-indole`.
- Added optional graph-proven omission of redundant constitutional locants.
  The public `name`, `name_mol`, and `name_many` APIs expose this through
  `omit_redundant_locants`, enabled by default.
- Added a reproducible 5,000-molecule, OPSIN-verified single-thread benchmark,
  corpus-generation tooling, and a pull-request performance gate that compares
  paired measurements against the target revision.
- Added retained-parent derivative, atom-order invariance, graph-isomorphism,
  OPSIN round-trip, locant-elision, and benchmark regression tests.

### Changed

- Unified fused parents and macrocycles behind one lazy, topology-indexed
  retained-graph registry with shared matching, numbering, metadata, and cache
  invalidation behavior.
- Generated regular acene and polyaphene templates from compact series data
  while retaining explicit graph templates for irregular parent systems.
- Moved morphology, functional-role, charge, suffix, and connection-boundary
  handling out of broad legacy literal post-processing and into structured
  assembly and data-backed naming rules.
- Made redundant-locant decisions conservative graph proofs based on labelled
  parent automorphisms and the complete set of suffix, unsaturation, and
  substituent features. Searches use strict work limits and retain explicit
  locants whenever uniqueness cannot be proven cheaply.
- Added graph mutation APIs that invalidate retained-parent and other derived
  caches, preventing stale template matches after atom or bond updates.

### Fixed

- Corrected additive-hydrogen accounting for retained parents that already
  contain inherent saturated positions, and prevented indicated-hydrogen
  relocation from consuming newly hydrogenated sites.
- Preserved exact retained-template proofs while keeping an audited von Baeyer
  fallback for relaxed fused-PAH topology matches that fail later retained-name
  chemistry checks.
- Corrected retained-parent metadata lookup for exact template names and
  unambiguous preferred output spellings while leaving tautomer-ambiguous
  aliases to graph-derived metadata.
- Prevented redundant-locant omission when parent symmetry, attachment
  capacity, supplied locant maps, or bounded-search limits leave more than one
  constitutional arrangement possible.

## [0.3.1] - 2026-08-14

### Changed

- Excluded the hosted website bundle, bundled Java runtime, and local lockfile
  from source distributions, reducing the archive size substantially.

### Fixed

- Replaced repository-relative README links with absolute GitHub URLs so the
  demo image and documentation links render correctly on PyPI.

## [0.3.0] - 2026-08-13

### Added

- Expanded retained fused-ring coverage with a substantially larger set of
  production templates, including `2H-1-benzothiopyran`.
- Added regression coverage for retained fused parents and their derivatives.
- Added a project demo and published QM9, PubChem, and ZINC22 round-trip
  coverage results to the README.

### Changed

- Reorganized parser resources as package data and split tests into unit and
  round-trip suites.
- Simplified retained-ring lookup and audit reconstruction around the expanded
  template data.

### Fixed

- Corrected indicated-hydrogen handling for oxo-substituted xanthene systems.
- Corrected amino, ammonio, and aminium selection for charged nitrogen groups,
  including zwitterions.
- Corrected spiro assembly when parent and side components contribute
  substituents or suffixes, preventing duplicated or separately rendered terms.
- Fixed retained fused-parent recognition and related QM9 naming regressions.

## [0.2.1] - 2026-08-09

### Added

- Added an OPSIN-free reconstruction audit with relative and parent-level
  stereochemistry checks.
- Added chain stems through 1,000 atoms and reverse stem lookup.
- Expanded retained parent, retained substituent, heterocycle, spiro, bridged,
  and skeletal-replacement coverage.
- Added skeletal replacement prefixes for magnesium, calcium, lithium, sodium,
  and potassium.

### Changed

- Made parent selection structural and independent of input atom order.
- Introduced unique-locant elision and a lighter naming pipeline.
- Extended the audit and rule tables to cover the names emitted by the naming
  engine more consistently.

### Fixed

- Corrected numerous stereochemistry, spiro, hydrazone, retained-parent,
  hypervalent-center, and prefix-assembly edge cases.
- Fixed QM9 naming regressions and retained-ring derivative handling.

## [0.2.0] - 2026-07-24

### Added

- Added the project website.
- Expanded substituent support for retained ring systems.

### Changed

- Simplified graph handling and removed unused naming classes.
- Improved naming throughput by approximately twofold.

### Fixed

- Corrected retained-ring graph definitions and updated amide test expectations
  for the revised nomenclature.

## [0.1.5] - 2026-07-22

### Added

- Added direct naming from existing RDKit molecule objects.
- Added recursive subgraph naming and structured rendered-substituent results.

### Fixed

- Removed unnecessary parentheses from substituent names without
  stereodescriptors.
- Made rendered substituent names immutable to prevent accidental mutation.

## [0.1.4] - 2026-07-20

### Added

- Expanded retained fused-heterocycle and derivative support, including
  phenazine, phenanthroline, acridine, carbazole, purine, indazole, and xanthene
  systems.
- Added contribution guidance and CI verification tests.

### Changed

- Optimized retained fused-system normalization.

### Fixed

- Improved RDKit normalization idempotence and hydrogen assignment.

## [0.1.3] - 2026-07-16

### Changed

- Updated the documentation and citation details.
- Marked the project as beta.

## [0.1.2] - 2026-07-03

### Changed

- Added diagnostics when normalized and input SMILES differ.
- Limited OPSIN error suppression so unexpected failures remain visible.

## [0.1.1] - 2026-07-01

### Added

- Added typed naming results, batch naming, a CLI, and optional OPSIN
  round-trip verification.
- Added natural-language descriptions and atom-level trace metadata.
- Added the FastAPI service, Docker support, and container CI checks.
- Added fuzz, dataset, golden-output, and RDKit compatibility tests.
- Expanded naming support for charges, spiro systems, stereochemistry, nitrogen
  functional groups, and retained biphenyl names.

### Changed

- Renamed the package to openclatura and adopted a `src` layout.
- Made OPSIN integration tolerate unavailable Java installations.

## [0.1.0] - 2026-05-08

### Added

- Initial deterministic IUPAC name generation from molecular structures.

[Unreleased]: https://github.com/lamalab-org/openclatura/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/lamalab-org/openclatura/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/lamalab-org/openclatura/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/lamalab-org/openclatura/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/lamalab-org/openclatura/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/lamalab-org/openclatura/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/lamalab-org/openclatura/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/lamalab-org/openclatura/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/lamalab-org/openclatura/compare/db7d1d4...7ed75cb
[0.2.0]: https://github.com/lamalab-org/openclatura/compare/8fe914e...db7d1d4
[0.1.5]: https://github.com/lamalab-org/openclatura/compare/4ee2ffb...8fe914e
[0.1.4]: https://github.com/lamalab-org/openclatura/compare/ba56a80...4ee2ffb
[0.1.3]: https://github.com/lamalab-org/openclatura/compare/4eba115...ba56a80
[0.1.2]: https://github.com/lamalab-org/openclatura/compare/1b0db76...4eba115
[0.1.1]: https://github.com/lamalab-org/openclatura/compare/0e86ce6...1b0db76
[0.1.0]: https://github.com/lamalab-org/openclatura/commit/0e86ce6

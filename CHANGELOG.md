# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/lamalab-org/openclatura/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/lamalab-org/openclatura/compare/v0.2.1...v0.3.0

# PR 90 retained-hydrogenation regression

- [x] Reproduce representative hidden-test failures and trace retained-template selection.
- [x] Design graph-derived additive-hydrogen operations tied to the selected locant map.
- [x] Implement the operation without rendered-name matching or unbounded graph search.
- [x] Add focused unit and round-trip regression tests for partial and extensive hydrogenation.
- [x] Recheck extracted PR 90 discrepancies and confirm no new naming regressions.
- [x] Run the complete test, lint, and focused performance suites.

## Review

- Corrected retained-parent hydrogen accounting so template-inherent saturated
  sites are excluded from both the observed pool and supported-H capacity.
- Kept indicated-H relocation based on the complete observed saturated set,
  preventing newly hydrogenated sites from replacing inherent parent sites.
- Restricted relaxed pre-descriptor matching to macrocycles while preserving
  exact retained-template proof for fused PAHs.
- Rechecked 2,493 extracted PR regressions: 2,485 reproduce the baseline name;
  the remaining 8 alternate names all round-trip through OPSIN; 0 naming errors.
- Full suite: 3,649 passed, 5 skipped, 1 xfailed. The sandbox-blocked process
  test passed separately outside the sandbox (2 passed).
- Ruff and `git diff --check` pass.
- CI-equivalent 5,000-molecule benchmark passes: head median 25.954 s versus
  base median 70.592 s (63.46% lower paired elapsed time).

# Structurally difficult PubChem failure analysis

## Scope

This rerun evaluates all 38,650 failures saved from the Openclatura 0.4.0
Orthonym PubChem scaffold run. Names are decoded with OPSIN 2.9.0 using
`allow_radicals=True` and compared by full InChIKey.

## Result

| Outcome | Before | After | Change |
| --- | ---: | ---: | ---: |
| Exact match | 13,923 | 14,803 | +880 |
| Constitutional match | 1,743 | 1,710 | -33 |
| Graph mismatch | 9,005 | 8,621 | -384 |
| Naming error | 9,335 | 7,052 | -2,283 |
| OPSIN structure invalid | 51 | 42 | -9 |
| OPSIN unparseable | 4,593 | 6,422 | +1,829 |

The apparent increase in OPSIN-unparseable rows is mostly a stage transition:
2,090 records containing an implicit `[HH]` component are now named as
`dihydrogen`, but OPSIN does not accept the resulting complete multicomponent
name. These are naming-coverage gains, not exact reconstruction gains.

## Generic fixes

- Structured parent charge operations now own the locants they render. Legacy
  rewrites no longer repeat `-ium` or `-ide` suffixes, including nested charged
  parents. The two explicit duplicate-signature families account for 102 exact
  recoveries; the same ownership correction also fixes broader nested charge
  loss and duplication patterns.
- Positive amide nitrogen is represented by data-backed `amidium` and
  `carboxamidium` suffixes. Of 112 changed names in this family, 108 become
  exact matches. Charge on the parent attachment atom is explicitly excluded.
- A one-atom `[HH]` RDKit representation is recognized generically as
  dihydrogen. This advances 2,090 records from naming failure to OPSIN
  evaluation.
- Partial retained-parent locant maps are rejected before they can be attached
  to a larger selected parent. All 226 affected records now fail at the honest
  audited-polycycle boundary rather than with an invalid retained proof.

## Remaining large families

- 2,673 naming failures are polycyclic parents without an audited descriptor.
  These are mostly mixed bridged/spiro cages and should remain fail-closed until
  descriptor construction and reconstruction auditing are extended together.
- 1,712 naming failures contain bonded elements outside the supported covalent
  nomenclature tier. The largest are Zr (374), Ru (212), Pt (155), Ti (123),
  Pd (109), V (81), Zn (74), and W (74). This is primarily a coordination
  nomenclature project, not an organic parent-selection patch.
- 209 molecules still hit the per-molecule timeout.
- Among OPSIN-unparseable names, 2,350 contain `dihydrogen`, 2,581 contain a
  polycycle descriptor, 746 contain phosphorus nomenclature, and 629 contain
  boron nomenclature. These overlap.
- OPSIN diagnostics include 979 uninterpretable grammar failures, 541 invalid
  bond/stereo references, 461 invalid locants, 426 unphysical valences, 327
  unsaturation-locant failures, and 270 failed unlocanted substitutions.

The next productive code targets are the repeated polycycle bond/locant
diagnostics and non-metal unphysical-valence families. Bonded transition metals
and OPSIN's multicomponent hydrogen parsing should be tracked separately from
ordinary organic naming correctness.

# Namer Refactor Notes

This refactor separates extendable nomenclature vocabulary from the recursive
graph algorithms in `namer.py`.

Blue Book source: https://iupac.qmul.ac.uk/BlueBook/ (IUPAC 2013
Recommendations and Preferred Names).

## What Changed

- Added `naming_data.py`, a cached JSON loader for nomenclature tables.
- Added `data/namer_rules.json`, an extendable table for retained-name
  exceptions, halogen names, metal/halide component names, prefix group names,
  ester/amide/sulfonyl prefix handling, and special component replacements.
- Added docstrings to every top-level function in `namer.py`; each docstring
  points to the Blue Book rule family that justifies the feature.
- Replaced repeated inline maps and hardcoded name lists in `namer.py` with
  named constants loaded from `data/namer_rules.json`.
- Added small helpers for repeated formatting behavior:
  `_strip_outer_parentheses`, `_is_complex_prefix`, `_count_names`,
  `_format_counted_prefixes`, `_oxy_prefix_from_branch`, and
  `_format_element_substituent`.
- Split `name_subgraph` into an explicit chemistry dispatch:
  oxygen, nitrogen, sulfur, selenium/chalcogen, phosphorus, silicon/boron, and
  halogen fragments now live in named helpers with Blue Book rule references.

## Data-Driven Tables

Each section in `data/namer_rules.json` has:

- `bluebook_rule`: the rule family to check before editing the table.
- `values`: the actual list or map consumed by `namer.py`.

Current sections include:

- `indicated_hydrogen_retained_names`: retained ring names requiring indicated
  hydrogen handling.
- `single_atom_cations`, `single_atom_anions`, and `salt_metal_names`: ionic
  component naming and ordering.
- `alkyl_oxy_prefixes`, `simple_sulfanyl_prefixes`, and
  `simple_selanyl_prefixes`: simple branch-derived prefix contractions.
- `halogen_prefixes` and `halogen_lambda_suffixes`: halogen prefix and
  hypervalent substituent names.
- `retained_ring_elements` and `peroxy_ester_groups`: supported retained-ring
  atom vocabulary and peroxy ester handling.
- `direct_group_prefixes`, `direct_prefix_groups`, and the functional-group
  family lists used to decide prefix handling.
- `special_component_names`: final component-level retained-name replacements.

## What Stayed in Code

The graph algorithms are still in Python because they are structural decisions,
not lookup data:

- RDKit SMILES parsing.
- Connected-component traversal.
- Parent skeleton selection and numbering.
- Recursive subgraph traversal.
- Locant comparison and stereochemical descriptor collection.

Those paths still refer to Blue Book rule families through function docstrings.

## Subgraph Naming Layout

`name_subgraph` now reads as:

1. Detect non-carbon, non-ring heteroatom starts.
2. Dispatch to the element-specific helper.
3. Fall back to perceived functional-group prefixes.
4. Select and number a carbon/ring parent for recursive substituent names.
5. Collect substituents, including spiro side rings.
6. Build `AssemblyParts`, add stereochemistry, indicated hydrogens,
   replacement prefixes, and unsaturation.
7. Assemble and apply recursive-substituent wrapping.

The element helpers connect implementation decisions to rule families:

- `_name_oxygen_subgraph`: P-61.2.2.1, P-63, P-65, P-67.
- `_name_nitrogen_subgraph`: P-62, P-63, P-66, P-67.
- `_name_sulfur_subgraph`: P-67, P-14.5, P-61.2.2.2.
- `_name_chalcogen_subgraph`: P-61.2.2.2 and P-14.5.
- `_name_phosphorus_subgraph`: heteroatom hydride-prefix behavior and P-14.5.
- `_name_group_13_14_subgraph`: P-61.2 and P-14.5.
- `_name_halogen_subgraph`: P-61.3 and P-14.5.

The non-element phases are also split out:

- `_subgraph_component`: recursive substituent component discovery.
- `_direct_subgraph_prefix`: direct prefix lookup for perceived functional groups.
- `_select_subgraph_parent`: chain/ring parent selection.
- `_retained_subgraph_ring`: retained ring validation and locant maps.
- `_collect_subgraph_substituents`: substituent and spiro side-ring collection.
- `_choose_subgraph_numbering`: retained locant-map choice or normal numbering.
- `_build_subgraph_parts`: assembly model construction.
- `_add_subgraph_parent_features`: skeletal replacement and unsaturation.
- `_finalize_subgraph_name`: parenthesized recursive-substituent formatting.

## Component Naming Layout

`name_component` now mirrors `name_subgraph` as an orchestration pipeline:

1. Handle one-atom ionic components.
2. Perceive characteristic groups and select the principal group.
3. Resolve anhydride components.
4. Retarget/exclude exocyclic suffix-group atoms before parent selection.
5. Select the parent chain or ring system.
6. Recompute principal and prefix groups on the selected parent.
7. Collect prefix, branch, and spiro substituents.
8. Choose numbering, build `AssemblyParts`, and add stereochemistry,
   indicated hydrogens, front modifiers, N-substituents, replacement prefixes,
   principal suffixes, substituents, and unsaturation.
9. Assemble the name and apply final retained-component replacements.

## How to Extend

1. Add or edit a row in `data/namer_rules.json`.
2. Include the closest Blue Book rule reference in `bluebook_rule` when adding a
   new section.
3. Run:

   ```bash
   .venv/bin/python -m compileall adri/iupac-name-generator/bluenamer
   ```

4. Add a SMILES example to the local tests when the new table entry changes
   naming behavior.

## Remaining Follow-Up

The sibling `rules/*.py` modules still contain Python dictionaries for stems,
suffixes, substituents, retained rings, elements, bonds, and multipliers. They
already isolate data from algorithms, but they can be moved to JSON with the
same loader pattern if the next step is to make the whole package externally
editable.

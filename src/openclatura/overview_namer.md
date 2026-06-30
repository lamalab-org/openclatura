I assume “muramid” means a Mermaid diagram.

Below is the current flow of `openclatura/namer.py` after the refactor. At a high level, `namer.py` is now an orchestrator: it parses a molecule, splits it into graph components, names each component, recursively names substituent subgraphs when needed, then assembles the final name.

```mermaid
flowchart TD
    A["name_smiles(smiles)"] --> B["read_smiles(smiles)<br/>RDKit -> Molecule graph"]
    B --> C{"empty molecule?"}
    C -->|yes| D["return ''"]
    C -->|no| E["get_connected_components(mol)"]

    E --> F["for each component:<br/>name_component(mol, comp)"]
    F --> G["sort component names<br/>salt metals first"]
    G --> H["join names with spaces"]
    H --> I["return final name"]

    J["analyze_smiles(smiles)"] --> K["same parse/component flow<br/>but records DecisionTrace"]
    K --> L["returns NameAnalysis<br/>name + trace_segments + decisions"]

    M["name_smiles_with_trace(smiles)"] --> J
```

**Main Public Entry Points**

`name_smiles(smiles)` is the fast path. It parses the SMILES, splits the graph into connected components, names each component with `name_component`, sorts salt-like components so supported metals come first, and joins the component names.

`analyze_smiles(smiles)` does the same naming work but also records decision trace phases: parse, component split, functional-group perception, priority, parent selection, numbering, and final assembly.

`name_smiles_with_trace(smiles)` is a compatibility wrapper over `analyze_smiles`; it returns only `(name, trace_segments)`.

```mermaid
flowchart TD
    A["name_component(mol, component_atoms)"] --> B{"single atom ion?"}
    B -->|yes| C["single_atom_component_name<br/>sodium, chloride, etc."]
    B -->|no| D["ComponentNamingState"]

    D --> E["component_groups()<br/>perceive functional groups in component"]
    E --> F["component_principal_key()<br/>choose senior suffix group"]
    F --> G{"anhydride shortcut?"}
    G -->|yes| H["try_name_anhydride_component()<br/>split acid halves"]
    G -->|no| I["prepare parent-search exclusions"]

    I --> I1["retarget_external_carbonyl_groups()"]
    I1 --> I2["partition_principal_and_prefix_groups()"]
    I2 --> I3["exclude_nonparent_group_atoms()"]

    I3 --> J["_select_component_parent()<br/>chains + ring systems"]
    J --> K{"parent found?"}
    K -->|no| L["fallback: methane"]
    K -->|yes| M["trace selected parent skeleton"]

    M --> N["filter_component_groups_to_parent()"]
    N --> O["_retained_subgraph_ring()<br/>optional retained parent name"]
    O --> P["principal_involved_atoms()"]
    P --> Q["collect prefixes and branches"]

    Q --> Q1["collect_component_prefix_substituents()<br/>data-driven PREFIX_HANDLERS"]
    Q1 --> Q2["_collect_component_branch_substituents()<br/>ordinary branches + spiro branches"]

    Q2 --> R["_choose_component_numbering()"]
    R --> S["build AssemblyParts"]
    S --> T["add assembly features"]
    T --> T1["emit_bond_stereo()"]
    T1 --> T2["add_indicated_hydrogens()"]
    T2 --> T3["add_component_front_modifiers()"]
    T3 --> T4["add_component_n_substituents()"]
    T4 --> T5["add_parent_features()"]
    T5 --> T6["add_component_principal_group()"]
    T6 --> T7["_add_component_substituents()"]

    T7 --> U["assemble_name(parts)"]
    U --> V["SPECIAL_COMPONENT_NAMES replacement"]
    V --> W["return component name"]
```

**Component Naming**

`name_component` is the core pipeline for one connected molecule component.

It first handles very small special cases. If the component is a single supported ion, it returns that directly. For example, `[Na+]` becomes `sodium`, and `[Cl-]` becomes `chloride`.

For normal components, it creates a `ComponentNamingState`. This state carries all the important intermediate data: component atoms, perceived groups, principal key, excluded atoms, selected parent, retained-name data, prefix groups, principal atoms, and exclusion sets for recursive substituent naming.

Functional groups are perceived first, then `principal_groups.py` chooses the senior principal group. That determines the suffix-bearing group, such as acid, alcohol, amide, nitrile, etc.

Then `component_group_rules.py` applies data-driven graph preprocessing:
- retargets some exocyclic carbonyl groups onto the parent atom,
- excludes non-parent linker atoms,
- computes principal involved atoms.

The parent skeleton is selected from carbon chains and ring systems. This delegates to `select_principal_parent`, which scores candidate chains/rings using the principal group and parent-selection rules.

Once a parent is selected, the code filters functional groups to only those attached to that parent, checks whether a retained ring name applies, computes substituent exclusions, collects prefixes and branches, chooses numbering, creates `AssemblyParts`, and finally calls `assemble_name(parts)`.

```mermaid
flowchart TD
    A["name_subgraph(mol, start_idx, exclude_atoms)"] --> B{"start atom is non-carbon<br/>and not cyclic?"}
    B -->|yes| C["name_heteroatom_subgraph()<br/>O/N/S/Se/P/Si/B/halogens"]
    C --> D{"heteroatom handler returned name?"}
    D -->|yes| E["return prefix name"]
    D -->|no| F["continue"]

    B -->|no| F["_subgraph_component()<br/>find connected branch atoms"]
    F --> G["_direct_subgraph_prefix()<br/>nitro, cyano, carboxy, etc."]
    G --> H{"direct prefix found?"}
    H -->|yes| I["return direct prefix"]
    H -->|no| J["_select_subgraph_parent()"]

    J --> K{"parent found?"}
    K -->|no| L["return ''"]
    K -->|yes| M["_retained_subgraph_ring()"]

    M --> N["perceive_groups(mol)"]
    N --> O["_collect_subgraph_substituents()<br/>nested branches + spiro side rings"]
    O --> P["_choose_subgraph_numbering()"]
    P --> Q["build subgraph AssemblyParts"]
    Q --> R["add stereo, indicated H,<br/>substituents, unsaturation/replacement"]
    R --> S["assemble_name(parts)"]
    S --> T["_finalize_subgraph_name()<br/>parentheses / yl rules"]
    T --> U["return recursive substituent name"]
```

**Recursive Subgraph Naming**

`name_subgraph` names substituent branches attached to a parent atom. It is called from component prefix handling, ordinary branch substituent collection, N-substituent handling, front modifiers, and heteroatom recursive naming.

It first checks whether the branch starts with a non-carbon atom outside a ring. If so, it delegates to `heteroatom_subgraphs.py`, which handles names like hydroxy, oxo, amino, sulfanyl, halogen prefixes, and related heteroatom-substituent forms.

If that does not apply, it finds the connected subgraph for the branch. If the branch is exactly a known direct functional prefix, `_direct_subgraph_prefix` returns it immediately.

Otherwise it selects a parent for the subgraph, just like component naming does, but with substituent-specific constraints. It then collects nested substituents, handles spiro side-ring substituents, chooses numbering, builds `AssemblyParts`, assembles the name, and wraps it in parentheses when required.

**Where The Data-Driven Rules Live**

The major simplification is that `namer.py` no longer owns every group-specific decision inline.

```mermaid
flowchart LR
    A["namer.py<br/>orchestration"] --> B["component_group_rules.py<br/>non-parent atom exclusion<br/>carbonyl retargeting"]
    A --> C["functional_prefixes.py<br/>PREFIX_HANDLERS registry"]
    A --> D["component_modifiers.py<br/>front modifiers<br/>N-substituents"]
    A --> E["principal_groups.py<br/>principal selection<br/>suffix attachment"]
    A --> F["special_cases.py<br/>single ions<br/>anhydrides"]
    A --> G["subgraph_tools.py<br/>shared graph mechanics"]
    A --> H["heteroatom_subgraphs.py<br/>O/N/S/halogen branch naming"]
    A --> I["numbering.py<br/>parent numbering"]
    A --> J["assembler.py<br/>AssemblyParts -> final string"]
```

`functional_prefixes.py` is the most data-driven example. Instead of a long `if/elif group.key` ladder in `namer.py`, it builds a `PREFIX_HANDLERS` registry from key groups loaded through `namer_config.py`. Adding a new prefix type should now usually mean adding a key to the data-backed category and, only when needed, adding one handler function.

`group_atom_roles.py` centralizes atom-role selectors. That prevents duplicated chemistry predicates like “find the ester single oxygen” or “find the amide nitrogen” from appearing in multiple naming steps.

`component_modifiers.py` owns post-parent modifier logic: front modifiers and N-substituents. It still receives `name_subgraph` as a callback, so it does not import `namer.py` and avoids circular dependencies.

**End-To-End Mental Model**

The code follows this chemistry pipeline:

1. Parse SMILES into a graph.
2. Split disconnected graph components.
3. For each component, perceive functional groups.
4. Pick the principal characteristic group.
5. Exclude functional-group atoms that should not be part of the parent skeleton.
6. Select the best parent chain/ring/polycycle.
7. Recompute principal/prefix groups against the selected parent.
8. Collect characteristic-group prefixes and ordinary substituent branches.
9. Number the parent.
10. Build `AssemblyParts`.
11. Add suffixes, prefixes, unsaturation, stereochemistry, retained names, and trace metadata.
12. Assemble the component name.
13. Sort disconnected component names and join them into the final molecule name.
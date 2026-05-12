# structure-to-iupac/api.py

import re

from .molecule import DecisionTrace, Molecule, NameAnalysis, TracePhase
from .naming_context import ComponentNamingState
from .perception import perceive_groups, PerceivedGroup
from .chains import find_all_carbon_paths, find_ring_systems, get_cyclic_atoms
from .parent_selection import select_principal_parent
from .assembler import assemble_name, AssemblyParts, SubstituentItem
from .component_group_rules import (
    exclude_nonparent_group_atoms as _exclude_nonparent_group_atoms,
    principal_involved_atoms as _principal_involved_atoms,
    retarget_external_carbonyl_groups as _retarget_external_carbonyl_groups,
)
from .component_modifiers import (
    add_component_front_modifiers as _add_component_front_modifiers,
    add_component_n_substituents as _add_component_n_substituents,
)
from .functional_prefixes import collect_component_prefix_substituents as _collect_component_prefix_substituents
from .graph_io import get_connected_components, read_smiles
from .heteroatom_subgraphs import name_heteroatom_subgraph, upstream_bond_order
from .locants import parse_locant
from .namer_config import (
    DIRECT_GROUP_PREFIXES,
    RETAINED_RING_ELEMENTS,
    SALT_METAL_NAMES,
    SPECIAL_COMPONENT_NAMES,
)
from .numbering import number_parent
from .principal_groups import (
    add_component_principal_group as _add_component_principal_group,
    component_groups as _component_groups,
    component_principal_key as _component_principal_key,
    filter_component_groups_to_parent as _filter_component_groups_to_parent,
    partition_principal_and_prefix_groups as _partition_principal_and_prefix_groups,
)
from .rules import substituents, retained
from .special_cases import (
    single_atom_component_name as _single_atom_component_name,
    try_name_anhydride_component as _try_name_anhydride_component,
)
from .subgraph_tools import (
    add_indicated_hydrogens as _add_indicated_hydrogens,
    add_parent_features as _add_subgraph_parent_features,
    emit_bond_stereo as _emit_bond_stereo,
    find_spiro_side_pair as _find_spiro_side_pair,
    spiro_side_component as _spiro_side_component,
    subgraph_component as _subgraph_component,
    subgraph_locant_getter as _subgraph_locant_getter,
)
from .trace_helpers import (
    add_substituent_trace as _add_substituent_trace,
    assembly_trace_segments as _assembly_trace_segments,
    bond_ids_within as _bond_ids_within,
    functional_group_trace_data as _functional_group_trace_data,
    trace_decision as _trace_decision,
)


def _direct_subgraph_prefix(mol: Molecule, start_idx: int, component: set[int]) -> str:
    """Return a direct functional-group prefix when the subgraph is one group.

    Blue Book references: P-63 through P-67 for direct detachable prefixes such
    as nitro, cyano, carboxy, carbamoyl, and halo-carbonyl prefixes.
    """

    for group in perceive_groups(mol):
        if start_idx in group.atoms_involved and group.atoms_involved.issubset(component):
            if group.key in DIRECT_GROUP_PREFIXES:
                return DIRECT_GROUP_PREFIXES[group.key]
            if group.key in substituents.SUBSTITUENTS:
                return substituents.get(group.key).prefix
    return ""


def _find_acyclic_subgraph_paths(
    mol: Molecule, start_idx: int, component: set[int], cyclic_atoms: set[int], sub_exclude: set[int]
) -> list[list[int]]:
    """Find carbon paths available for recursive acyclic substituent naming.

    Blue Book references: P-44 and P-45 for parent hydride selection in
    substituent names.
    """

    valid_nodes = {n for n in component if n not in cyclic_atoms and mol.atoms[n].is_carbon and n not in sub_exclude}
    paths = []

    def dfs_sub(curr, path, visited_nodes):
        neighbors = [n for n in mol.get_neighbors(curr) if n in valid_nodes and n not in visited_nodes]
        if not neighbors:
            if start_idx in path:
                paths.append(path)
            return
        for n in neighbors:
            dfs_sub(n, path + [n], visited_nodes | {n})

    endpoints = [n for n in valid_nodes if sum(1 for x in mol.get_neighbors(n) if x in valid_nodes) <= 1]
    start_nodes = endpoints if endpoints else valid_nodes
    for start in start_nodes:
        dfs_sub(start, [start], {start})
    return paths


def _select_subgraph_parent(mol: Molecule, start_idx: int, component: set[int], sub_exclude: set[int]):
    """Select parent candidates for a recursive substituent component.

    Blue Book references: P-44, P-45, P-52, and P-53 for parent selection in
    chains, rings, fused systems, and retained parents.
    """

    cyclic_atoms = get_cyclic_atoms(mol, sub_exclude)
    if start_idx in cyclic_atoms:
        ring_systems = find_ring_systems(mol, sub_exclude)
        valid_rings = [rs for rs in ring_systems if start_idx in rs.atoms]
        if not valid_rings:
            return None
        selection = select_principal_parent(mol, [], valid_rings, [start_idx])
        if selection is None:
            return None
        return selection.with_fixed_start(selection.requires_fixed_substituent_start)

    paths = _find_acyclic_subgraph_paths(mol, start_idx, component, cyclic_atoms, sub_exclude)
    if not paths:
        return None
    selection = select_principal_parent(mol, paths, [], [start_idx])
    if selection is None:
        return None
    return selection.with_fixed_start(False)


def _retained_subgraph_ring(mol: Molecule, path: list[int], is_ring: bool, is_bicycle: bool, is_polycycle: bool):
    """Return a retained ring name and locant maps when valid for this subgraph.

    Blue Book references: P-52 and P-53 for retained names, and P-22/P-25 for
    supported heterocycle retained names.
    """

    temp_retained = retained.get_retained_ring(mol, path) if is_ring else None
    if not temp_retained:
        return None, None
    retained_name_val, locant_maps = temp_retained
    if locant_maps is None and (is_bicycle or is_polycycle):
        return None, None
    if any(mol.atoms[idx].symbol not in RETAINED_RING_ELEMENTS for idx in path):
        return None, None
    return retained_name_val, locant_maps


def _spiro_subgraph_name(mol: Molecule, c_idx: int, sub_comp: set[int]) -> str:
    """Name a side ring as a synthetic spiro substituent marker.

    Blue Book references: P-24 and P-52.3; the attachment atom is temporarily
    represented as silicon so the side ring can be named independently, then the
    silane marker is stripped back out.
    """

    sub_mol = Molecule()
    for n in sub_comp:
        atom = mol.atoms[n]
        symbol = "Si" if n == c_idx else atom.symbol
        sub_mol.add_atom(symbol=symbol, idx=n, charge=atom.charge, stereo=atom.stereo)
    for n in sub_comp:
        for nxt in mol.get_neighbors(n):
            if nxt in sub_comp and n < nxt:
                bond = mol.get_bond(n, nxt)
                sub_mol.add_bond(u=n, v=nxt, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring)

    sub_name_raw = name_component(sub_mol, sub_comp, is_substituent=False)
    match = re.search(r"(?:(^|-)(\d+)-)?sil[a]?", sub_name_raw)
    if not match:
        return f"[SPIRO]-1-{sub_name_raw}"

    loc = match.group(2) if match.group(2) else "1"
    if match.group(2):
        sub_name_clean = re.sub(rf"(^|-){loc}-sil[a]?-?", r"\1", sub_name_raw)
    else:
        sub_name_clean = re.sub(r"sil[a]?-?", "", sub_name_raw)

    sub_name_clean = sub_name_clean.replace("--", "-").strip("-")
    sub_name_clean = sub_name_clean.replace("-cyclo", "cyclo")
    if not sub_name_clean:
        sub_name_clean = "methane"
    return f"[SPIRO]-{loc}-{sub_name_clean}"


def _collect_subgraph_substituents(
    mol: Molecule,
    candidate_path: list[int],
    sub_perceived: list[PerceivedGroup],
    sub_exclude: set[int],
) -> dict[int, list[SubstituentItem]]:
    """Collect prefixes attached to a recursive subgraph parent.

    Blue Book references: P-14.2, P-16.5, P-44, P-61 through P-67, and P-24
    for multiplicative prefixes, complex prefixes, parent substituents, and
    spiro side-ring substituents.
    """

    main_set = set(candidate_path)
    subst_mapping: dict[int, list[SubstituentItem]] = {}
    sub_handled_atoms = set()

    for group in sub_perceived:
        if group.attachment_carbon in main_set and not group.is_principal_candidate:
            name = substituents.get(group.key).prefix if group.key in substituents.SUBSTITUENTS else ""
            if name:
                subst_mapping.setdefault(group.attachment_carbon, []).append(
                    SubstituentItem(
                        name=name,
                        locants=[],
                        atom_ids=set(group.atoms_involved),
                        bond_ids=_bond_ids_within(mol, set(group.atoms_involved) | {group.attachment_carbon}),
                    )
                )
                sub_handled_atoms.update(group.atoms_involved)

    for c_idx in candidate_path:
        n_subs = [
            n_idx
            for n_idx in mol.get_neighbors(c_idx)
            if n_idx not in main_set and n_idx not in sub_handled_atoms and n_idx not in sub_exclude
        ]

        spiro_pair = _find_spiro_side_pair(mol, c_idx, n_subs, main_set, sub_exclude)
        if spiro_pair:
            sub_comp = _spiro_side_component(mol, c_idx, spiro_pair[0], main_set, sub_exclude)
            subst_mapping.setdefault(c_idx, []).append(
                SubstituentItem(
                    name=_spiro_subgraph_name(mol, c_idx, sub_comp),
                    locants=[],
                    atom_ids=sub_comp - {c_idx},
                    bond_ids=_bond_ids_within(mol, sub_comp),
                )
            )
            sub_handled_atoms.update(sub_comp - {c_idx})
            n_subs = [n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in sub_handled_atoms and n_idx not in sub_exclude:
                branch_name, branch_trace = name_subgraph(
                    mol, n_idx, sub_exclude | main_set, upstream_atom=c_idx, return_trace=True
                )
                if branch_name:
                    branch_atoms = _subgraph_component(mol, n_idx, sub_exclude | main_set)
                    branch_with_attachment = branch_atoms | {c_idx}
                    subst_mapping.setdefault(c_idx, []).append(
                        SubstituentItem(
                            name=branch_name,
                            locants=[],
                            atom_ids=branch_atoms,
                            bond_ids=_bond_ids_within(mol, branch_with_attachment),
                            trace_segments=branch_trace,
                        )
                    )

    return subst_mapping


def _choose_subgraph_numbering(
    mol: Molecule,
    candidate_paths: list[list[int]],
    start_idx: int,
    subst_mapping: dict[int, list[str]],
    locant_maps,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    fixed_start: bool,
    retained_name_val: str | None,
):
    """Choose a numbering and locant map for recursive substituent assembly.

    Blue Book references: P-14.4, P-44, and P-45; retained-ring locant maps are
    compared before falling back to normal parent numbering.
    """

    if locant_maps:

        def evaluate_map(lmap):
            def get_val(idx):
                return parse_locant(lmap[idx])

            principal = sorted([get_val(start_idx)])
            het_by_priority = {}
            for atom in mol:
                if atom.idx in lmap and not atom.is_carbon:
                    priority = atom.element.hw_priority or 99
                    het_by_priority.setdefault(priority, []).append(atom.idx)
            heteroatom_eval = tuple(
                sorted([get_val(idx) for idx in het_by_priority[priority]])
                for priority in sorted(het_by_priority.keys())
            )
            substituent_eval = sorted([get_val(idx) for idx in set(subst_mapping.keys()) if idx in lmap])
            return heteroatom_eval + (principal, substituent_eval)

        locant_map = min(locant_maps, key=evaluate_map)
        return list(locant_map.keys()), locant_map

    return (
        number_parent(
            mol,
            candidate_paths,
            {start_idx},
            subst_mapping,
            is_ring,
            is_bicycle,
            is_spiro,
            is_polycycle=is_polycycle,
            fixed_start=fixed_start,
            retained_name=retained_name_val,
        ),
        None,
    )


def _add_subgraph_substituents(parts: AssemblyParts, subst_mapping: dict[int, list[SubstituentItem]], get_loc) -> None:
    """Add collected substituent prefixes to assembly parts.

    Blue Book references: P-14.2 and P-16.5 for locants, multiplicative
    prefixes, and complex substituent citation.
    """

    for c_idx, items in subst_mapping.items():
        locant = get_loc(c_idx)
        for item in items:
            _add_substituent_trace(parts, item.name, locant, item.atom_ids, item.bond_ids, item.trace_segments)


def _build_subgraph_parts(
    mol: Molecule,
    start_idx: int,
    upstream_atom: int | None,
    numbered_path: list[int],
    get_loc,
    retained_name_val: str | None,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    xyz,
    polycycle_descriptor,
) -> AssemblyParts:
    """Create assembly parts for a recursive substituent parent.

    Blue Book references: P-13.6, P-14.3, P-31, P-44, P-45, and P-91/P-93 for
    substituent suffixes, locants, parent descriptors, and stereochemistry.
    """

    attach_locant = get_loc(start_idx)
    upstream_order = upstream_bond_order(mol, start_idx, upstream_atom)
    parts = AssemblyParts(
        parent_length=len(numbered_path),
        is_ring=is_ring,
        is_bicycle=is_bicycle,
        is_spiro=is_spiro,
        is_polycycle=is_polycycle,
        bicycle_xyz=xyz if is_bicycle else (0, 0, 0),
        spiro_xy=(xyz[0], xyz[1]) if is_spiro else (0, 0),
        polycycle_descriptor=polycycle_descriptor,
        is_substituent=True,
        is_double_attach=upstream_order == 2,
        is_triple_attach=upstream_order == 3,
        attachment_locant=attach_locant,
        retained_name=retained_name_val,
        parent_atom_ids=set(numbered_path),
        parent_bond_ids=_bond_ids_within(mol, set(numbered_path)),
    )

    for atom_idx in numbered_path:
        if mol.atoms[atom_idx].stereo:
            parts.stereo_features.append((get_loc(atom_idx), mol.atoms[atom_idx].stereo))
    return parts


def _finalize_subgraph_name(name: str, parts: AssemblyParts) -> str:
    """Apply recursive-substituent wrapping rules to an assembled name.

    Blue Book references: P-13.6 and P-16.5 for substituent suffix citation and
    parentheses around complex substituent prefixes.
    """

    if name == "phenyl" and not parts.substituents:
        return name
    if (
        (name.endswith("yl") or name.endswith("ylidene") or name.endswith("ylidyne"))
        and not parts.substituents
        and not parts.unsaturations
        and str(parts.attachment_locant) == "1"
        and not name.startswith("bicyclo")
        and not name.startswith("spiro")
        and not name.startswith("tricyclo")
    ):
        return name
    return f"({name})"


def name_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int = None,
    return_trace: bool = False,
):
    """Name a recursive substituent subgraph attached to the current parent.

    Blue Book references: P-13.6, P-14.2, P-16.5, P-61, P-62, P-63, P-65,
    P-66, and P-67.  Extendable prefix vocabularies are loaded from
    ``data/namer_rules.json``.
    """

    start_atom = mol.atoms[start_idx]
    cyclic_atoms_global = get_cyclic_atoms(mol, exclude_atoms)

    if not start_atom.is_carbon and start_idx not in cyclic_atoms_global:
        name = name_heteroatom_subgraph(mol, start_idx, exclude_atoms, upstream_atom, name_subgraph)
        if name is not None:
            return (name, []) if return_trace else name

    component = _subgraph_component(mol, start_idx, exclude_atoms)
    direct_prefix = _direct_subgraph_prefix(mol, start_idx, component)
    if direct_prefix:
        return (direct_prefix, []) if return_trace else direct_prefix

    sub_exclude = set(mol.atoms.keys()) - component
    parent_selection = _select_subgraph_parent(mol, start_idx, component, sub_exclude)
    if parent_selection is None:
        return ("", []) if return_trace else ""

    retained_name_val, locant_maps = _retained_subgraph_ring(
        mol,
        parent_selection.primary_path,
        parent_selection.is_ring,
        parent_selection.is_bicycle,
        parent_selection.is_polycycle,
    )
    sub_perceived = perceive_groups(mol)
    subst_mapping = _collect_subgraph_substituents(mol, parent_selection.primary_path, sub_perceived, sub_exclude)
    numbered_path, locant_map = _choose_subgraph_numbering(
        mol,
        parent_selection.paths,
        start_idx,
        subst_mapping,
        locant_maps,
        parent_selection.is_ring,
        parent_selection.is_bicycle,
        parent_selection.is_spiro,
        parent_selection.is_polycycle,
        parent_selection.fixed_start_required,
        retained_name_val,
    )
    get_loc = _subgraph_locant_getter(numbered_path, locant_map)

    parts = _build_subgraph_parts(
        mol,
        start_idx,
        upstream_atom,
        numbered_path,
        get_loc,
        retained_name_val,
        parent_selection.is_ring,
        parent_selection.is_bicycle,
        parent_selection.is_spiro,
        parent_selection.is_polycycle,
        parent_selection.xyz,
        parent_selection.polycycle_descriptor,
    )
    _emit_bond_stereo(mol, parts, numbered_path, get_loc, sub_exclude, upstream_atom)
    _add_indicated_hydrogens(mol, parts, numbered_path, get_loc)
    _add_subgraph_substituents(parts, subst_mapping, get_loc)
    _add_subgraph_parent_features(
        mol,
        parts,
        numbered_path,
        get_loc,
        parent_selection.is_bicycle,
        parent_selection.is_spiro,
        parent_selection.is_polycycle,
    )

    name = _finalize_subgraph_name(assemble_name(parts), parts)
    if return_trace:
        return name, _assembly_trace_segments(parts)
    return name


def _select_component_parent(mol: Molecule, exclude_atoms: set[int], principal_carbons: list[int]):
    """Select the parent chain or ring system for a connected component.

    Blue Book references: P-44 and P-45 for parent hydride selection.
    """

    chains = find_all_carbon_paths(mol, exclude_atoms)
    ring_systems = find_ring_systems(mol, exclude_atoms)
    if not chains and not ring_systems:
        return None
    return select_principal_parent(mol, chains, ring_systems, principal_carbons)


def _collect_component_branch_substituents(
    mol: Molecule,
    parent_path: list[int],
    subst_mapping: dict[int, list[SubstituentItem]],
    handled_prefix_atoms: set[int],
    principal_involved_atoms: set[int],
    base_exclude: set[int],
    sub_exclude: set[int],
) -> None:
    """Collect ordinary branch and spiro substituents from the component parent.

    Blue Book references: P-14.2, P-16.5, P-24, P-44, and P-61 through P-67.
    """

    main_set = set(parent_path)
    for c_idx in parent_path:
        n_subs = [
            n_idx
            for n_idx in mol.get_neighbors(c_idx)
            if n_idx not in main_set
            and n_idx not in principal_involved_atoms
            and n_idx not in handled_prefix_atoms
            and n_idx not in base_exclude
        ]

        spiro_pair = _find_spiro_side_pair(mol, c_idx, n_subs, main_set, base_exclude)
        if spiro_pair:
            sub_comp = _spiro_side_component(mol, c_idx, spiro_pair[0], main_set, base_exclude)
            subst_mapping.setdefault(c_idx, []).append(
                SubstituentItem(
                    name=_spiro_subgraph_name(mol, c_idx, sub_comp),
                    locants=[],
                    atom_ids=sub_comp - {c_idx},
                    bond_ids=_bond_ids_within(mol, sub_comp),
                )
            )
            handled_prefix_atoms.update(sub_comp - {c_idx})
            n_subs = [n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in principal_involved_atoms and n_idx not in handled_prefix_atoms:
                branch_name, branch_trace = name_subgraph(
                    mol, n_idx, sub_exclude | main_set, upstream_atom=c_idx, return_trace=True
                )
                if branch_name:
                    branch_atoms = _subgraph_component(mol, n_idx, sub_exclude | main_set)
                    subst_mapping.setdefault(c_idx, []).append(
                        SubstituentItem(
                            name=branch_name,
                            locants=[],
                            atom_ids=branch_atoms,
                            bond_ids=_bond_ids_within(mol, branch_atoms | {c_idx}),
                            trace_segments=branch_trace,
                        )
                    )


def _choose_component_numbering(
    mol: Molecule,
    best_paths: list[list[int]],
    principal_carbons: list[int],
    subst_mapping: dict[int, list[str]],
    locant_maps,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    retained_name_val: str | None,
):
    """Choose component numbering from retained maps or normal parent rules.

    Blue Book references: P-14.4, P-44, and P-45 for lowest locant sets and
    retained-name locant maps.
    """

    if locant_maps:

        def evaluate_map(lmap):
            def get_val(idx):
                return parse_locant(lmap[idx])

            principal_eval = sorted([get_val(c) for c in principal_carbons if c in lmap])
            het_by_priority = {}
            for atom in mol:
                if atom.idx in lmap and not atom.is_carbon:
                    priority = atom.element.hw_priority or 99
                    het_by_priority.setdefault(priority, []).append(atom.idx)
            heteroatom_eval = tuple(
                sorted([get_val(idx) for idx in het_by_priority[priority]])
                for priority in sorted(het_by_priority.keys())
            )
            substituent_eval = sorted([get_val(idx) for idx in set(subst_mapping.keys()) if idx in lmap])
            return heteroatom_eval + (principal_eval, substituent_eval)

        locant_map = min(locant_maps, key=evaluate_map)
        return list(locant_map.keys()), locant_map

    return (
        number_parent(
            mol,
            best_paths,
            principal_carbons,
            subst_mapping,
            is_ring,
            is_bicycle,
            is_spiro,
            is_polycycle=is_polycycle,
            retained_name=retained_name_val,
        ),
        None,
    )


def _build_component_parts(
    mol: Molecule,
    numbered_path: list[int],
    get_loc,
    retained_name_val: str | None,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    xyz,
    polycycle_descriptor,
) -> AssemblyParts:
    """Create assembly parts for a complete connected component.

    Blue Book references: P-13, P-14, P-31, P-44, P-45, P-52/P-53, and P-91/P-93.
    """

    parts = AssemblyParts(
        parent_length=len(numbered_path),
        is_ring=is_ring,
        is_bicycle=is_bicycle,
        is_spiro=is_spiro,
        is_polycycle=is_polycycle,
        bicycle_xyz=xyz if is_bicycle else (0, 0, 0),
        spiro_xy=(xyz[0], xyz[1]) if is_spiro else (0, 0),
        polycycle_descriptor=polycycle_descriptor,
        retained_name=retained_name_val,
        parent_atom_ids=set(numbered_path),
        parent_bond_ids=_bond_ids_within(mol, set(numbered_path)),
    )
    for atom_idx in numbered_path:
        if mol.atoms[atom_idx].stereo:
            parts.stereo_features.append((get_loc(atom_idx), mol.atoms[atom_idx].stereo))
    return parts


def _add_component_substituents(
    parts: AssemblyParts, subst_mapping: dict[int, list[SubstituentItem]], numbered_path: list[int], get_loc
) -> None:
    """Add collected component substituents to assembly parts.

    Blue Book references: P-14.2 and P-16.5 for substituent locants and complex
    prefix citation.
    """

    for c_idx, items in subst_mapping.items():
        if c_idx in numbered_path:
            locant = get_loc(c_idx)
            for item in items:
                _add_substituent_trace(parts, item.name, locant, item.atom_ids, item.bond_ids, item.trace_segments)


def name_component(
    mol: Molecule,
    component_atoms: set[int],
    is_substituent: bool = False,
    return_trace: bool = False,
    decision_trace: DecisionTrace | None = None,
):
    """Name one connected component or recursive component of a molecule.

    Blue Book references: P-44 and P-45 for parent selection, P-52/P-53 for
    retained names, P-61 through P-67 for prefixes and characteristic group
    suffixes, and P-72 for one-atom ionic components.
    """

    single_atom_name = _single_atom_component_name(mol, component_atoms)
    if single_atom_name:
        _trace_decision(
            decision_trace,
            TracePhase.COMPONENT,
            "named single-atom component",
            "A one-atom ionic component is named directly before parent selection.",
            atoms=component_atoms,
            data={"name": single_atom_name},
        )
        if return_trace:
            return single_atom_name, []
        return single_atom_name

    state = ComponentNamingState(component_atoms=set(component_atoms), is_substituent=is_substituent)
    state.perceived_groups = _component_groups(mol, state.component_atoms)
    _trace_decision(
        decision_trace,
        TracePhase.PERCEPTION,
        "identified functional groups",
        "Functional-group perception binds matched subgroups to graph atoms before priority selection.",
        atoms=state.component_atoms,
        data={"groups": _functional_group_trace_data(state.perceived_groups)},
    )
    state.principal_key = _component_principal_key(state.perceived_groups, state.is_substituent)
    _trace_decision(
        decision_trace,
        TracePhase.PRIORITY,
        "selected principal group",
        "Principal candidates are ranked with the seniority order from rules.suffixes.",
        atoms={group.attachment_carbon for group in state.perceived_groups if group.key == state.principal_key},
        data={"principal_key": state.principal_key, "is_substituent": state.is_substituent},
    )

    anhydride_name = _try_name_anhydride_component(mol, state.perceived_groups, state.principal_key, name_component)
    if anhydride_name:
        _trace_decision(
            decision_trace,
            TracePhase.ASSEMBLY,
            "used anhydride component shortcut",
            "Anhydrides are split into acid halves and assembled before normal parent selection.",
            atoms=state.component_atoms,
            data={"name": anhydride_name},
        )
        if return_trace:
            return anhydride_name, []
        return anhydride_name

    state.exclude_atoms = set(mol.atoms.keys()) - state.component_atoms
    state.cyclic_atoms_all = get_cyclic_atoms(mol, set())
    _retarget_external_carbonyl_groups(
        mol,
        state.perceived_groups,
        state.principal_key,
        state.exclude_atoms,
        state.cyclic_atoms_all,
    )
    state.principal_carbons, _ = _partition_principal_and_prefix_groups(state.perceived_groups, state.principal_key)
    _exclude_nonparent_group_atoms(mol, state.perceived_groups, state.exclude_atoms, state.cyclic_atoms_all)

    state.parent_selection = _select_component_parent(mol, state.exclude_atoms, state.principal_carbons)
    if state.parent_selection is None:
        _trace_decision(
            decision_trace,
            TracePhase.PARENT_SELECTION,
            "used methane fallback",
            "No supported chain or ring parent was found after exclusions.",
            atoms=state.component_atoms,
            data={"excluded_atoms": sorted(state.exclude_atoms)},
        )
        if return_trace:
            return "methane", []
        return "methane"

    _trace_decision(
        decision_trace,
        TracePhase.PARENT_SELECTION,
        "selected parent skeleton",
        "The parent is chosen after splitting ring systems/chains and scoring principal-group coverage and parent size.",
        atoms=state.parent_set,
        bonds=_bond_ids_within(mol, state.parent_set),
        data={
            "parent_atoms": state.parent_path,
            "is_ring": state.parent_selection.is_ring,
            "is_bicycle": state.parent_selection.is_bicycle,
            "is_spiro": state.parent_selection.is_spiro,
            "is_polycycle": state.parent_selection.is_polycycle,
            "polycycle_descriptor": state.parent_selection.polycycle_descriptor,
        },
    )

    state.perceived_groups, state.principal_key, state.principal_carbons, state.prefix_groups = (
        _filter_component_groups_to_parent(state.perceived_groups, state.parent_set, state.is_substituent)
    )
    state.retained_name, state.locant_maps = _retained_subgraph_ring(
        mol,
        state.parent_path,
        state.parent_selection.is_ring,
        state.parent_selection.is_bicycle,
        state.parent_selection.is_polycycle,
    )
    if state.retained_name:
        _trace_decision(
            decision_trace,
            TracePhase.PARENT_SELECTION,
            "used retained parent name",
            "The selected ring parent matched a retained ring table with locant maps.",
            atoms=state.parent_set,
            data={"retained_name": state.retained_name, "locant_map_count": len(state.locant_maps or [])},
        )

    state.principal_involved_atoms = _principal_involved_atoms(
        state.perceived_groups,
        state.principal_key,
        state.parent_path,
    )
    state.base_exclude = set(mol.atoms.keys()) - state.component_atoms
    state.sub_exclude = state.base_exclude | state.parent_set | state.principal_involved_atoms

    subst_mapping, handled_prefix_atoms = _collect_component_prefix_substituents(
        mol,
        state.prefix_groups,
        state.parent_path,
        state.sub_exclude,
        name_subgraph,
    )
    _collect_component_branch_substituents(
        mol,
        state.parent_path,
        subst_mapping,
        handled_prefix_atoms,
        state.principal_involved_atoms,
        state.base_exclude,
        state.sub_exclude,
    )

    numbered_path, locant_map = _choose_component_numbering(
        mol,
        state.parent_selection.paths,
        state.principal_carbons,
        subst_mapping,
        state.locant_maps,
        state.parent_selection.is_ring,
        state.parent_selection.is_bicycle,
        state.parent_selection.is_spiro,
        state.parent_selection.is_polycycle,
        state.retained_name,
    )
    _trace_decision(
        decision_trace,
        TracePhase.NUMBERING,
        "selected numbering",
        "Numbering minimizes principal-group, heteroatom, substituent, and unsaturation locants.",
        atoms=set(numbered_path),
        data={"numbered_path": numbered_path, "locants": locant_map or {atom: i + 1 for i, atom in enumerate(numbered_path)}},
    )
    get_loc = _subgraph_locant_getter(numbered_path, locant_map)

    parts = _build_component_parts(
        mol,
        numbered_path,
        get_loc,
        state.retained_name,
        state.parent_selection.is_ring,
        state.parent_selection.is_bicycle,
        state.parent_selection.is_spiro,
        state.parent_selection.is_polycycle,
        state.parent_selection.xyz,
        state.parent_selection.polycycle_descriptor,
    )
    _emit_bond_stereo(mol, parts, numbered_path, get_loc, state.base_exclude)
    _add_indicated_hydrogens(mol, parts, numbered_path, get_loc)
    _add_component_front_modifiers(mol, parts, state.perceived_groups, state.principal_key, state.sub_exclude, name_subgraph)
    _add_component_n_substituents(
        mol,
        parts,
        state.perceived_groups,
        state.principal_key,
        numbered_path,
        get_loc,
        state.sub_exclude,
        name_subgraph,
    )
    _add_subgraph_parent_features(
        mol,
        parts,
        numbered_path,
        get_loc,
        state.parent_selection.is_bicycle,
        state.parent_selection.is_spiro,
        state.parent_selection.is_polycycle,
    )
    _add_component_principal_group(
        mol,
        parts,
        state.perceived_groups,
        state.principal_key,
        state.principal_carbons,
        numbered_path,
        get_loc,
    )
    _add_component_substituents(parts, subst_mapping, numbered_path, get_loc)

    name = assemble_name(parts)
    name = SPECIAL_COMPONENT_NAMES.get(name, name)
    _trace_decision(
        decision_trace,
        TracePhase.ASSEMBLY,
        "assembled component name",
        "AssemblyParts combines parent, suffix group, prefixes, unsaturation, stereochemistry, and retained-name replacements.",
        atoms=set(numbered_path),
        data={
            "name": name,
            "principal_key": state.principal_key,
            "substituent_count": len(parts.substituents),
            "unsaturation_count": len(parts.unsaturations),
        },
    )
    if return_trace:
        return name, _assembly_trace_segments(parts)
    return name


def name_smiles_with_trace(smiles: str) -> tuple[str, list[dict]]:
    """Return a generated name and AssemblyParts-derived trace annotations.

    Blue Book references are carried by each trace segment. This API keeps the
    original ``name_smiles`` function unchanged while exposing atom and bond IDs
    selected during parent, prefix, unsaturation, and suffix assembly.
    """

    analysis = analyze_smiles(smiles)
    return analysis.name, analysis.trace_segments


def analyze_smiles(smiles: str) -> NameAnalysis:
    """Return a generated name with structure annotations and decision traces.

    The default ``name_smiles`` path stays minimal. This analysis API turns on
    explainability and records the major choices in the pipeline: parsing,
    component splitting, functional-group perception, priority, parent
    selection, numbering, and final assembly.
    """

    decisions = DecisionTrace()
    mol = read_smiles(smiles)
    _trace_decision(
        decisions,
        TracePhase.PARSE,
        "parsed SMILES",
        "RDKit parsing populated the internal Molecule graph used by the namer.",
        atoms=set(mol.atoms.keys()),
        bonds=set(mol.bonds.keys()),
        data={"smiles": smiles, "atom_count": len(mol.atoms), "bond_count": len(mol.bonds)},
    )
    if not mol.atoms:
        return NameAnalysis(name="", trace_segments=[], decisions=decisions.steps)

    components = get_connected_components(mol)
    _trace_decision(
        decisions,
        TracePhase.COMPONENT,
        "split molecule into components",
        "Each connected graph component is named independently before final component ordering.",
        atoms=set(mol.atoms.keys()),
        data={"components": [sorted(component) for component in components]},
    )

    named_components = []
    for comp in components:
        comp_name, trace = name_component(mol, comp, return_trace=True, decision_trace=decisions)
        if comp_name:
            named_components.append((comp_name, trace))

    def sort_key(item):
        name, _ = item
        return (0 if name in SALT_METAL_NAMES else 1, name)

    named_components.sort(key=sort_key)
    final_name = " ".join(name for name, _ in named_components)
    traces = []
    for _, trace in named_components:
        traces.extend(trace)
    _trace_decision(
        decisions,
        TracePhase.ASSEMBLY,
        "assembled final molecule name",
        "Named components are sorted with supported salt metals first, then joined.",
        atoms=set(mol.atoms.keys()),
        data={"name": final_name, "components": [name for name, _ in named_components]},
    )
    return NameAnalysis(name=final_name, trace_segments=traces, decisions=decisions.steps)

def name_smiles(smiles: str) -> str:
    """Return an IUPAC-style name for a SMILES string.

    Blue Book references: P-13 for name construction, P-44/P-45 for parent
    selection and numbering, and P-72 for ordering disconnected ionic
    components.  Component-order metal names are data-backed in
    ``data/namer_rules.json``.
    """

    mol = read_smiles(smiles)
    if not mol.atoms:
        return ""
    components = get_connected_components(mol)

    names =[]
    for comp in components:
        comp_name = name_component(mol, comp)
        if comp_name:
            names.append(comp_name)

    def sort_key(name):
        return (0 if name in SALT_METAL_NAMES else 1, name)

    names.sort(key=sort_key)

    return " ".join(names)

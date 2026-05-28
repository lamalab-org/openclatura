"""Connected-component naming pipeline."""

from collections.abc import Callable

from .assembly_parts import AssemblyParts, SubstituentItem
from .chains import find_all_carbon_paths, find_ring_systems, get_cyclic_atoms
from .component_group_rules import (
    exclude_nonparent_group_atoms,
    principal_involved_atoms,
    retarget_external_carbonyl_groups,
)
from .component_modifiers import add_component_front_modifiers, add_component_n_substituents
from .functional_prefixes import collect_component_prefix_substituents
from .molecule import DecisionTrace, Molecule, TracePhase
from .name_bindings import binding_trace_data, refresh_name_atom_bindings
from .naming_audit import UnnamedAtomError, assert_component_fully_named
from .naming_context import ComponentNamingState, NamingIntent
from .parent_pipeline import build_parent_assembly_plan, resolve_retained_parent
from .parent_selection import select_principal_parent
from .principal_groups import (
    add_component_principal_group,
    component_groups,
    component_principal_key,
    filter_component_groups_to_parent,
    partition_principal_and_prefix_groups,
)
from .special_cases import single_atom_component_name, structural_replacement_parent_name, try_name_anhydride_component
from .spiro_assembly import SpiroAssembly
from .stereo_audit import audit_stereochemistry
from .subgraph_tools import (
    add_indicated_hydrogens,
    add_parent_features,
    emit_bond_stereo,
    find_spiro_side_pair,
    spiro_side_component,
    subgraph_component,
)
from .trace_helpers import (
    add_substituent_trace,
    assembly_trace_segments,
    bond_ids_within,
    functional_group_trace_data,
    trace_decision,
)

SubgraphNamer = Callable[..., str]
SpiroSubgraphNamer = Callable[[Molecule, int, set[int]], SpiroAssembly]
ParentAssembler = Callable[..., str]


def select_component_parent(mol: Molecule, exclude_atoms: set[int], principal_carbons: list[int]):
    """Select the parent chain or ring system for a connected component."""

    chains = find_all_carbon_paths(mol, exclude_atoms)
    ring_systems = find_ring_systems(mol, exclude_atoms)
    if not chains and not ring_systems:
        return None
    return select_principal_parent(mol, chains, ring_systems, principal_carbons)


def collect_component_branch_substituents(
    mol: Molecule,
    parent_path: list[int],
    subst_mapping: dict[int, list[SubstituentItem]],
    handled_prefix_atoms: set[int],
    principal_involved_atom_ids: set[int],
    base_exclude: set[int],
    sub_exclude: set[int],
    *,
    name_subgraph: SubgraphNamer,
    name_spiro_subgraph: SpiroSubgraphNamer,
) -> None:
    """Collect ordinary branch and spiro substituents from the component parent."""

    main_set = set(parent_path)
    for c_idx in parent_path:
        n_subs = [
            n_idx
            for n_idx in mol.get_neighbors(c_idx)
            if n_idx not in main_set
            and n_idx not in principal_involved_atom_ids
            and n_idx not in handled_prefix_atoms
            and n_idx not in base_exclude
        ]

        spiro_pair = find_spiro_side_pair(mol, c_idx, n_subs, main_set, base_exclude)
        if spiro_pair:
            sub_comp = spiro_side_component(mol, c_idx, spiro_pair[0], main_set, base_exclude)
            subst_mapping.setdefault(c_idx, []).append(
                SubstituentItem(
                    name="",
                    locants=[],
                    atom_ids=sub_comp - {c_idx},
                    bond_ids=bond_ids_within(mol, sub_comp),
                    spiro=name_spiro_subgraph(mol, c_idx, sub_comp),
                )
            )
            handled_prefix_atoms.update(sub_comp - {c_idx})
            n_subs = [n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in principal_involved_atom_ids and n_idx not in handled_prefix_atoms:
                branch_name, branch_trace = name_subgraph(
                    mol, n_idx, sub_exclude | main_set, upstream_atom=c_idx, return_trace=True
                )
                if branch_name:
                    branch_atoms = subgraph_component(mol, n_idx, sub_exclude | main_set)
                    subst_mapping.setdefault(c_idx, []).append(
                        SubstituentItem(
                            name=branch_name,
                            locants=[],
                            atom_ids=branch_atoms,
                            bond_ids=bond_ids_within(mol, branch_atoms | {c_idx}),
                            trace_segments=branch_trace,
                        )
                    )


def add_component_substituents(
    parts: AssemblyParts, subst_mapping: dict[int, list[SubstituentItem]], numbered_path: list[int], get_loc
) -> None:
    """Add collected component substituents to assembly parts."""

    for c_idx, items in subst_mapping.items():
        if c_idx in numbered_path:
            locant = get_loc(c_idx)
            for item in items:
                add_substituent_trace(
                    parts,
                    item.name,
                    locant,
                    item.atom_ids,
                    item.bond_ids,
                    item.trace_segments,
                    spiro=item.spiro,
                )


def name_component(
    mol: Molecule,
    component_atoms: set[int],
    *,
    is_substituent: bool = False,
    return_trace: bool = False,
    decision_trace: DecisionTrace | None = None,
    name_subgraph: SubgraphNamer,
    name_spiro_subgraph: SpiroSubgraphNamer,
    assemble_parent_name: ParentAssembler,
):
    """Name one connected component or recursive component of a molecule."""

    single_atom_name = single_atom_component_name(mol, component_atoms)
    if single_atom_name:
        trace_decision(
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

    def name_component_again(next_mol: Molecule, next_atoms: set[int], is_substituent: bool = False):
        return name_component(
            next_mol,
            next_atoms,
            is_substituent=is_substituent,
            name_subgraph=name_subgraph,
            name_spiro_subgraph=name_spiro_subgraph,
            assemble_parent_name=assemble_parent_name,
        )

    structural_parent_name = structural_replacement_parent_name(mol, component_atoms, name_subgraph)
    if structural_parent_name:
        trace_decision(
            decision_trace,
            TracePhase.COMPONENT,
            "named structural replacement parent",
            "A graph-derived replacement-parent hydride class matched the full component graph.",
            atoms=component_atoms,
            data={"name": structural_parent_name},
        )
        if return_trace:
            return structural_parent_name, []
        return structural_parent_name

    state = ComponentNamingState(component_atoms=set(component_atoms), is_substituent=is_substituent)
    state.perceived_groups = component_groups(mol, state.component_atoms)
    trace_decision(
        decision_trace,
        TracePhase.PERCEPTION,
        "identified functional groups",
        "Functional-group perception binds matched subgroups to graph atoms before priority selection.",
        atoms=state.component_atoms,
        data={"groups": functional_group_trace_data(state.perceived_groups)},
    )
    state.principal_key = component_principal_key(state.perceived_groups, state.is_substituent)
    trace_decision(
        decision_trace,
        TracePhase.PRIORITY,
        "selected principal group",
        "Principal candidates are ranked with the functional-group registry seniority order.",
        atoms={group.attachment_carbon for group in state.perceived_groups if group.key == state.principal_key},
        data={"principal_key": state.principal_key, "is_substituent": state.is_substituent},
    )

    anhydride_name = try_name_anhydride_component(
        mol, state.perceived_groups, state.principal_key, name_component_again
    )
    if anhydride_name:
        trace_decision(
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
    retarget_external_carbonyl_groups(
        mol,
        state.perceived_groups,
        state.principal_key,
        state.exclude_atoms,
        state.cyclic_atoms_all,
    )
    state.principal_carbons, _ = partition_principal_and_prefix_groups(state.perceived_groups, state.principal_key)
    exclude_nonparent_group_atoms(mol, state.perceived_groups, state.exclude_atoms, state.cyclic_atoms_all)

    state.parent_selection = select_component_parent(mol, state.exclude_atoms, state.principal_carbons)
    if state.parent_selection is None:
        trace_decision(
            decision_trace,
            TracePhase.PARENT_SELECTION,
            "failed parent selection",
            "No supported chain or ring parent was found after exclusions.",
            atoms=state.component_atoms,
            data={"excluded_atoms": sorted(state.exclude_atoms)},
        )
        details = ", ".join(f"{idx}:{mol.atoms[idx].symbol}" for idx in sorted(state.component_atoms))
        raise UnnamedAtomError(f"No supported parent skeleton for component atoms: {details}")

    trace_decision(
        decision_trace,
        TracePhase.PARENT_SELECTION,
        "selected parent skeleton",
        "The parent is chosen after splitting ring systems/chains and scoring principal-group coverage and parent size.",
        atoms=state.parent_set,
        bonds=bond_ids_within(mol, state.parent_set),
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
        filter_component_groups_to_parent(state.perceived_groups, state.parent_set, state.is_substituent)
    )
    state.retained_name, state.locant_maps = resolve_retained_parent(
        mol,
        state.parent_path,
        state.parent_selection.is_ring,
        state.parent_selection.is_bicycle,
        state.parent_selection.is_polycycle,
    )
    if state.retained_name:
        trace_decision(
            decision_trace,
            TracePhase.PARENT_SELECTION,
            "used retained parent name",
            "The selected ring parent matched a retained ring table with locant maps.",
            atoms=state.parent_set,
            data={"retained_name": state.retained_name, "locant_map_count": len(state.locant_maps or [])},
        )

    state.principal_involved_atoms = principal_involved_atoms(
        state.perceived_groups,
        state.principal_key,
        state.parent_path,
    )
    state.base_exclude = set(mol.atoms.keys()) - state.component_atoms
    state.sub_exclude = state.base_exclude | state.parent_set | state.principal_involved_atoms

    subst_mapping, handled_prefix_atoms = collect_component_prefix_substituents(
        mol,
        state.prefix_groups,
        state.parent_path,
        state.sub_exclude,
        name_subgraph,
    )
    collect_component_branch_substituents(
        mol,
        state.parent_path,
        subst_mapping,
        handled_prefix_atoms,
        state.principal_involved_atoms,
        state.base_exclude,
        state.sub_exclude,
        name_subgraph=name_subgraph,
        name_spiro_subgraph=name_spiro_subgraph,
    )
    if (
        state.retained_name
        and state.locant_maps is None
        and state.parent_selection.is_polycycle
        and (subst_mapping or state.perceived_groups)
    ):
        state.retained_name = None

    parent_plan = build_parent_assembly_plan(
        mol,
        state.parent_selection,
        NamingIntent.component(state.principal_carbons),
        subst_mapping,
        state.locant_maps,
        state.retained_name,
    )
    numbered_path = parent_plan.numbered_path
    locant_map = parent_plan.locant_map
    trace_decision(
        decision_trace,
        TracePhase.NUMBERING,
        "selected numbering",
        "Numbering minimizes principal-group, heteroatom, substituent, and unsaturation locants.",
        atoms=set(numbered_path),
        data={
            "numbered_path": numbered_path,
            "locants": locant_map or {atom: i + 1 for i, atom in enumerate(numbered_path)},
        },
    )
    get_loc = parent_plan.get_loc
    parts = parent_plan.parts
    emit_bond_stereo(mol, parts, numbered_path, get_loc, state.base_exclude)
    add_indicated_hydrogens(mol, parts, numbered_path, get_loc)
    add_component_front_modifiers(
        mol, parts, state.perceived_groups, state.principal_key, state.sub_exclude, name_subgraph
    )
    add_component_n_substituents(
        mol,
        parts,
        state.perceived_groups,
        state.principal_key,
        numbered_path,
        get_loc,
        state.sub_exclude,
        name_subgraph,
    )
    add_parent_features(
        mol,
        parts,
        numbered_path,
        get_loc,
        state.parent_selection.is_bicycle,
        state.parent_selection.is_spiro,
        state.parent_selection.is_polycycle,
    )
    add_component_principal_group(
        mol,
        parts,
        state.perceived_groups,
        state.principal_key,
        state.principal_carbons,
        numbered_path,
        get_loc,
    )
    add_component_substituents(parts, subst_mapping, numbered_path, get_loc)

    refresh_name_atom_bindings(parts)
    parts.stereo_audit_issues = list(audit_stereochemistry(mol, parts).issues)
    assert_component_fully_named(mol, state.component_atoms, parts, "<component>")
    name = assemble_parent_name(mol, parts, numbered_path, get_loc, apply_special_component_names=True)
    trace_decision(
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
            "stereo_audit_issues": parts.stereo_audit_issues,
            "name_atom_bindings": binding_trace_data(parts.name_atom_bindings),
        },
    )
    if return_trace:
        return name, assembly_trace_segments(parts)
    return name

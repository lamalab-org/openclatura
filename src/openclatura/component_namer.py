"""Connected-component naming pipeline."""

from collections.abc import Callable
from typing import Literal, Protocol, overload

from .assembly_parts import AssemblyParts, NameAtomBinding, SubstituentItem
from .chains import find_all_carbon_paths, find_ring_systems, get_cyclic_atoms
from .component_group_rules import (
    exclude_nonparent_group_atoms,
    principal_involved_atoms,
    retarget_external_carbonyl_groups,
)
from .component_modifiers import add_component_front_modifiers, add_component_n_substituents
from .functional_prefixes import collect_component_prefix_substituents
from .molecule import DecisionTrace, Molecule, TracePhase
from .name_assembly import NameAssemblyResult, assert_final_name_assembly, token_span_trace_data
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
from .reconstruction_audit import audit_component_reconstruction
from .retained_fused_production import production_retained_fused_parent
from .special_cases import (
    single_atom_component_name,
    structural_replacement_parent_result,
    try_name_anhydride_component_result,
)
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
from .substituent_tokens import graph_bound_substituent_tokens
from .trace_helpers import (
    add_substituent_trace,
    assembly_substituent_tree,
    assembly_trace_segments,
    bond_ids_within,
    decision_trace_data,
    functional_group_trace_data,
    trace_decision,
)


class SubgraphNamer(Protocol):
    """Recursive subgraph namer with simple and traced/tree return modes."""

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[False] = False,
        return_tree: Literal[False] = False,
        decision_trace: DecisionTrace | None = None,
    ) -> str: ...

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[True],
        return_tree: Literal[False] = False,
        decision_trace: DecisionTrace | None = None,
    ) -> tuple[str, list[dict]]: ...

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[True],
        return_tree: Literal[True],
        decision_trace: DecisionTrace | None = None,
    ) -> tuple[str, list[dict], dict | None]: ...

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[False] = False,
        return_tree: Literal[True],
        decision_trace: DecisionTrace | None = None,
    ) -> tuple[str, dict | None]: ...


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
    emit_metadata: bool = True,
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
                    charge_atom_ids=_charged_atoms(mol, sub_comp - {c_idx}),
                    spiro=name_spiro_subgraph(mol, c_idx, sub_comp),
                )
            )
            handled_prefix_atoms.update(sub_comp - {c_idx})
            n_subs = [n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in principal_involved_atom_ids and n_idx not in handled_prefix_atoms:
                branch_decisions = DecisionTrace() if emit_metadata else None
                if emit_metadata:
                    branch_name, branch_trace, branch_tree = name_subgraph(
                        mol,
                        n_idx,
                        sub_exclude | main_set,
                        upstream_atom=c_idx,
                        return_trace=True,
                        return_tree=True,
                        decision_trace=branch_decisions,
                    )
                else:
                    branch_name = name_subgraph(
                        mol,
                        n_idx,
                        sub_exclude | main_set,
                        upstream_atom=c_idx,
                    )
                    branch_trace = []
                    branch_tree = None
                if branch_name:
                    branch_exclude = sub_exclude | main_set
                    branch_atoms = subgraph_component(mol, n_idx, branch_exclude)
                    subst_mapping.setdefault(c_idx, []).append(
                        SubstituentItem(
                            name=branch_name,
                            locants=[],
                            atom_ids=branch_atoms,
                            bond_ids=bond_ids_within(mol, branch_atoms | {c_idx}),
                            charge_atom_ids=_charged_atoms(mol, branch_atoms),
                            emitted_tokens=(
                                graph_bound_substituent_tokens(
                                    mol,
                                    n_idx,
                                    branch_atoms,
                                    branch_name,
                                    c_idx,
                                    branch_exclude,
                                    name_subgraph,
                                )
                                if emit_metadata
                                else ()
                            ),
                            trace_segments=branch_trace if emit_metadata else [],
                            nested_decisions=decision_trace_data(branch_decisions) if emit_metadata else [],
                            substituent_tree=branch_tree if emit_metadata else None,
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
                    item.charge_atom_ids,
                    item.trace_segments,
                    item.nested_decisions,
                    item.emitted_tokens,
                    substituent_tree=item.substituent_tree,
                    spiro=item.spiro,
                )


def _charged_atoms(mol: Molecule, atom_ids: set[int]) -> set[int]:
    """Return formally charged atoms from an already named graph fragment."""

    return {atom_idx for atom_idx in atom_ids if mol.atoms[atom_idx].charge != 0}


def _shortcut_component_result(
    mol: Molecule,
    component_atoms: set[int],
    name: str,
    *,
    stage: str,
    role: str,
    bindings: list[NameAtomBinding] | tuple[NameAtomBinding, ...] | None = None,
    emit_metadata: bool = True,
    token_debug: bool = False,
) -> tuple[str, list[dict], list[dict], list[dict]]:
    """Build audited metadata for a component shortcut name."""

    if not emit_metadata:
        return name, [], [], []
    parts = AssemblyParts(parent_length=max(1, len(component_atoms)), parent_atom_ids=set(component_atoms))
    parts.name_atom_bindings = (
        list(bindings)
        if bindings is not None
        else [
            NameAtomBinding(
                stage=stage,
                role=role,
                term=name,
                atom_ids=set(component_atoms),
                bond_ids=bond_ids_within(mol, component_atoms),
                charge_atom_ids=_charged_atoms(mol, component_atoms),
            )
        ]
    )
    result = NameAssemblyResult.from_final_name(name, parts.name_atom_bindings)
    parts.name_atom_bindings = list(result.bindings)
    parts.name_token_spans = token_span_trace_data(result)
    assert_final_name_assembly(mol, component_atoms, parts, result)
    return (
        name,
        binding_trace_data(parts.name_atom_bindings, include_emitted_tokens=token_debug),
        parts.name_token_spans if token_debug else [],
        parts.name_rewrite_history,
    )


def name_component(
    mol: Molecule,
    component_atoms: set[int],
    *,
    is_substituent: bool = False,
    return_trace: bool = False,
    return_tree: bool = False,
    decision_trace: DecisionTrace | None = None,
    name_subgraph: SubgraphNamer,
    name_spiro_subgraph: SpiroSubgraphNamer,
    assemble_parent_name: ParentAssembler,
    token_debug: bool = False,
):
    """Name one connected component or recursive component of a molecule."""

    emit_metadata = return_trace or return_tree or decision_trace is not None

    single_atom_name = single_atom_component_name(mol, component_atoms)
    if single_atom_name:
        name, bindings, token_spans, rewrite_history = _shortcut_component_result(
            mol,
            component_atoms,
            single_atom_name,
            stage="shortcut",
            role="single_atom_component",
            emit_metadata=emit_metadata,
            token_debug=token_debug,
        )
        trace_decision(
            decision_trace,
            TracePhase.COMPONENT,
            "named single-atom component",
            "A one-atom ionic component is named directly before parent selection.",
            atoms=component_atoms,
            data={
                "name": name,
                "name_atom_bindings": bindings,
                "name_token_spans": token_spans,
                "name_rewrite_history": rewrite_history,
            },
        )
        if return_trace and return_tree:
            return name, [], _shortcut_tree(name, component_atoms, bindings, token_spans)
        if return_trace:
            return name, []
        if return_tree:
            return name, _shortcut_tree(name, component_atoms, bindings, token_spans)
        return name

    def name_component_again(next_mol: Molecule, next_atoms: set[int], is_substituent: bool = False):
        return name_component(
            next_mol,
            next_atoms,
            is_substituent=is_substituent,
            name_subgraph=name_subgraph,
            name_spiro_subgraph=name_spiro_subgraph,
            assemble_parent_name=assemble_parent_name,
            token_debug=token_debug,
        )

    structural_parent_result = structural_replacement_parent_result(mol, component_atoms, name_subgraph)
    if structural_parent_result is not None:
        name, bindings, token_spans, rewrite_history = _shortcut_component_result(
            mol,
            component_atoms,
            structural_parent_result.name,
            stage="shortcut",
            role=structural_parent_result.role,
            bindings=structural_parent_result.bindings,
            emit_metadata=emit_metadata,
            token_debug=token_debug,
        )
        trace_decision(
            decision_trace,
            TracePhase.COMPONENT,
            "named structural replacement parent",
            "A graph-derived replacement-parent hydride class matched the full component graph.",
            atoms=component_atoms,
            data={
                "name": name,
                "name_atom_bindings": bindings,
                "name_token_spans": token_spans,
                "name_rewrite_history": rewrite_history,
            },
        )
        if return_trace and return_tree:
            return name, [], _shortcut_tree(name, component_atoms, bindings, token_spans)
        if return_trace:
            return name, []
        if return_tree:
            return name, _shortcut_tree(name, component_atoms, bindings, token_spans)
        return name

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

    anhydride_result = try_name_anhydride_component_result(
        mol, state.perceived_groups, state.principal_key, name_component_again
    )
    if anhydride_result is not None:
        name, bindings, token_spans, rewrite_history = _shortcut_component_result(
            mol,
            state.component_atoms,
            anhydride_result.name,
            stage="shortcut",
            role="anhydride_component",
            bindings=anhydride_result.bindings,
            emit_metadata=emit_metadata,
            token_debug=token_debug,
        )
        trace_decision(
            decision_trace,
            TracePhase.ASSEMBLY,
            "used anhydride component shortcut",
            "Anhydrides are split into acid halves and assembled before normal parent selection.",
            atoms=state.component_atoms,
            data={
                "name": name,
                "name_atom_bindings": bindings,
                "name_token_spans": token_spans,
                "name_rewrite_history": rewrite_history,
            },
        )
        if return_trace and return_tree:
            return name, [], _shortcut_tree(name, state.component_atoms, bindings, token_spans)
        if return_trace:
            return name, []
        if return_tree:
            return name, _shortcut_tree(name, state.component_atoms, bindings, token_spans)
        return name

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
        mol,
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
        emit_metadata=emit_metadata,
    )
    retained_fused = production_retained_fused_parent(
        mol,
        state.parent_path,
        state.component_atoms,
        state.perceived_groups,
        state.principal_key,
        subst_mapping,
    )
    if retained_fused is not None:
        state.retained_name = retained_fused.name
        state.locant_maps = retained_fused.locant_maps
        state.retained_parent_metadata = retained_fused.metadata
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
        state.retained_parent_metadata,
    )
    numbered_path = parent_plan.numbered_path
    locant_map = parent_plan.locant_map
    get_loc = parent_plan.get_loc
    trace_decision(
        decision_trace,
        TracePhase.NUMBERING,
        "selected numbering",
        "Numbering minimizes principal-group, heteroatom, substituent, and unsaturation locants.",
        atoms=set(numbered_path),
        bonds=bond_ids_within(mol, set(numbered_path)),
        data={
            "numbered_path": numbered_path,
            "locants": locant_map or {atom: i + 1 for i, atom in enumerate(numbered_path)},
            "atom_to_locant": {atom: get_loc(atom) for atom in numbered_path},
        },
    )
    parts = parent_plan.parts
    emit_bond_stereo(mol, parts, numbered_path, get_loc, state.base_exclude)
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
    add_indicated_hydrogens(mol, parts, numbered_path, get_loc)
    add_component_substituents(parts, subst_mapping, numbered_path, get_loc)

    refresh_name_atom_bindings(parts)
    parts.stereo_audit_issues = list(audit_stereochemistry(mol, parts).issues)
    reconstruction = audit_component_reconstruction(mol, parts)
    parts.reconstruction_audit_status = reconstruction.status
    parts.reconstruction_audit_issues = list(reconstruction.issues)
    assert_component_fully_named(mol, state.component_atoms, parts, "<component>")
    name = assemble_parent_name(mol, parts, numbered_path, get_loc, emit_metadata=emit_metadata)
    if emit_metadata:
        final_result = NameAssemblyResult.from_final_name(name, parts.name_atom_bindings)
        parts.name_token_spans = token_span_trace_data(final_result)
        assert_final_name_assembly(mol, state.component_atoms, parts, final_result)
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
            "reconstruction_audit": {
                "status": parts.reconstruction_audit_status,
                "issues": parts.reconstruction_audit_issues,
            },
            "name_atom_bindings": binding_trace_data(parts.name_atom_bindings, include_emitted_tokens=token_debug),
            "name_token_spans": parts.name_token_spans if token_debug else [],
            "name_rewrite_history": parts.name_rewrite_history,
        },
    )
    trace_segments = assembly_trace_segments(parts) if return_trace or return_tree else []
    tree = None
    if return_tree:
        tree = assembly_substituent_tree(
            parts,
            name=name,
            atom_ids=state.component_atoms,
            bond_ids=bond_ids_within(mol, state.component_atoms),
            trace_segments=trace_segments,
        )
        tree["kind"] = "component"
    if return_trace and return_tree:
        return name, trace_segments, tree
    if return_trace:
        return name, trace_segments
    if return_tree:
        return name, tree
    return name


def _shortcut_tree(name: str, component_atoms: set[int], bindings: list[dict], token_spans: list[dict]) -> dict:
    """Return a minimal component tree for shortcut component names."""

    return {
        "kind": "component",
        "name": name,
        "atoms": sorted(component_atoms),
        "bonds": [],
        "parent": None,
        "principal_group": None,
        "substituents": [],
        "replacement_prefixes": [],
        "unsaturations": [],
        "trace_segments": [],
        "nested_decisions": [],
        "name_atom_bindings": bindings,
        "name_token_spans": token_spans,
    }

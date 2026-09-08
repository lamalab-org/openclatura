"""Component-proved carbon hydrogen conventions and fusion relocation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache

from ..locants import SystemLocant, system_locant_sort_key
from ..molecule import Molecule
from ..retained_graph_model import RetainedGraphAtomTemplate
from .charges import fusion_charge_lone_pair_sites
from .mancude import (
    _nitrogen_composition_parent_model,
    _single_site_parent_model,
    compare_actual_parent_to_implied_parent,
)
from .model import (
    FusionComponentSpec,
    FusionGraph,
    FusionGraphAtom,
    FusionGraphBond,
    FusionNameAst,
    ParentBondModel,
)
from .numbering import parent_bond_model


def _component_graph(spec: FusionComponentSpec, atoms: tuple[RetainedGraphAtomTemplate, ...]) -> FusionGraph:
    ids = {atom.locant: index for index, atom in enumerate(atoms)}
    return FusionGraph(
        atoms=tuple(
            FusionGraphAtom(
                ids[atom.locant],
                atom.symbol,
                atom.charge,
                pi_capacity=atom.resolved_pi_capacity,
                forced_single=atom.forced_single,
                saturated=atom.saturated,
            )
            for atom in atoms
        ),
        bonds=tuple(
            FusionGraphBond(tuple(ids[locant] for locant in bond.locants), bond.bond_class) for bond in spec.bonds
        ),
    )


@lru_cache(maxsize=512)
def component_parent_atoms(spec: FusionComponentSpec) -> tuple[RetainedGraphAtomTemplate, ...]:
    """Release a declared carbon H tautomer without changing component pi capacity."""

    atoms = spec.atoms
    template = spec.template
    if template.mancude_double_bonds is None:
        return atoms
    movable = {
        atom.locant
        for atom in atoms
        if atom.locant in template.default_indicated_h
        and atom.symbol == "C"
        and atom.charge == 0
        and atom.saturated
        and not atom.forced_single
        and atom.pi_capacity != 0
    }
    if not movable or any(bond.bond_class not in {"aromatic", "mancude", "fusion"} for bond in spec.bonds):
        return atoms
    relaxed = tuple(
        replace(atom, saturated=False, default_h=False) if atom.locant in movable else atom for atom in atoms
    )
    original_model = parent_bond_model(_component_graph(spec, atoms))
    relaxed_model = parent_bond_model(_component_graph(spec, relaxed))
    if original_model.maximum_non_cumulative_double_bonds != template.mancude_double_bonds:
        return atoms
    if relaxed_model.maximum_non_cumulative_double_bonds != template.mancude_double_bonds:
        return atoms
    movable_ids = {index for index, atom in enumerate(atoms) if atom.locant in movable}
    if not any(
        all(order == 1 for edge, order in assignment.orders if movable_ids.intersection(edge))
        for assignment in relaxed_model.allowed_kekule_assignments
    ):
        return atoms
    return relaxed


@lru_cache(maxsize=512)
def _component_carbon_h_locants(spec: FusionComponentSpec) -> frozenset[str]:
    atoms = component_parent_atoms(spec)
    graph = _component_graph(spec, atoms)
    model = parent_bond_model(graph)
    if model.maximum_non_cumulative_double_bonds != spec.template.mancude_double_bonds:
        return frozenset()
    candidates = set()
    for assignment in model.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        candidates.update(
            atoms[atom.id].locant
            for atom in graph.atoms
            if atom.symbol == "C"
            and atom.pi_capacity
            and not atom.forced_single
            and not atom.saturated
            and atom.id not in paired
        )
    return frozenset(candidates)


def intrinsic_carbon_fusion_scope(ast: FusionNameAst, specs: Mapping[int, FusionComponentSpec]) -> bool:
    """Require a connected composition of components with proved pi budgets.

    Join order and component ring count do not establish hydrogen capacity.
    Component and completed-parent maximum matchings establish it instead.
    """

    occurrences = {match.occurrence_id for match in ast.component_occurrences}
    if len(occurrences) < 2 or occurrences != set(specs):
        return False
    if any(spec.template.mancude_double_bonds is None for spec in specs.values()):
        return False
    neighbors = {occurrence: set() for occurrence in occurrences}
    for join in ast.joins:
        if join.host_occurrence not in neighbors or join.attached_occurrence not in neighbors:
            return False
        neighbors[join.host_occurrence].add(join.attached_occurrence)
        neighbors[join.attached_occurrence].add(join.host_occurrence)
    visited = set()
    pending = [next(iter(occurrences))]
    while pending:
        occurrence = pending.pop()
        if occurrence not in visited:
            visited.add(occurrence)
            pending.extend(neighbors[occurrence] - visited)
    return visited == occurrences


def component_carbon_h_relocation_scope(ast: FusionNameAst, specs: Mapping[int, FusionComponentSpec]) -> bool:
    """Carbon-only composition preserves the proved component pi capacities.

    Heteroatom donor composition needs its separate intrinsic-H proof and
    explicit component pi budgets before additive operations.
    """

    return intrinsic_carbon_fusion_scope(ast, specs) or all(
        atom.symbol == "C" and atom.charge == 0 for spec in specs.values() for atom in spec.atoms
    )


def intrinsic_carbon_candidate_atoms(
    ast: FusionNameAst,
    specs: Mapping[int, FusionComponentSpec],
    mol: Molecule,
) -> frozenset[int]:
    """Map component-local carbon H capacity into a demonstrated fusion class."""

    if not intrinsic_carbon_fusion_scope(ast, specs):
        return frozenset()
    parent_atoms = {atom for match in ast.component_occurrences for atom in match.input_atom_by_locant.values()}
    if not any(is_intrinsic_carbon_h_site(mol, atom, parent_atoms) for atom in parent_atoms):
        return frozenset()
    candidates = set()
    for match in ast.component_occurrences:
        spec = specs[match.occurrence_id]
        candidates.update(match.input_atom_by_locant[locant] for locant in _component_carbon_h_locants(spec))
    return frozenset(candidates)


def is_intrinsic_carbon_h_site(mol: Molecule, atom: int, parent_atoms: set[int] | frozenset[int]) -> bool:
    """Recognize a parent CH2 site, including H replaced by branches or bridges."""

    site = mol.atoms[atom]
    if site.symbol != "C" or site.charge or site.is_aromatic:
        return False
    neighbors = mol.get_neighbors(atom)
    external_count = sum(neighbor not in parent_atoms for neighbor in neighbors)
    return (
        len(neighbors) - external_count == 2
        and site.total_h_count + external_count == 2
        and all(mol.get_bond(atom, neighbor).order == 1 for neighbor in neighbors)
    )


def intrinsic_carbon_parent_model(
    mol: Molecule,
    graph: FusionGraph,
    model: ParentBondModel,
    locants: Mapping[int, SystemLocant],
    candidates: frozenset[int],
    *,
    intrinsic_hydrogen_atom_ids: frozenset[int] = frozenset(),
) -> tuple[ParentBondModel, frozenset[int]]:
    """Choose jointly proved intrinsic CH2 sites without consuming hydro bonds.

    Intrinsic H restricts the maximum-bond assignments themselves. It is not
    an operation deleting a double bond from an otherwise chosen assignment.
    """

    if not candidates or any(
        mol.atoms[atom].symbol != "C" and mol.atoms[atom].total_h_count and atom not in intrinsic_hydrogen_atom_ids
        for atom in locants
    ):
        return model, frozenset()
    # Resolve fusion-N valence before deciding whether a carbon is intrinsically
    # unpaired. Reassigning N afterwards can turn an additive pair into false H.
    parent_atoms = frozenset(locants)
    model = _nitrogen_composition_parent_model(mol, parent_atoms, model, intrinsic_hydrogen_atom_ids)
    eligible = {atom for atom in candidates if is_intrinsic_carbon_h_site(mol, atom, parent_atoms)}
    carbon_atoms = {
        atom.id
        for atom in graph.atoms
        if atom.symbol == "C" and atom.pi_capacity and not atom.saturated and not atom.forced_single
    }
    groups = {}
    for assignment in model.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        unpaired = carbon_atoms - paired
        # A proved N-H donor can already account for the unpaired parent
        # valence. Do not introduce a competing carbon tautomer in that case.
        if intrinsic_hydrogen_atom_ids and carbon_atoms <= paired:
            return model, frozenset()
        if unpaired and unpaired <= eligible:
            unpaired = frozenset(unpaired)
            groups.setdefault(unpaired, []).append(assignment)
    choices = []
    for sites, assignments in groups.items():
        constrained = replace(model, allowed_kekule_assignments=tuple(assignments))
        delta = compare_actual_parent_to_implied_parent(
            mol,
            parent_atoms,
            constrained,
            atom_to_locant=locants,
            indicated_hydrogen_atom_ids=intrinsic_hydrogen_atom_ids,
        )
        if delta is None or not delta.compatible or delta.additional_multiple_bond_ids:
            continue
        rank = (len(delta.hydrogenated_edges), tuple(sorted(system_locant_sort_key(locants[atom]) for atom in sites)))
        choices.append((rank, constrained, sites))
    if not choices:
        return model, frozenset()
    _, constrained, sites = min(choices, key=lambda choice: choice[0])
    result = _single_site_parent_model(model, sites)
    if result.maximum_non_cumulative_double_bonds != model.maximum_non_cumulative_double_bonds:
        return model, frozenset()
    return result, sites


def intrinsic_parent_lone_pair_sites(mol: Molecule, graph: FusionGraph) -> frozenset[int]:
    """Shared planner/auditor intrinsic donors, including charged derivatives."""

    return aromatic_nitrogen_hydrogen_atoms(mol, graph) | fusion_charge_lone_pair_sites(mol, graph)


def aromatic_nitrogen_hydrogen_atoms(mol: Molecule, graph: FusionGraph) -> frozenset[int]:
    """Neutral aromatic [nH] is an intrinsic lone-pair site, not additive H."""

    atoms = {atom.id for atom in graph.atoms}
    if any(
        mol.atoms[atom].symbol == "N" and not mol.atoms[atom].is_aromatic and mol.atoms[atom].total_h_count
        for atom in atoms
    ):
        return frozenset()
    if any(
        neighbor not in atoms and mol.get_bond(atom, neighbor).order > 1
        for atom in atoms
        for neighbor in mol.get_neighbors(atom)
    ):
        return frozenset()
    return frozenset(
        atom.id
        for atom in graph.atoms
        if atom.symbol == "N"
        and mol.atoms[atom.id].is_aromatic
        and mol.atoms[atom.id].charge == 0
        and mol.atoms[atom.id].total_h_count == 1
    )

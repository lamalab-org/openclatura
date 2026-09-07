"""Component-proved intrinsic carbon H for ordinary ortho bicycles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache

from ..locants import SystemLocant, system_locant_sort_key
from ..molecule import Molecule
from ..retained_graph_model import RetainedGraphAtomTemplate
from .mancude import compare_actual_parent_to_implied_parent, indicated_hydrogen_parent_bond_model
from .model import (
    FusionComponentSpec,
    FusionGraph,
    FusionGraphAtom,
    FusionGraphBond,
    FusionJoinKind,
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
    """Release only a declared carbon H tautomer of a monocyclic mancude template."""

    atoms = spec.atoms
    template = spec.template
    if len(spec.rings) != 1 or template.mancude_double_bonds is None:
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
    if len(movable) != 1 or any(bond.bond_class not in {"aromatic", "mancude", "fusion"} for bond in spec.bonds):
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
    candidates = set()
    for assignment in model.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        candidates.update(
            atoms[atom.id].locant
            for atom in graph.atoms
            if atom.symbol == "C" and atom.pi_capacity and atom.id not in paired
        )
    return frozenset(candidates)


def intrinsic_carbon_fusion_scope(ast: FusionNameAst, specs: Mapping[int, FusionComponentSpec]) -> bool:
    return (
        len(ast.component_occurrences) == 2
        and len(ast.joins) == 1
        and ast.joins[0].kind is FusionJoinKind.ORTHO
        and all(len(spec.rings) == 1 for spec in specs.values())
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
    if not any(
        mol.atoms[atom].symbol == "C"
        and mol.atoms[atom].total_h_count == 2
        and not mol.atoms[atom].is_aromatic
        and mol.atoms[atom].charge == 0
        for atom in parent_atoms
    ):
        return frozenset()
    candidates = set()
    for match in ast.component_occurrences:
        spec = specs[match.occurrence_id]
        candidates.update(match.input_atom_by_locant[locant] for locant in _component_carbon_h_locants(spec))
    return frozenset(candidates)


def intrinsic_carbon_parent_model(
    mol: Molecule,
    graph: FusionGraph,
    model: ParentBondModel,
    locants: Mapping[int, SystemLocant],
    candidates: frozenset[int],
) -> tuple[ParentBondModel, frozenset[int]]:
    """Choose one intrinsic CH2 site without consuming additive hydrogenation.

    Intrinsic H restricts the maximum-bond assignments themselves. It is not
    an operation deleting a double bond from an otherwise chosen assignment.
    """

    if not candidates or any(mol.atoms[atom].symbol != "C" and mol.atoms[atom].total_h_count for atom in locants):
        return model, frozenset()
    eligible = {
        atom
        for atom in candidates
        if mol.atoms[atom].charge == 0
        and mol.atoms[atom].total_h_count == 2
        and not mol.atoms[atom].is_aromatic
        and len([neighbor for neighbor in mol.get_neighbors(atom) if neighbor in locants]) == 2
        and all(mol.get_bond(atom, neighbor).order == 1 for neighbor in mol.get_neighbors(atom))
    }
    groups = {}
    for assignment in model.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        unpaired = candidates - paired
        if len(unpaired) == 1 and unpaired <= eligible:
            groups.setdefault(unpaired, []).append(assignment)
    choices = []
    for sites, assignments in groups.items():
        constrained = replace(model, allowed_kekule_assignments=tuple(assignments))
        delta = compare_actual_parent_to_implied_parent(mol, frozenset(locants), constrained, atom_to_locant=locants)
        if delta is None or not delta.compatible or delta.additional_multiple_bond_ids:
            continue
        rank = (len(delta.hydrogenated_edges), tuple(sorted(system_locant_sort_key(locants[atom]) for atom in sites)))
        choices.append((rank, constrained, sites))
    if not choices:
        return model, frozenset()
    _, constrained, sites = min(choices, key=lambda choice: choice[0])
    return indicated_hydrogen_parent_bond_model(graph, sites), sites


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

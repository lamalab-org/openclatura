"""Exact component-matching witnesses for carbon H consumed by fusion."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache

from .model import FusionComponentMatch, FusionComponentSpec, FusionNameAst, ParentBondModel


@dataclass(frozen=True)
class ComponentHydrogenConsumption:
    """Completed maximum assignments witnessing all local component budgets."""

    default_sites: frozenset[tuple[int, str]]
    junction_sites: frozenset[tuple[int, int]]
    parent_model: ParentBondModel


def component_hydrogen_consumption(
    ast: FusionNameAst, specs: Mapping[int, FusionComponentSpec]
) -> ComponentHydrogenConsumption | None:
    """Prove consumption without assuming the original H locant is a junction."""
    from .indicated_hydrogen import intrinsic_carbon_fusion_scope

    if not any(
        atom.symbol == "C" and atom.saturated and atom.locant in spec.template.default_indicated_h
        for spec in specs.values()
        for atom in spec.atoms
    ):
        return None
    if not intrinsic_carbon_fusion_scope(ast, specs):
        return None
    components = tuple((match, specs[match.occurrence_id]) for match in ast.component_occurrences)
    junctions = frozenset(atom for join in ast.joins for atom in join.shared_input_atoms)
    return _component_hydrogen_consumption(components, junctions)


@lru_cache(maxsize=512)
def _component_hydrogen_consumption(
    components: tuple[tuple[FusionComponentMatch, FusionComponentSpec], ...], junctions: frozenset[int]
) -> ComponentHydrogenConsumption | None:
    from .indicated_hydrogen import (
        _component_graph,
        _component_parent_graph,
        component_h_locants,
        component_parent_atoms,
    )
    from .numbering import parent_bond_model

    if any(
        bond.bond_class not in {"aromatic", "mancude", "fusion"} for _match, spec in components for bond in spec.bonds
    ):
        return None
    graph = _component_parent_graph(components, True)
    # This proof owns only neutral, fully pi-paired completed parents. Residual
    # carbon H, lone-pair N-H, fixed saturation and ions keep their own proofs.
    if any(
        atom.formal_charge or atom.saturated or (atom.symbol == "C" and (atom.forced_single or atom.pi_capacity != 1))
        for atom in graph.atoms
    ):
        return None
    model = parent_bond_model(graph)
    if not model.allowed_kekule_assignments:
        return None
    pi_atoms = {atom for edge in model.pi_eligible_edges | model.required_double_bonds for atom in edge}
    local_domains = []
    defaults = set()
    for match, spec in components:
        atoms = component_parent_atoms(spec)
        local_model = parent_bond_model(_component_graph(spec, atoms))
        declared = spec.template.mancude_double_bonds
        if declared is not None and local_model.maximum_non_cumulative_double_bonds != declared:
            return None
        mapping = match.input_atom_by_locant
        edges = frozenset(tuple(sorted(mapping[locant] for locant in bond.locants)) for bond in spec.bonds)
        assignments = frozenset(
            frozenset(
                tuple(sorted(mapping[atoms[atom].locant] for atom in edge))
                for edge, order in assignment.orders
                if order == 2
            )
            for assignment in local_model.allowed_kekule_assignments
        )
        # A carbon between two fixed single-bond heteroatoms cannot take part
        # in any pi assignment. Its implicit saturation is not a hydrogen
        # obligation consumed by fusion (for example an acetal carbon).
        local_pi = frozenset(
            mapping[atoms[atom].locant]
            for edge in local_model.pi_eligible_edges | local_model.required_double_bonds
            for atom in edge
        )
        carbon_h = frozenset(mapping[locant] for locant in component_h_locants(spec, "C"))
        for locant in spec.template.default_indicated_h:
            # Never use this proof to waive a heteroatom H obligation.
            if mapping[locant] not in carbon_h:
                return None
            defaults.add((match.occurrence_id, locant))
        local_domains.append((match.occurrence_id, edges, assignments, local_pi, carbon_h))

    consumed = set()
    witnesses = []
    for assignment in model.allowed_kekule_assignments:
        doubles = frozenset(edge for edge, order in assignment.orders if order == 2)
        paired = {atom for edge in doubles for atom in edge}
        if paired != pi_atoms:
            return None
        assignment_consumed = set()
        for occurrence, edges, assignments, local_pi, carbon_h in local_domains:
            local_doubles = doubles & edges
            # Pi bonds may cross the boundary or shift along alternating paths.
            # A local maximum must preserve every nonjunction atom's occupancy:
            # only fusion junctions may be paired outside this component.
            local_paired = frozenset(atom for edge in local_doubles for atom in edge)
            local_witnesses = tuple(
                candidate
                for candidate in assignments
                if local_paired.symmetric_difference(atom for edge in candidate for atom in edge) <= junctions
            )
            if not local_witnesses:
                break
            unpaired = local_pi - local_paired
            if not any(
                unpaired - {atom for edge in candidate for atom in edge} <= carbon_h & junctions
                for candidate in local_witnesses
            ):
                break
            assignment_consumed.update((occurrence, atom) for atom in unpaired & carbon_h)
        else:
            witnesses.append(assignment)
            consumed.update(assignment_consumed)
    if not defaults or not consumed or not witnesses:
        return None
    return ComponentHydrogenConsumption(
        frozenset(defaults), frozenset(consumed), replace(model, allowed_kekule_assignments=tuple(witnesses))
    )

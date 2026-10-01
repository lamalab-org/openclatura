"""Graph witnesses for an uncompletable fusion-citation pi assignment."""

from __future__ import annotations

from ..locants import system_locant_sort_key
from .model import FusionJoinKind, FusionParentPlan


def citation_pi_dead_end(plan: FusionParentPlan) -> tuple[int, int] | None:
    """Prove when OPSIN's first nonterminal pi choice cannot be completed.

    OPSIN 2.9 retains host atoms at an ortho fusion. A nonshared child
    atom's surviving internal bond consequently precedes its recreated bond
    to the host. Its non-backtracking pi matcher visits numbered atoms,
    consumes terminal pairs first, then chooses the first neighbor of the
    first nonfusion atom. Reject only a forced choice absent from *every*
    complete assignment of the already proved carbon-H parent.

    This is a citation compatibility check, not a different chemical model.
    Other constructions and derivative operations remain outside this proof.
    """
    ast = plan.ast
    state = plan.derivative_state
    if (
        len(ast.component_occurrences) != 2
        or len(ast.parent_occurrences) != 1
        or ast.multiplicative_groups
        or len(ast.joins) != 1
        or ast.joins[0].interface.kind != FusionJoinKind.ORTHO
        or ast.joins[0].host_occurrence != ast.parent_occurrences[0]
        or not plan.indicated_hydrogens
        or state.hydro_operations
        or state.unsaturation_operations
        or state.external_pi_operations
        or state.added_hydrogen_operations
        or state.intrinsic_hydro_operations
        or state.pi_redistribution
        or plan.charge_operations
        or plan.lambda_descriptors
    ):
        return None
    graph = plan.abstract_parent_graph
    symbols = {atom.id: atom.symbol for atom in graph.atoms}
    if any(atom.symbol not in {"C", "N"} or atom.formal_charge for atom in graph.atoms):
        return None
    locants = plan.numbering.string_input_locant_maps()[0]
    indicated = {atom for atom, locant in locants.items() if locant in set(map(str, plan.indicated_hydrogens))}
    if any(symbols[atom] != "C" for atom in indicated):
        return None
    model = plan.bond_model
    active = set(symbols) - indicated
    if model.required_double_bonds or any(not indicated.intersection(edge) for edge in model.required_single_bonds):
        return None
    if not model.allowed_kekule_assignments or any(
        {atom for edge, order in assignment.orders if order == 2 for atom in edge} != active
        for assignment in model.allowed_kekule_assignments
    ):
        return None
    return forced_surviving_child_edge(plan, frozenset(active))


def forced_surviving_child_edge(plan: FusionParentPlan, active: frozenset[int]) -> tuple[int, int] | None:
    """Prove the first leaf-edge choice is absent from every parent matching.

    Callers certify the chemical scope and the complete active pi domain.
    Only an untouched, neutral imine N can anchor this parent-only scan:
    external substitution at carbon can change the parser's visit order.
    """
    ast, model, graph = plan.ast, plan.bond_model, plan.abstract_parent_graph
    symbols = {atom.id: atom.symbol for atom in graph.atoms}
    neighbors = {atom: set() for atom in symbols}
    for bond in graph.bonds:
        left, right = bond.atoms
        neighbors[left].add(right)
        neighbors[right].add(left)
    locants = plan.numbering.string_input_locant_maps()[0]
    # No terminal pair may preempt the first nonfusion choice.
    if any(len(neighbors[atom] & active) < 2 for atom in active):
        return None
    first = min(
        (atom for atom in active if len(neighbors[atom]) == 2),
        key=lambda atom: system_locant_sort_key(locants[atom]),
        default=None,
    )
    if first is None or symbols[first] != "N" or first in {operation.atom_id for operation in plan.charge_operations}:
        return None
    all_shared = frozenset().union(*(join.shared_input_atoms for join in ast.joins))
    matches = {match.occurrence_id: match for match in ast.component_occurrences}
    for join in ast.joins:
        child = matches[join.attached_occurrence]
        child_atoms = set(child.input_atom_by_locant.values())
        if (
            first not in child_atoms - all_shared
            or not neighbors[first] <= child_atoms
            or any(other.host_occurrence == child.occurrence_id for other in ast.joins)
        ):
            continue
        retained = neighbors[first] - all_shared
        if len(retained) != 1 or len(neighbors[first] & set(join.shared_input_atoms)) != 1:
            continue
        edge = tuple(sorted((first, next(iter(retained)))))
        if (
            edge in model.pi_eligible_edges
            and model.allowed_kekule_assignments
            and all(dict(assignment.orders)[edge] == 1 for assignment in model.allowed_kekule_assignments)
        ):
            return edge
    return None

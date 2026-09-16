"""Construction-order pi witnesses for charged parents with sigma donors."""

from ..molecule import Molecule
from .citation_pi import forced_surviving_child_edge
from .model import FusionJoinKind, FusionParentPlan


def charged_donor_citation_pi_dead_end(mol: Molecule, plan: FusionParentPlan) -> tuple[int, int] | None:
    """Reject a forced parser edge only against a complete, proved pi domain.

    The neutral carbon-H witness does not describe N-oxide/sigma-N parents.
    Here the input graph proves their parser valence roles before the same
    surviving-child-edge argument is applied to a leaf of an ortho tree.
    No matching, charge, hydrogen, or numbering operation is changed.
    """
    ast, state, model = plan.ast, plan.derivative_state, plan.bond_model
    if (
        not plan.charge_operations
        or plan.indicated_hydrogens
        or plan.lambda_descriptors
        or state.hydro_operations
        or state.unsaturation_operations
        or state.external_pi_operations
        or state.added_hydrogen_operations
        or state.intrinsic_hydro_operations
        or state.pi_redistribution
        or len(ast.parent_occurrences) != 1
        or ast.multiplicative_groups
        or len(ast.joins) != len(ast.component_occurrences) - 1
        or any(
            join.kind not in {FusionJoinKind.ORTHO, FusionJoinKind.HIGHER_ORDER} or len(join.shared_input_atoms) != 2
            for join in ast.joins
        )
        or model.required_double_bonds
        or not model.allowed_kekule_assignments
    ):
        return None
    graph = plan.abstract_parent_graph
    atoms = frozenset(atom.id for atom in graph.atoms)
    if any(atom.symbol not in {"C", "N"} or atom.formal_charge for atom in graph.atoms):
        return None
    neighbors = {atom: atoms.intersection(mol.get_neighbors(atom)) for atom in atoms}
    active = {atom for edge, order in model.allowed_kekule_assignments[0].orders if order == 2 for atom in edge}
    donors = atoms - active
    if not donors or any(
        {atom for edge, order in assignment.orders if order == 2 for atom in edge} != active
        for assignment in model.allowed_kekule_assignments
    ):
        return None
    if any(not donors.intersection(edge) for edge in model.required_single_bonds):
        return None
    charged = {operation.atom_id for operation in plan.charge_operations}
    if charged != {atom for atom in atoms if mol.atoms[atom].charge}:
        return None
    for atom in atoms:
        observed = mol.atoms[atom]
        external = set(mol.get_neighbors(atom)) - atoms
        if any(mol.get_bond(atom, other).order != 1 for other in external):
            return None
        if atom in donors:
            if (
                observed.symbol != "N"
                or observed.charge
                or observed.total_h_count
                or len(neighbors[atom]) != 2
                or len(external) != 1
                or any(mol.get_bond(atom, other).order != 1 for other in neighbors[atom])
            ):
                return None
        elif atom in charged:
            if (
                observed.symbol != "N"
                or observed.charge != 1
                or observed.total_h_count
                or len(neighbors[atom]) != 2
                or len(external) != 1
                or sum(mol.get_bond(atom, other).order for other in neighbors[atom]) != 3
            ):
                return None
            oxide = mol.atoms[next(iter(external))]
            if oxide.symbol != "O" or oxide.charge != -1 or len(mol.get_neighbors(oxide.idx)) != 1:
                return None
        elif observed.symbol == "N" and (external or observed.total_h_count or len(neighbors[atom]) != 2):
            return None
    return forced_surviving_child_edge(plan, frozenset(active))

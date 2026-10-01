"""Prove paired external-pi consumption without assigning extra hydrogen."""

from ..molecule import Molecule
from ..polycycle_topology import normalize_edge
from .exocyclic import is_neutral_external_pi_ligand
from .mancude import ParentDerivativeState


def has_paired_external_pi_consumption(mol: Molecule, atoms: frozenset[int], state: ParentDerivativeState) -> bool:
    """Match alternating carbon paths ending at two external-pi ligands.

    Each path loses one skeletal double bond overall. Its two endpoints gain
    external pi occupancy, while every interior vertex retains its occupancy.
    Alternating cycles only change the Kekule representation. No H operation
    or heteroatom valence change may pay for a missing path endpoint.
    """
    if (
        not state.bond_delta.compatible
        or state.hydro_operations
        or state.unsaturation_operations
        or state.added_hydrogen_operations
        or state.intrinsic_hydro_operations
    ):
        return False
    endpoints = set()
    for operation in state.external_pi_operations:
        parent, external = operation.parent_atom_id, operation.external_atom_id
        bond = mol.bonds.get(operation.bond_id)
        if (
            parent not in atoms
            or external in atoms
            or external not in mol.atoms
            or parent in endpoints
            or mol.atoms[parent].symbol != "C"
            or len(atoms.intersection(mol.get_neighbors(parent))) != 2
            or bond is None
            or normalize_edge(bond.u, bond.v) != normalize_edge(parent, external)
            or bond.order != 2
            or mol.atoms[external].symbol != operation.external_atom_symbol
            or not is_neutral_external_pi_ligand(mol, parent, external)
        ):
            return False
        endpoints.add(parent)
    if not endpoints or len(endpoints) % 2:
        return False
    expected = dict(state.bond_delta.assignment.orders)
    observed = {
        normalize_edge(bond.u, bond.v): bond.order for bond in mol.bonds.values() if bond.u in atoms and bond.v in atoms
    }
    if set(expected) != set(observed) or any(order not in {1, 2} for order in (*expected.values(), *observed.values())):
        return False
    changed = {}
    parent_pi = dict.fromkeys(atoms, 0)
    actual_pi = dict.fromkeys(atoms, 0)
    for edge, order in expected.items():
        for atom in edge:
            parent_pi[atom] += order - 1
            actual_pi[atom] += observed[edge] - 1
            if order != observed[edge]:
                changed.setdefault(atom, []).append(order - observed[edge])
    if not endpoints <= changed.keys():
        return False
    for atom, differences in changed.items():
        value = mol.atoms[atom]
        if (
            value.symbol != "C"
            or value.charge
            or value.total_h_count + sum(mol.get_bond(atom, other).order for other in mol.get_neighbors(atom)) != 4
            or parent_pi[atom] != 1
        ):
            return False
        if atom in endpoints:
            if differences != [1] or actual_pi[atom] != 0:
                return False
        elif sorted(differences) != [-1, 1] or actual_pi[atom] != 1:
            return False
    return True

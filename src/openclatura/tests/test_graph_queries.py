from openclatura.graph_queries import (
    bond_ids_within,
    bond_order,
    charged_atom_ids,
    component_atoms_until_blocked,
    edges_within_atoms,
    normalize_edges,
)
from openclatura.molecule import Molecule


def test_graph_queries_return_induced_bonds_edges_and_charges():
    mol = Molecule()
    mol.add_atom("C", idx=0)
    mol.add_atom("N", idx=1, charge=1)
    mol.add_atom("O", idx=2, charge=-1)
    internal = mol.add_bond(0, 1, idx=4)
    mol.add_bond(1, 2, idx=5)

    selected = {0, 1}

    assert charged_atom_ids(mol, selected) == {1}
    assert bond_order(mol, 0, 1) == 1
    assert bond_order(mol, 0, 2) == 0
    assert bond_order(mol, 0, None) == 0
    assert bond_ids_within(mol, selected) == {internal.idx}
    assert edges_within_atoms(mol, selected) == {(0, 1)}


def test_normalize_edges_canonicalizes_and_deduplicates_undirected_pairs():
    assert normalize_edges([(2, 1), (1, 2), (2, 3)]) == {(1, 2), (2, 3)}


def test_component_atoms_until_blocked_stops_at_graph_boundary():
    mol = Molecule()
    for atom_idx in range(4):
        mol.add_atom("C", idx=atom_idx)
    mol.add_bond(0, 1)
    mol.add_bond(1, 2)
    mol.add_bond(2, 3)

    assert component_atoms_until_blocked(mol, set(mol.atoms), 0, {2}) == {0, 1}
    assert component_atoms_until_blocked(mol, set(mol.atoms), 2, {2}) == set()

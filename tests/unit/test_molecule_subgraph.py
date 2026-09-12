"""Graph projections retain identities used by numbering, stereo and name bindings."""

import pytest

from openclatura.molecule import Molecule


@pytest.mark.parametrize("order", [(4, 12, 21), (21, 12, 4)])
def test_subgraph_preserves_sparse_id_chemistry_and_stereo(order):
    mol = Molecule()
    mol.add_atom("C", idx=4, isotope=13, stereo="R", raw_stereo="@", cip="R", total_h_count=1)
    mol.add_atom("N", idx=12, charge=1, explicit_h_count=1, total_h_count=1)
    mol.add_atom("C", idx=21, is_aromatic=True)
    mol.add_atom("O", idx=30)
    mol.add_bond(12, 4, idx=8, order=2, stereo="E", cip="E", in_small_ring=True)
    mol.add_bond(12, 21, idx=19)
    mol.add_bond(21, 30, idx=2)
    mol.accurate_cip = {4: "R", 30: "S"}

    fragment = mol.subgraph(order)

    assert fragment.atoms == {idx: mol.atoms[idx] for idx in order}
    assert set(fragment.bonds) == {8, 19}
    for idx, bond in fragment.bonds.items():
        original = mol.bonds[idx]
        assert bond == original
        assert fragment.get_bond(original.u, original.v) == bond
    assert fragment.accurate_cip == {4: "R"}
    assert fragment.get_neighbors(21) == [12]
    fragment.update_atom(12, charge=0)
    fragment.update_bond(8, order=1)
    assert mol.atoms[12].charge == 1
    assert mol.bonds[8].order == 2


def test_symbol_projection_keeps_graph_identity():
    mol = Molecule()
    mol.add_atom("N", idx=5)
    mol.add_atom("C", idx=9)
    mol.add_bond(9, 5, idx=42)

    fragment = mol.subgraph({5, 9}, symbols={5: "C"})

    assert fragment.atoms[5].symbol == "C"
    assert fragment.substituted_symbols == {5}
    assert fragment.get_bond(5, 9).idx == 42
    assert mol.atoms[5].symbol == "N"

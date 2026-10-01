"""A Kekule comparison must not admit proton, charge, stereo or topology changes."""

import pytest
from rdkit import Chem

from openclatura.resonance_compare import equivalent_smiles


def _bridged_conjugated_graph(ligand_length):
    graph = Chem.RWMol()
    for symbol in ("O", "C", "C", "C", "C", "C", "C", "C", "C", "O", "C", "C", "O", "C", "C", "C", "C", "C"):
        graph.AddAtom(Chem.Atom(symbol))
    double_edges = {(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (13, 14), (15, 16)}
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (8, 10),
        (10, 11),
        (11, 12),
        (11, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (6, 17),
        (16, 1),
        (7, 2),
        (16, 10),
        (17, 3),
    )
    for edge in edges:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in double_edges else Chem.BondType.SINGLE)
    last = 14
    for _ in range(ligand_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.parametrize("ligand_length", (0, 1, 2))
def test_bridged_conjugated_kekule_drawings_are_equivalent(ligand_length):
    graph = _bridged_conjugated_graph(ligand_length)
    original = Chem.MolToSmiles(graph)
    alternatives = {
        Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(candidate)))
        for candidate in Chem.ResonanceMolSupplier(graph, Chem.KEKULE_ALL)
    } - {original}
    assert alternatives
    for other in alternatives:
        assert equivalent_smiles(original, other)
        assert equivalent_smiles(other, original)


@pytest.mark.parametrize(
    "left,right",
    [
        ("CC=O", "C=CO"),  # Proton relocation.
        ("C=CC=C", "CC#CC"),  # Same formula, different H/bond distribution.
        ("[O-]C=O", "O=C[O-]"),  # Equivalent atom permutation remains permitted below.
        ("C/C=C/C", "C/C=C\\C"),
        ("C[C@H](O)C=O", "C[C@@H](O)C=O"),
        ("[NH3+]CC(=O)[O-]", "NCC(=O)O"),
        ("CC(C)C", "CCCC"),
        ("[13CH2]=CO", "C=[13CH]O"),
    ],
)
def test_resonance_does_not_hide_other_graph_changes(left, right):
    assert equivalent_smiles(left, right) == (Chem.CanonSmiles(left) == Chem.CanonSmiles(right))

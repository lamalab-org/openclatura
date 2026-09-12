"""An oxo-consumed parent pi bond retains ownership of its other H endpoint."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin


def _bridged_lactone(bridge_oxo, sidechain_length):
    graph = Chem.RWMol()
    for symbol in ("O", "C", "O", "C", "C", "C", "C", "C", "C", "C", "C", "C"):
        graph.AddAtom(Chem.Atom(symbol))
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
        (6, 10),
        (10, 11),
        (9, 1),
        (9, 4),
        (11, 4),
    )
    for left, right in edges:
        graph.AddBond(left, right, Chem.BondType.DOUBLE if (left, right) in {(0, 1), (7, 8)} else Chem.BondType.SINGLE)
    if bridge_oxo:
        oxygen = graph.AddAtom(Chem.Atom("O"))
        graph.AddBond(11, oxygen, Chem.BondType.DOUBLE)
    last = 3
    for _ in range(sidechain_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    for atom in (4, 6, 9):
        graph.GetAtomWithIdx(atom).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("bridge_oxo", (False, True))
@pytest.mark.parametrize("sidechain_length", (0, 1, 2))
def test_retained_bridge_oxo_added_hydrogen_preserves_graph_and_stereo(bridge_oxo, sidechain_length):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _bridged_lactone(bridge_oxo, sidechain_length)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert result.parent_nomenclature == "bridged_fusion"
        assert "(7aH)" in result.name
        check = verify_with_opsin(result.name, expected, standardize_smiles=False)
        assert check.ok, check.to_dict()

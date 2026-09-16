"""Proved chalcogen-oxo spectators do not suppress carbonyl added hydrogen."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin


def _graph(element, oxo_count, branch_length):
    graph = Chem.RWMol()
    for atom in range(9):
        graph.AddAtom(Chem.Atom(element if atom == 8 else "C"))
    for edge in tuple((atom, atom + 1) for atom in range(8)) + ((8, 0), (0, 4)):
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in {(2, 3), (6, 7)} else Chem.BondType.SINGLE)
    for parent in (1, 5, *((8,) * oxo_count)):
        oxygen = graph.AddAtom(Chem.Atom("O"))
        graph.AddBond(parent, oxygen, Chem.BondType.DOUBLE)
    last = 0
    for _ in range(branch_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("element", ("S", "Se"))
@pytest.mark.parametrize("oxo_count", (1, 2))
@pytest.mark.parametrize("branch_length", (1, 2))
def test_lambda_oxo_spectator_and_carbonyl_hydrogen_are_composed_together(element, oxo_count, branch_length):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _graph(element, oxo_count, branch_length)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert "cyclopenta[b]" in result.name
        assert "(4aH,7aH)" in result.name
        check = verify_with_opsin(result.name, expected, standardize_smiles=False)
        assert check.ok, check.to_dict()

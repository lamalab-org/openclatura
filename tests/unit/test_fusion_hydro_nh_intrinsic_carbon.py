"""Hydrogenated N-H does not hide an independently unpaired carbon-H site."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin


def _graph(branch_length, amino):
    graph = Chem.RWMol()
    for atom in range(10):
        graph.AddAtom(Chem.Atom("N" if atom in {0, 2} else "O" if atom == 8 else "C"))
    for edge in tuple((atom, atom + 1) for atom in range(9)) + ((9, 0), (4, 9)):
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in {(3, 4), (9, 0)} else Chem.BondType.SINGLE)
    last = 1
    for _ in range(branch_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    if amino:
        nitrogen = graph.AddAtom(Chem.Atom("N"))
        graph.AddBond(3, nitrogen, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("branch_length", (0, 1, 2))
@pytest.mark.parametrize("amino", (False, True))
def test_additive_nh_and_intrinsic_carbon_h_have_separate_ownership(branch_length, amino):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _graph(branch_length, amino)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert "5H-pyrano[2,3-d]pyrimidin" in result.name
        check = verify_with_opsin(result.name, expected, standardize_smiles=False)
        assert check.ok, check.to_dict()

"""External carbonyl ownership is distinct from intrinsic carbon hydrogen."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin


def _pyrano_oxo_graph(second_oxo, branch_length):
    graph = Chem.RWMol()
    symbols = ["C", "N", "C", "N" if second_oxo else "C", "C", "C", "C", "C", "C", "O"]
    for symbol in symbols:
        graph.AddAtom(Chem.Atom(symbol))
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4))
    doubles = {(4, 5), (7, 8)} | (set() if second_oxo else {(2, 3)})
    for edge in edges:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in doubles else Chem.BondType.SINGLE)
    for site in (0, 2) if second_oxo else (0,):
        oxygen = graph.AddAtom(Chem.Atom("O"))
        graph.AddBond(site, oxygen, Chem.BondType.DOUBLE)
    for nitrogen in (1, 3) if second_oxo else (1,):
        methyl = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(nitrogen, methyl, Chem.BondType.SINGLE)
    last = 6
    for _ in range(branch_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    if branch_length:
        graph.GetAtomWithIdx(6).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("second_oxo", (False, True))
@pytest.mark.parametrize("branch_length", (0, 1, 2))
def test_oxo_donor_fusion_keeps_carbon_h_and_exact_bond_state(second_oxo, branch_length):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _pyrano_oxo_graph(second_oxo, branch_length)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert "pyrano[" in result.name
        assert "bicyclo" not in result.name
        check = verify_with_opsin(result.name, expected, standardize_smiles=False)
        assert check.ok, check.to_dict()

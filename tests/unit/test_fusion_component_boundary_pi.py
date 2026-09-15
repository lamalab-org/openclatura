"""Shared boundary pi occupancy is proved across the completed fused graph."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin


def _graph(sidechain_length, acetal_substituent):
    graph = Chem.RWMol()
    for symbol in ("C", "O", "C", "C", "O", "C", "O", "C", "C", "C", "C", "C", "N", "C"):
        graph.AddAtom(Chem.Atom(symbol))
    edges = tuple((atom, atom + 1) for atom in range(13)) + ((13, 0), (7, 2), (13, 8))
    for edge in edges:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in {(9, 10), (13, 8)} else Chem.BondType.SINGLE)
    oxygen = graph.AddAtom(Chem.Atom("O"))
    graph.AddBond(11, oxygen, Chem.BondType.DOUBLE)
    methyl = graph.AddAtom(Chem.Atom("C"))
    graph.AddBond(12, methyl, Chem.BondType.SINGLE)
    last = 0
    for _ in range(sidechain_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    if acetal_substituent:
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(5, carbon, Chem.BondType.SINGLE)
    for atom in (2, 7):
        graph.GetAtomWithIdx(atom).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("sidechain_length", (0, 1, 2))
@pytest.mark.parametrize("acetal_substituent", (False, True))
def test_component_boundary_pi_matching_keeps_junction_stereochemistry(sidechain_length, acetal_substituent):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _graph(sidechain_length, acetal_substituent)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert "dioxino[" in result.name
        assert "pyrano[" in result.name
        check = verify_with_opsin(result.name, expected, standardize_smiles=False)
        assert check.ok, check.to_dict()


def _four_component_graph(sidechain_length):
    graph = Chem.RWMol()
    for atom in range(18):
        graph.AddAtom(Chem.Atom("O" if atom in {1, 16} else "N" if atom == 10 else "C"))
    edges = tuple((atom, atom + 1) for atom in range(17)) + ((9, 0), (17, 12), (7, 2), (17, 8))
    for edge in edges:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in {(2, 3), (4, 5), (6, 7), (8, 9)} else Chem.BondType.SINGLE)
    oxygen = graph.AddAtom(Chem.Atom("O"))
    graph.AddBond(0, oxygen, Chem.BondType.DOUBLE)
    last = 11
    for _ in range(sidechain_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    for atom in (12, 17):
        graph.GetAtomWithIdx(atom).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("sidechain_length", (0, 1, 2))
def test_alternating_component_projection_preserves_nonjunction_pi_occupancy(sidechain_length):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _four_component_graph(sidechain_length)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert result.name.count("pyrano[") == 2
        check = verify_with_opsin(result.name, expected, standardize_smiles=False)
        assert check.ok, check.to_dict()

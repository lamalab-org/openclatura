"""Fusion junctions cannot acquire pi bonds beyond their fixed valence."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.model import FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.fusion.numbering import parent_bond_model


@pytest.mark.parametrize("degree", (3, 4))
def test_fixed_valence_junction_limits_parent_pi_capacity(degree):
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in range(degree + 1)),
        bonds=tuple(FusionGraphBond((0, atom)) for atom in range(1, degree + 1)),
    )
    model = parent_bond_model(graph)
    assert model.maximum_non_cumulative_double_bonds == (1 if degree == 3 else 0)
    for assignment in model.allowed_kekule_assignments:
        assert sum(order for edge, order in assignment.orders if 0 in edge) <= 4


@pytest.mark.opsin
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("ethyl_derivative", (False, True))
def test_epoxide_fused_parent_hydrogenation_roundtrips(reverse, ethyl_derivative):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = Chem.RWMol(
        Chem.MolFromSmiles("CC(C)[C@@H]1C[C@@H]([Si](C)(C)C)[C@]2(C)CC[C@@H]([Se]c3ccccc3)[C@@H]3O[C@@]32C1")
    )
    if ethyl_derivative:
        terminal = next(
            atom.GetIdx()
            for atom in graph.GetAtoms()
            if atom.GetSymbol() == "C" and atom.GetDegree() == 1 and atom.GetNeighbors()[0].GetSymbol() == "Si"
        )
        graph.AddBond(terminal, graph.AddAtom(Chem.Atom("C")), Chem.BondType.SINGLE)
        Chem.SanitizeMol(graph)
    reference = Chem.MolToSmiles(graph)
    if reverse:
        graph = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    result = name_mol(graph)
    assert result.error is None, result.error
    assert "naphtho" in result.name
    assert "octahydro" in result.name
    check = verify_with_opsin(result.name, reference, standardize_smiles=False)
    assert check.status == "matched", (result.name, check.to_dict())

"""Fixed acetal saturation does not consume the completed parent's pi budget."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _graph(branch_length, ester):
    graph = Chem.RWMol()
    for symbol in ("C", "C", "C", "O", "C", "C", "O", "C", "O", "C"):
        graph.AddAtom(Chem.Atom(symbol))
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 9), (9, 0), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)):
        graph.AddBond(left, right, Chem.BondType.SINGLE)
    oxygen = graph.AddAtom(Chem.Atom("O"))
    graph.AddBond(1, oxygen, Chem.BondType.DOUBLE)
    hydroxyl = graph.AddAtom(Chem.Atom("O"))
    graph.AddBond(2, hydroxyl, Chem.BondType.SINGLE)
    previous = 7
    for _ in range(branch_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(previous, carbon, Chem.BondType.SINGLE)
        previous = carbon
    if ester:
        oxygen, carbon, oxo, methyl = [graph.AddAtom(Chem.Atom(s)) for s in ("O", "C", "O", "C")]
        for left, right, order in ((0, oxygen, 1), (oxygen, carbon, 1), (carbon, oxo, 2), (carbon, methyl, 1)):
            graph.AddBond(left, right, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.parametrize("branch_length", (0, 1, 2))
@pytest.mark.parametrize("ester", (False, True))
def test_component_acetal_does_not_suppress_fusion_junction_pi(branch_length, ester):
    graph = _graph(branch_length, ester)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        reordered = Chem.RenumberAtoms(graph, order)
        mol = read_rdkit_mol(reordered)
        atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
        plan = plan_fusion_parent(mol, atoms, mode="audited_pin")
        assert isinstance(plan, FusionConfirmed), plan
        assert plan.plan.bond_model.maximum_non_cumulative_double_bonds == 3
        result = name_mol(reordered, include_trace=True)
        assert result.error is None
        assert "pyrano[" in result.name
        if opsin_available():
            check = verify_with_opsin(result.name, expected, standardize_smiles=False)
            assert check.ok, check.to_dict()

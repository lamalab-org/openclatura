"""A retained monocycle must not activate the fused-parent bridge route."""

import pytest
from rdkit import Chem

from openclatura import name, name_mol
from openclatura.fusion.model import FusionMode
from openclatura.fusion.wrappers import plan_bridged_fusion_wrapper
from openclatura.graph_io import read_rdkit_mol


@pytest.mark.parametrize(
    "smiles",
    [
        "CC1C2C3CC2N13",
        "CC1C2C3CN2C13",
        "CCCC1C2C3CC2N13",
        "CCCC1C2C3CN2C13",
        "CCOC1C2C3CN2C13",
        "COCC1C2C3CC2N13",
    ],
)
def test_reported_cage_substituents_remain_named(smiles):
    result = name(smiles, include_trace=True, verify_opsin=True)

    assert result.error is None
    assert "tricyclo" in result.name
    assert result.opsin_check.status == "matched"
    assert not any(step.decision == "selected audited bridged fusion parent" for step in result.decisions)


def _substituted_cage(nitrogen, ligand):
    graph = Chem.RWMol()
    for index in range(6):
        graph.AddAtom(Chem.Atom("N" if index == nitrogen else "C"))
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (4, 1), (5, 2)):
        graph.AddBond(left, right, Chem.BondType.SINGLE)
    previous = 0
    for symbol in ligand:
        atom = graph.AddAtom(Chem.Atom(symbol))
        graph.AddBond(previous, atom, Chem.BondType.SINGLE)
        previous = atom
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("nitrogen", [4, 5])
@pytest.mark.parametrize("ligand", [("C",), ("C", "C"), ("C", "C", "C"), ("O", "C", "C"), ("C", "O", "C")])
@pytest.mark.parametrize("reverse_atoms", [False, True])
@pytest.mark.parametrize("mode", [FusionMode.AUDITED_PIN, FusionMode.GENERAL])
def test_graph_built_small_cages_do_not_become_bridged_monocycles(nitrogen, ligand, reverse_atoms, mode):
    mol = _substituted_cage(nitrogen, ligand)
    ring_atoms = frozenset(range(6))
    if reverse_atoms:
        count = mol.GetNumAtoms()
        mol = Chem.RenumberAtoms(mol, list(reversed(range(count))))
        ring_atoms = frozenset(count - 1 - atom for atom in ring_atoms)
    graph = read_rdkit_mol(mol)

    assert plan_bridged_fusion_wrapper(graph, ring_atoms, mode=mode) is None
    result = name_mol(mol, fusion_mode=mode, verify_opsin=True)
    assert result.error is None
    assert "tricyclo" in result.name
    assert result.opsin_check.status == "matched"

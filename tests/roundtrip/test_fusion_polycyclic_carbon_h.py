"""Local component hydrogen cannot fix the completed polycyclic pi state."""

import random

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol

CORE = "C1=CC2=C3C4=C(C2C=C1)C1C=CC=CC1=C4c1ccccc13"
FUSION = "dibenzo[1',2':1,2;1'',2'':5,6]pentaleno[3,3a,4-ab]indene"


@pytest.mark.opsin
@pytest.mark.parametrize("mode", [FusionMode.AUDITED_PIN, FusionMode.GENERAL])
@pytest.mark.parametrize("ordering", ["original", "reversed", "shuffled"])
@pytest.mark.parametrize("substituent", [None, "C", "O", "F"])
def test_polycyclic_component_h_relocates_before_hydrogenation(mode, ordering, substituent):
    mol = Chem.MolFromSmiles(CORE)
    if substituent:
        graph = Chem.RWMol(mol)
        atom = graph.AddAtom(Chem.Atom(substituent))
        graph.AddBond(0, atom, Chem.BondType.SINGLE)
        mol = graph.GetMol()
        Chem.SanitizeMol(mol)
    order = list(range(mol.GetNumAtoms()))
    if ordering == "reversed":
        order.reverse()
    elif ordering == "shuffled":
        random.Random(53).shuffle(order)
    result = name_mol(Chem.RenumberAtoms(mol, order), fusion_mode=mode, verify_opsin=True)

    assert result.error is None
    assert (FUSION.removesuffix("e") if substituent == "O" else FUSION) in result.name
    assert "1H-indene" not in result.name
    if substituent is None:
        assert result.name == "4a,4c-dihydro" + FUSION
    assert result.opsin_check is not None and result.opsin_check.status == "matched"
    decoded = Chem.MolFromSmiles(result.opsin_check.opsin_smiles)
    assert decoded is not None
    assert Chem.MolToSmiles(decoded, isomericSmiles=True) == Chem.MolToSmiles(mol, isomericSmiles=True)


def test_pi_redistribution_trace_keeps_atom_and_bond_proof():
    mol = Chem.MolFromSmiles(CORE)
    result = name_mol(mol, include_trace=True)
    step = next(
        step for step in result.decisions if step.decision == "proved fusion hydrogenation with pi redistribution"
    )
    assert set(step.atoms) == {6, 9}
    assert step.data["hydrogenated_atom_ids"] == [6, 9]
    assert step.data["removed_pi_bond_ids"]
    assert step.data["added_pi_bond_ids"]
    assert len(step.data["final_bond_orders"]) == mol.GetNumBonds()


@pytest.mark.opsin
def test_substituent_on_hydrogenated_fusion_junction_is_not_lost():
    graph = Chem.RWMol(Chem.MolFromSmiles(CORE))
    methyl = graph.AddAtom(Chem.Atom("C"))
    graph.AddBond(6, methyl, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    result = name_mol(graph, verify_opsin=True)
    assert result.name == "4a-methyl-4a,4c-dihydro" + FUSION
    assert result.opsin_check is not None and result.opsin_check.status == "matched"
    decoded = Chem.MolFromSmiles(result.opsin_check.opsin_smiles)
    assert Chem.MolToSmiles(decoded) == Chem.MolToSmiles(graph)

"""Completed numbering must agree with the independently parsed fusion graph."""

import random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol
from openclatura.opsin_verify import verify_with_opsin

# Maps verified independently against OPSIN's completed-parent CML locants.
# Names and structures are fixtures, never production recognition keys.
CASES = (
    pytest.param(
        "Cc1cccc(Cl)c1N1CC=C[C@]23S[C@@]4(C)C=CCCOC(=O)[C@H]4[C@H]2C(=O)N([C@H](C)CO)C3C1=O",
        {
            16: "13",
            17: "12",
            18: "11",
            19: "10",
            20: "9",
            21: "8",
            23: "7b",
            24: "7a",
            25: "7",
            27: "6",
            32: "5a",
            33: "5",
            8: "4",
            9: "3",
            10: "2",
            11: "1",
            12: "14a",
            13: "14",
            14: "13a",
        },
        id="pubchem-2067",
    ),
    pytest.param(
        "O=C1C2C=CC=CC2CC2C3C=CC=CC3=CC=CN12",
        {
            6: "13",
            5: "12",
            4: "11",
            3: "10",
            2: "9a",
            1: "9",
            19: "8",
            18: "7",
            17: "6",
            16: "5",
            15: "4a",
            14: "4",
            13: "3",
            12: "2",
            11: "1",
            10: "14b",
            9: "14a",
            8: "14",
            7: "13a",
        },
        id="pubchem-2783",
    ),
    pytest.param(
        "CO[C@@H]1CC[C@H]2CCC[C@H]3C(=O)OC[C@H]3C[C@H]21",
        {
            10: "3",
            12: "2",
            13: "1",
            14: "10a",
            15: "10",
            16: "9a",
            2: "9",
            3: "8",
            4: "7",
            5: "6a",
            6: "6",
            7: "5",
            8: "4",
            9: "3a",
        },
        id="pubchem-3555",
    ),
    pytest.param(
        "CCCN(CCC)CC(=O)N1c2cc(F)ccc2-n2c(n[nH]c2=O)-c2cccnc21",
        {
            20: "3",
            21: "2",
            22: "1",
            18: "13",
            17: "12a",
            16: "12",
            15: "11",
            13: "10",
            12: "9",
            11: "8a",
            10: "8",
            29: "7a",
            28: "7",
            27: "6",
            26: "5",
            25: "4",
            24: "3b",
            19: "3a",
        },
        id="pubchem-3883",
    ),
)


def _permuted(smiles, order):
    mol = Chem.MolFromSmiles(smiles)
    indices = list(range(mol.GetNumAtoms()))
    if order == "reversed":
        indices.reverse()
    elif order == "shuffled":
        random.Random(19).shuffle(indices)
    return Chem.RenumberAtoms(mol, indices), indices


@pytest.mark.parametrize("smiles,expected", CASES)
@pytest.mark.parametrize("order", ("original", "reversed", "shuffled"))
def test_completed_locants_match_parsed_parent(smiles, expected, order):
    graph, indices = _permuted(smiles, order)
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed), result
    assert result.plan.audit.confirmed
    actual = {atom: str(locant) for atom, locant in result.plan.numbering.input_locant_maps[0]}
    assert actual == {new: expected[old] for new, old in enumerate(indices) if old in expected}
    assert len(set(actual.values())) == len(atoms)
    state = result.plan.derivative_state
    for operation in (*state.hydro_operations, *state.added_hydrogen_operations):
        assert operation.locants == tuple(actual[atom] for atom in operation.atom_ids)
    for operation in state.oxo_operations:
        assert operation.locant == actual[operation.parent_atom_id]


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("smiles,expected", CASES)
@pytest.mark.parametrize("order", ("original", "reversed", "shuffled"))
def test_completed_fusion_roundtrips_with_stereo_under_permutations(smiles, expected, order):
    mol, _ = _permuted(smiles, order)
    result = name_mol(mol, fusion_mode=FusionMode.AUDITED_PIN)
    assert result.error is None
    assert "cyclo[" not in result.name
    check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
    assert check.status == "matched", (result.name, check)
    assert check.canonical_original == check.canonical_roundtrip

"""Round-trip tests for test_principal_group_breadth.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "CCS(=O)O",
    "CCP(=O)O",
    "CC[Se](=O)O",
    "CC[Se](=O)(=O)O",
    "CC[Te](=O)O",
    "CC[Te](=O)(=O)O",
    "CCB(O)O",
    "CC(=N)N",
    "CC(=N)NCC",
    "CC(=NO)C",
    "CC=NO",
    "CC(=NOC)C",
    "CC(=NN)C",
    "CC=NN",
    "CC(=NNC)C",
    "COC(=N)C",
    "COC(=N)NCC",
    "O=S(O)c1oc2ncccc2c1",
    "OB(O)c1oc2ncccc2c1",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

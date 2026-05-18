"""Round-trip tests for test_oxyacids.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'CCS(=O)(=O)O',
    'CCP(=O)(O)O',
    'CO[N+](=O)[O-]',
    'CCO[N+](=O)[O-]',
    'CON=O',
    'CCON=O',
    'CCOS(=O)(=O)O',
    'CCOP(=O)(O)O',
    'CCCCC(CC)CCC(CC(C)C)OS(=O)(=O)O',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

"""Round-trip tests for test_p13_conjunctive.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'c1ccccc1-c1ccccc1',
    'n1ccccc1-c1ccccn1',
    'C1CC1-C1CC1',
    'c1ccccc1-c2ccc(cc2)-c2ccccc2',
    'C1=CCCCC1-C1=CCCCC1',
    'C1=COC=CC1-C1=COC=CC1',
    'OCCc1ccncc1',
    'OCCc1ccccc1-c1ccccc1',
    'C1CCCCC1CCO',
    'OCCC1CCCCC1',
    'O=C(O)CC1CCCC1',
    'c1ccccc1CC(=O)O',
    'c1ccccc1CC=O',
    'c1ccccc1CC#N',
    'c1ccccc1C(C)O',
    'c1ccccc1CC(C)O',
    'n1ccccc1C(C)O',
    'OCCC1=CC=NC=C1',
    'O=C(O)Cc1ccccn1',
    'O=C(O)Cc1ccncc1',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

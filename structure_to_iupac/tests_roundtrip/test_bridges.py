"""Round-trip tests for test_bridges.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'c1ccccc1',
    'C1CC1',
    'CC(C)C',
    'CC1CC1CC',
    'CCCC(C)(C)CC',
    'CC1CC1CC(C(C)C)C',
    'c1ccnnc1',
    'c1cncnc1',
    'c1c[nH]cn1',
    'c1cocn1',
    'c1cnoc1',
    'C1CCNCC1',
    'C1COCCN1',
    'c1ccc2ccccc2c1',
    'c1ccc2ncccc2c1',
    'c1ccc2occc2c1',
    'c1oc2ncccc2c1',
    'C1CC2CCC1C2',
    'C1CCC2(CC1)CCC2',
    'C1CO1',
    'O1COCC1',
    'O1CC=CCC1',
    'C1CCCCCC1',
    'C1CCCCCCC1',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

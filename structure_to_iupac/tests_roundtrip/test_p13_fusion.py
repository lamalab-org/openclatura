"""Round-trip tests for test_p13_fusion.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'c1ccc2ccccc2c1',
    'c1ccc2cc3ncccc3cc2c1',
    'c1ccc2cc3ncncc3cc2c1',
    'c1oc2cc3nc4ccccc4nc3cc2c1',
    'c1oc2cc3[nH]ncc3cc2c1',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

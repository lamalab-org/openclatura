"""Round-trip tests for test_composite_bridges.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'CCCCC',
    'CC1CC1C',
    'CC1CC1CC',
    'CC(C)CC',
    'CCCC(C)(C)CC',
    'CCCC(C(C)C)CC',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

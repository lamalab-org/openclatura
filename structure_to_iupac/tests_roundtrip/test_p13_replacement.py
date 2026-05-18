"""Round-trip tests for test_p13_replacement.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'NOC',
    'Cl[SiH]=[SiH]Cl',
    'N1CC2CCC1C2',
    'N1CC2CNCC1C2',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

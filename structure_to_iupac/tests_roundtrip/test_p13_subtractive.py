"""Round-trip tests for test_p13_subtractive.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'CC=C',
    'C#CC',
    'C1CCCC=C1',
    'C1CC#CCC1',
    'Cl[SiH]=[SiH]Cl',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

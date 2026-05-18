"""Round-trip tests for test_p13_multiplicative.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'CC(C)(C)C',
    'c1oc2nc3occc3cc2c1',
    'c1oc2nc3occc3nc2c1',
    'CC(C)(C)Oc1nc(Cl)c(Cl)c(OC(C)(C)C)c1Cl',
    'CC(O)c1cc(OC(C)(C)C)cc(OC(C)(C)C)c1',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

"""Round-trip tests for test_p13_substitutive.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'CCO',
    'O=C(O)CC1CCCC1',
    'CCNC(CN1CCC(C)(CC)CC1)C(C)C',
    'CCc1ccc(-c2nc(C3(CN)CC3)[nH]c2C)cc1',
    'CC=NO',
    'CC=NN',
    'CC[SeH]',
    'CC[TeH]',
    'Cc1ccc(C(=O)N[C@H](C)CC(=O)NC(C)C)c(Br)c1',
    'CC(C)(C)Oc1nc(Cl)c(Cl)c(OC(C)(C)C)c1Cl',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

"""Round-trip tests for test_p13_additive.py."""

import pytest

from openclatura.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "N1CCCC2=CC=CC=C12",
    "C1=CCNCC1",
    "CP(F)(OC)c1ccccc1",
    "C1COc2ncccc21",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

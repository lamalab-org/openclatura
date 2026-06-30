"""Round-trip tests for test_heteroatom_substituents.py."""

import pytest

from openclatura.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "COCCO",
    "CN(C)CCO",
    "C[PH]CCO",
    "C[SeH]CCO",
    "C[SiH2]CCO",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

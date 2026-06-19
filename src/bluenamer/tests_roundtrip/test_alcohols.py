"""Round-trip tests for test_alcohols.py."""

import pytest

from bluenamer.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "CCO",
    "CCCO",
    "CC(O)C",
    "NCCO",
    "OCCO",
    "OCC(O)CO",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

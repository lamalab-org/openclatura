"""Round-trip tests for test_locant_display.py."""

import pytest

from bluenamer.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "CCO",
    "CC(C)CC",
    "CCCC",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

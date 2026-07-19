"""Round-trip tests for test_nitro.py."""

import pytest

from openclatura.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "C[NH3+]",
    "C[N+](=O)[O-]",
    "CC[N+](=O)[O-]",
    "O=[N+]([O-])c1ccccc1",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

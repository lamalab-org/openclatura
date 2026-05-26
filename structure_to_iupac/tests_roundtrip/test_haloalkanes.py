"""Round-trip tests for test_haloalkanes.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "CCCl",
    "ClCCO",
    "CC(F)C",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

"""Round-trip tests for test_ketones.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "CC(=O)C",
    "CCC(=O)C",
    "O=CCO",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

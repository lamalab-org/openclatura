"""Round-trip tests for test_alkanes.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "C",
    "CC",
    "CCC",
    "CCCC",
    "CC(C)C",
    "CC(C)(C)C",
    "C1CCCCC1",
    "c1ccccc1",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

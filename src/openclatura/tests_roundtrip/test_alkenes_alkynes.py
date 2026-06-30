"""Round-trip tests for test_alkenes_alkynes.py."""

import pytest

from bluenamer.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "C=C",
    "CC=C",
    "C#C",
    "C=CC=C",
    "C=CCC#C",
    "C1=CC=CCC1",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

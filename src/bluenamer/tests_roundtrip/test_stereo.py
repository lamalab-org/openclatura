"""Round-trip tests for test_stereo.py."""

import pytest

from bluenamer.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "F[C@H](Cl)Br",
    "C/C=C/C",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

"""Round-trip tests for test_fused_multiplicity.py."""

import pytest

from bluenamer.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "c1oc2nc3occc3cc2c1",
    "c1sc2nc3sccc3cc2c1",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

import pytest

from openclatura.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

CASES = [
    pytest.param("[N-]1[NH+]=CC=C2C=CN=C12", id="qm9-129001"),
    pytest.param("[N-]1[NH+]=CC=C2N=CC=C12", id="qm9-129002"),
    pytest.param("[N-]1[NH+]=CC=C2N=CN=C12", id="qm9-129003"),
    pytest.param("[N-]1[NH+]=CN=C2C=CN=C12", id="qm9-129004"),
    pytest.param("[N-]1[NH+]=CN=C2N=CN=C12", id="qm9-129006"),
    pytest.param("[N-]1[NH+]=NC=C2N=CN=C12", id="qm9-129021"),
]


@pytest.mark.parametrize("smiles", CASES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

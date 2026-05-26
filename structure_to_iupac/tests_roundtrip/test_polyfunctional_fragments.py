"""Round-trip tests for test_polyfunctional_fragments.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "Cc1ccc(C(=O)N[C@H](C)CC(=O)NC(C)C)c(Br)c1",
    "NC(=O)Nc1ccc(CCC=O)cc1",
    "CCNC(=NCCc1cnn(C)c1)N[C@@H]1CCN(C2CCCC2)C1",
    "CC(=O)c1ccnc2c([C@H](C)CNc3cc(-c4cnc(C)nc4)ncn3)cccc12",
    "COc1ccc(Cl)c2sc(N3CCN(C(=O)c4nn(C)cc4C)CC3)nc12",
    "CN(C)CCNC(=O)c1ccnc(N2CCOCC2)n1",
    "COc1cc(OC)cc(C(=O)NC(C(=O)NCc2ccc(C)cc2C)C(C)C)c1",
    "CC(=O)c1ccnc2c(C(C)CNc3cc(-c4cnc(C)nc4)ncn3)cccc12",
    "CCNC(CN1CCC(C)(CC)CC1)C(C)C",
    "CCCOc1ccnc(NC(C)(CC)CCN)n1",
    "NC1CN(C(CO)c2ccccc2)C1",
    "COc1cccc(S(=O)(=O)NC(=O)[C@@H](C)NC(=O)c2ccco2)c1",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

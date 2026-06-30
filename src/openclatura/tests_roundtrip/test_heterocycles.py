"""Round-trip tests for test_heterocycles.py."""

import pytest

from openclatura.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "c1c[nH]cn1",
    "c1n[nH]cn1",
    "OC1=CC=CCN1",
    "N1=C=CCCC1",
    "C1CO1",
    "C1CN1",
    "C1COC1",
    "C1CNC1",
    "C1CCOC1",
    "C1CCNC1",
    "C1CCNCC1",
    "O1COCC1",
    "O1CCOCC1",
    "O1CCSCC1",
    "O1CCNCC1",
    "N1CCNCC1",
    "O1CCCOCC1",
    "c1ccoc1",
    "c1ccsc1",
    "c1cc[nH]c1",
    "c1ccncc1",
    "c1ccnnc1",
    "c1cncnc1",
    "c1cnccn1",
    "c1c[nH]nn1",
    "c1cocn1",
    "c1cnoc1",
    "c1cscn1",
    "c1cnsc1",
    "O1CCCCCCCCCCC1",
    "O1CCNCCCCCCCCC1",
    "c1cc[se]c1",
    "c1cc[te]c1",
    "N1NCCC1",
    "N1CNCC1",
    "O1CNCC1",
    "S1NCCC1",
    "S1CCNCC1",
    "[Se]1CCNCC1",
    "[Te]1CCNCC1",
    "Oc1ccncc1",
    "Nc1ccncc1",
    "Sc1ccncc1",
    "Cc1ccncc1",
    "Clc1ccncc1",
    "O=C(O)c1ccncc1",
    "N#Cc1ccncc1",
    "O=C1CO1",
    "O=C1NCCCCC1",
    "O=C1NC=CC=C1",
    "N1CC=CC=C1",
    "O1CC=CCC1",
    "N1C=CCCCC1",
    "N1CC=CCCCCCCC1",
    "N1CCCC=CCCCCCC1",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")

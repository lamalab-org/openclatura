import pytest

from bluenamer.tests_roundtrip.roundtrip_helpers import roundtrip_smiles


@pytest.mark.parametrize(
    "smiles",
    [
        "CP(C)(N=C=O)(N=C=O)N=C=O",
        "NNc1cn[nH]c1",
        "CN(C)Nc1cccc2ncccc12",
        "CN(C)C(=O)OP(=O)(O)O",
        "COC(CO)(COC(=O)O)OC",
        "NNCCOS(=O)(=O)O",
        "N=S1C=CC1",
        "N=[Se]1C=CC=C1",
        "N=S(Cl)CF",
        "[S-][n+]1ccc(-c2ccncc2)cc1",
        "[S-][n+]1ccc2ncccn21",
        "C[S+](C)[CH-]C",
    ],
)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

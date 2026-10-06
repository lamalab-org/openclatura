"""A carbamimidoyl prefix keeps the configuration of its own C=N bond."""

import pytest

from openclatura import name_smiles, opsin_available, verify_with_opsin

# The C=N of the prefix has no locant, so its descriptor is cited unlocanted in
# front of the prefix word, as the amidoxime parent already does
# (lamalab-org/openclatura#130).
CASES = {
    "COC(=O)c1ccc(cc1)/C(=N\\O)/N": "methyl 4-((E)-N'-hydroxycarbamimidoyl)benzoate",
    "COC(=O)c1ccc(cc1)/C(=N/O)/N": "methyl 4-((Z)-N'-hydroxycarbamimidoyl)benzoate",
    "N/C(=N/O)c1ccc(cc1)C(=O)O": "4-((E)-N'-hydroxycarbamimidoyl)benzoic acid",
    "COC(=O)c1ccc(cc1)/C(=N/C)/NC": "methyl 4-((Z)-N,N'-dimethylcarbamimidoyl)benzoate",
    "OC(=O)c1ccc(cc1)/C(=N/[H])/N": "4-((Z)-carbamimidoyl)benzoic acid",
}


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_carbamimidoyl_prefix_cites_its_c_n_configuration(smiles):
    assert name_smiles(smiles) == CASES[smiles]


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        # no configuration given: no descriptor
        ("COC(=O)c1ccc(cc1)C(=NO)N", "methyl 4-(N'-hydroxycarbamimidoyl)benzoate"),
        # the amidoxime as parent keeps its descriptor
        ("N/C(=N/O)c1ccccc1", "(E)-N'-hydroxybenzenecarboximidamide"),
    ],
)
def test_carbamimidoyl_guards(smiles, expected):
    assert name_smiles(smiles) == expected


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES))
def test_carbamimidoyl_stereo_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin(CASES[smiles], smiles)
    assert check.status == "matched", check.to_dict()

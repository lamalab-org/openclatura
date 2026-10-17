"""A heteroatom centre scrambled by proton exchange gets no R/S descriptor."""

import pytest

from openclatura import name_smiles, opsin_available, verify_with_opsin
from openclatura.graph_io import read_smiles

# RDKit keeps the chiral tag on an N-H+ or on the P of P(=O)(OH), but proton
# exchange inverts these centres, and OPSIN does not place a descriptor on them
# (lamalab-org/openclatura#132).  Each input is named as if the tag were absent;
# the second SMILES is that untagged structure, which OPSIN must rebuild.
CASES = {
    "C[P@](=O)(O)c1ccccc1": ("CP(=O)(O)c1ccccc1", "methyl(phenyl)phosphinic acid"),
    "C[P@@](=O)(O)CC": ("CP(=O)(O)CC", "ethyl(methyl)phosphinic acid"),
    "CC[N@H+]1CCC(=O)[C@@H]([C@@H]1[C@H](O)C)O": (
        "CC[NH+]1CCC(=O)[C@@H]([C@@H]1[C@H](O)C)O",
        "(2S,3R)-1-ethyl-3-hydroxy-2-((1R)-1-hydroxyethyl)piperidinium-4-one",
    ),
    "C[N@@H+]1CCC[C@H]1c1cccnc1": ("C[NH+]1CCC[C@H]1c1cccnc1", "3-((2S)-1-methylpyrrolidinium-2-yl)pyridine"),
    "C[N@H+]1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)c1ccccc1)C(=O)OC": (
        "C[NH+]1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)c1ccccc1)C(=O)OC",
        "methyl (1R,2R,3S,5S)-3-(benzoyloxy)-8-methyl-8-azoniabicyclo[3.2.1]octane-2-carboxylate",
    ),
    "C[N@H+](CC)c1ccccc1": ("C[NH+](CC)c1ccccc1", "N-ethyl-N-methylbenzenaminium"),
    "C[P@](=O)([O-])c1ccccc1": ("CP(=O)([O-])c1ccccc1", None),
    "C[P@](=S)(S)c1ccccc1": ("CP(=S)(S)c1ccccc1", None),
    "C[S@](=O)O": ("CS(=O)O", "methanesulfinic acid"),
    "C[N@@H+]1CC[C@H](C)CC1": ("C[NH+]1CCC(C)CC1", "1,4-dimethylpiperidinium"),
}

# Real heteroatom stereocentres keep their descriptor: no hydrogen on the
# centre, and no two terminal neighbours of the same element.
GUARDS = {
    "C[N@+]1(Cc2ccccc2)CCC[C@H]1C": "(1R,2R)-1-benzyl-1,2-dimethylpyrrolidinium",
    "C[S@](=O)c1ccccc1": "((S)-methylsulfinyl)benzene",
    "C[P@](=O)(CC)c1ccccc1": "((S)-ethyl(methyl)oxophosphanyl)benzene",
    "C[P@](=O)(OC)c1ccccc1": "((S)-methoxy(methyl)phosphoryl)benzene",
    "C[P@](=O)(S)c1ccccc1": "((S)-methyl(sulfanyl)oxophosphanyl)benzene",
    "C[P@+](CC)(CCC)c1ccccc1": "((S)-ethyl(methyl)(propyl)phosphaniumyl)benzene",
    "CC[N@+]1([O-])CCC[C@H]1C": "(1S,2R)-1-ethyl-2-methyl-1-oxidopyrrolidinium",
}


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_exchangeable_centre_is_named_as_untagged(smiles):
    untagged, expected = CASES[smiles]
    assert name_smiles(smiles) == name_smiles(untagged)
    if expected is not None:
        assert name_smiles(smiles) == expected


@pytest.mark.parametrize("smiles", ["C[N@H+]1CCC[C@H]1C", "C[N@+]1([2H])CCC[C@H]1C", "C[P@](=O)(O)c1ccccc1"])
def test_exchangeable_centre_carries_no_stereo(smiles):
    mol = read_smiles(smiles)
    centre = next(atom for atom in mol if atom.symbol in {"N", "P"})
    assert centre.stereo is None
    assert centre.raw_stereo is None


@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_heteroatom_stereocentre_guards(smiles):
    assert name_smiles(smiles) == GUARDS[smiles]


def test_aziridine_nitrogen_keeps_its_stereo():
    mol = read_smiles("C[N@]1C[C@H]1C")
    assert mol.atoms[1].stereo is not None


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES))
def test_exchangeable_centre_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    untagged, _ = CASES[smiles]
    check = verify_with_opsin(name_smiles(smiles), untagged)
    assert check.status == "matched", check.to_dict()


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_heteroatom_stereocentre_guards_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin(GUARDS[smiles], smiles)
    assert check.status == "matched", check.to_dict()

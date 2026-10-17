"""The R/S of a stereogenic quaternary nitrogen, -aminium or ammonio, is cited in the name."""

import pytest

from openclatura import name_smiles, opsin_available, verify_with_opsin

# A quaternary -aminium nitrogen with four different groups is a stereocentre,
# but it has only an N locant, so its descriptor was left out and both
# enantiomers got the same name (lamalab-org/openclatura#139).  It is cited
# unlocanted when it is the only descriptor, and at its N locant beside others.
CASES = {
    "C[N@+](CC)(CCC)c1ccccc1": "(S)-N-ethyl-N-methyl-N-propylbenzenaminium",
    "C[N@@+](CC)(CCC)c1ccccc1": "(R)-N-ethyl-N-methyl-N-propylbenzenaminium",
    "C[N@+](CC)(CCC)Cc1ccccc1": "(S)-N-benzyl-N-ethyl-N-methylpropan-1-aminium",
    "C[N@+](CC)(CCC)CC(=O)O": "(S)-N-ethyl-N-(2-hydroxy-2-oxoethyl)-N-methylpropan-1-aminium",
    "C[N@+](CC)([O-])c1ccccc1": "(R)-N-ethyl-N-methyl-N-oxidobenzenaminium",
    "C[N@+](CC)(CCC)C1CCCCC1": "(S)-N-ethyl-N-methyl-N-propylcyclohexanaminium",
    # A descriptor inside a substituent prefix leaves the nitrogen's unlocanted.
    "C[N@+](CC)(CCC)[C@@H](C)c1ccccc1": "(S)-N-ethyl-N-methyl-N-((1S)-1-phenylethyl)propan-1-aminium",
    # Beside parent-carbon descriptors the nitrogen's takes its N locant.
    "C[N@+](CC)(CCC)[C@@H](C)CC": "(NS,2S)-N-ethyl-N-methyl-N-propylbutan-2-aminium",
    "C[N@@+](CC)(CCC)[C@@H](C)CC": "(NR,2S)-N-ethyl-N-methyl-N-propylbutan-2-aminium",
    "C[N@+](CC)(CCC)CC[C@@H](C)O": "(NS,3R)-N-ethyl-3-hydroxy-N-methyl-N-propylbutan-1-aminium",
    "C[N@+](CC)(CCC)[C@H]1CCC[C@H](C)C1": "(NS,1S,3S)-N-ethyl-N,3-dimethyl-N-propylcyclohexanaminium",
    "C[N@@+](CC)(CCC)[C@@H]1CCC[C@H](C)C1": "(NR,1R,3S)-N-ethyl-N,3-dimethyl-N-propylcyclohexanaminium",
    # Two aminium nitrogens are told apart by N and N'.
    "C[N@+](CC)(CCC)CCCC[N@@+](C)(CC)CC(C)C": (
        "(NS,N'R)-N,N'-diethyl-N,N'-dimethyl-N'-(2-methylpropyl)-N-propylbutane-1,4-diaminium"
    ),
    "C[N@+](CC)(CCC)c1ccc(cc1)[N+](C)(C)C": "(S)-N-ethyl-N,N',N',N'-tetramethyl-N-propylbenzene-1,4-diaminium",
    # Cited as an ammonio prefix, the nitrogen's descriptor goes inside the prefix.
    "C[N@+](CC)(CCC)c1ccc(cc1)S(=O)(=O)[O-]": "4-((S)-ethyl(methyl)(propyl)ammonio)benzene-1-sulfonate",
    "C[N@+](CC)(CCC)CC(=O)[O-]": "2-((S)-ethyl(methyl)(propyl)ammonio)acetate",
    "C[N@@+](CC)(CCC)CC(=O)[O-]": "2-((R)-ethyl(methyl)(propyl)ammonio)acetate",
    "C[N@+](CC)(CCC)C[C@@H](C)C(=O)[O-]": "(2R)-3-((S)-ethyl(methyl)(propyl)ammonio)-2-methylpropanoate",
    "C[N@+](CC)(CCC)CC1=CC=CC=C1C[N@@+](C)(CC)CCC": (
        "(S)-N-ethyl-N-((2-(((S)-ethyl(methyl)(propyl)ammonio)methyl)phenyl)methyl)-N-methylpropan-1-aminium"
    ),
}

# Stereo that was already cited, and nitrogens that are not stereocentres.
GUARDS = {
    "C[N@+]1(Cc2ccccc2)CCC[C@H]1C": "(1R,2R)-1-benzyl-1,2-dimethylpyrrolidinium",
    "C[S@+](CC)c1ccccc1": "((R)-ethyl(methyl)sulfaniumyl)benzene",
    "C[P@+](CC)(CCC)c1ccccc1": "((S)-ethyl(methyl)(propyl)phosphaniumyl)benzene",
    "C[N@+](C)(CC)c1ccccc1": "N-ethyl-N,N-dimethylbenzenaminium",
    "C[N+](C)(C)c1ccccc1": "N,N,N-trimethylbenzenaminium",
    "C[C@@H](CC)[N+](C)(C)C": "(2S)-N,N,N-trimethylbutan-2-aminium",
    "C[N+](C)(C)c1ccc(cc1)S(=O)(=O)[O-]": "4-(trimethylammonio)benzene-1-sulfonate",
}


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_aminium_nitrogen_configuration_is_cited(smiles):
    assert name_smiles(smiles) == CASES[smiles]


@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_aminium_stereo_guards(smiles):
    assert name_smiles(smiles) == GUARDS[smiles]


def test_protonated_amine_nitrogen_takes_no_descriptor():
    # An N-H aminium nitrogen is not cited here, whatever RDKit perceives for it.
    assert name_smiles("C[N@H+](CC)c1ccccc1") == "N-ethyl-N-methylbenzenaminium"
    assert name_smiles("C[N@H+](CC)c1ccc(cc1)S(=O)(=O)[O-]") == "4-(ethyl(methyl)ammonio)benzene-1-sulfonate"


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES | GUARDS))
def test_aminium_stereo_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin((CASES | GUARDS)[smiles], smiles)
    assert check.status == "matched", check.to_dict()

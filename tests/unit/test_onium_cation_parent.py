"""An acyclic S, P, O, Se or Te onium cation is named on the cationic parent hydride."""

import pytest

from openclatura import name_smiles, opsin_available, verify_with_opsin

# Cations are senior to acids, ketones and hydroxy compounds (P-41, P-70.2), so the
# onium centre belongs in the parent, as the -aminium nitrogen already is.  It was
# cited as a "...iumyl" prefix instead, and beside an anion "...phenol" then ran into
# "...sulfonate" (lamalab-org/openclatura#135).
CASES = {
    "Oc1ccc(cc1)[S+](C)C": "(4-hydroxyphenyl)dimethylsulfanium",
    "Oc1ccc(cc1)[S+](C)C.Cc1ccc(cc1)S(=O)(=O)[O-]": "(4-hydroxyphenyl)dimethylsulfanium 4-methylbenzene-1-sulfonate",
    "OC(=O)CC[S+](C)C": "(3-hydroxy-3-oxopropyl)dimethylsulfanium",
    "C[S+](C)CC(=O)O.[Cl-]": "(2-hydroxy-2-oxoethyl)dimethylsulfanium chloride",
    "OCC[P+](C)(C)C": "(2-hydroxyethyl)trimethylphosphanium",
    "OCC[P+](C)(C)C.[Cl-]": "(2-hydroxyethyl)trimethylphosphanium chloride",
    "CC(=O)c1ccc(cc1)[S+](C)C": "(4-acetylphenyl)dimethylsulfanium",
    "N#Cc1ccc(cc1)[S+](C)C": "(4-cyanophenyl)dimethylsulfanium",
    "C[S+](C)c1ccc(O)c(O)c1": "(3,4-dihydroxyphenyl)dimethylsulfanium",
    "OCC[S+](CCO)CCO": "tris(2-hydroxyethyl)sulfanium",
    "C[S+](C)C[C@H](C)O": "((2S)-2-hydroxypropyl)dimethylsulfanium",
    "C[S+](C)CC[C@H](N)C(=O)O": "((3S)-3-amino-4-hydroxy-4-oxobutyl)dimethylsulfanium",
    "NC(=O)C[S+](C)C": "(2-amino-2-oxoethyl)dimethylsulfanium",
    "COC(=O)C[S+](C)C.[Br-]": "(2-methoxy-2-oxoethyl)dimethylsulfanium bromide",
    "CC(=O)[S+](C)C": "acetyldimethylsulfanium",
    "c1ccccc1C(=O)[S+](C)C": "benzoyldimethylsulfanium",
    "c1ccccc1[P+](c1ccccc1)(c1ccccc1)CC(=O)OCC.[Br-]": "(2-ethoxy-2-oxoethyl)triphenylphosphanium bromide",
    "c1ccccc1[P+](c1ccccc1)(c1ccccc1)CC(=O)c1ccccc1.[Br-]": "(2-oxo-2-phenylethyl)triphenylphosphanium bromide",
    "OC(=O)CCCC[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]": "(5-hydroxy-5-oxopentyl)triphenylphosphanium bromide",
    "OCC[O+](C)C": "(2-hydroxyethyl)dimethyloxidanium",
    # Without a principal group the centre was a prefix on a hydrocarbon as well.
    "C[S+](C)C": "trimethylsulfanium",
    "C[S+](C)c1ccccc1": "dimethyl(phenyl)sulfanium",
    "C[S+](C)CC=C": "dimethyl(prop-2-en-1-yl)sulfanium",
    "C[O+](C)C": "trimethyloxidanium",
    "CC[O+](CC)CC": "triethyloxidanium",
    "C[Se+](C)C": "trimethylselanium",
    "C[Te+](C)C": "trimethyltellanium",
    # A hydrogen on the centre is implied by the parent hydride, as in -aminium.
    "C[SH+]C": "dimethylsulfanium",
    "C[PH+](C)C": "trimethylphosphanium",
    "CC[OH2+]": "ethyloxidanium",
    # A stereogenic centre is cited before the parent.
    "C[S@+](CC)c1ccccc1": "(R)-ethyl(methyl)(phenyl)sulfanium",
    "C[P@+](CC)(CCC)c1ccccc1": "(S)-ethyl(methyl)(phenyl)(propyl)phosphanium",
}

# Names that do not change: aminium cations, zwitterions (the anion is senior),
# ring onium atoms, two cationic centres, ylides and neutral S/P/Si centres.
GUARDS = {
    "OCC[N+](C)(C)C": "2-hydroxy-N,N,N-trimethylethan-1-aminium",
    "C[S+](C)CCC(=O)[O-]": "3-(dimethylsulfaniumyl)propanoate",
    "[BH3-][P+](C)(C)C": "(trimethylphosphaniumyl)boranuide",
    "C[S+](C)[CH-]C": "1-(dimethylsulfaniumyl)ethan-1-ide",
    "C[S+]1CCCC1.[I-]": "1-methylthiolan-1-ium iodide",
    "C[S+](C)CCCCCC[N+](C)(C)C": "6-(dimethylsulfaniumyl)-N,N,N-trimethylhexan-1-aminium",
    "c1ccccc1[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]": "tetraphenylphosphanium bromide",
    "C(Cl)[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Cl-]": "(chloromethyl)triphenylphosphanium chloride",
    "C[As+](C)(C)C": "tetramethylarsanium",
    "CSC": "dimethylsulfane",
    "CP(C)C": "trimethylphosphane",
    "Oc1ccc(cc1)P(c1ccccc1)c1ccccc1": "4-(diphenylphosphanyl)phenol",
    "OCC[Si](C)(C)C": "2-(trimethylsilyl)ethan-1-ol",
}


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_onium_cation_is_the_parent(smiles):
    assert name_smiles(smiles) == CASES[smiles]


@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_onium_cation_parent_guards(smiles):
    assert name_smiles(smiles) == GUARDS[smiles]


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES | GUARDS))
def test_onium_cation_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin((CASES | GUARDS)[smiles], smiles)
    assert check.status == "matched", check.to_dict()

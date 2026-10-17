"""Double-bond E/Z is named by the modern CIP labeler, as atom R/S already is."""

import pytest

from openclatura import name_smiles, opsin_available, verify_with_opsin

# RDKit's legacy perception ranks the substituents on the Kekule form, where it
# can put a heteroaryl group such as 1H-benzimidazol-2-yl below a nitrile or an
# alkynyl group, and so reports the opposite configuration
# (lamalab-org/openclatura#137).
CASES = {
    "C1=CC=C2C(=C1)NC(=N2)/C(=C/C3=CSC=C3)/C#N": "(2E)-2-(1H-benzimidazol-2-yl)-3-(thiophen-3-yl)prop-2-enenitrile",
    "N#C/C(=C\\c1ccccc1)c1ncc[nH]1": "(2E)-2-(1H-imidazol-2-yl)-3-phenylprop-2-enenitrile",
    "N#C/C(=C/c1ccccc1)c1ncc[nH]1": "(2Z)-2-(1H-imidazol-2-yl)-3-phenylprop-2-enenitrile",
    "C/C(=C(/C#N)\\C1=NC2=CC=CC=C2N1)/O": "(2E)-2-(1H-benzimidazol-2-yl)-3-hydroxybut-2-enenitrile",
    "C1CC2=C3C1=CC=CC3=CC(=C2)/C=C(/C#N)\\C4=NC5=CC=CC=C5N4": (
        "(2Z)-2-(1H-benzimidazol-2-yl)-3-(1,2-dihydroacenaphthylen-4-yl)prop-2-enenitrile"
    ),
    "CC\\1=C(C2=CC=CC=C2/C1=C\\C3=CC=C(C=C3)Br)CC(=O)O": (
        "2-((1Z)-1-((4-bromophenyl)methylidene)-2-methyl-1H-inden-3-yl)acetic acid"
    ),
    "CC(=CCC/C(=C/COC/C=C(\\C#C)/C1=CC=CC=C1)/C)C": (
        "((3E)-5-(((2E)-3,7-dimethylocta-2,6-dien-1-yl)oxy)pent-3-en-1-yn-3-yl)benzene"
    ),
    "C1=CC(/C(=N\\O)/C=C1)CCC2C3=CC=CN3C4=C(C=CS4)C(=O)N2": (
        "6-(2-((6Z)-6-(hydroxyimino)cyclohexa-2,4-dien-1-yl)ethyl)"
        "-5H-pyrrolo[1,2-a]thieno[3,2-f][1,4]diazepin-4(6H)-one"
    ),
}

# Bonds where the two perceptions already agree keep their descriptor.
GUARDS = {
    "N#C/C(=C\\c1ccccc1)c1nccs1": "(2E)-3-phenyl-2-(1,3-thiazol-2-yl)prop-2-enenitrile",
    "C/C=C/C(=O)O": "(2E)-but-2-enoic acid",
    "C/C(=N/O)c1ccccc1": "(1Z)-N-hydroxy-1-phenylethan-1-imine",
    "F/C=C/Cl": "(1E)-1-chloro-2-fluoroethene",
}


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_bond_configuration_comes_from_modern_cip(smiles):
    assert name_smiles(smiles) == CASES[smiles]


@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_bond_configuration_guards(smiles):
    assert name_smiles(smiles) == GUARDS[smiles]


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES | GUARDS))
def test_bond_configuration_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin((CASES | GUARDS)[smiles], smiles)
    assert check.status == "matched", check.to_dict()

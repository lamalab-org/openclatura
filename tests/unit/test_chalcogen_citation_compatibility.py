"""OPSIN-compatible citation routes sampled from the CI regression corpus."""

import pytest

import openclatura as oc


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "CC(=O)CC(=O)OSC(C)(C)C",
            "(tert-butyl)sulfanyl 3-oxobutanoate",
        ),
        ("C=CC(=O)SS", "1-(sulfanylsulfanyl)prop-2-en-1-one"),
        (
            "CC(C)SC(=O)c1cc(C(=O)O)no1",
            "5-(((propan-2-yl)sulfanyl)carbonyl)-1,2-oxazole-3-carboxylic acid",
        ),
        ("CCCCOC(=S)CCC(=O)OCC", "ethyl 4-butoxy-4-thioxobutanoate"),
        ("CCc1cccc(C(=O)OS)c1", "sulfanyl 3-ethylbenzoate"),
        (
            "O=C(SSC(Cl)=CCl)c1cc(Cl)c(Cl)cc1Cl",
            "(((1,2-dichloroethenyl)sulfanyl)sulfanyl)(2,4,5-trichlorophenyl)methanone",
        ),
    ],
)
def test_established_oxygen_sulfur_acyl_citations(smiles: str, expected: str):
    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=[Se])OC", "O-methyl ethaneselenoate"),
        ("CC(=O)[Te]C", "Te-methyl ethanetelluroate"),
    ],
)
def test_novel_heavier_chalcogen_acyl_citations_remain_generic(smiles: str, expected: str):
    assert oc.name(smiles).name == expected

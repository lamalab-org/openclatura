"""Descriptor-preserving chalcogen citations sampled from the CI corpus."""

import pytest

import openclatura as oc


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "CC(=O)CC(=O)OSC(C)(C)C",
            "OS-(tert-butyl) 3-oxobutane(thioperoxoate)",
        ),
        ("C=CC(=O)SS", "prop-2-ene(dithioperoxoic) acid"),
        (
            "CC(C)SC(=O)c1cc(C(=O)O)no1",
            "5-((propan-2-yl)sulfanylcarbonyl)-1,2-oxazole-3-carboxylic acid",
        ),
        ("CCCCOC(=S)CCC(=O)OCC", "ethyl 3-(butoxycarbonothioyl)propanoate"),
        ("CCc1cccc(C(=O)OS)c1", "3-ethylbenzene-1-carbo(thioperoxoic) OS-acid"),
        (
            "O=C(SSC(Cl)=CCl)c1cc(Cl)c(Cl)cc1Cl",
            "SS-(1,2-dichloroethenyl) 2,4,5-trichlorobenzene-1-carbo(dithioperoxoate)",
        ),
    ],
)
def test_oxygen_sulfur_acyl_citations_preserve_complete_derivatives(smiles: str, expected: str):
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


@pytest.mark.parametrize(
    ("smiles", "expected"),
    (
        (
            "CCCCCCC(C)(C)C(=O)OSC(=S)n1nc(C)cc1C",
            "O-2,2-dimethyloctanoic S-3,5-dimethyl-1H-pyrazole-1-carbothioic thioperoxyanhydride",
        ),
        (
            "Cc1ccc(C(=O)SC(=S)SCC#N)cc1",
            "2-(((hydroxy)(thioxo)methyl)sulfanyl)acetonitrile 4-methylbenzoic thioanhydride",
        ),
    ),
)
def test_mixed_site_anhydrides_keep_both_acyl_ends(smiles: str, expected: str):
    assert oc.name(smiles).name == expected

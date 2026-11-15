"""Metallocene parents: metal(2+) plus two cyclopentadienide rings named as one parent."""

import pytest

from openclatura import NamingEngine, NamingRequest, name_smiles
from openclatura.resonance_compare import equivalent_smiles


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]", "ferrocene"),
        ("[cH-]1cccc1.[cH-]1cccc1.[Fe+2]", "ferrocene"),
        ("[cH-]1cccc1.[cH-]1cccc1.[Ru+2]", "ruthenocene"),
        ("[cH-]1cccc1.[cH-]1cccc1.[Os+2]", "osmocene"),
        ("[cH-]1cccc1.[cH-]1cccc1.[Co+2]", "cobaltocene"),
        ("[cH-]1cccc1.[cH-]1cccc1.[Ni+2]", "nickelocene"),
        ("[cH-]1cccc1.[cH-]1cccc1.[Cr+2]", "chromocene"),
        ("[cH-]1cccc1.[cH-]1cccc1.[V+2]", "vanadocene"),
        ("[cH-]1cccc1.[cH-]1cccc1.[Ti+2]", "titanocene"),
    ],
)
def test_unsubstituted_metallocene_parents(smiles, expected):
    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        # dppf: one substituent on each ring takes 1 and 1'.
        (
            "c1ccc(cc1)P(c1ccccc1)[c-]1cccc1.c1ccc(cc1)P(c1ccccc1)[c-]1cccc1.[Fe+2]",
            "1,1'-bis(diphenylphosphanyl)ferrocene",
        ),
        # A lone substituent needs no locant.
        ("CC(=O)[c-]1cccc1.[cH-]1cccc1.[Fe+2]", "acetylferrocene"),
        ("[c-]1(cccc1)c1ncccc1.[cH-]1cccc1.[Fe+2]", "(pyridin-2-yl)ferrocene"),
        # Same ring: 1,2 and 1,3 are different compounds.
        ("C[c-]1c(C)ccc1.[cH-]1cccc1.[Fe+2]", "1,2-dimethylferrocene"),
        ("C[c-]1cc(C)cc1.[cH-]1cccc1.[Fe+2]", "1,3-dimethylferrocene"),
        # The more substituted ring is unprimed: {1,1',2} is lower than {1,1',2'}.
        ("C[c-]1cccc1.C[c-]1c(C)ccc1.[Fe+2]", "1,1',2-trimethylferrocene"),
        # Equal locant sets: the first-cited prefix takes the unprimed ring.
        ("Br[c-]1cccc1.CC(=O)[c-]1cccc1.[Fe+2]", "1-acetyl-1'-bromoferrocene"),
        ("OC[c-]1cccc1.OC[c-]1cccc1.[Ru+2]", "1,1'-bis(hydroxymethyl)ruthenocene"),
    ],
)
def test_substituted_metallocenes(smiles, expected):
    assert name_smiles(smiles) == expected


def test_metallocene_is_named_through_the_analysis_path():
    result = NamingEngine().run(NamingRequest(smiles="C[c-]1cccc1.C[c-]1cccc1.[Fe+2]", include_trace=True))
    assert result.error is None
    assert result.name == "1,1'-dimethylferrocene"


def test_other_components_are_still_named():
    assert name_smiles("[cH-]1cccc1.[cH-]1cccc1.[Fe+2].CO") == "ferrocene methanol"


def test_a_ring_fused_to_the_cyclopentadienide_is_not_a_metallocene():
    result = NamingEngine().run(NamingRequest(smiles="C1C[c-]2cccc2C1.[cH-]1cccc1.[Fe+2]"))
    assert result.name != "ferrocene"
    assert "ferrocene" not in result.name


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # The ring charge may be drawn on any of the five carbons.
        ("O=C[c-]1cccc1", "O=CC1=CC=C[CH-]1", True),
        ("Clc1c(Br)c[c-](C=O)c1", "Br[C-]1C(=CC(=C1)C=O)Cl", True),
        ("C[c-]1cc(C)cc1", "Cc1cc(C)c[cH-]1", True),
        # ...but the substitution pattern and the charge itself still count.
        ("C[c-]1cc(C)cc1", "C[c-]1c(C)ccc1", False),
        ("[CH-]1C=CC=C1.[Na+]", "C1=CC=CC1.[Na+]", False),
        ("c1ccc2[cH-]ccc2c1", "c1ccc2c[cH-]cc2c1", False),
    ],
)
def test_cyclopentadienide_charge_position_is_resonance(left, right, expected):
    assert equivalent_smiles(left, right) is expected
    assert equivalent_smiles(right, left) is expected

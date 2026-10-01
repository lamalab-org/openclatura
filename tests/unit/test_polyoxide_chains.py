"""Oxygen forms parent-hydride chains like the other heteroatoms (P-21.1.1)."""

import pytest

from openclatura import name_smiles

UNBRANCHED = {
    2: "hydrogen peroxide",  # the retained name wins over dioxidane
    3: "trioxidane",
    4: "tetraoxidane",
    5: "pentaoxidane",
    6: "hexaoxidane",
    7: "heptaoxidane",
    8: "octaoxidane",
    10: "decaoxidane",
}


@pytest.mark.parametrize("length,expected", sorted(UNBRANCHED.items()))
def test_unbranched_oxygen_chains_name_as_oxidanes(length, expected):
    assert name_smiles("O" * length) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("COOC", "1,2-dimethyldioxidane"),
        ("COOOC", "1,3-dimethyltrioxidane"),
        ("COOOOC", "1,4-dimethyltetraoxidane"),
        ("COOOOOC", "1,5-dimethylpentaoxidane"),
        ("CC(C)OOC(C)C", "1,2-bis(propan-2-yl)dioxidane"),
        ("c1ccccc1OOOc1ccccc1", "1,3-diphenyltrioxidane"),
        ("COOO", "1-methyltrioxidane"),
    ],
)
def test_substituted_oxygen_chains_use_the_oxidane_parent(smiles, expected):
    """These used to be spelled as peroxy prefixes on carbon instead."""

    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # a lone oxygen is an ether or an alcohol, never a one-atom chain
        ("COC", "methoxymethane"),
        ("CCOCC", "ethoxyethane"),
        ("CCO", "ethanol"),
        ("OCCO", "ethane-1,2-diol"),
        ("c1ccccc1O", "phenol"),
        ("CC(=O)O", "acetic acid"),
        ("COCCOC", "1,2-dimethoxyethane"),
    ],
)
def test_lone_oxygen_atoms_are_untouched(smiles, expected):
    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # A candidate element must carry a chain of its own: a methoxy oxygen
        # beside a hydrazine must not cost the molecule its N-N parent.
        ("NNCCOC", "1-(2-methoxyethyl)hydrazine"),
        ("SSCCOC", "1-(2-methoxyethyl)disulfane"),
        ("CC(OC)NN", "1-(1-methoxyethyl)hydrazine"),
        # When two elements each carry a chain, the senior one is the parent
        # (P-41): nitrogen outranks oxygen.
        ("OONN", "1-hydroperoxyhydrazine"),
        ("NNOO", "1-hydroperoxyhydrazine"),
        ("OOCCNN", "1-(2-hydroperoxyethyl)hydrazine"),
    ],
)
def test_oxygen_chains_do_not_displace_a_senior_heteroatom_chain(smiles, expected):
    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [("SSS", "trisulfane"), ("SSSS", "tetrasulfane"), ("NNN", "triazane"), ("P" * 25, "pentacosaphosphane")],
)
def test_the_other_heteroatom_chains_are_unchanged(smiles, expected):
    assert name_smiles(smiles) == expected

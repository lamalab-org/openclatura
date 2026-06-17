"""Tests for the human-oriented metadata descriptor."""

from __future__ import annotations

from bluenamer import HumanDescription, describe_human


def test_human_descriptor_uses_parent_metadata_without_token_spans():
    d = describe_human("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    assert isinstance(d, HumanDescription)
    text = str(d)
    assert "9-membered bicyclic [4.3.0] heteroskeleton" in text
    assert "nitrogen at positions 2, 4, 7, and 9" in text
    assert "double bond between positions 1 and 6" in text
    assert "double bond between positions 8 and 9" in text
    assert "oxo groups at positions 3 and 5" in text
    assert "methyl groups at positions 2, 4, and 7" in text
    assert "token" not in text.lower()
    assert "span" not in text.lower()


def test_human_descriptor_recurses_into_substituent_parents():
    d = describe_human("CC(=O)Nc1ccccc1")

    text = str(d)
    assert "N-phenylacetamide" in text
    assert "an amide group at position 1" in text
    assert "a phenyl group at position N" in text
    assert "phenyl substituent at position N is built around the retained benzene parent" in text


def test_human_descriptor_handles_nested_substituent_trees_generically():
    d = describe_human("O=S(=O)(Nc1nc(cc(n1)C)C)c2ccc(N)cc2")

    text = str(d)
    assert "N-(4-aminophenylsulfonyl)-4,6-dimethylpyrimidin-2-amine" in text
    assert "4-aminophenylsulfonyl" in text
    assert "4-aminophenyl" in text
    assert "retained benzene parent" in text
    assert "amino group" in text

"""Tests for the human-oriented metadata descriptor."""

from __future__ import annotations

from openclatura import HumanDescription, describe_human


def test_human_descriptor_uses_parent_metadata_without_token_spans():
    d = describe_human("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    assert isinstance(d, HumanDescription)
    text = str(d)
    assert "9-membered bicyclic [4.3.0] heteroskeleton" in text
    assert "nitrogen at positions 2 (atom id" in text
    assert "4 (atom id" in text
    assert "7 (atom id" in text
    assert "9 (atom id" in text
    assert "double bond between position 1 (atom id" in text
    assert "and position 6 (atom id" in text
    assert "double bond between position 8 (atom id" in text
    assert "and position 9 (atom id" in text
    assert "oxo groups at positions 3 (atom id" in text
    assert "5 (atom id" in text
    assert "methyl groups at positions 2 (atom id" in text
    assert "token" not in text.lower()
    assert "span" not in text.lower()


def test_human_descriptor_starts_with_processed_smiles_atom_ids():
    d = describe_human("C[C@@H](Cl)C(=O)c1ccccc1")

    first = d.paragraphs[0]
    assert first.startswith("Processed SMILES: C[C@@H](Cl)C(=O)c1ccccc1\n")
    assert "C{0}[C@@H]{1}(Cl{2})C{3}(=O{4})c{5}1c{6}c{7}c{8}c{9}c{10}1" in first


def test_human_descriptor_recurses_into_substituent_parents():
    d = describe_human("CC(=O)Nc1ccccc1")

    text = str(d)
    assert "N-phenylacetamide" in text
    assert "an amide group at position 1 (atom id" in text
    assert "a phenyl group at position N" in text
    assert "phenyl substituent at position N is built around the retained benzene parent" in text
    assert "\nThe principal characteristic feature is an amide group" in text


def test_human_descriptor_uses_local_substituent_names_and_atom_ids():
    d = describe_human("CC(=O)c1cccc(Nc2ccccc2CC)c1")

    text = str(d)
    assert "1-(3-((2-ethylphenyl)amino)phenyl)ethan-1-one" in text
    assert "a phenyl group at position 1 (atom id" in text
    assert "3-((2-ethylphenyl)amino)phenyl group" not in text
    assert "an amino group at position 3 (atom id" in text
    assert "a phenyl group." in text
    assert "a ethyl group" not in text
    assert "an ethyl group" in text


def test_human_descriptor_handles_nested_substituent_trees_generically():
    d = describe_human("O=S(=O)(Nc1nc(cc(n1)C)C)c2ccc(N)cc2")

    text = str(d)
    assert "N-(4-aminophenylsulfonyl)-4,6-dimethylpyrimidin-2-amine" in text
    assert "4-aminophenylsulfonyl" in text
    assert "4-aminophenyl" in text
    assert "retained benzene parent" in text
    assert "amino group" in text

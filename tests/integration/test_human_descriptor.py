"""Tests for the human-oriented metadata descriptor."""

from __future__ import annotations

from openclatura import HumanDescription, describe_human, name


def test_human_descriptor_uses_parent_metadata_without_token_spans():
    d = describe_human("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    assert isinstance(d, HumanDescription)
    text = str(d)
    assert "9-membered bicyclic [4.3.0] heteroskeleton" in text
    assert "retained purine parent" in text
    assert "equivalent von Baeyer representation" in text
    assert "descriptor bicyclo[4.3.0]" in text
    assert "nitrogen at positions 1 (atom id" in text
    assert "3 (atom id" in text
    assert "7 (atom id" in text
    assert "9 (atom id" in text
    assert "oxo groups at positions 2 (atom id" in text
    assert "6 (atom id" in text
    assert "methyl groups at positions 1 (atom id" in text
    assert "token" not in text.lower()
    assert "span" not in text.lower()


def test_human_descriptor_adds_audited_von_baeyer_view_to_fusion_parent():
    d = describe_human("O1C=CC2=NC3=C(C=C21)SC=C3")

    text = str(d)
    assert d.name == "furo[3,2-b]thieno[2,3-e]pyridine"
    assert "12-membered polycyclic heteroskeleton" in text
    assert "descriptor tricyclo[7.3.0.0^{3,7}]" in text
    view = d.result.substituent_tree[0]["parent"]["von_baeyer_topology"]
    assert view["cycle_count"] == 3
    assert len(view["atom_ids_by_locant"]) == 12
    assert view["proof_source"] == "audited_von_baeyer_candidate"


def test_human_descriptor_adds_audited_von_baeyer_view_to_tetracyclic_retained_parent():
    d = describe_human("c1cc2ccc3cccc4ccc(c1)c2c34")

    assert d.name == "pyrene"
    assert "descriptor tetracyclo[6.6.2.0^{4,16}.0^{11,15}]" in str(d)
    assert d.result.substituent_tree[0]["parent"]["von_baeyer_topology"]["cycle_count"] == 4


def test_human_descriptor_adds_von_baeyer_view_to_nested_retained_polycycle():
    d = describe_human("CC(=O)c1ccc2c(c1)ccc1ccccc12")

    text = str(d)
    assert d.name == "1-(phenanthren-2-yl)ethan-1-one"
    assert "phenanthrene-derived substituent" in text
    assert "descriptor tricyclo[8.4.0.0^{2,7}]" in text


def test_human_descriptor_does_not_add_von_baeyer_view_to_nonpolycycle():
    d = describe_human("CC(=O)Nc1ccccc1")

    assert "equivalent von Baeyer representation" not in str(d)


def test_plain_naming_does_not_compute_description_only_topology(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("description-only topology entered ordinary naming")

    monkeypatch.setattr("openclatura.descriptive_topology.von_baeyer_topology_view", fail_if_called)

    assert name("c1ccc2c(c1)ccc1ccccc12").name == "phenanthrene"


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
    assert "4-amino-N-(4,6-dimethylpyrimidin-2-yl)benzene-1-sulfonamide" in text
    assert "sulfonamide group" in text
    assert "pyrimidine-derived" in text
    assert "retained benzene parent" in text
    assert "amino group" in text

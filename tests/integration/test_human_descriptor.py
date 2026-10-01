"""Tests for the human-oriented metadata descriptor."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from openclatura import HumanDescription, describe_human, name
from openclatura.graph_io import read_smiles
from openclatura.human_descriptor import _describe_node, _retained_fusion_framework_sentence
from openclatura.retained_fused_templates import _smallest_ring_basis


def test_human_descriptor_uses_parent_metadata_without_token_spans():
    d = describe_human("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    assert isinstance(d, HumanDescription)
    text = str(d)
    assert "9-atom polycyclic heteroskeleton" in text
    assert "retained purine parent" in text
    assert "von Baeyer" not in text
    assert "bicyclic [4.3.0]" not in text
    assert "5-membered ring of aromatic atoms" in text
    assert "6-membered ring of aromatic atoms" in text
    assert "share parent bonds between position 4 (atom id 4) and position 5 (atom id 5)" in text
    assert "nitrogen at positions 1 (atom id" in text
    assert "3 (atom id" in text
    assert "7 (atom id" in text
    assert "9 (atom id" in text
    assert "oxo groups at positions 2 (atom id" in text
    assert "6 (atom id" in text
    assert "methyl groups at positions 1 (atom id" in text
    assert "token" not in text.lower()
    assert "span" not in text.lower()


def test_human_descriptor_describes_fusion_parent_in_selected_numbering():
    d = describe_human("O1C=CC2=NC3=C(C=C21)SC=C3")

    text = str(d)
    assert d.name == "furo[3,2-b]thieno[2,3-e]pyridine"
    assert f"fusion {d.name} parent" in text
    assert "12-atom polycyclic heteroskeleton" in text
    assert "von Baeyer" not in text
    assert "tricyclo" not in text
    assert text.count("5-membered ring of aromatic atoms") == 2
    assert text.count("6-membered ring of aromatic atoms") == 1
    assert "position 3a (atom id 3) and position 8a (atom id 8)" in text
    assert "position 4a (atom id 5) and position 7a (atom id 6)" in text
    assert "nitrogen at position 4 (atom id 4)" in text
    assert "oxygen at position 1 (atom id 0)" in text
    assert "sulfur at position 7 (atom id 9)" in text
    assert "furan component 0" in text
    assert "thiophene component 1" in text
    assert "pyridine component 2 (fusion parent component)" in text
    assert (
        "The furan component 0 joins the pyridine component 2 at positions 3a (atom id 3) and 8a (atom id 8)." in text
    )


def test_human_descriptor_describes_tetracyclic_retained_parent():
    d = describe_human("c1cc2ccc3cccc4ccc(c1)c2c34")

    assert d.name == "pyrene"
    assert "retained pyrene parent, 16-atom polycyclic carbon skeleton" in str(d)
    assert str(d).count("6-membered ring of aromatic atoms") == 4
    assert "von Baeyer" not in str(d)
    assert "tetracyclo" not in str(d)


def test_human_descriptor_describes_nested_retained_polycycle():
    d = describe_human("CC(=O)c1ccc2c(c1)ccc1ccccc12")

    text = str(d)
    assert d.name == "1-(phenanthren-2-yl)ethan-1-one"
    assert "phenanthrene-derived substituent" in text
    assert "retained phenanthrene parent, 14-atom polycyclic carbon skeleton" in text
    assert text.count("6-membered ring of aromatic atoms") == 3
    assert "von Baeyer" not in text
    assert "tricyclo" not in text


@pytest.mark.parametrize("smiles", ["c1ccc2ccccc2c1", "O1C=CC2=NC3=C(C=C21)SC=C3"])
def test_parent_description_ignores_rendered_text_and_auxiliary_topology(smiles):
    d = describe_human(smiles)
    mol = read_smiles(smiles)
    node = deepcopy(d.result.substituent_tree[0])
    expected = _describe_node(node, mol, subject="The molecule", depth=0)
    node["name"] = "unrelated rendered name"
    node["trace_segments"] = [{"name_terms": ["unrelated"]}]
    node["name_token_spans"] = [{"text": "unrelated"}]
    node["parent"]["von_baeyer_topology"] = {"descriptor": "unrelated", "atom_ids_by_locant": {"99": 0}}
    assert _describe_node(node, mol, subject="The molecule", depth=0) == expected
    del node["parent"]["von_baeyer_topology"]
    assert _describe_node(node, mol, subject="The molecule", depth=0) == expected


def test_parent_ring_details_exclude_substituent_atoms_and_bonds():
    smiles = "CC(=O)Nc1ccccc1"
    d = describe_human(smiles)
    text = str(d)
    details = next(line for line in text.splitlines() if line.startswith("The parent ring basis"))
    assert details.count("6-membered ring of aromatic atoms") == 1
    for atom in range(4):
        assert f"(atom id {atom})" not in details
    assert "share parent bonds" not in text


def test_parent_ring_details_handle_missing_metadata():
    mol = read_smiles("c1ccccc1")
    assert _retained_fusion_framework_sentence({}, mol) == ""
    assert _retained_fusion_framework_sentence({"retained_name": "benzene"}, mol) == ""


def test_parent_ring_details_follow_graph_and_locant_metadata_not_parent_label():
    mol = read_smiles("C1CCCCC1")
    parent = {
        "parent_nomenclature": "systematic_fusion",
        "parent_hydride_name": "arbitrary label",
        "atoms": list(mol.atoms),
        "bonds": list(mol.bonds),
        "atom_ids_by_locant": {"3a": 0, "4": 1, "4a": 2, "5": 3, "6": 4, "7": 5},
    }
    text = _retained_fusion_framework_sentence(parent, mol)
    assert "6-membered ring containing position 3a (atom id 0), position 4 (atom id 1), position 4a (atom id 2)" in text
    assert "aromatic" not in text
    assert "share parent bonds" not in text
    parent["atom_ids_by_locant"] = {}
    assert "ring containing atom id 0" in _retained_fusion_framework_sentence(parent, mol)
    parent["bonds"] = parent["bonds"][:-1]
    assert _retained_fusion_framework_sentence(parent, mol) == ""


def test_parent_ring_details_reuse_existing_basis_without_rdkit_reconstruction(monkeypatch):
    smiles = "c1ccc2ccccc2c1"
    parent = describe_human(smiles).result.substituent_tree[0]["parent"]
    mol = read_smiles(smiles)
    calls = []

    def record_basis(atoms, edges, **kwargs):
        calls.append((atoms, edges))
        return _smallest_ring_basis(atoms, edges, **kwargs)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("human ring details must not rebuild an RDKit graph")

    monkeypatch.setattr("openclatura.human_descriptor._smallest_ring_basis", record_basis)
    monkeypatch.setattr("openclatura.human_descriptor.Chem.RWMol", fail_if_called)
    monkeypatch.setattr("openclatura.human_descriptor.Chem.GetSSSR", fail_if_called)
    text = _retained_fusion_framework_sentence(parent, mol)

    assert text.count("6-membered ring of aromatic atoms") == 2
    assert len(calls) == 1
    assert {int(atom) for atom in calls[0][0]} == set(parent["atoms"])
    assert len(calls[0][1]) == len(parent["bonds"])
    shared = text.split("These rings share parent bonds between ")[1]
    assert shared.count("(atom id") == 2


@pytest.mark.parametrize(
    ("smiles", "ring_size", "count"),
    [("C1CCCCCCCC1", 9, 1), ("C1CCCCC1CC2CCCCC2", 6, 2)],
)
def test_parent_ring_details_handle_large_monocycles_and_separate_blocks(smiles, ring_size, count):
    mol = read_smiles(smiles)
    parent = {"retained_name": "arbitrary label", "atoms": list(mol.atoms), "bonds": list(mol.bonds)}
    text = _retained_fusion_framework_sentence(parent, mol)
    assert text.count(f"{ring_size}-membered ring containing") == count
    assert "share parent bonds" not in text


def test_parent_ring_details_do_not_render_an_incomplete_basis(monkeypatch):
    mol = read_smiles("C1CCCCC1")
    parent = {"retained_name": "arbitrary label", "atoms": list(mol.atoms), "bonds": list(mol.bonds)}

    def incomplete_basis(*args, **kwargs):
        raise ValueError("Could not construct a complete bounded ring basis for retained fused graph.")

    monkeypatch.setattr("openclatura.human_descriptor._smallest_ring_basis", incomplete_basis)
    assert _retained_fusion_framework_sentence(parent, mol) == ""


@pytest.mark.parametrize(
    "smiles",
    [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "c1cc2ccc3cccc4ccc(c1)c2c34",
        "CC(=O)c1ccc2c(c1)ccc1ccccc12",
        "O1C=CC2=NC3=C(C=C21)SC=C3",
    ],
)
def test_retained_and_fusion_descriptions_do_not_compute_auxiliary_von_baeyer(smiles, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("retained/fusion description must not compute auxiliary von Baeyer topology")

    monkeypatch.setattr("openclatura.descriptive_topology.von_baeyer_topology_view", fail_if_called)
    d = describe_human(smiles)
    assert d.name
    assert "von_baeyer_topology" not in json.dumps(d.result.substituent_tree)


def test_selected_fusion_metadata_reuses_faces_and_preserves_graph_ids(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("selected fusion faces must not trigger another ring-basis search")

    monkeypatch.setattr("openclatura.human_descriptor._smallest_ring_basis", fail_if_called)
    smiles = "O1C=CC2=NC3=C(C=C21)SC=C3"
    d = describe_human(smiles)
    parent = d.result.substituent_tree[0]["parent"]
    selected = parent["selected_fusion"]
    assert json.loads(json.dumps(selected)) == selected
    assert selected["proof_source"] == "selected_fusion_plan"
    assert {atom for face in selected["faces"] for atom in face["atoms"]} == set(parent["atoms"])
    assert {bond for face in selected["faces"] for bond in face["bonds"]} == set(parent["bonds"])
    components = {item["occurrence_id"]: item for item in selected["components"]}
    mol = read_smiles(smiles)
    for join in selected["joins"]:
        attached_atoms = set(components[join["attached_occurrence"]]["atom_ids_by_locant"].values())
        host_atoms = set(components[join["host_occurrence"]]["atom_ids_by_locant"].values())
        assert set(join["atoms"]) == attached_atoms & host_atoms
        assert all({mol.bonds[bond].u, mol.bonds[bond].v} <= set(join["atoms"]) for bond in join["bonds"])


def test_incomplete_selected_faces_fall_back_to_complete_parent_basis():
    smiles = "O1C=CC2=NC3=C(C=C21)SC=C3"
    parent = deepcopy(describe_human(smiles).result.substituent_tree[0]["parent"])
    parent["selected_fusion"]["faces"] = parent["selected_fusion"]["faces"][:1]
    text = _retained_fusion_framework_sentence(parent, read_smiles(smiles))
    assert text.count("5-membered ring") == 2
    assert text.count("6-membered ring") == 1


def test_systematic_bicyclic_parent_keeps_its_selected_descriptor():
    d = describe_human("C1CC2CCC1C2")
    text = str(d)
    assert "bicyclic [2.2.1] carbon skeleton" in text
    assert "parent ring basis" not in text
    assert "equivalent von Baeyer" not in text
    assert d.result.substituent_tree[0]["parent"]["von_baeyer_topology"]["descriptor"] == "bicyclo[2.2.1]"


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

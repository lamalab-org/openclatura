"""Graph-backed retained-name coverage for tetrapyrrole macrocycles."""

from __future__ import annotations

import pytest
from rdkit import Chem

from openclatura import name, name_smiles
from openclatura.graph_io import read_smiles
from openclatura.retained_macrocycle_templates import (
    match_retained_macrocycle,
    match_retained_macrocycles,
    retained_macrocycle_templates,
)

MACROCYCLE_CASES = (
    (
        "porphyrin",
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3",
    ),
    (
        "corrin",
        "C1=C2CCC(=N2)C=C2CCC(N2)C2CCC(=N2)C=C2CCC1=N2",
    ),
)


@pytest.mark.parametrize(("expected_name", "smiles"), MACROCYCLE_CASES)
def test_retained_macrocycle_parent_names(expected_name, smiles):
    assert name_smiles(smiles) == expected_name


@pytest.mark.parametrize(("expected_name", "smiles"), MACROCYCLE_CASES)
def test_retained_macrocycle_names_are_invariant_to_atom_order(expected_name, smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    for _ in range(4):
        reordered = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        assert name_smiles(reordered) == expected_name


def test_porphyrin_aliases_are_policy_data_for_one_graph_template():
    templates = {template.name: template for template in retained_macrocycle_templates()}
    assert set(templates) == {"porphyrin", "corrin"}
    assert templates["porphyrin"].output_name == "porphyrin"
    assert templates["porphyrin"].aliases == ("porphine", "21H,23H-porphine")
    assert templates["porphyrin"].numbering_policy == "retained_macrocycle_template"
    assert all(template.pre_descriptor_selection for template in templates.values())


@pytest.mark.parametrize(("expected_name", "smiles"), MACROCYCLE_CASES)
def test_retained_macrocycle_match_carries_complete_conventional_locants(expected_name, smiles):
    mol = read_smiles(smiles)
    match = match_retained_macrocycle(mol, set(mol.atoms))
    assert match is not None
    assert match.name == expected_name
    assert set(match.atom_to_locant) == set(mol.atoms)
    assert set(match.locant_to_atom) == set(match.template.locants)
    assert {match.template.atom_by_locant[locant].symbol for locant in ("21", "22", "23", "24")} == {"N"}


def test_similarly_sized_hydrogenated_graph_does_not_false_match_porphyrin():
    smiles = MACROCYCLE_CASES[0][1]
    rd_mol = Chem.MolFromSmiles(smiles)
    assert rd_mol is not None
    editable = Chem.RWMol(rd_mol)
    double_bond = next(bond for bond in editable.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
    double_bond.SetBondType(Chem.BondType.SINGLE)
    altered = read_smiles(Chem.MolToSmiles(editable))
    assert match_retained_macrocycle(altered, set(altered.atoms)) is None


def test_macrocycle_match_exposes_all_numbering_orientations():
    mol = read_smiles(MACROCYCLE_CASES[0][1])
    matches = match_retained_macrocycles(mol, set(mol.atoms))
    assert len(matches) > 1
    assert len({tuple(sorted(match.atom_to_locant.items())) for match in matches}) == len(matches)


def test_exact_macrocycle_bond_policy_rejects_a_rearranged_corrin_pattern():
    mol = read_smiles(MACROCYCLE_CASES[1][1])
    match = match_retained_macrocycle(mol, set(mol.atoms))
    assert match is not None
    by_class = {
        bond_class: next(bond for bond in match.template.bonds if bond.bond_class == bond_class)
        for bond_class in ("single", "double")
    }
    for bond_class, new_order in (("single", 2), ("double", 1)):
        left, right = by_class[bond_class].locants
        mol.get_bond(match.locant_to_atom[left], match.locant_to_atom[right]).order = new_order
    mol._retained_fused_cache.clear()
    assert match_retained_macrocycle(mol, set(mol.atoms)) is None


@pytest.mark.parametrize(("expected_name", "smiles"), MACROCYCLE_CASES)
def test_retained_macrocycle_round_trips_through_opsin(expected_name, smiles):
    result = name(smiles, verify_opsin=True, include_trace=True, token_debug=True)
    assert result.name == expected_name
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched"

    retained = next(step for step in result.decisions if step.decision == "used retained parent name")
    assert retained.data["retained_name"] == expected_name
    assert retained.data["locant_map_count"] >= 1
    numbering = next(step for step in result.decisions if step.decision == "selected numbering")
    assert len(numbering.data["atom_to_locant"]) == len(Chem.MolFromSmiles(smiles).GetAtoms())
    assembly = next(step for step in result.decisions if step.decision == "assembled component name")
    token_spans = assembly.data["name_token_spans"]
    assert len(token_spans) == 1
    token = token_spans[0]
    assert token["text"] == expected_name
    assert token["start"] == 0
    assert token["end"] == len(expected_name)
    assert token["token_kind"] == "parent"
    assert token["ownership"] == "exact"
    assert token["confidence"] == "derived"
    assert token["source"] == "typed_rewrite"
    assert token["grammar_role"] == "parent"
    assert token["binding_key"] == "parent:parent"
    assert token["atoms"] == list(range(Chem.MolFromSmiles(smiles).GetNumAtoms()))
    assert token["bonds"] == list(range(1, Chem.MolFromSmiles(smiles).GetNumBonds() + 1))

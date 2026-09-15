"""Separate completed-system numbering from missing carbon indicated H."""

import pytest
from rdkit import Chem

from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.numbering import (
    completed_system_numberings,
    indicated_hydrogen_candidate_atoms,
    observed_parent_matches_bond_model,
)
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles
from openclatura.opsin_verify import opsin_available, verify_with_opsin

SMILES = "C1N=COC2=NON=C12"
CORE_NAME = "[1,2,5]oxadiazolo[3,4-e][1,3]oxazine"


def _general_plan(smiles=SMILES):
    mol = read_smiles(smiles)
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert isinstance(result, FusionConfirmed)
    return mol, result.plan


@pytest.mark.parametrize("offset", range(9))
def test_hw_layout_numbering_preserves_carbon_h_site_under_atom_permutation(offset):
    rdkit_mol = Chem.MolFromSmiles(SMILES)
    order = list(reversed(range(rdkit_mol.GetNumAtoms())))
    order = order[offset:] + order[:offset]
    reordered = Chem.RenumberAtoms(rdkit_mol, order)
    mol, plan = _general_plan(Chem.MolToSmiles(reordered, canonical=False))

    assert plan.numbering.selected_layout.atom_positions
    assert observed_parent_matches_bond_model(mol, plan.bond_model)
    assert plan.bond_model.maximum_non_cumulative_double_bonds == 3
    for items in plan.numbering.input_locant_maps:
        locants = dict(items)
        assert {str(locant): mol.atoms[atom].symbol for atom, locant in items} == {
            "1": "N",
            "2": "O",
            "3": "N",
            "3a": "C",
            "4": "O",
            "5": "C",
            "6": "N",
            "7": "C",
            "7a": "C",
        }
        candidates = indicated_hydrogen_candidate_atoms(mol, locants)
        assert len(candidates) == 1
        assert mol.atoms[candidates[0]].symbol == "C"
        assert mol.atoms[candidates[0]].total_h_count == 2
        assert str(locants[candidates[0]]) == "7"


def test_graph_only_numbering_agrees_with_layout_numbering_for_hw_bicycle():
    mol, plan = _general_plan()
    faces = select_bounded_face_model(mol, mol.atoms)
    assert faces is not None
    graph_maps = {numbering.atom_to_locant for numbering in completed_system_numberings(mol, faces)}
    assert graph_maps == set(plan.numbering.input_locant_maps)


def test_hw_mancude_parent_cites_its_carbon_indicated_hydrogen():
    _, plan = _general_plan()
    assert tuple(map(str, plan.indicated_hydrogens)) == ("7",)
    assert plan.rendered_base_name == f"7H-{CORE_NAME}"


def test_all_generated_hw_composition_accepts_proved_intrinsic_carbon_h():
    mol = read_smiles(SMILES)
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    assert result.plan.rendered_base_name == f"7H-{CORE_NAME}"
    assert tuple(map(str, result.plan.indicated_hydrogens)) == ("7",)


@pytest.mark.opsin
@pytest.mark.parametrize("prefix, status", [("", "mismatched"), ("4H-", "mismatched"), ("7H-", "matched")])
def test_opsin_requires_carbon_h_at_layout_locant_not_graph_heuristic_locant(prefix, status):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin(f"{prefix}{CORE_NAME}", SMILES, standardize_smiles=False)
    assert check.status == status, check.to_dict()

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from rdkit import Chem

from openclatura import FusionMode, name, name_many, name_mol, opsin_available
from openclatura.fusion.context import current_fusion_mode, reset_fusion_mode, set_fusion_mode
from openclatura.fusion.faces import FaceSearchBudgetExceeded
from openclatura.fusion.model import (
    AuditStatus,
    FusionAuditFailed,
    FusionConfirmed,
    FusionNotApplicable,
    FusionUnsupported,
)
from openclatura.fusion.numbering import MancudeSearchBudgetExceeded
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles
from openclatura.molecule import OperationClass

SYSTEMATIC_FUSION_CASES = json.loads(
    (Path(__file__).parents[1] / "data" / "systematic_fusion_cases.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "case",
    SYSTEMATIC_FUSION_CASES,
    ids=[case["id"] for case in SYSTEMATIC_FUSION_CASES],
)
def test_audited_systematic_fusion_names(case):
    result = name(case["smiles"], fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)

    assert result.error is None
    assert result.name == case["name"]
    assert result.parent_nomenclature == case["parent_nomenclature"]
    assert result.pin_status == case["pin_status"]
    assert result.fusion_support_tier == case["support_tier"]
    assert result.proof_source == case["proof_source"]
    assert result.to_dict()["parent_nomenclature"] == "systematic_fusion"
    decisions = [step for step in result.decisions if step.decision == "selected audited systematic fusion parent"]
    assert len(decisions) == 1
    assert decisions[0].data["parent_nomenclature"] == "systematic_fusion"
    assert "input_graph_identity" in decisions[0].data["audit_checks"]
    assert decisions[0].data["proof_counts"]["bounded_faces"] >= 2
    assert decisions[0].data["proof_counts"]["audit_checks"] == len(decisions[0].data["audit_checks"])
    assert any(operation.operation_class is OperationClass.FUSION for operation in result.analysis.operations)


def test_fusion_trace_exposes_each_existing_proof_stage():
    result = name(
        "O1C2=C(C=C1)C=CS2",
        fusion_mode=FusionMode.GENERAL,
        include_trace=True,
    )

    decisions = {step.decision: step for step in result.decisions}
    assert decisions["selected fusion face model"].data["fusion_edges"]
    assert len([step for step in result.decisions if step.decision == "matched fusion component"]) == 2
    assert decisions["selected fusion parent location"].data["parent_occurrences"]
    assert decisions["constructed fusion descriptor"].data["descriptor"] == "[2,3-b]"
    assert decisions["selected preferred fusion orientation"].data["face_shapes"]
    assert decisions["selected completed fusion numbering"].data["atom_to_locant"]
    assert decisions["selected completed fusion numbering"].data["proof_counts"] == decisions[
        "audited systematic fusion parent"
    ].data["proof_counts"]
    assert decisions["audited systematic fusion parent"].data["status"] == "confirmed"


def test_fusion_tokens_are_owned_by_ast_components_and_interfaces():
    result = name(
        "O1C2=C(C=C1)C=CS2",
        fusion_mode=FusionMode.GENERAL,
        include_trace=True,
        token_debug=True,
    )
    assembly = next(step for step in reversed(result.decisions) if "name_token_spans" in step.data)
    tokens = {token["text"]: token for token in assembly.data["name_token_spans"]}

    assert tokens["thieno"]["source"] == "fusion_renderer"
    assert tokens["furan"]["source"] == "fusion_renderer"
    assert tokens["2,3"]["atoms"] == tokens["b"]["atoms"]
    assert tokens["2,3"]["bonds"] == tokens["b"]["bonds"]
    assert len(tokens["2,3"]["atoms"]) == 2
    assert len(tokens["2,3"]["bonds"]) == 1
    assert tokens["2,3"]["binding_key"].startswith("fusion:descriptor:interfaces=")


def test_multiplicative_fusion_token_keeps_grammar_only_ownership():
    result = name(
        "O1C=CC2=C1C=C1C(=N2)C=CO1",
        fusion_mode=FusionMode.GENERAL,
        include_trace=True,
        token_debug=True,
    )
    assembly = next(step for step in reversed(result.decisions) if "name_token_spans" in step.data)
    multiplier = next(token for token in assembly.data["name_token_spans"] if token["text"] == "di")

    assert multiplier["source"] == "fusion_renderer"
    assert multiplier["token_kind"] == "grammar"
    assert multiplier["atoms"] == []
    assert multiplier["bonds"] == []


def test_legacy_mode_preserves_previous_ring_name():
    result = name("O1C2=C(C=C1)C=CS2", fusion_mode=FusionMode.LEGACY)
    assert result.name == "2-oxa-8-thiabicyclo[3.3.0]octa-1(5),3,6-triene"


def test_legacy_mode_does_not_invoke_fusion_planner(monkeypatch):
    calls = 0

    def unexpected_planner(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("legacy naming must not invoke systematic fusion planning")

    monkeypatch.setattr("openclatura.parent_pipeline.plan_fusion_parent", unexpected_planner)

    result = name("O1C2=C(C=C1)C=CS2", fusion_mode=FusionMode.LEGACY)

    assert result.error is None
    assert calls == 0


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC1=CC2=C(O1)SC=C2", "2-methylthieno[2,3-b]furan"),
        ("O1C2=C(C=C1)C=C(S2)O", "thieno[2,3-b]furan-5-ol"),
        (
            "O1C=CC2=NC3=C(C=C21)SC(=C3)C(=O)O",
            "furo[3,2-b]thieno[2,3-e]pyridine-6-carboxylic acid",
        ),
    ],
)
def test_derivative_locants_use_completed_system_map(smiles, expected):
    assert name(smiles, fusion_mode=FusionMode.GENERAL).name == expected


def test_simple_hydrogenation_is_derived_from_parent_bond_model():
    result = name("O1C2=C(C=C1)CCS2", fusion_mode=FusionMode.GENERAL)
    assert result.name == "4,5-dihydrothieno[2,3-b]furan"


def test_generated_carbocycle_component_uses_existing_retained_polycycle_parent():
    result = name(
        "C1C=CC2=C1C1=CC=CC=C1C=1C=CC=CC21",
        fusion_mode=FusionMode.GENERAL,
        include_trace=True,
    )

    assert result.name == "1H-cyclopenta[l]phenanthrene"
    assert result.parent_nomenclature == "systematic_fusion"
    assert result.pin_status == "valid_general_name"


def test_generated_component_with_retained_polycycle_is_atom_order_invariant():
    mol = Chem.MolFromSmiles("C1C=CC2=C1C1=CC=CC=C1C=1C=CC=CC21")
    renumbered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    assert name_mol(mol, fusion_mode=FusionMode.GENERAL).name == name_mol(
        renumbered,
        fusion_mode=FusionMode.GENERAL,
    ).name


def test_saturated_carbon_uses_the_correct_completed_system_proof_locant():
    smiles = "C1C=CC2=C1C1=CC=CC=C1C=1C=CC=CC21"
    mol = read_smiles(smiles)
    result = name(
        smiles,
        fusion_mode=FusionMode.GENERAL,
        include_trace=True,
    )

    assert result.name == "1H-cyclopenta[l]phenanthrene"
    numbering = next(step for step in result.decisions if step.decision == "selected completed fusion numbering")
    saturated_carbon = next(atom for atom, value in mol.atoms.items() if value.total_h_count == 2)
    assert numbering.data["atom_to_locant"][saturated_carbon] == "1"


def test_carbon_indicated_hydrogen_is_not_added_to_two_monocycle_fusion():
    result = name("O1C=2C(=CC1)C=CC2", fusion_mode=FusionMode.GENERAL)

    assert result.name == "cyclopenta[b]furan"


def test_retained_parent_precedes_systematic_fusion():
    result = name("c1ccc2ccccc2c1", fusion_mode=FusionMode.GENERAL, include_trace=True)
    assert result.name == "naphthalene"
    assert not any(step.decision == "selected audited systematic fusion parent" for step in result.decisions)


def test_issue_78_retained_fusion_parent_precedes_the_new_planner():
    result = name("N1C=CC2=NC=CC=C21", fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)

    assert result.name == "1H-pyrrolo[3,2-b]pyridine"
    assert result.parent_nomenclature is None
    assert not any(step.decision == "selected audited systematic fusion parent" for step in result.decisions)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("c1ccc2cc3ccccc3cc2c1", "anthracene"),
        ("c1ccc2c(c1)ccc1ccccc12", "phenanthrene"),
        ("C1=CC=C2C=CC=CC=C12", "azulene"),
        ("C1=CC=C2C=CC3=CC=CC4=CC=C1C2=C34", "pyrene"),
        ("C1=CC=CC2=NC3=CC=CC=C3C=C12", "acridine"),
        ("C1=CC=CC=2C3=CC=CC=C3NC12", "9H-carbazole"),
        ("N1=CN=C2N=CNC2=C1", "7H-purine"),
        ("C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3", "porphyrin"),
        ("C1=C2CCC(=N2)C=C2CCC(N2)C2CCC(=N2)C=C2CCC1=N2", "corrin"),
    ],
)
def test_retained_complete_system_matrix_precedes_systematic_fusion(smiles, expected):
    result = name(smiles, fusion_mode=FusionMode.GENERAL, include_trace=True)

    assert result.name == expected
    assert result.parent_nomenclature is None
    assert not any(step.decision == "selected audited systematic fusion parent" for step in result.decisions)


def test_issue_71_aromatic_and_kekule_inputs_choose_the_same_fusion_parent():
    aromatic = name("c1cc2ccsc2o1", fusion_mode=FusionMode.GENERAL).name
    kekule = name("O1C2=C(C=C1)C=CS2", fusion_mode=FusionMode.GENERAL).name
    assert aromatic == kekule == "thieno[2,3-b]furan"


def test_issue_89_retained_fused_hydrocarbon_precedes_systematic_fusion():
    result = name("C1=CC=C2C=CC3=CC=CC4=CC=C1C2=C34", fusion_mode=FusionMode.GENERAL)

    assert result.name == "pyrene"
    assert result.parent_nomenclature is None


def test_fusion_name_is_invariant_to_graph_atom_renumbering():
    mol = Chem.MolFromSmiles("O1C=CC2=NC3=C(C=C21)SC=C3")
    renumbered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    assert (
        name_mol(mol, fusion_mode=FusionMode.GENERAL).name == name_mol(renumbered, fusion_mode=FusionMode.GENERAL).name
    )


@pytest.mark.parametrize(
    "smiles",
    (
        "O1C2=C(C=C1)C=CS2",
        "O1C=CC2=NC3=C(C=C21)SC=C3",
        "C1C=CC2=C1C1=CC=CC=C1C=1C=CC=CC21",
    ),
)
def test_fusion_name_is_invariant_to_random_atom_renumberings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    expected = name_mol(mol, fusion_mode=FusionMode.GENERAL).name

    for seed in range(4):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        renumbered = Chem.RenumberAtoms(mol, order)
        assert name_mol(renumbered, fusion_mode=FusionMode.GENERAL).name == expected


def test_batch_request_propagates_fusion_mode_without_cross_request_state():
    values = name_many(
        ["O1C2=C(C=C1)C=CS2", "CCO"],
        fusion_mode=FusionMode.GENERAL,
        processes=1,
    )
    assert [result.name for result in values] == ["thieno[2,3-b]furan", "ethanol"]
    assert name("O1C2=C(C=C1)C=CS2").name == "2-oxa-8-thiabicyclo[3.3.0]octa-1(5),3,6-triene"


def test_nested_legacy_request_restores_the_active_fusion_policy():
    token = set_fusion_mode(FusionMode.GENERAL)
    try:
        assert name("CCO", fusion_mode=FusionMode.LEGACY).name == "ethanol"
        assert current_fusion_mode() is FusionMode.GENERAL
    finally:
        reset_fusion_mode(token)


def test_planner_cache_is_request_policy_scoped_and_mutation_invalidated():
    mol = read_smiles("O1C2=C(C=C1)C=CS2")
    first = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    second = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(first, FusionConfirmed)
    assert first is second
    assert first.plan.audit.status is AuditStatus.CONFIRMED
    mol.update_atom(next(iter(mol.atoms)), total_h_count=mol.atoms[next(iter(mol.atoms))].total_h_count)
    assert not mol._fusion_plan_cache


def test_pin_mode_abstains_when_only_one_ring_meets_size_gate():
    mol = read_smiles("C1CC2=C1CCC2")
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionUnsupported)
    assert "ring-size gate" in result.reason


def test_charged_fused_parent_abstains_before_component_planning():
    mol = read_smiles("O1C2=C(C=C1)C=CS2")
    atom_id = next(iter(mol.atoms))
    mol.update_atom(atom_id, charge=1)

    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionUnsupported)
    assert "charged fused parents" in result.reason


def test_nonstandard_valence_fused_parent_abstains_before_component_planning():
    mol = read_smiles("O1C2=C(C=C1)C=CS2")
    oxygen = next(atom_id for atom_id, atom in mol.atoms.items() if atom.symbol == "O")
    mol.update_atom(oxygen, total_h_count=1)

    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionUnsupported)
    assert "nonstandard-valence fused parents" in result.reason


def test_spiro_parent_is_explicitly_not_applicable_to_fusion_nomenclature():
    mol = read_smiles("C1CCC2(CC1)CC2")

    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionNotApplicable)
    assert "spiro-only" in result.reason


def test_bridged_parent_explicitly_abstains_from_ordinary_fusion_nomenclature():
    mol = read_smiles("C1CC2CCC1C2")

    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionUnsupported)
    assert "bridged" in result.reason


def test_face_search_budget_exhaustion_becomes_a_typed_abstention(monkeypatch):
    mol = read_smiles("O1C2=C(C=C1)C=CS2")

    def exhausted(*args, **kwargs):
        raise FaceSearchBudgetExceeded("cycle enumeration", 1)

    monkeypatch.setattr("openclatura.fusion.planner.select_bounded_face_model", exhausted)
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionUnsupported)
    assert result.reason == "bounded-face search budget exhausted"
    assert "cycle enumeration" in result.details[0]


def test_mancude_search_budget_exhaustion_becomes_a_typed_abstention(monkeypatch):
    mol = read_smiles("O1C2=C(C=C1)C=CS2")

    def exhausted(*args, **kwargs):
        raise MancudeSearchBudgetExceeded(1)

    monkeypatch.setattr("openclatura.fusion.planner.parent_bond_model", exhausted)
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionUnsupported)
    assert result.reason == "mancude assignment search budget exhausted"
    assert "budget of 1 states" in result.details[0]


def test_component_graph_merge_failure_becomes_a_typed_audit_result(monkeypatch):
    mol = read_smiles("O1C2=C(C=C1)C=CS2")

    def inconsistent_graph(*args, **kwargs):
        raise ValueError("shared interface bond classes disagree")

    monkeypatch.setattr("openclatura.fusion.planner._abstract_graph", inconsistent_graph)
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionAuditFailed)
    assert result.reason == "fusion component graphs could not be merged consistently"
    assert result.candidate_summary == ("shared interface bond classes disagree",)


@pytest.mark.parametrize(
    "smiles",
    [
        "C1=CC=C2C=CC3=CC=CC4=CC=C1C2=C34",  # pyrene
        "c1cc2ccc3ccc4ccc5ccc6ccc1c2c3c4c56",  # coronene
    ],
)
def test_interior_atom_fused_parent_abstains_before_component_planning(smiles):
    mol = read_smiles(smiles)

    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionUnsupported)
    assert "interior-atom fused systems" in result.reason


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C12=CC=CC=C2C1", "bicyclo[4.1.0]hepta-1,3,5-triene"),
        ("C12=CC=CC=C2C=C1", "bicyclo[4.2.0]octa-1,3,5,7-tetraene"),
    ],
)
def test_pin_ring_size_gate_preserves_small_ring_von_baeyer_names(smiles, expected):
    result = name(smiles, fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)

    assert result.name == expected
    assert result.parent_nomenclature is None
    assert not any(step.decision == "selected audited systematic fusion parent" for step in result.decisions)


@pytest.mark.parametrize("smiles", ["C1CCC2(CC1)CC2", "C1CC12CCC2"])
def test_non_ortho_topologies_do_not_emit_systematic_fusion(smiles):
    result = name(smiles, fusion_mode=FusionMode.GENERAL, include_trace=True)
    assert "[" not in result.name or "spiro" in result.name or "bicyclo" in result.name
    assert not any(step.decision == "selected audited systematic fusion parent" for step in result.decisions)


def test_unsupported_topology_reason_is_exposed_in_the_public_trace():
    result = name("C1CC2CCC1C2", fusion_mode=FusionMode.GENERAL, include_trace=True)

    fallback = next(step for step in result.decisions if step.decision == "systematic fusion fallback")
    assert fallback.data["result"] == "FusionUnsupported"
    assert "bridged" in fallback.data["reason"]
    assert not any(step.decision == "selected audited systematic fusion parent" for step in result.decisions)


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    "smiles",
    [
        "O1C2=C(C=C1)C=CS2",
        "O1C=2C(=CC1)C=CC2",
        "S1C=2N(C=C1)C=CN2",
        "O1C=CC2=NC3=C(C=C21)SC=C3",
        "O1C=CC2=C1C=C1C(=N2)C=CO1",
        "CC1=CC2=C(O1)SC=C2",
        "O1C2=C(C=C1)CCS2",
        "C1C=CC2=C1C1=CC=CC=C1C=1C=CC=CC21",
    ],
)
def test_systematic_fusion_round_trips_through_opsin(smiles):
    result = name(smiles, fusion_mode=FusionMode.GENERAL, verify_opsin=True)
    assert result.opsin_check is not None
    assert result.opsin_check.ok, result.opsin_check.to_dict()

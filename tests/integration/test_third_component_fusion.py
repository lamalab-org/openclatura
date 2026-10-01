"""Production integration tests for cyclic third-component fusion systems."""

import pytest
from rdkit import Chem

from openclatura import FusionMode, name, name_mol, opsin_available
from openclatura.molecule import OperationClass

THIRD_COMPONENT_CASES = (
    (
        "N1=C2C3=C(N=NC3=N1)N=N2",
        "1,2,3,4,5,6-hexaazacyclopenta[cd]pentalene",
    ),
    (
        "N1=C2C3=C(N=NC3=N1)C=C2",
        "1,2,3,4-tetraazacyclopenta[cd]pentalene",
    ),
    (
        "CC1=CC=2C3=C1N=NC3=NN2",
        "5-methyl-1,2,3,4-tetraazacyclopenta[cd]pentalene",
    ),
    (
        "FC1=CC=2C3=C1N=NC3=NN2",
        "5-fluoro-1,2,3,4-tetraazacyclopenta[cd]pentalene",
    ),
)


@pytest.mark.parametrize(("smiles", "expected"), THIRD_COMPONENT_CASES)
def test_third_component_fusion_uses_corresponding_carbon_parent(smiles, expected):
    result = name(
        smiles,
        fusion_mode=FusionMode.AUDITED_PIN,
        include_trace=True,
    )

    assert result.error is None
    assert result.name == expected
    assert result.parent_nomenclature == "skeletal_replacement_fusion"
    assert result.pin_status == "confirmed"
    assert result.proof_source == "p25_5_skeletal_replacement"
    assert any(
        operation.operation_class is OperationClass.FUSION and operation.detail == "skeletal_replacement_fusion_parent"
        for operation in result.analysis.operations
    )
    selected = next(step for step in result.decisions if step.decision == "selected skeletal-replacement fusion parent")
    assert selected.data["cover_topology"] == "unicyclic"
    assert selected.data["prohibited_cycle_closing_joins"]
    assert "ordinary_pairwise_citation_not_emitted" in selected.data["audit_checks"]
    numbering = next(step for step in result.decisions if step.decision == "selected numbering")
    assert numbering.data["parent_nomenclature"] == "skeletal_replacement_fusion"


def test_third_component_name_is_atom_order_and_input_resonance_invariant():
    source = Chem.MolFromSmiles(THIRD_COMPONENT_CASES[0][0])
    renumbered = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    canonical = Chem.MolToSmiles(source, canonical=True)

    expected = THIRD_COMPONENT_CASES[0][1]
    assert name_mol(source, fusion_mode=FusionMode.GENERAL).name == expected
    assert name_mol(renumbered, fusion_mode=FusionMode.GENERAL).name == expected
    assert name(canonical, fusion_mode=FusionMode.GENERAL).name == expected


def test_third_component_tokens_use_replacement_and_fusion_graph_scopes():
    result = name(
        THIRD_COMPONENT_CASES[0][0],
        fusion_mode=FusionMode.GENERAL,
        include_trace=True,
        token_debug=True,
    )
    assembly = next(step for step in reversed(result.decisions) if "name_token_spans" in step.data)
    tokens = assembly.data["name_token_spans"]
    nitrogen_ids = {
        atom.GetIdx() for atom in Chem.MolFromSmiles(THIRD_COMPONENT_CASES[0][0]).GetAtoms() if atom.GetSymbol() == "N"
    }

    aza = next(token for token in tokens if token["text"] == "aza")
    assert set(aza["atoms"]) == nitrogen_ids
    assert aza["grammar_role"] == "replacement_prefix"
    assert all(
        token["source"] == "fusion_renderer" for token in tokens if token["text"] in {"cyclopenta", "cd", "pentalene"}
    )
    descriptor = next(token for token in tokens if token["text"] == "cd")
    assert descriptor["atoms"]
    assert descriptor["bonds"]


@pytest.mark.skipif(not opsin_available(), reason="OPSIN verification is unavailable")
@pytest.mark.parametrize(("smiles", "expected"), THIRD_COMPONENT_CASES)
def test_third_component_fusion_round_trips_through_opsin(smiles, expected):
    result = name(
        smiles,
        fusion_mode=FusionMode.AUDITED_PIN,
        verify_opsin=True,
    )

    assert result.name == expected
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched"

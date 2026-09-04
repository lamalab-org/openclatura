"""Focused tests for the unified ring parent-hydride handoff."""

import pytest

from openclatura import name, name_smiles
from openclatura.assembly_parts import AssemblyParts, RetainedParentMetadata
from openclatura.component_namer import select_component_parent
from openclatura.graph_io import read_smiles
from openclatura.parent_pipeline import resolve_parent_hydride_plan, resolve_retained_parent
from openclatura.ring_parent import ParentHydrideKind, ParentHydridePlan, RingParent
from openclatura.rules import retained


def _resolved_ring_parent(smiles: str) -> ParentHydridePlan:
    mol = read_smiles(smiles)
    selection = select_component_parent(mol, set(), [])
    assert selection is not None
    retained_name, locant_maps = resolve_retained_parent(
        mol,
        selection.primary_path,
        selection.is_ring,
        selection.is_bicycle,
        selection.is_polycycle,
    )
    plan = resolve_parent_hydride_plan(
        mol,
        selection,
        retained_name=retained_name,
        locant_maps=locant_maps,
        retained_parent_metadata=(
            retained.parent_metadata(retained_name) if retained_name is not None else None
        ),
    )
    assert plan is not None
    return plan


def test_retained_parent_owns_name_locants_stem_and_hydride_metadata():
    plan = _resolved_ring_parent("c1ccc2ncccc2c1")

    assert plan.hydride_kind is ParentHydrideKind.RETAINED
    assert plan.base_name == "quinoline"
    assert plan.derivative_stem == "quinolin"
    assert plan.metadata is not None
    assert plan.metadata.fusion_locants == ("4a", "8a")
    assert plan.metadata.mancude_double_bonds == 5
    assert plan.proof_locant_maps
    assert all(set(mapping) == plan.atoms for mapping in plan.proof_locant_maps)


def test_generated_and_von_baeyer_parents_use_the_same_handoff_type():
    monocycle = _resolved_ring_parent("C1CCCCC1")
    bicyclo = _resolved_ring_parent("C1CC2CCC1C2")

    assert isinstance(monocycle, RingParent)
    assert monocycle.hydride_kind is ParentHydrideKind.GENERATED_MONOCYCLE
    assert monocycle.base_hydride_name(6) == "cyclohexane"
    assert bicyclo.hydride_kind is ParentHydrideKind.VON_BAEYER
    assert bicyclo.base_hydride_name(7) == "bicyclo[2.2.1]heptane"
    assert bicyclo.binding_term == "bicyclo[2.2.1]"
    assert bicyclo.proof_locant_maps


def test_assembly_parts_exposes_ring_parent_as_compatibility_alias():
    plan = _resolved_ring_parent("c1ccccc1")
    parts = AssemblyParts(parent_length=6, parent_hydride=plan)

    assert parts.parent_hydride is plan
    assert parts.ring_parent is plan
    assert parts.retained_name == "benzene"
    assert parts.retained_parent_metadata is plan.metadata

    other = _resolved_ring_parent("C1CCCCC1")
    with pytest.raises(ValueError, match="same plan"):
        AssemblyParts(parent_length=6, parent_hydride=plan, ring_parent=other)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C1CCCCC1", "cyclohexane"),
        ("c1ccccc1", "benzene"),
        ("C1CC2CCC1C2", "bicyclo[2.2.1]heptane"),
        ("c1ccc2ncccc2c1", "quinoline"),
        ("CC(=O)Nc1ccccc1", "N-phenylacetamide"),
        ("NC(=O)c1ccc(C(=O)O)cc1", "4-carbamoylbenzoic acid"),
    ],
)
def test_parent_hydride_handoff_preserves_names(smiles: str, expected: str):
    assert name_smiles(smiles) == expected


def test_parent_hydride_kind_is_exposed_in_numbering_and_assembly_trace():
    result = name("c1ccc2ncccc2c1", include_trace=True)

    assert result.error is None
    relevant = [
        step.data["parent_hydride_kind"]
        for step in result.decisions
        if step.decision in {"selected numbering", "assembled component name"}
    ]
    assert relevant == ["retained", "retained"]


def test_retained_metadata_compatibility_name_is_the_canonical_type():
    assert RetainedParentMetadata.__name__ == "ParentHydrideMetadata"

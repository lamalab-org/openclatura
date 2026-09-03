"""Focused tests for the pipeline-independent fusion nomenclature model."""

from dataclasses import FrozenInstanceError, replace

import pytest

from openclatura.fusion import (
    AuditStatus,
    BondAssignment,
    ComponentAtom,
    ComponentBond,
    ComponentLocant,
    Face,
    FaceModel,
    FusedLayout,
    FusionAuditResult,
    FusionCitationNode,
    FusionComponentMatch,
    FusionComponentSpec,
    FusionConfirmed,
    FusionDescriptor,
    FusionGraph,
    FusionGraphAtom,
    FusionGraphBond,
    FusionJoin,
    FusionJoinKind,
    FusionMode,
    FusionNameAst,
    FusionNotApplicable,
    FusionNumberingProof,
    FusionParentPlan,
    FusionSide,
    FusionUnsupported,
    ParentBondModel,
    PinStatus,
    SystemLocant,
    component_seniority_key,
    explain_component_comparison,
    fusion_mode_allows_planning,
    pin_ring_size_gate,
)
from openclatura.retained_fused_templates import RetainedGraphTemplate, validate_retained_fused_template
from openclatura.ring_parent import RingParent


def _match(occurrence_id: int, spec_key: str, offset: int = 0) -> FusionComponentMatch:
    return FusionComponentMatch(
        occurrence_id=occurrence_id,
        spec_key=spec_key,
        covered_face_ids=frozenset({occurrence_id}),
        local_to_input_atom=(("1", offset + 1), ("2", offset + 2)),
        local_to_skeleton_atom=(("1", offset + 10), ("2", offset + 11)),
        topology_key=(2, 1),
    )


def _face_model() -> FaceModel:
    return FaceModel(
        faces=(Face(id=0, atom_cycle=(0, 1, 2), edge_cycle=(0, 1, 2), size=3),),
        edge_to_faces=((0, (0,)), (1, (0,)), (2, (0,))),
        perimeter_edges=frozenset({0, 1, 2}),
        fusion_edges=frozenset(),
        outer_boundary=(0, 1, 2),
        face_adjacency=(),
    )


def _bond_model() -> ParentBondModel:
    return ParentBondModel(
        allowed_kekule_assignments=(BondAssignment(orders=((((0, 1)), 2), (((1, 2)), 1), (((0, 2)), 1))),),
        required_single_bonds=frozenset(),
        pi_eligible_edges=frozenset({(0, 1), (1, 2), (0, 2)}),
        maximum_non_cumulative_double_bonds=1,
    )


def _confirmed_fusion_plan() -> FusionParentPlan:
    attached = _match(0, "furan")
    parent = _match(1, "pyridine", offset=2)
    join = FusionJoin(
        attached_occurrence=0,
        host_occurrence=1,
        order=1,
        kind=FusionJoinKind.ORTHO,
        attached_locants=(ComponentLocant(0, "2"), ComponentLocant(0, "3")),
        host_sides=(FusionSide(1, "b"),),
        shared_input_atoms=frozenset({2, 3}),
        shared_input_bonds=frozenset({2}),
    )
    ast = FusionNameAst(
        plan_kind="two_component",
        parent_occurrences=(1,),
        component_occurrences=(attached, parent),
        joins=(join,),
        citation_tree=FusionCitationNode(1, children=(FusionCitationNode(0),)),
        descriptors=(
            FusionDescriptor(
                attached_locants=join.attached_locants,
                parent_sides=join.host_sides,
            ),
        ),
    )
    locants = (SystemLocant(1), SystemLocant(2), SystemLocant(2, "a"))
    numbering = FusionNumberingProof(
        selected_face_model=_face_model(),
        selected_layout=FusedLayout(face_positions=((0, 0, 0),)),
        orientation_score=(1, 0, 0, 0),
        abstract_atom_to_locant=tuple(enumerate(locants)),
        input_locant_maps=(((10, locants[0]), (11, locants[1]), (12, locants[2])),),
    )
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(index, "C") for index in range(3)),
        bonds=(FusionGraphBond((0, 1)), FusionGraphBond((1, 2)), FusionGraphBond((2, 0))),
    )
    return FusionParentPlan(
        ast=ast,
        rendered_base_name="furo[2,3-b]pyridine",
        abstract_parent_graph=graph,
        numbering=numbering,
        bond_model=_bond_model(),
        indicated_hydrogens=(),
        pin_status=PinStatus.CONFIRMED,
        rule_trace=(),
        audit=FusionAuditResult(AuditStatus.CONFIRMED, checks=("reconstruction", "numbering")),
    )


def _component_spec(key: str, symbols: tuple[str, ...], *, rings: int = 1) -> FusionComponentSpec:
    locants = tuple(str(index) for index in range(1, len(symbols) + 1))
    ring = tuple(locants)
    template = RetainedGraphTemplate(
        name=key,
        pin=True,
        priority=0,
        aliases=(),
        attached_prefix=f"{key}o",
        derivative_stem=None,
        default_indicated_h=(),
        locants=locants,
        atoms=tuple(ComponentAtom(locant, symbol) for locant, symbol in zip(locants, symbols, strict=True)),
        bonds=tuple(ComponentBond((locants[index - 1], locants[index])) for index in range(len(locants))),
        rings=tuple(ring for _ in range(rings)),
        fusion_atoms=(),
        peripheral_atoms=locants,
        interior_atoms=(),
    )
    return FusionComponentSpec(
        key=key,
        parent_name=key,
        attached_prefix=f"{key}o",
        template=template,
        usable_as_parent=True,
        usable_as_attached=True,
        rule_reference="P-25",
    )


def test_locant_namespaces_render_without_becoming_interchangeable():
    component = ComponentLocant(component_id=2, text="3", prime_depth=1)
    side = FusionSide(component_id=1, letter="b", prime_depth=2)
    fusion = SystemLocant(base=4, fusion_suffix="a")
    interior = SystemLocant(base=3, interior_distance=2)

    assert str(component) == "3'"
    assert str(side) == "b''"
    assert str(fusion) == "4a"
    assert str(interior) == "3²"
    assert interior.render(unicode_superscript=False) == "3^2"
    assert component != ComponentLocant(component_id=1, text="3", prime_depth=1)


def test_locants_reject_ambiguous_or_invalid_states():
    with pytest.raises(ValueError, match="prime marks"):
        ComponentLocant(0, "3'")
    with pytest.raises(ValueError, match="mutually exclusive"):
        SystemLocant(3, fusion_suffix="a", interior_distance=2)
    with pytest.raises(ValueError, match="positive"):
        SystemLocant(0)


def test_component_spec_validates_complete_local_graph():
    spec = _component_spec("pyridine", ("N", "C", "C", "C", "C", "C"))
    assert len(spec.rings) == 1

    with pytest.raises(ValueError, match="atom locants"):
        validate_retained_fused_template(replace(spec.template, atoms=spec.atoms[:-1]))


def test_component_match_requires_two_complete_bijective_maps():
    with pytest.raises(ValueError, match="bijective"):
        FusionComponentMatch(
            occurrence_id=0,
            spec_key="furan",
            covered_face_ids=frozenset({0}),
            local_to_input_atom=(("1", 4), ("2", 4)),
            local_to_skeleton_atom=(("1", 0), ("2", 1)),
            topology_key=(),
        )
    with pytest.raises(ValueError, match="same complete"):
        FusionComponentMatch(
            occurrence_id=0,
            spec_key="furan",
            covered_face_ids=frozenset({0}),
            local_to_input_atom=(("1", 4), ("2", 5)),
            local_to_skeleton_atom=(("1", 0), ("3", 1)),
            topology_key=(),
        )


def test_join_and_descriptor_preserve_attached_locant_direction():
    join = FusionJoin(
        attached_occurrence=0,
        host_occurrence=1,
        order=1,
        kind=FusionJoinKind.ORTHO,
        attached_locants=(ComponentLocant(0, "3"), ComponentLocant(0, "2")),
        host_sides=(FusionSide(1, "b"),),
    )
    descriptor = FusionDescriptor(join.attached_locants, parent_sides=join.host_sides)

    assert descriptor.render() == "[3,2-b]"
    assert tuple(locant.text for locant in join.attached_locants) == ("3", "2")


def test_numbering_proof_rejects_incomplete_input_map():
    locants = (SystemLocant(1), SystemLocant(2), SystemLocant(2, "a"))
    with pytest.raises(ValueError, match="completely cover"):
        FusionNumberingProof(
            selected_face_model=_face_model(),
            selected_layout=FusedLayout(face_positions=((0, 0, 0),)),
            orientation_score=(),
            abstract_atom_to_locant=tuple(enumerate(locants)),
            input_locant_maps=(((10, locants[0]), (11, locants[1])),),
        )


def test_fusion_parent_plan_requires_confirmed_audit():
    confirmed = _confirmed_fusion_plan()
    with pytest.raises(ValueError, match="confirmed independent audit"):
        FusionParentPlan(
            ast=confirmed.ast,
            rendered_base_name=confirmed.rendered_base_name,
            abstract_parent_graph=confirmed.abstract_parent_graph,
            numbering=confirmed.numbering,
            bond_model=confirmed.bond_model,
            indicated_hydrogens=(),
            pin_status=PinStatus.UNSUPPORTED,
            rule_trace=(),
            audit=FusionAuditResult(AuditStatus.ABSTAIN),
        )


def test_ring_parent_from_fusion_preserves_proof_maps_and_is_immutable():
    fusion = _confirmed_fusion_plan()
    parent = RingParent.from_fusion_plan(fusion)

    assert parent.kind == "systematic_fusion"
    assert parent.proof_locant_maps == ({10: "1", 11: "2", 12: "2a"},)
    assert parent.proof_source == "fusion_reconstruction"
    assert FusionConfirmed(fusion).plan is fusion
    with pytest.raises(FrozenInstanceError):
        parent.descriptor = "changed"


def test_planning_outcomes_represent_normal_abstention_without_partial_plan():
    assert FusionNotApplicable("spiro_only").reason == "spiro_only"
    assert FusionUnsupported("unsupported_ring_size", ("ring_size=9",)).details == ("ring_size=9",)


def test_pin_gate_and_modes_are_explicit():
    assert pin_ring_size_gate((5, 6))
    assert not pin_ring_size_gate((4, 6))
    assert not fusion_mode_allows_planning(FusionMode.LEGACY)
    assert fusion_mode_allows_planning(FusionMode.AUDITED_PIN)
    assert fusion_mode_allows_planning(FusionMode.GENERAL)


def test_component_seniority_is_lexicographic_and_explainable():
    carbon = _component_spec("benzene", ("C",) * 6)
    nitrogen = _component_spec("pyridine", ("N", "C", "C", "C", "C", "C"))
    registry = {carbon.key: carbon, nitrogen.key: nitrogen}
    carbon_match = _match(0, carbon.key)
    nitrogen_match = _match(1, nitrogen.key, offset=2)

    assert component_seniority_key(nitrogen_match, registry) < component_seniority_key(carbon_match, registry)
    decision = explain_component_comparison(nitrogen_match, carbon_match, registry)
    assert decision.outcome == "left"
    assert decision.criterion == "earliest_special_heteroatom"


def test_parent_bond_model_rejects_incomplete_kekule_assignment():
    with pytest.raises(ValueError, match="every parent bond"):
        ParentBondModel(
            allowed_kekule_assignments=(BondAssignment(orders=((((0, 1)), 2),)),),
            required_single_bonds=frozenset(),
            pi_eligible_edges=frozenset({(0, 1), (1, 2)}),
            maximum_non_cumulative_double_bonds=1,
        )

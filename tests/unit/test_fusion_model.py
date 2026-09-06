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
    PinDecision,
    PinStatus,
    SystemLocant,
    component_seniority_key,
    component_spec_seniority_key,
    explain_component_comparison,
    fusion_mode_allows_planning,
    pin_ring_size_gate,
)
from openclatura.fusion.model import FusionCitationPlan, OrderedFusionInterface
from openclatura.retained_fused_templates import RetainedGraphTemplate, validate_retained_fused_template
from openclatura.ring_parent import ParentHydrideKind, RingParent


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
        order=1,
        interface=OrderedFusionInterface(
            kind=FusionJoinKind.ORTHO,
            attached_occurrence=0,
            host_occurrence=1,
            attached_path=(ComponentLocant(0, "2"), ComponentLocant(0, "3")),
            host_path=(ComponentLocant(1, "1"), ComponentLocant(1, "2")),
            cited_attached_locants=(ComponentLocant(0, "2"), ComponentLocant(0, "3")),
            host_sides=(FusionSide(1, "b"),),
            ordered_input_atoms=(2, 3),
            ordered_input_edges=((2, 3),),
            ordered_input_bonds=(2,),
        ),
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
        pin_eligibility="fusion_rules_satisfied",
        rule_trace=(),
        audit=FusionAuditResult(AuditStatus.CONFIRMED, checks=("reconstruction", "numbering")),
    )


def test_citation_plan_uses_join_indices_when_attachment_depths_repeat():
    ast = _confirmed_fusion_plan().ast
    original = ast.joins[0]
    second = replace(
        original,
        interface=replace(
            original.interface,
            attached_occurrence=2,
            attached_path=(ComponentLocant(2, "2"), ComponentLocant(2, "3")),
            cited_attached_locants=(ComponentLocant(2, "2"), ComponentLocant(2, "3")),
        ),
    )
    tree = FusionCitationNode(1, children=(FusionCitationNode(0), FusionCitationNode(2)))
    repeated_depth_ast = FusionNameAst(
        plan_kind="polycomponent_tree",
        parent_occurrences=(1,),
        component_occurrences=(*ast.component_occurrences, _match(2, "furan", offset=4)),
        joins=(original, second),
        citation_tree=tree,
        descriptors=(
            FusionDescriptor.from_interface(original.interface),
            FusionDescriptor.from_interface(second.interface),
        ),
    )

    plan = FusionCitationPlan.from_tree(tree, repeated_depth_ast.joins)

    assert tuple(join.order for join in repeated_depth_ast.joins) == (1, 1)
    assert plan.primary_join_indices == (0, 1)
    assert replace(repeated_depth_ast, parent_occurrences=(0,)).parent_occurrences == (0,)
    explicit = replace(repeated_depth_ast, citation_plan=plan)
    corrupted = replace(
        explicit,
        parent_occurrences=(0,),
        citation_tree=FusionCitationNode(0, children=(FusionCitationNode(1), FusionCitationNode(2))),
    )
    assert corrupted.parent_occurrences == (0,)


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
    assert str(SystemLocant(3, fusion_suffix="a", interior_distance=2)) == "3a²"
    with pytest.raises(ValueError, match="positive"):
        SystemLocant(0)


def test_component_spec_validates_complete_local_graph():
    spec = _component_spec("pyridine", ("N", "C", "C", "C", "C", "C"))
    assert len(spec.rings) == 1

    with pytest.raises(ValueError, match="atom locants"):
        validate_retained_fused_template(replace(spec.template, atoms=spec.atoms[:-1]))


def test_retained_template_stereo_policy_requires_locant_level_declarations():
    spec = _component_spec("stereoparent", ("C", "C", "C"))

    with pytest.raises(ValueError, match="declares no stereo locants"):
        validate_retained_fused_template(replace(spec.template, implied_stereo=True))

    atoms = (replace(spec.atoms[0], required_stereo=True), *spec.atoms[1:])
    with pytest.raises(ValueError, match="without implied stereo"):
        validate_retained_fused_template(replace(spec.template, atoms=atoms))


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
        order=1,
        interface=OrderedFusionInterface(
            kind=FusionJoinKind.ORTHO,
            attached_occurrence=0,
            host_occurrence=1,
            attached_path=(ComponentLocant(0, "3"), ComponentLocant(0, "2")),
            host_path=(ComponentLocant(1, "2"), ComponentLocant(1, "3")),
            cited_attached_locants=(ComponentLocant(0, "3"), ComponentLocant(0, "2")),
            host_sides=(FusionSide(1, "b"),),
            ordered_input_atoms=(2, 3),
            ordered_input_edges=((2, 3),),
            ordered_input_bonds=(7,),
        ),
    )
    descriptor = FusionDescriptor.from_interface(join.interface)

    assert descriptor.render() == "[3,2-b]"
    assert tuple(locant.text for locant in join.attached_locants) == ("3", "2")


def test_ordered_ortho_peri_interface_keeps_complete_paths_and_descriptor_projection():
    interface = OrderedFusionInterface(
        kind=FusionJoinKind.ORTHO_PERI,
        attached_occurrence=0,
        host_occurrence=1,
        attached_path=tuple(ComponentLocant(0, text) for text in ("1", "2", "3")),
        host_path=tuple(ComponentLocant(1, text) for text in ("3", "4", "5")),
        cited_attached_locants=tuple(ComponentLocant(0, text) for text in ("1", "2", "3")),
        host_sides=(FusionSide(1, "c"), FusionSide(1, "d")),
        ordered_input_atoms=(10, 11, 12),
        ordered_input_edges=((10, 11), (11, 12)),
        ordered_input_bonds=(20, 21),
    )
    join = FusionJoin(order=1, interface=interface)
    descriptor = FusionDescriptor.from_interface(join.interface)

    assert descriptor.render() == "[1,2,3-cd]"
    assert join.shared_input_atoms == frozenset({10, 11, 12})
    assert join.shared_input_bonds == frozenset({20, 21})


def test_ordered_interface_rejects_disconnected_or_misordered_graph_evidence():
    with pytest.raises(ValueError, match="follow the ordered input atom path"):
        OrderedFusionInterface(
            kind=FusionJoinKind.ORTHO_PERI,
            attached_occurrence=0,
            host_occurrence=1,
            attached_path=tuple(ComponentLocant(0, text) for text in ("1", "2", "3")),
            host_path=tuple(ComponentLocant(1, text) for text in ("3", "4", "5")),
            cited_attached_locants=tuple(ComponentLocant(0, text) for text in ("1", "2", "3")),
            host_sides=(FusionSide(1, "c"), FusionSide(1, "d")),
            ordered_input_atoms=(10, 11, 12),
            ordered_input_edges=((10, 11), (10, 12)),
            ordered_input_bonds=(20, 21),
        )


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
            pin_eligibility="fusion_rules_satisfied",
            rule_trace=(),
            audit=FusionAuditResult(AuditStatus.ABSTAIN),
        )


def test_ring_parent_from_fusion_preserves_proof_maps_and_is_immutable():
    fusion = _confirmed_fusion_plan()
    parent = RingParent.from_fusion_plan(
        fusion,
        pin_decision=PinDecision(
            PinStatus.CONFIRMED,
            (
                "no_preferred_retained_complete_parent",
                "no_preferred_independently_systematic_complete_parent",
                "fusion_rules_satisfied",
            ),
        ),
    )

    assert parent.kind == "systematic_fusion"
    assert parent.proof_locant_maps == ({10: "1", 11: "2", 12: "2a"},)
    assert parent.proof_source == "fusion_reconstruction"
    assert parent.hydride_kind is ParentHydrideKind.SYSTEMATIC_FUSION
    assert parent.base_name == fusion.rendered_base_name
    assert parent.bond_model is fusion.bond_model
    assert parent.metadata is not None
    assert parent.metadata.mancude_double_bonds == fusion.bond_model.maximum_non_cumulative_double_bonds
    assert parent.parent_name is None
    assert parent.descriptor is None
    assert not parent.retained_locant_maps
    assert parent.parent_bond_model is None
    assert parent.base_name == fusion.rendered_base_name
    assert parent.proof_locant_maps == fusion.numbering.string_input_locant_maps()
    assert parent.bond_model is fusion.bond_model
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


def test_confirmed_pin_decision_requires_precedence_and_fusion_evidence():
    with pytest.raises(ValueError, match="precedence and fusion-rule evidence"):
        PinDecision(PinStatus.CONFIRMED, ("fusion_rules_satisfied",))


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


def _seniority_variant(
    key: str,
    symbols: tuple[str, ...],
    *,
    rings: tuple[tuple[str, ...], ...] | None = None,
    fusion_atoms: tuple[str, ...] = (),
    horizontal_ring_count: int = 1,
) -> FusionComponentSpec:
    base = _component_spec(key, symbols)
    template = replace(
        base.template,
        rings=rings or base.template.rings,
        fusion_atoms=fusion_atoms,
    )
    return replace(base, template=template, horizontal_ring_count=horizontal_ring_count)


@pytest.mark.parametrize(
    ("preferred", "other", "criterion"),
    [
        (
            _seniority_variant("two-rings", ("C",) * 6, rings=(("1", "2", "3"), ("3", "4", "5", "6"))),
            _seniority_variant("one-ring", ("C",) * 6),
            "ring_count",
        ),
        (
            _seniority_variant("seven-ring", ("C",) * 7),
            _seniority_variant("six-ring", ("C",) * 6),
            "ring_size_vector",
        ),
        (
            _seniority_variant("two-nitrogens", ("N", "N", "C", "C", "C", "C")),
            _seniority_variant("one-nitrogen", ("N", "C", "C", "C", "C", "C")),
            "heteroatom_count",
        ),
        (
            _seniority_variant("two-kinds", ("N", "O", "C", "C", "C", "C")),
            _seniority_variant("one-kind", ("N", "N", "C", "C", "C", "C")),
            "heteroatom_kind_count",
        ),
        (
            _seniority_variant("nitrogen-oxygen", ("N", "O", "C", "C", "C", "C")),
            _seniority_variant("nitrogen-sulfur", ("N", "S", "C", "C", "C", "C")),
            "heteroatom_counts_by_priority",
        ),
        (
            _seniority_variant("linear", ("C",) * 6, horizontal_ring_count=2),
            _seniority_variant("angular", ("C",) * 6, horizontal_ring_count=1),
            "horizontal_row_count",
        ),
        (
            _seniority_variant("lower-set", ("N", "C", "C", "C", "C", "C")),
            _seniority_variant("higher-set", ("C", "N", "C", "C", "C", "C")),
            "all_heteroatom_locants",
        ),
        (
            _seniority_variant("senior-at-one", ("O", "N", "C", "C", "C", "C")),
            _seniority_variant("senior-at-two", ("N", "O", "C", "C", "C", "C")),
            "per_element_locants",
        ),
        (
            _seniority_variant("fusion-at-one", ("C",) * 6, fusion_atoms=("1",)),
            _seniority_variant("fusion-at-two", ("C",) * 6, fusion_atoms=("2",)),
            "peripheral_fusion_carbon_locants",
        ),
    ],
)
def test_each_component_seniority_criterion_is_ordered_independently(preferred, other, criterion):
    assert component_spec_seniority_key(preferred) < component_spec_seniority_key(other)
    decision = explain_component_comparison(
        _match(0, preferred.key),
        _match(1, other.key, offset=10),
        {preferred.key: preferred, other.key: other},
    )
    assert decision.criterion == criterion


def test_resolved_component_variant_controls_seniority_without_key_collapse():
    lower = _seniority_variant("shared", ("N", "C", "C", "C", "C", "C"))
    higher = _seniority_variant("shared", ("C", "N", "C", "C", "C", "C"))

    assert component_spec_seniority_key(lower) < component_spec_seniority_key(higher)


def test_registry_key_is_not_a_component_seniority_criterion():
    original = _seniority_variant("original-key", ("N", "C", "C", "C", "C", "C"))
    renamed = replace(original, key="renamed-key")

    assert component_spec_seniority_key(original) == component_spec_seniority_key(renamed)
    decision = explain_component_comparison(
        _match(0, original.key),
        _match(1, renamed.key, offset=10),
        {original.key: original, renamed.key: renamed},
    )
    assert decision.criterion == "complete_tie"
    assert decision.outcome == "tie"


def test_parent_bond_model_rejects_incomplete_kekule_assignment():
    with pytest.raises(ValueError, match="every parent bond"):
        ParentBondModel(
            allowed_kekule_assignments=(BondAssignment(orders=((((0, 1)), 2),)),),
            required_single_bonds=frozenset(),
            pi_eligible_edges=frozenset({(0, 1), (1, 2)}),
            maximum_non_cumulative_double_bonds=1,
        )

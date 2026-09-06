"""Graph-native tests for independent systematic-fusion auditing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.model import (
    AuditStatus,
    BondAssignment,
    ComponentAtom,
    ComponentBond,
    ComponentLocant,
    Face,
    FaceModel,
    FusedLayout,
    FusionCitationNode,
    FusionCitationPlan,
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
    FusionMultiplicityGroup,
    FusionNameAst,
    FusionNumberingProof,
    FusionSide,
    OrderedFusionInterface,
    ParentBondModel,
    SystemLocant,
)
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles
from openclatura.molecule import Molecule
from openclatura.retained_fused_templates import RetainedGraphTemplate


@dataclass(frozen=True)
class _Candidate:
    mol: Molecule
    parent_atoms: frozenset[int]
    ast: FusionNameAst
    graph: FusionGraph
    numbering: FusionNumberingProof
    bond_model: ParentBondModel
    registry: dict[str, FusionComponentSpec]


def _two_fused_rings(
    left_size: int = 6,
    right_size: int = 6,
    *,
    interface_atom_count: int = 2,
) -> _Candidate:
    """Build two carbocycles sharing one edge, without line notation."""

    if not 2 <= interface_atom_count < min(left_size, right_size):
        raise ValueError("interface_atom_count must define a proper ring path")
    shared = tuple(range(left_size - interface_atom_count, left_size))
    left_cycle = tuple(range(left_size))
    right_cycle = shared + tuple(range(left_size, left_size + right_size - interface_atom_count))

    mol = Molecule()
    for atom in range(left_size + right_size - interface_atom_count):
        mol.add_atom("C", idx=atom)
    next_bond = 100
    for cycle in (left_cycle, right_cycle):
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            if mol.get_bond(left, right) is None:
                mol.add_bond(left, right, idx=next_bond)
                next_bond += 1

    bounded = select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None and bounded.audit.ok
    faces = tuple(
        Face(
            id=index,
            atom_cycle=cycle.atoms,
            edge_cycle=tuple(
                _bond_id(mol, left, right) for left, right in zip(cycle.atoms, cycle.atoms[1:] + cycle.atoms[:1])
            ),
            size=len(cycle.atoms),
        )
        for index, cycle in enumerate(bounded.faces)
    )
    edge_owners = tuple(
        sorted(
            (
                _bond_id(mol, *edge),
                tuple(index for index, cycle in enumerate(bounded.faces) if edge in cycle.edges),
            )
            for edge in bounded.edge_ids
        )
    )
    fusion_edges = frozenset(edge for edge, owners in edge_owners if len(owners) == 2)
    perimeter_edges = frozenset(edge for edge, owners in edge_owners if len(owners) == 1)
    face_model = FaceModel(
        faces=faces,
        edge_to_faces=edge_owners,
        perimeter_edges=perimeter_edges,
        fusion_edges=fusion_edges,
        outer_boundary=bounded.outer_boundary.atoms,
        face_adjacency=tuple(sorted((owners[0], owners[1], edge) for edge, owners in edge_owners if len(owners) == 2)),
    )

    specs = {
        "left": _ring_spec("left", left_size),
        # Give the synthetic host a real chemical seniority distinction.
        "right": replace(_ring_spec("right", right_size), horizontal_ring_count=1),
    }
    face_by_atoms = {frozenset(face.atom_cycle): face.id for face in faces}
    left_match = _match(0, "left", left_cycle, face_by_atoms[frozenset(left_cycle)])
    right_match = _match(1, "right", right_cycle, face_by_atoms[frozenset(right_cycle)])
    join = FusionJoin(
        order=1,
        interface=OrderedFusionInterface(
            kind=(FusionJoinKind.ORTHO if interface_atom_count == 2 else FusionJoinKind.ORTHO_PERI),
            attached_occurrence=0,
            host_occurrence=1,
            attached_path=tuple(
                ComponentLocant(0, str(left_size - interface_atom_count + index + 1))
                for index in range(interface_atom_count)
            ),
            host_path=tuple(ComponentLocant(1, str(index + 1)) for index in range(interface_atom_count)),
            cited_attached_locants=tuple(
                ComponentLocant(0, str(left_size - interface_atom_count + index + 1))
                for index in range(interface_atom_count)
            ),
            host_sides=tuple(FusionSide(1, chr(ord("a") + index)) for index in range(interface_atom_count - 1)),
            ordered_input_atoms=shared,
            ordered_input_edges=tuple(_edge(left, right) for left, right in zip(shared, shared[1:])),
            ordered_input_bonds=tuple(_bond_id(mol, left, right) for left, right in zip(shared, shared[1:])),
        ),
    )
    ast = FusionNameAst(
        plan_kind="two_component",
        parent_occurrences=(1,),
        component_occurrences=(left_match, right_match),
        joins=(join,),
        citation_tree=FusionCitationNode(1, children=(FusionCitationNode(0),)),
        descriptors=(FusionDescriptor.from_interface(join.interface),),
    )

    locant_map = _completed_locants(bounded.outer_boundary.atoms, frozenset(shared))
    next_locant = max(locant.base for locant in locant_map.values()) + 1
    for atom in sorted(set(mol.atoms) - locant_map.keys()):
        locant_map[atom] = SystemLocant(next_locant, interior_distance=1)
        next_locant += 1
    numbering = FusionNumberingProof(
        selected_face_model=face_model,
        selected_layout=FusedLayout(face_positions=((faces[0].id, 0, 0), (faces[1].id, 4, 0))),
        orientation_score=(),
        abstract_atom_to_locant=tuple(sorted(locant_map.items())),
        input_locant_maps=(tuple(sorted(locant_map.items())),),
    )
    graph_edges = frozenset(_edge(bond.u, bond.v) for bond in mol.bonds.values())
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in sorted(mol.atoms)),
        bonds=tuple(FusionGraphBond(edge, "single") for edge in sorted(graph_edges)),
    )
    bond_model = ParentBondModel(
        allowed_kekule_assignments=(BondAssignment(tuple((edge, 1) for edge in sorted(graph_edges))),),
        required_single_bonds=graph_edges,
        pi_eligible_edges=frozenset(),
        maximum_non_cumulative_double_bonds=0,
    )
    return _Candidate(
        mol=mol,
        parent_atoms=frozenset(mol.atoms),
        ast=ast,
        graph=graph,
        numbering=numbering,
        bond_model=bond_model,
        registry=specs,
    )


def _ring_spec(key: str, size: int) -> FusionComponentSpec:
    locants = tuple(str(index) for index in range(1, size + 1))
    template = RetainedGraphTemplate(
        name=f"{key}-parent",
        pin=True,
        priority=0,
        aliases=(),
        attached_prefix=f"{key}o",
        derivative_stem=None,
        default_indicated_h=(),
        locants=locants,
        atoms=tuple(ComponentAtom(locant, "C", aromatic=False, saturated=True) for locant in locants),
        bonds=tuple(ComponentBond((left, right), "single") for left, right in zip(locants, locants[1:] + locants[:1])),
        rings=(locants,),
        fusion_atoms=locants,
        peripheral_atoms=locants,
        interior_atoms=(),
    )
    return FusionComponentSpec(
        key=key,
        parent_name=f"{key}-parent",
        attached_prefix=f"{key}o",
        template=template,
        usable_as_parent=True,
        usable_as_attached=True,
        rule_reference="P-25",
    )


def _match(occurrence: int, key: str, cycle: tuple[int, ...], face_id: int) -> FusionComponentMatch:
    mapping = tuple((str(index), atom) for index, atom in enumerate(cycle, start=1))
    return FusionComponentMatch(
        occurrence_id=occurrence,
        spec_key=key,
        covered_face_ids=frozenset({face_id}),
        local_to_input_atom=mapping,
        local_to_skeleton_atom=mapping,
        topology_key=(len(cycle),),
    )


def _completed_locants(perimeter: tuple[int, ...], fusion_atoms: frozenset[int]) -> dict[int, SystemLocant]:
    result = {}
    integer = 0
    for atom in perimeter:
        if atom in fusion_atoms:
            result[atom] = SystemLocant(integer, "a")
        else:
            integer += 1
            result[atom] = SystemLocant(integer)
    return result


def _audit(candidate: _Candidate, **changes):
    arguments = {
        "ast": candidate.ast,
        "abstract_parent_graph": candidate.graph,
        "numbering": candidate.numbering,
        "bond_model": candidate.bond_model,
        "mode": FusionMode.GENERAL,
        "registry": candidate.registry,
    }
    arguments.update(changes)
    return audit_fusion_plan(candidate.mol, candidate.parent_atoms, **arguments)


def _bond_id(mol: Molecule, left: int, right: int) -> int:
    bond = mol.get_bond(left, right)
    assert bond is not None
    return bond.idx


def _edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def test_audit_confirms_independent_component_join_numbering_and_bond_reconstruction():
    candidate = _two_fused_rings()

    result = _audit(candidate, mode=FusionMode.AUDITED_PIN)

    assert result.status is AuditStatus.CONFIRMED
    assert result.checks == (
        "pin_ring_size_gate",
        "pin_component_policy",
        "nomenclature_selection",
        "component_coverage",
        "descriptor_interfaces",
        "abstract_graph_reconstruction",
        "input_graph_identity",
        "completed_numbering",
        "charge_operations",
        "parent_bond_model",
        "indicated_hydrogens",
        "lambda_descriptors",
    )
    assert result.errors == ()


def test_audit_rejects_a_rendered_parent_that_does_not_reproduce_the_ast():
    candidate = _two_fused_rings()

    result = _audit(candidate, rendered_core_name="wrong-parent")

    assert result.status is AuditStatus.MISMATCH
    assert "context_free_rendering" in result.checks
    assert "rendered fusion parent does not reproduce its context-free AST" in result.errors


def test_audit_confirms_graph_built_ortho_peri_interface_without_descriptor_inference():
    candidate = _two_fused_rings(interface_atom_count=3)
    join = candidate.ast.joins[0]

    result = _audit(candidate)

    assert result.status is AuditStatus.CONFIRMED
    assert result.errors == ()
    assert join.kind is FusionJoinKind.ORTHO_PERI
    assert tuple(side.letter for side in join.interface.host_sides) == ("a", "b")
    assert candidate.ast.descriptors[0].render() == "[4,5,6-ab]"


def test_audit_reconstructs_every_interface_of_a_cyclic_component_cover():
    cycles = ((0, 1, 2, 3), (1, 4, 5, 2), (2, 5, 6, 3))
    mol = Molecule()
    for atom in range(7):
        mol.add_atom("C", idx=atom)
    for cycle in cycles:
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            if mol.get_bond(left, right) is None:
                mol.add_bond(left, right, idx=100 + len(mol.bonds))

    spec = _ring_spec("ring", 4)
    matches = tuple(
        FusionComponentMatch(
            occurrence_id=occurrence,
            spec_key="ring",
            covered_face_ids=frozenset({occurrence}),
            local_to_input_atom=tuple(zip(spec.locants, cycle, strict=True)),
            local_to_skeleton_atom=tuple(zip(spec.locants, cycle, strict=True)),
            topology_key=(4,),
        )
        for occurrence, cycle in enumerate(cycles)
    )

    def interface(attached, host, atoms, attached_locants, host_locants, *, higher=False):
        edges = tuple(_edge(left, right) for left, right in zip(atoms, atoms[1:]))
        evidence = OrderedFusionInterface(
            kind=FusionJoinKind.HIGHER_ORDER if higher else FusionJoinKind.ORTHO,
            attached_occurrence=attached,
            host_occurrence=host,
            attached_path=tuple(ComponentLocant(attached, value) for value in attached_locants),
            host_path=tuple(ComponentLocant(host, value) for value in host_locants),
            cited_attached_locants=tuple(ComponentLocant(attached, value) for value in attached_locants),
            host_sides=() if higher else (FusionSide(host, chr(ord("a") + int(host_locants[0]) - 1)),),
            host_locants=(tuple(ComponentLocant(host, value) for value in host_locants) if higher else ()),
            ordered_input_atoms=atoms,
            ordered_input_edges=edges,
            ordered_input_bonds=tuple(_bond_id(mol, *edge) for edge in edges),
        )
        return FusionJoin(order=2 if higher else 1, interface=evidence)

    joins = (
        interface(1, 0, (1, 2), ("1", "4"), ("2", "3")),
        interface(2, 0, (2, 3), ("1", "4"), ("3", "4")),
        interface(2, 1, (2, 5), ("1", "2"), ("4", "3"), higher=True),
    )
    tree = FusionCitationNode(0, (FusionCitationNode(1), FusionCitationNode(2)))
    citation = FusionCitationPlan(
        roots=(tree,),
        primary_join_indices=(0, 1),
        cycle_closing_join_indices=(2,),
        render_order=(1, 2),
    )
    ast = FusionNameAst(
        plan_kind="cyclic_component_cover",
        parent_occurrences=(0,),
        component_occurrences=matches,
        joins=joins,
        citation_tree=tree,
        descriptors=tuple(FusionDescriptor.from_interface(join.interface) for join in joins),
        citation_plan=citation,
    )
    edge_owners = {
        _bond_id(mol, left, right): tuple(
            occurrence
            for occurrence, cycle in enumerate(cycles)
            if _edge(left, right) in {_edge(a, b) for a, b in zip(cycle, cycle[1:] + cycle[:1])}
        )
        for left, right in {_edge(bond.u, bond.v) for bond in mol.bonds.values()}
    }
    face_model = FaceModel(
        faces=tuple(
            Face(
                occurrence,
                cycle,
                tuple(_bond_id(mol, left, right) for left, right in zip(cycle, cycle[1:] + cycle[:1])),
                4,
            )
            for occurrence, cycle in enumerate(cycles)
        ),
        edge_to_faces=tuple(sorted(edge_owners.items())),
        perimeter_edges=frozenset(edge for edge, owners in edge_owners.items() if len(owners) == 1),
        fusion_edges=frozenset(edge for edge, owners in edge_owners.items() if len(owners) == 2),
        outer_boundary=(0, 1, 4, 5, 6, 3),
        face_adjacency=tuple(
            sorted((owners[0], owners[1], edge) for edge, owners in edge_owners.items() if len(owners) == 2)
        ),
    )
    fusion_atoms = {1, 2, 3, 5}
    locants = {atom: SystemLocant(atom + 1, "a" if atom in fusion_atoms else "") for atom in mol.atoms}
    numbering = FusionNumberingProof(
        selected_face_model=face_model,
        selected_layout=FusedLayout(face_positions=((0, 0, 0), (1, 4, 0), (2, 2, 4))),
        orientation_score=(),
        abstract_atom_to_locant=tuple(locants.items()),
        input_locant_maps=(tuple(locants.items()),),
    )
    graph_edges = frozenset(_edge(bond.u, bond.v) for bond in mol.bonds.values())
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in mol.atoms),
        bonds=tuple(FusionGraphBond(edge, "single") for edge in graph_edges),
    )
    bond_model = ParentBondModel(
        allowed_kekule_assignments=(BondAssignment(tuple((edge, 1) for edge in graph_edges)),),
        required_single_bonds=graph_edges,
        pi_eligible_edges=frozenset(),
        maximum_non_cumulative_double_bonds=0,
    )

    result = audit_fusion_plan(
        mol,
        mol.atoms,
        ast=ast,
        abstract_parent_graph=graph,
        numbering=numbering,
        bond_model=bond_model,
        registry={"ring": spec},
    )

    assert result.status is AuditStatus.CONFIRMED, result.errors


def test_audit_rejects_a_descriptor_whose_ordered_interface_is_reversed():
    candidate = _two_fused_rings()
    join = candidate.ast.joins[0]
    corrupted_join = replace(
        join,
        interface=replace(
            join.interface,
            attached_path=tuple(reversed(join.interface.attached_path)),
        ),
    )
    corrupted_ast = replace(
        candidate.ast,
        joins=(corrupted_join,),
        descriptors=(FusionDescriptor.from_interface(corrupted_join.interface),),
    )

    result = _audit(candidate, ast=corrupted_ast)

    assert result.status is AuditStatus.MISMATCH
    assert any("ordered input atoms" in error for error in result.errors)


def test_audit_rejects_a_wrong_parent_side_letter():
    candidate = _two_fused_rings()
    join = candidate.ast.joins[0]
    original_side = join.interface.host_sides[0]
    shifted_letter = chr(ord("a") + ((ord(original_side.letter) - ord("a") + 1) % 6))
    corrupted_join = replace(
        join,
        interface=replace(
            join.interface,
            host_sides=(replace(original_side, letter=shifted_letter),),
        ),
    )
    corrupted_ast = replace(
        candidate.ast,
        joins=(corrupted_join,),
        descriptors=(FusionDescriptor.from_interface(corrupted_join.interface),),
    )

    result = _audit(candidate, ast=corrupted_ast)

    assert result.status is AuditStatus.MISMATCH
    assert any("host sides disagree" in error for error in result.errors)


def test_audit_rejects_an_incomplete_ortho_peri_parent_side_path():
    candidate = _two_fused_rings(left_size=6, right_size=6, interface_atom_count=3)
    join = candidate.ast.joins[0]
    assert join.kind is FusionJoinKind.ORTHO_PERI
    assert len(join.interface.host_sides) == 2

    with pytest.raises(ValueError, match="two or more ordered shared sides"):
        replace(join.interface, host_sides=join.interface.host_sides[:1])


def test_audit_rejects_a_wrong_multiplicative_count():
    mol = read_smiles("O1C=CC2=C1C=C1C(=N2)C=CO1")
    planned = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert isinstance(planned, FusionConfirmed)
    plan = planned.plan
    group = plan.ast.multiplicative_groups[0]
    corrupted_ast = replace(
        plan.ast,
        multiplicative_groups=(replace(group, multiplier="tri"),),
    )

    result = audit_fusion_plan(
        mol,
        mol.atoms,
        ast=corrupted_ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        mode=FusionMode.GENERAL,
    )

    assert result.status is AuditStatus.MISMATCH
    assert "multiplicative groups do not match exact sibling interface orbits" in result.errors


def test_audit_rejects_a_wrong_multiplicative_prime_depth():
    mol = read_smiles("O1C=CC2=C1C=C1C(=N2)C=CO1")
    planned = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert isinstance(planned, FusionConfirmed)
    plan = planned.plan
    group = plan.ast.multiplicative_groups[0]
    primed_occurrence = group.occurrence_ids[1]
    joins = list(plan.ast.joins)
    position = next(index for index, join in enumerate(joins) if join.attached_occurrence == primed_occurrence)
    join = joins[position]
    corrupted_interface = replace(
        join.interface,
        attached_path=tuple(replace(locant, prime_depth=0) for locant in join.interface.attached_path),
        cited_attached_locants=tuple(
            replace(locant, prime_depth=0) for locant in join.interface.cited_attached_locants
        ),
    )
    joins[position] = replace(join, interface=corrupted_interface)
    corrupted_ast = replace(
        plan.ast,
        joins=tuple(joins),
        descriptors=tuple(FusionDescriptor.from_interface(item.interface) for item in joins),
    )

    result = audit_fusion_plan(
        mol,
        mol.atoms,
        ast=corrupted_ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        mode=FusionMode.GENERAL,
    )

    assert result.status is AuditStatus.MISMATCH
    assert any("prime depth" in error for error in result.errors)


def test_audit_rejects_nonidentical_components_under_one_multiplier():
    mol = read_smiles("O1C=CC2=NC3=C(C=C21)SC=C3")
    planned = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert isinstance(planned, FusionConfirmed)
    plan = planned.plan
    attached = tuple(
        occurrence.occurrence_id
        for occurrence in plan.ast.component_occurrences
        if occurrence.occurrence_id not in plan.ast.parent_occurrences
    )
    assert len(attached) == 2
    assert (
        len(
            {
                occurrence.spec_key
                for occurrence in plan.ast.component_occurrences
                if occurrence.occurrence_id in attached
            }
        )
        == 2
    )
    corrupted_ast = replace(
        plan.ast,
        plan_kind="multiplicative_tree",
        multiplicative_groups=(FusionMultiplicityGroup(attached, "di"),),
    )

    result = audit_fusion_plan(
        mol,
        mol.atoms,
        ast=corrupted_ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        mode=FusionMode.GENERAL,
    )

    assert result.status is AuditStatus.MISMATCH
    assert "multiplicative groups do not match exact sibling interface orbits" in result.errors


def test_audit_rejects_a_non_senior_declared_parent():
    candidate = _two_fused_rings()
    explicit = replace(
        candidate.ast,
        citation_plan=FusionCitationPlan.from_tree(
            candidate.ast.citation_tree,
            candidate.ast.joins,
        ),
    )
    corrupted = replace(
        explicit,
        parent_occurrences=(0,),
        citation_tree=FusionCitationNode(0, children=(FusionCitationNode(1),)),
    )

    result = _audit(candidate, ast=corrupted)

    assert result.status is AuditStatus.MISMATCH
    assert "fusion citation roots do not match the declared parent occurrences" in result.errors
    assert "declared fusion parent is not the intrinsically senior eligible component" in result.errors


def test_audit_rejects_element_and_formal_charge_graph_mismatch():
    candidate = _two_fused_rings()
    atoms = list(candidate.graph.atoms)
    atoms[0] = FusionGraphAtom(atoms[0].id, "N", formal_charge=1)

    result = _audit(candidate, abstract_parent_graph=replace(candidate.graph, atoms=tuple(atoms)))

    assert result.status is AuditStatus.MISMATCH
    assert any("declared abstract parent graph" in error for error in result.errors)


def test_audit_rejects_incomplete_or_duplicate_component_face_coverage():
    candidate = _two_fused_rings()
    left, right = candidate.ast.component_occurrences
    corrupted_left = replace(left, covered_face_ids=right.covered_face_ids)
    corrupted_ast = replace(candidate.ast, component_occurrences=(corrupted_left, right))

    result = _audit(candidate, ast=corrupted_ast)

    assert result.status is AuditStatus.MISMATCH
    assert "component occurrences do not cover every selected face exactly once" in result.errors


def test_audit_rejects_shared_interface_metadata_that_disagrees_with_the_graph():
    candidate = _two_fused_rings()
    original = candidate.ast.joins[0]
    join = replace(
        original,
        interface=replace(original.interface, ordered_input_bonds=(999,)),
    )
    corrupted_ast = replace(candidate.ast, joins=(join,))

    result = _audit(candidate, ast=corrupted_ast)

    assert result.status is AuditStatus.MISMATCH
    assert any("stores wrong shared bonds" in error for error in result.errors)


def test_audit_rejects_a_non_graph_preserving_completed_numbering_map():
    candidate = _two_fused_rings()
    locants = dict(candidate.numbering.input_locant_maps[0])
    fusion_atom = next(atom for atom in candidate.parent_atoms if len(candidate.mol.get_neighbors(atom)) == 3)
    nonfusion_atom = next(atom for atom in candidate.parent_atoms if len(candidate.mol.get_neighbors(atom)) == 2)
    locants[fusion_atom], locants[nonfusion_atom] = locants[nonfusion_atom], locants[fusion_atom]
    corrupted = replace(candidate.numbering, input_locant_maps=(tuple(sorted(locants.items())),))

    result = _audit(candidate, numbering=corrupted)

    assert result.status is AuditStatus.MISMATCH
    assert any("not graph preserving" in error for error in result.errors)


def test_audit_rejects_layout_that_does_not_match_completed_numbering():
    mol = read_smiles("O1C2=C(C=C1)C=CS2")
    planned = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert isinstance(planned, FusionConfirmed)
    plan = planned.plan
    positions = list(plan.numbering.selected_layout.atom_positions)
    atom, _, _ = positions[1]
    _, duplicate_x, duplicate_y = positions[0]
    positions[1] = (atom, duplicate_x, duplicate_y)
    bad_layout = replace(plan.numbering.selected_layout, atom_positions=tuple(positions))
    bad_numbering = replace(plan.numbering, selected_layout=bad_layout)

    result = audit_fusion_plan(
        mol,
        mol.atoms,
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=bad_numbering,
        bond_model=plan.bond_model,
        mode=FusionMode.GENERAL,
    )

    assert result.status is AuditStatus.MISMATCH
    assert "selected layout assigns the same position to multiple parent atoms" in result.errors


def test_audit_rejects_parent_bond_model_that_cannot_describe_the_input():
    candidate = _two_fused_rings()
    edges = tuple(sorted(_edge(*bond.atoms) for bond in candidate.graph.bonds))
    orders = tuple((edge, 2 if position == 0 else 1) for position, edge in enumerate(edges))
    corrupted = ParentBondModel(
        allowed_kekule_assignments=(BondAssignment(orders),),
        required_single_bonds=frozenset(edges[1:]),
        pi_eligible_edges=frozenset({edges[0]}),
        maximum_non_cumulative_double_bonds=1,
    )

    result = _audit(candidate, bond_model=corrupted)

    assert result.status is AuditStatus.MISMATCH
    assert "selected input bond orders are not allowed by the parent bond model" in result.errors


def test_pin_mode_abstains_when_only_one_ring_has_at_least_five_atoms():
    candidate = _two_fused_rings(6, 4)

    result = _audit(candidate, mode=FusionMode.AUDITED_PIN)

    assert result.status is AuditStatus.ABSTAIN
    assert result.checks == ("pin_ring_size_gate",)
    assert result.errors == ("fewer than two rings of size at least five",)


def test_missing_component_policy_is_an_explicit_audit_error():
    candidate = _two_fused_rings()

    result = _audit(candidate, registry={})

    assert result.status is AuditStatus.ERROR
    assert result.errors == ("'unknown fusion component spec: left'",)

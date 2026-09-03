"""Graph-native tests for independent systematic-fusion auditing."""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    FusionComponentMatch,
    FusionComponentSpec,
    FusionDescriptor,
    FusionGraph,
    FusionGraphAtom,
    FusionGraphBond,
    FusionJoin,
    FusionJoinKind,
    FusionMode,
    FusionNameAst,
    FusionNumberingProof,
    FusionSide,
    ParentBondModel,
    SystemLocant,
)
from openclatura.molecule import Molecule


@dataclass(frozen=True)
class _Candidate:
    mol: Molecule
    parent_atoms: frozenset[int]
    ast: FusionNameAst
    graph: FusionGraph
    numbering: FusionNumberingProof
    bond_model: ParentBondModel
    registry: dict[str, FusionComponentSpec]


def _two_fused_rings(left_size: int = 6, right_size: int = 6) -> _Candidate:
    """Build two carbocycles sharing one edge, without line notation."""

    shared = (left_size - 2, left_size - 1)
    left_cycle = tuple(range(left_size))
    right_cycle = shared + tuple(range(left_size, left_size + right_size - 2))

    mol = Molecule()
    for atom in range(left_size + right_size - 2):
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

    specs = {"left": _ring_spec("left", left_size), "right": _ring_spec("right", right_size)}
    face_by_atoms = {frozenset(face.atom_cycle): face.id for face in faces}
    left_match = _match(0, "left", left_cycle, face_by_atoms[frozenset(left_cycle)])
    right_match = _match(1, "right", right_cycle, face_by_atoms[frozenset(right_cycle)])
    join = FusionJoin(
        attached_occurrence=0,
        host_occurrence=1,
        order=1,
        kind=FusionJoinKind.ORTHO,
        attached_locants=(ComponentLocant(0, str(left_size - 1)), ComponentLocant(0, str(left_size))),
        host_sides=(FusionSide(1, "a"),),
        shared_input_atoms=frozenset(shared),
        shared_input_bonds=frozenset({_bond_id(mol, *shared)}),
    )
    ast = FusionNameAst(
        plan_kind="two_component",
        parent_occurrences=(1,),
        component_occurrences=(left_match, right_match),
        joins=(join,),
        citation_tree=FusionCitationNode(1, children=(FusionCitationNode(0),)),
        descriptors=(FusionDescriptor(join.attached_locants, parent_sides=join.host_sides),),
    )

    locant_map = _completed_locants(bounded.outer_boundary.atoms, frozenset(shared))
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
    return FusionComponentSpec(
        key=key,
        parent_name=f"{key}-parent",
        attached_prefix=f"{key}o",
        derivative_stem=None,
        locants=locants,
        atoms=tuple(ComponentAtom(locant, "C", pi_capacity=0, forced_single=True) for locant in locants),
        bonds=tuple(ComponentBond((left, right), "single") for left, right in zip(locants, locants[1:] + locants[:1])),
        rings=(locants,),
        peripheral_order=locants,
        usable_as_parent=True,
        usable_as_attached=True,
        pin_component=True,
        retained_complete_name=False,
        benzoheterocycle=False,
        traditional_numbering=False,
        ring_sizes=(size,),
        fusion_carbon_locants=locants,
        preferred_layouts=(),
        seniority_override=None,
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
        "component_coverage",
        "descriptor_interfaces",
        "abstract_graph_reconstruction",
        "input_graph_identity",
        "completed_numbering",
        "parent_bond_model",
    )
    assert result.errors == ()


def test_audit_rejects_a_descriptor_whose_ordered_interface_is_reversed():
    candidate = _two_fused_rings()
    join = candidate.ast.joins[0]
    corrupted_join = replace(join, attached_locants=tuple(reversed(join.attached_locants)))
    corrupted_ast = replace(
        candidate.ast,
        joins=(corrupted_join,),
        descriptors=(FusionDescriptor(corrupted_join.attached_locants, parent_sides=corrupted_join.host_sides),),
    )

    result = _audit(candidate, ast=corrupted_ast)

    assert result.status is AuditStatus.MISMATCH
    assert any("ordered input atoms" in error for error in result.errors)


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
    join = replace(candidate.ast.joins[0], shared_input_bonds=frozenset({999}))
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

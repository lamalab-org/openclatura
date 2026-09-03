import pytest

from openclatura.fusion.faces import (
    FaceSearchBudgetExceeded,
    GraphCycle,
    audit_bounded_face_model,
    enumerate_chordless_cycles,
    select_bounded_face_model,
)
from openclatura.molecule import Molecule


def _molecule(edges: list[tuple[int, int]]) -> Molecule:
    mol = Molecule()
    for atom in sorted({value for edge in edges for value in edge}):
        mol.add_atom("C", idx=atom)
    for index, (left, right) in enumerate(edges, start=1):
        mol.add_bond(left, right, idx=index)
    return mol


def _linear_fused_hexagons() -> Molecule:
    # Two six-membered faces share edge 4-5; the outside is a ten-cycle.
    return _molecule(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (4, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 5),
        ]
    )


def _molecule_from_faces(*faces: tuple[int, ...]) -> Molecule:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        edges.update(
            tuple(sorted((left, right)))
            for left, right in zip(face, face[1:] + face[:1], strict=True)
        )
    return _molecule(sorted(edges))


@pytest.mark.parametrize(
    ("faces", "expected_sizes", "expected_outer_size", "expected_fusion_edges", "expected_dual"),
    [
        (
            (
                (0, 1, 2, 3, 4, 5),
                (3, 4, 6, 7, 8, 13),
                (8, 9, 10, 11, 12, 13),
            ),
            (6, 6, 6),
            14,
            2,
            ((0, (1,)), (1, (0, 2)), (2, (1,))),
        ),
        (
            ((0, 1, 2, 3, 9), (3, 4, 5, 6, 7, 8, 9)),
            (5, 7),
            10,
            1,
            ((0, (1,)), (1, (0,))),
        ),
        (
            (
                (0, 1, 2, 3, 14, 13),
                (3, 4, 5, 6, 15, 14),
                (6, 7, 8, 9, 10, 15),
                (10, 11, 12, 13, 14, 15),
            ),
            (6, 6, 6, 6),
            14,
            5,
            ((0, (1, 3)), (1, (0, 2, 3)), (2, (1, 3)), (3, (0, 1, 2))),
        ),
    ],
)
def test_representative_fused_face_models_are_complete_and_deterministic(
    faces,
    expected_sizes,
    expected_outer_size,
    expected_fusion_edges,
    expected_dual,
):
    mol = _molecule_from_faces(*faces)

    model = select_bounded_face_model(mol, reversed(tuple(mol.atoms)))

    assert model is not None
    assert model.audit.ok
    assert tuple(sorted(len(face.atoms) for face in model.faces)) == expected_sizes
    assert len(model.outer_boundary.atoms) == expected_outer_size
    assert sum(count == 2 for _, count in model.audit.edge_multiplicity) == expected_fusion_edges
    assert model.audit.dual_adjacency == expected_dual
    assert model.audit.reconstructed_edges == model.edge_ids


def test_pericondensed_face_model_exposes_interior_atoms_for_the_production_gate():
    mol = _molecule_from_faces(
        (0, 1, 2, 3, 14, 13),
        (3, 4, 5, 6, 15, 14),
        (6, 7, 8, 9, 10, 15),
        (10, 11, 12, 13, 14, 15),
    )

    model = select_bounded_face_model(mol, mol.atoms)

    assert model is not None
    assert set(model.outer_boundary.atoms) < set(model.atom_ids)
    assert set(model.atom_ids) - set(model.outer_boundary.atoms) == {14, 15}


def test_chordless_cycle_enumeration_is_deterministic_and_excludes_outer_chorded_cycle():
    mol = _linear_fused_hexagons()

    cycles = enumerate_chordless_cycles(mol, reversed(range(10)))

    assert [cycle.atoms for cycle in cycles] == [(0, 1, 2, 3, 4, 5), (4, 5, 9, 8, 7, 6)]


def test_bounded_face_model_proves_fused_edge_and_outer_boundary():
    mol = _linear_fused_hexagons()

    model = select_bounded_face_model(mol, range(10))

    assert model is not None
    assert model.audit.ok
    assert model.cycle_rank == 2
    assert model.outer_boundary.atoms == (0, 1, 2, 3, 4, 6, 7, 8, 9, 5)
    assert dict(model.audit.edge_multiplicity)[(4, 5)] == 2
    assert model.audit.dual_adjacency == ((0, (1,)), (1, (0,)))
    assert model.audit.reconstructed_edges == model.edge_ids


def test_bounded_face_selection_is_invariant_to_candidate_cycle_order(monkeypatch):
    mol = _linear_fused_hexagons()
    expected = select_bounded_face_model(mol, range(10))
    assert expected is not None
    from openclatura.fusion import faces as faces_module

    enumerate_original = faces_module.enumerate_chordless_cycles

    def reversed_candidates(*args, **kwargs):
        return tuple(reversed(enumerate_original(*args, **kwargs)))

    monkeypatch.setattr(faces_module, "enumerate_chordless_cycles", reversed_candidates)
    actual = select_bounded_face_model(mol, range(10))

    assert actual is not None
    assert {face.edges for face in actual.faces} == {face.edges for face in expected.faces}
    assert actual.outer_boundary.edges == expected.outer_boundary.edges
    assert actual.audit.reconstructed_edges == expected.audit.reconstructed_edges


def test_face_selection_ranks_only_the_selected_parent_subgraph():
    mol = _linear_fused_hexagons()
    parent_atoms = frozenset(mol.atoms)
    expected = select_bounded_face_model(mol, parent_atoms)
    mol.add_atom("Cl", idx=20)
    mol.add_bond(0, 20, idx=20)

    actual = select_bounded_face_model(mol, parent_atoms)

    assert expected is not None and actual is not None
    assert {face.edges for face in actual.faces} == {face.edges for face in expected.faces}
    assert actual.outer_boundary.edges == expected.outer_boundary.edges


def test_face_audit_rejects_incomplete_and_overcovered_models():
    mol = _linear_fused_hexagons()
    left = GraphCycle.from_atoms((0, 1, 2, 3, 4, 5))
    right = GraphCycle.from_atoms((4, 5, 9, 8, 7, 6))

    incomplete = audit_bounded_face_model(mol, range(10), (left,))
    overcovered = audit_bounded_face_model(mol, range(10), (left, right, left))

    assert not incomplete.ok
    assert "bounded faces do not exactly reconstruct the ring-graph edges" in incomplete.errors
    assert not overcovered.ok
    assert "a ring-graph edge belongs to more than two bounded faces" in overcovered.errors


def test_nonplanar_or_bridged_graph_without_simple_xor_boundary_is_not_accepted():
    # K3,3 cannot supply a planar bounded-face model satisfying Euler's cycle
    # rank together with exact edge reconstruction and a simple XOR boundary.
    mol = _molecule([(left, right) for left in range(3) for right in range(3, 6)])

    assert select_bounded_face_model(mol, range(6)) is None


def test_cycle_enumeration_budget_exhaustion_is_explicit():
    mol = _linear_fused_hexagons()

    with pytest.raises(FaceSearchBudgetExceeded, match="cycle enumeration"):
        enumerate_chordless_cycles(mol, range(10), search_budget=1)


@pytest.mark.parametrize("minimum,maximum", [(2, 6), (3, 9), (7, 6)])
def test_cycle_size_contract_is_limited_to_first_production_slice(minimum: int, maximum: int):
    with pytest.raises(ValueError, match="3 <= min_size <= max_size <= 8"):
        enumerate_chordless_cycles(_molecule([(0, 1), (1, 2), (2, 0)]), min_size=minimum, max_size=maximum)

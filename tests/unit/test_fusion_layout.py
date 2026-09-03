import pytest

from openclatura.fusion.layout import (
    RING_SHAPE_TEMPLATES,
    LayoutSearchBudgetExceeded,
    intrinsic_fused_layouts,
    preferred_intrinsic_layout,
)
from openclatura.fusion.model import Face, FaceModel


def _fused_faces(ring_size: int, permutation: dict[int, int] | None = None) -> FaceModel:
    first = tuple(range(ring_size))
    second = (ring_size - 2, ring_size - 1, *range(ring_size, 2 * ring_size - 2))
    mapping = permutation or {atom: atom for atom in set(first + second)}
    cycles = tuple(tuple(mapping[atom] for atom in cycle) for cycle in (first, second))
    edge_ids: dict[frozenset[int], int] = {}

    def edges(cycle: tuple[int, ...]) -> tuple[int, ...]:
        result = []
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            key = frozenset((left, right))
            result.append(edge_ids.setdefault(key, len(edge_ids) + 1))
        return tuple(result)

    face_edges = tuple(edges(cycle) for cycle in cycles)
    shared = set(face_edges[0]) & set(face_edges[1])
    assert len(shared) == 1
    shared_edge = shared.pop()
    owners = {
        edge: tuple(face_id for face_id, cycle_edges in enumerate(face_edges) if edge in cycle_edges)
        for edge in edge_ids.values()
    }
    perimeter = frozenset(edge for edge, face_ids in owners.items() if len(face_ids) == 1)
    outer = (
        mapping[ring_size - 2],
        *(mapping[atom] for atom in reversed(range(ring_size - 2))),
        mapping[ring_size - 1],
        *(mapping[atom] for atom in range(ring_size, 2 * ring_size - 2)),
    )
    return FaceModel(
        faces=tuple(
            Face(id=face_id, atom_cycle=cycle, edge_cycle=face_edges[face_id], size=ring_size)
            for face_id, cycle in enumerate(cycles)
        ),
        edge_to_faces=tuple(sorted(owners.items())),
        perimeter_edges=perimeter,
        fusion_edges=frozenset((shared_edge,)),
        outer_boundary=outer,
        face_adjacency=((0, 1, shared_edge),),
    )


def _geometry_signature(layout) -> tuple:
    return (
        layout.orientation_score,
        tuple(sorted(shape for _, shape in layout.face_shapes)),
        tuple(sorted((x, y) for _, x, y in layout.atom_positions)),
    )


def test_shape_registry_covers_every_supported_ring_size_with_exact_templates():
    assert {shape.ring_size for shape in RING_SHAPE_TEMPLATES} == set(range(3, 9))
    assert all(shape.vertices[:2] == ((0, 0), (4, 0)) for shape in RING_SHAPE_TEMPLATES)
    assert {shape.ring_size for shape in RING_SHAPE_TEMPLATES if shape.distortion_rank} == {5, 7}


@pytest.mark.parametrize("ring_size", range(3, 9))
def test_two_ortho_fused_supported_faces_receive_an_audited_layout(ring_size: int):
    model = _fused_faces(ring_size)

    layout = preferred_intrinsic_layout(model)

    assert layout is not None
    assert {atom for atom, _, _ in layout.atom_positions} == {atom for face in model.faces for atom in face.atom_cycle}
    assert {face for face, _, _ in layout.face_positions} == {0, 1}
    assert {face for face, _ in layout.face_shapes} == {0, 1}
    assert layout.audit_evidence == (
        "all face boundaries represented",
        "shared edge coordinates agree",
        "unrelated edges do not cross",
        "nonadjacent face interiors do not overlap",
        "geometric and topological perimeters agree",
    )


def test_layout_preference_is_invariant_to_input_atom_ids():
    original = _fused_faces(6)
    atom_ids = sorted({atom for face in original.faces for atom in face.atom_cycle})
    permutation = dict(zip(atom_ids, reversed([atom * 7 + 3 for atom in atom_ids])))
    renumbered = _fused_faces(6, permutation)

    left = preferred_intrinsic_layout(original)
    right = preferred_intrinsic_layout(renumbered)

    assert left is not None and right is not None
    assert _geometry_signature(left) == _geometry_signature(right)


def test_inconsistent_face_adjacency_abstains():
    model = _fused_faces(6)
    inconsistent = FaceModel(
        faces=model.faces,
        edge_to_faces=model.edge_to_faces,
        perimeter_edges=model.perimeter_edges,
        fusion_edges=model.fusion_edges,
        outer_boundary=model.outer_boundary,
        face_adjacency=((0, 1, min(model.perimeter_edges)),),
    )

    assert preferred_intrinsic_layout(inconsistent) is None


def test_layout_search_budget_exhaustion_is_explicit():
    with pytest.raises(LayoutSearchBudgetExceeded, match="intrinsic layout search"):
        intrinsic_fused_layouts(_fused_faces(6), search_budget=1)


def test_completed_layout_limit_never_returns_a_partial_preference_set():
    with pytest.raises(LayoutSearchBudgetExceeded, match="completed layouts"):
        intrinsic_fused_layouts(_fused_faces(6), max_layouts=1)

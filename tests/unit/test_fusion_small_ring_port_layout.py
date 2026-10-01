"""Small-ring proxies preserve graph ports, orientation axes, and audit bounds."""

from dataclasses import replace

import pytest

from openclatura.fusion import layout as layout_module
from openclatura.fusion.faces import select_bounded_face_model, typed_face_model
from openclatura.fusion.layout import (
    LayoutSearchBudgetExceeded,
    _audit_layout,
    _direction_grid_centers,
    _expand_proxy_layouts,
    _two_port_large_ring_layouts,
    _two_port_pentagon_axes,
    intrinsic_fused_layouts,
    preferred_intrinsic_layouts,
)
from openclatura.molecule import Molecule


def _model(cycles):
    mol = Molecule()
    for atom in sorted({atom for cycle in cycles for atom in cycle}):
        mol.add_atom("C", idx=atom)
    edges = {tuple(sorted((a, b))) for cycle in cycles for a, b in zip(cycle, cycle[1:] + cycle[:1])}
    for index, (a, b) in enumerate(sorted(edges)):
        mol.add_bond(a, b, idx=index)
    bounded = select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None
    return typed_face_model(mol, bounded)


def _angular(size, distance=None):
    distance = size // 2 - 1 if distance is None else distance
    return _model(
        (
            tuple(range(size)),
            (0, 1, size),
            (distance, distance + 1, size + 1, size + 2, size + 3),
        )
    )


def _assert_audited(model, layouts):
    assert layouts
    orders = {face.id: face.atom_cycle for face in model.faces}
    for layout in layouts:
        positions = {atom: (x, y) for atom, x, y in layout.atom_positions}
        assert set(positions) == {atom for face in model.faces for atom in face.atom_cycle}
        assert _audit_layout(model, orders, positions)


@pytest.mark.parametrize("size", (7, 8))
def test_supported_ring_angular_ports_do_not_become_a_straight_row(size):
    model = _angular(size)
    layouts = preferred_intrinsic_layouts(model)
    _assert_audited(model, layouts)
    assert {layout.orientation_score[1] for layout in layouts} == {-2}
    assert all(any(shape.startswith(f"two-port-{size}:") for _, shape in layout.face_shapes) for layout in layouts)


@pytest.mark.parametrize("size", (7, 8))
def test_adjacent_ports_are_not_promoted_to_the_angular_proxy(size):
    model = _angular(size, 1)
    face = next(face for face in model.faces if face.size == size)
    assert _two_port_large_ring_layouts(model, search_budget=10000, max_layouts=1000, face_id=face.id) == ()


@pytest.mark.parametrize("size", (7, 8))
def test_terminal_subdivision_restores_the_complete_original_graph(size):
    model = _model((tuple(range(6)), (2, 3, 6, 7, 8), (7, 8, *range(9, size + 7))))
    layouts = intrinsic_fused_layouts(model)
    _assert_audited(model, layouts)
    assert all(any(shape.startswith(f"terminal-{size}:") for _, shape in layout.face_shapes) for layout in layouts)


@pytest.mark.parametrize("size", (7, 8))
def test_terminal_subdivision_preserves_edge_ownership_with_reversed_entrance(size, monkeypatch):
    model = _model((tuple(range(6)), (2, 3, 6, 7, 8), (7, 8, *range(9, size + 7))))
    original_endpoints = layout_module._edge_endpoints
    original_rebuild = layout_module._subdivided_face_model
    checked = []

    def reversed_entrance(face, edge):
        endpoints = original_endpoints(face, edge)
        return tuple(reversed(endpoints)) if face.size == size else endpoints

    def checked_rebuild(original, faces, removed):
        for face in faces:
            for left, right, edge in zip(face.atom_cycle, face.atom_cycle[1:] + face.atom_cycle[:1], face.edge_cycle):
                owner = next((old for old in original.faces if edge in old.edge_cycle), None)
                if owner is not None:
                    assert frozenset((left, right)) == frozenset(original_endpoints(owner, edge))
        checked.append(True)
        return original_rebuild(original, faces, removed)

    monkeypatch.setattr(layout_module, "_edge_endpoints", reversed_entrance)
    monkeypatch.setattr(layout_module, "_subdivided_face_model", checked_rebuild)
    _assert_audited(model, intrinsic_fused_layouts(model))
    assert checked


@pytest.mark.parametrize("limits", ({"search_budget": 1}, {"max_layouts": 1}))
def test_supported_ring_proxy_keeps_the_existing_hard_limits(limits):
    with pytest.raises(LayoutSearchBudgetExceeded):
        intrinsic_fused_layouts(_angular(8), **limits)


def test_proxy_expansion_rejects_a_corrupted_geometry_witness():
    model = _angular(8)
    layout = intrinsic_fused_layouts(model)[0]
    positions = layout.atom_positions
    corrupted = replace(layout, atom_positions=((positions[0][0], *positions[1][1:]), *positions[1:]))
    path = model.faces[0].atom_cycle[:2]
    assert _expand_proxy_layouts(model, (corrupted,), (path,), {}, ()) == ()


def test_numbering_axes_retain_the_same_relative_grid_used_for_scoring():
    model = _angular(8)
    adjacent = frozenset(frozenset((a, b)) for a, b, _ in model.face_adjacency)
    for layout in intrinsic_fused_layouts(model):
        centers = {face: (x, y) for face, x, y in layout.face_positions}
        grid = _direction_grid_centers(centers, adjacent)
        assert grid is not None
        # Common scaling and translation may differ, but row membership and
        # the first face used by completed numbering must be identical.
        assert max(centers, key=lambda face: (centers[face][1], centers[face][0])) == max(
            grid, key=lambda face: (grid[face][1], grid[face][0])
        )
        assert all((centers[a][1] == centers[b][1]) == (grid[a][1] == grid[b][1]) for a in centers for b in centers)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("offset", range(5))
def test_nonconsecutive_house_ports_have_a_straight_axis_independent_of_cycle_origin(offset, reverse):
    cycle = (0, 1, 2, 3, 4)
    if reverse:
        cycle = tuple(reversed(cycle))
    cycle = cycle[offset:] + cycle[:offset]
    orders = {0: cycle, 1: (0, 1, 5, 6, 7, 8), 2: (2, 3, 9, 10, 11, 12)}
    adjacent = frozenset((frozenset((0, 1)), frozenset((0, 2))))
    centers = {0: (4, 2), 1: (0, 0), 2: (8, 0)}
    assert _two_port_pentagon_axes(orders, centers, adjacent) == {**centers, 0: (4, 0)}
    assert centers[0] == (4, 2)


def test_consecutive_house_ports_do_not_acquire_a_straight_axis():
    orders = {0: (0, 1, 2, 3, 4), 1: (0, 1, 5, 6, 7, 8), 2: (1, 2, 9, 10, 11, 12)}
    adjacent = frozenset((frozenset((0, 1)), frozenset((0, 2))))
    centers = {0: (4, 2), 1: (0, 0), 2: (8, 0)}
    assert _two_port_pentagon_axes(orders, centers, adjacent) == centers


def test_coupled_house_axes_remain_outside_the_independent_projection():
    orders = {0: (0, 1, 2, 3, 4), 1: (2, 3, 5, 6, 7), 2: (0, 1, 8), 3: (5, 6, 9)}
    adjacent = frozenset((frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 3))))
    centers = {0: (0, 0), 1: (4, 0), 2: (-2, 2), 3: (6, 2)}
    assert _two_port_pentagon_axes(orders, centers, adjacent) == centers


def test_house_axis_projection_does_not_collapse_distinct_face_centers():
    orders = {0: (0, 1, 2, 3, 4), 1: (0, 1, 5, 6, 7, 8), 2: (2, 3, 9, 10, 11, 12), 3: (5, 6, 13)}
    adjacent = frozenset((frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 3))))
    centers = {0: (4, 2), 1: (0, 0), 2: (8, 0), 3: (4, 0)}
    assert _two_port_pentagon_axes(orders, centers, adjacent) == centers

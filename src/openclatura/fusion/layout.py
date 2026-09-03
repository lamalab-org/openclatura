"""Intrinsic, graph-derived layouts for bounded fused-ring face models.

The layout search uses exact rational arithmetic and fixed shape templates. It
never reads molecular drawing coordinates. A candidate is exposed only after
its shared edges, graph edges, crossings, overlaps, and topological perimeter
have all been audited.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

from .config import RingShapeSpec, fusion_nomenclature_config
from .model import Face, FaceModel, FusedLayout

Point = tuple[Fraction, Fraction]
Edge = tuple[int, int]


class LayoutSearchBudgetExceeded(RuntimeError):
    """Raised instead of returning a partial intrinsic-layout search."""

    def __init__(self, budget: int, *, resource: str = "states") -> None:
        super().__init__(f"intrinsic layout search exceeded its budget of {budget} {resource}")
        self.budget = budget
        self.resource = resource


@dataclass(slots=True)
class _Budget:
    limit: int
    used: int = 0

    def spend(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise LayoutSearchBudgetExceeded(self.limit)


_CONFIG = fusion_nomenclature_config()
RING_SHAPE_TEMPLATES: tuple[RingShapeSpec, ...] = _CONFIG.ring_shapes
_SHAPES_BY_SIZE = {
    size: tuple(shape for shape in RING_SHAPE_TEMPLATES if shape.ring_size == size)
    for size in range(_CONFIG.search.minimum_ring_size, _CONFIG.search.maximum_ring_size + 1)
}


def intrinsic_fused_layouts(
    model: FaceModel,
    *,
    search_budget: int = _CONFIG.search.layout_states,
    max_layouts: int = _CONFIG.search.maximum_layouts,
) -> tuple[FusedLayout, ...]:
    """Enumerate audited intrinsic layouts in nomenclatural preference order.

    An empty tuple is an explicit abstention: the face model is unsupported or
    inconsistent with the standard 3--8 member shape vocabulary.
    """

    if search_budget < 1 or max_layouts < 1:
        raise ValueError("layout search budget and result limit must be positive")
    if any(face.size not in _SHAPES_BY_SIZE for face in model.faces):
        return ()
    face_by_id = {face.id: face for face in model.faces}
    if not _valid_face_adjacency(model, face_by_id):
        return ()
    budget = _Budget(search_budget)
    completed: dict[tuple, FusedLayout] = {}

    # Enumerating every seed avoids making the selected geometry depend on the
    # input atom IDs used to assign face IDs. The hard state/result budgets keep
    # this bounded for larger fused systems.
    for root in sorted(model.faces, key=lambda face: (face.size, face.id)):
        for shape in _SHAPES_BY_SIZE[root.size]:
            for offset in range(root.size):
                for reverse in (False, True):
                    budget.spend()
                    order = _oriented_cycle(root.atom_cycle, offset, reverse)
                    atom_positions = {atom: (Fraction(x), Fraction(y)) for atom, (x, y) in zip(order, shape.vertices)}
                    _search_layouts(
                        model,
                        face_by_id,
                        {root.id: order},
                        {root.id: shape},
                        atom_positions,
                        budget,
                        completed,
                        max_layouts,
                    )
    return tuple(sorted(completed.values(), key=_layout_sort_key))


def preferred_intrinsic_layout(
    model: FaceModel,
    *,
    search_budget: int = _CONFIG.search.layout_states,
    max_layouts: int = _CONFIG.search.maximum_layouts,
) -> FusedLayout | None:
    """Return the preferred audited layout, or ``None`` to abstain."""

    layouts = preferred_intrinsic_layouts(
        model,
        search_budget=search_budget,
        max_layouts=max_layouts,
    )
    return layouts[0] if layouts else None


def preferred_intrinsic_layouts(
    model: FaceModel,
    *,
    search_budget: int = _CONFIG.search.layout_states,
    max_layouts: int = _CONFIG.search.maximum_layouts,
) -> tuple[FusedLayout, ...]:
    """Return every layout tied on the intrinsic orientation criteria.

    Retaining the tied embeddings is essential: completed-system heteroatom
    locant criteria are applied after preferred orientation and may select a
    reflected embedding without changing the preferred layout score.
    """

    layouts = intrinsic_fused_layouts(
        model,
        search_budget=search_budget,
        max_layouts=max_layouts,
    )
    if not layouts:
        return ()
    best_score = layouts[0].orientation_score
    return tuple(layout for layout in layouts if layout.orientation_score == best_score)


def _search_layouts(
    model: FaceModel,
    face_by_id: dict[int, Face],
    placed_orders: dict[int, tuple[int, ...]],
    placed_shapes: dict[int, RingShapeSpec],
    atom_positions: dict[int, Point],
    budget: _Budget,
    completed: dict[tuple, FusedLayout],
    max_layouts: int,
) -> None:
    if len(placed_orders) == len(model.faces):
        if _audit_layout(model, placed_orders, atom_positions):
            for layout in _materialize_layouts(placed_orders, placed_shapes, atom_positions):
                completed.setdefault(_layout_geometry_key(layout), layout)
            if len(completed) > max_layouts:
                raise LayoutSearchBudgetExceeded(max_layouts, resource="completed layouts")
        return

    next_face, placed_neighbor, shared_edge = _next_face(model, placed_orders)
    if next_face is None or placed_neighbor is None or shared_edge is None:
        return
    face = face_by_id[next_face]
    shared_endpoints = _edge_endpoints(face_by_id[placed_neighbor], shared_edge)
    if shared_endpoints is None or any(atom not in atom_positions for atom in shared_endpoints):
        return
    existing_center = _face_center(placed_orders[placed_neighbor], atom_positions)

    for endpoints in (shared_endpoints, tuple(reversed(shared_endpoints))):
        for order in _orders_starting_with_edge(face.atom_cycle, endpoints):
            for shape in _SHAPES_BY_SIZE[face.size]:
                budget.spend()
                candidate = _place_shape(shape, order, atom_positions[endpoints[0]], atom_positions[endpoints[1]])
                candidate_center = _points_center(candidate.values())
                if not _opposite_side(
                    atom_positions[endpoints[0]],
                    atom_positions[endpoints[1]],
                    existing_center,
                    candidate_center,
                ):
                    continue
                if any(atom in atom_positions and atom_positions[atom] != point for atom, point in candidate.items()):
                    continue
                merged = dict(atom_positions)
                merged.update(candidate)
                new_orders = {**placed_orders, face.id: order}
                if not _partial_layout_is_valid(model, new_orders, merged):
                    continue
                _search_layouts(
                    model,
                    face_by_id,
                    new_orders,
                    {**placed_shapes, face.id: shape},
                    merged,
                    budget,
                    completed,
                    max_layouts,
                )


def _valid_face_adjacency(model: FaceModel, face_by_id: dict[int, Face]) -> bool:
    known = set(face_by_id)
    seen_edges: set[int] = set()
    for left, right, edge in model.face_adjacency:
        if left not in known or right not in known or left == right or edge in seen_edges:
            return False
        if edge not in face_by_id[left].edge_cycle or edge not in face_by_id[right].edge_cycle:
            return False
        seen_edges.add(edge)
    adjacency = defaultdict(set)
    for left, right, _ in model.face_adjacency:
        adjacency[left].add(right)
        adjacency[right].add(left)
    reached = {min(known)}
    pending = deque(reached)
    while pending:
        current = pending.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached == known


def _next_face(model: FaceModel, placed: dict[int, tuple[int, ...]]) -> tuple[int | None, int | None, int | None]:
    options = []
    for left, right, edge in model.face_adjacency:
        if (left in placed) == (right in placed):
            continue
        unplaced, neighbor = (right, left) if left in placed else (left, right)
        placed_neighbors = sum(
            1 for a, b, _ in model.face_adjacency if unplaced in (a, b) and (b if a == unplaced else a) in placed
        )
        options.append((-placed_neighbors, unplaced, neighbor, edge))
    if not options:
        return None, None, None
    _, face, neighbor, edge = min(options)
    return face, neighbor, edge


def _edge_endpoints(face: Face, edge_id: int) -> Edge | None:
    try:
        index = face.edge_cycle.index(edge_id)
    except ValueError:
        return None
    return face.atom_cycle[index], face.atom_cycle[(index + 1) % face.size]


def _oriented_cycle(cycle: tuple[int, ...], offset: int, reverse: bool) -> tuple[int, ...]:
    order = tuple(reversed(cycle)) if reverse else cycle
    return order[offset:] + order[:offset]


def _orders_starting_with_edge(cycle: tuple[int, ...], endpoints: Edge) -> tuple[tuple[int, ...], ...]:
    variants = []
    for reverse in (False, True):
        order = tuple(reversed(cycle)) if reverse else cycle
        for offset in range(len(order)):
            candidate = order[offset:] + order[:offset]
            if candidate[:2] == endpoints:
                variants.append(candidate)
    return tuple(variants)


def _place_shape(shape: RingShapeSpec, order: tuple[int, ...], start: Point, end: Point) -> dict[int, Point]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return {
        atom: (
            start[0] + Fraction(x, 4) * dx - Fraction(y, 4) * dy,
            start[1] + Fraction(x, 4) * dy + Fraction(y, 4) * dx,
        )
        for atom, (x, y) in zip(order, shape.vertices)
    }


def _opposite_side(start: Point, end: Point, left: Point, right: Point) -> bool:
    left_cross = _cross(start, end, left)
    right_cross = _cross(start, end, right)
    return left_cross != 0 and right_cross != 0 and (left_cross > 0) != (right_cross > 0)


def _partial_layout_is_valid(
    model: FaceModel,
    placed_orders: dict[int, tuple[int, ...]],
    positions: dict[int, Point],
) -> bool:
    if len(set(positions.values())) != len(positions):
        return False
    drawn_edges: dict[frozenset[int], tuple[Point, Point]] = {}
    for face_id, order in placed_orders.items():
        face = next(face for face in model.faces if face.id == face_id)
        for edge_id, left, right in zip(face.edge_cycle, face.atom_cycle, face.atom_cycle[1:] + face.atom_cycle[:1]):
            if left not in positions or right not in positions:
                return False
            key = frozenset((left, right))
            segment = (positions[left], positions[right])
            previous = drawn_edges.setdefault(key, segment)
            if set(previous) != set(segment):
                return False
    edges = list(drawn_edges.items())
    for index, (left_atoms, left_segment) in enumerate(edges):
        for right_atoms, right_segment in edges[index + 1 :]:
            if left_atoms & right_atoms:
                continue
            if _segments_intersect(*left_segment, *right_segment):
                return False
    polygons = [(face_id, tuple(positions[atom] for atom in order)) for face_id, order in placed_orders.items()]
    for index, (left_id, left_polygon) in enumerate(polygons):
        for right_id, right_polygon in polygons[index + 1 :]:
            if _face_ids_adjacent(model, left_id, right_id):
                continue
            if _point_strictly_inside(_points_center(left_polygon), right_polygon):
                return False
            if _point_strictly_inside(_points_center(right_polygon), left_polygon):
                return False
    return True


def _audit_layout(
    model: FaceModel,
    placed_orders: dict[int, tuple[int, ...]],
    positions: dict[int, Point],
) -> bool:
    if set(placed_orders) != {face.id for face in model.faces} or not _partial_layout_is_valid(
        model, placed_orders, positions
    ):
        return False
    graph_edges = {
        frozenset((left, right))
        for face in model.faces
        for left, right in zip(face.atom_cycle, face.atom_cycle[1:] + face.atom_cycle[:1])
    }
    perimeter_edges = {
        frozenset(_edge_endpoints(face, edge) or ())
        for face in model.faces
        for edge in face.edge_cycle
        if edge in model.perimeter_edges
    }
    face_edges = [
        {frozenset((left, right)) for left, right in zip(face.atom_cycle, face.atom_cycle[1:] + face.atom_cycle[:1])}
        for face in model.faces
    ]
    geometric_perimeter = {edge for edge in graph_edges if sum(edge in edges for edges in face_edges) == 1}
    declared_perimeter = {
        frozenset((left, right))
        for left, right in zip(
            model.outer_boundary,
            model.outer_boundary[1:] + model.outer_boundary[:1],
        )
    }
    return perimeter_edges == geometric_perimeter == declared_perimeter and all(len(edge) == 2 for edge in graph_edges)


def _materialize_layouts(
    placed_orders: dict[int, tuple[int, ...]],
    shapes: dict[int, RingShapeSpec],
    positions: dict[int, Point],
) -> tuple[FusedLayout, ...]:
    fractional_centers = {
        face: _points_center(positions[atom] for atom in order)
        for face, order in placed_orders.items()
    }
    denominators = [coordinate.denominator for point in positions.values() for coordinate in point]
    denominators.extend(
        coordinate.denominator
        for point in fractional_centers.values()
        for coordinate in point
    )
    scale = lcm(*denominators) if denominators else 1
    integer = {atom: (int(x * scale), int(y * scale)) for atom, (x, y) in positions.items()}
    centers = {
        face: (int(x * scale), int(y * scale))
        for face, (x, y) in fractional_centers.items()
    }
    integer, centers = _normalize_integer_layout(integer, centers)

    # A generated embedding has an arbitrary horizontal seed edge.  P-25
    # orientation instead chooses the axis that contains the greatest row of
    # rings.  Every row containing two or more ring centers is parallel to a
    # center-pair vector, so these graph-derived directions are a complete
    # finite set for the supported all-peripheral tier.
    directions = {(1, 0)}
    center_values = tuple(centers.values())
    for index, (left_x, left_y) in enumerate(center_values):
        for right_x, right_y in center_values[index + 1 :]:
            dx, dy = right_x - left_x, right_y - left_y
            divisor = gcd(abs(dx), abs(dy))
            if divisor == 0:
                continue
            dx, dy = dx // divisor, dy // divisor
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            directions.add((dx, dy))

    candidates: dict[tuple, FusedLayout] = {}
    for dx, dy in sorted(directions):
        for x_sign in (-1, 1):
            for y_sign in (-1, 1):
                oriented = {
                    atom: (
                        x_sign * (x * dx + y * dy),
                        y_sign * (-x * dy + y * dx),
                    )
                    for atom, (x, y) in integer.items()
                }
                oriented_centers = {
                    face: (
                        x_sign * (x * dx + y * dy),
                        y_sign * (-x * dy + y * dx),
                    )
                    for face, (x, y) in centers.items()
                }
                oriented, oriented_centers = _normalize_integer_layout(oriented, oriented_centers)
                layout = FusedLayout(
                    face_positions=tuple(
                        (face, *oriented_centers[face]) for face in sorted(oriented_centers)
                    ),
                    atom_positions=tuple((atom, *oriented[atom]) for atom in sorted(oriented)),
                    face_shapes=tuple((face, shapes[face].shape_id) for face in sorted(shapes)),
                    orientation_score=_orientation_score(tuple(oriented_centers.values()), shapes),
                    audit_evidence=(
                        "all face boundaries represented",
                        "shared edge coordinates agree",
                        "unrelated edges do not cross",
                        "nonadjacent face interiors do not overlap",
                        "geometric and topological perimeters agree",
                        "preferred axis derived from ring-center rows",
                    ),
                )
                candidates.setdefault(_layout_geometry_key(layout), layout)

    best_score = min(layout.orientation_score for layout in candidates.values())
    return tuple(
        sorted(
            (layout for layout in candidates.values() if layout.orientation_score == best_score),
            key=_layout_sort_key,
        )
    )


def _normalize_integer_layout(
    positions: dict[int, tuple[int, int]],
    centers: dict[int, tuple[int, int]],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    min_x = min(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())
    positions = {atom: (x - min_x, y - min_y) for atom, (x, y) in positions.items()}
    centers = {face: (x - min_x, y - min_y) for face, (x, y) in centers.items()}
    divisor = 0
    for point in (*positions.values(), *centers.values()):
        divisor = gcd(divisor, point[0])
        divisor = gcd(divisor, point[1])
    if divisor > 1:
        positions = {atom: (x // divisor, y // divisor) for atom, (x, y) in positions.items()}
        centers = {face: (x // divisor, y // divisor) for face, (x, y) in centers.items()}
    return positions, centers


def _orientation_score(centers: tuple[tuple[int, int], ...], shapes: dict[int, RingShapeSpec]) -> tuple[int, ...]:
    ys = [y for _, y in centers]
    row_count = max(sum(y == row for _, y in centers) for row in set(ys))
    orientation = min(
        _row_orientation_score(centers, row)
        for row in set(ys)
        if sum(y == row for _, y in centers) == row_count
    )
    distortion = sum(shape.distortion_rank for shape in shapes.values())
    # Distorted shapes are disfavored before applying the ordinary P-25
    # orientation criteria; see the separate distortion precedence rule.
    return distortion, -row_count, *orientation


def _row_orientation_score(centers: tuple[tuple[int, int], ...], axis_y: int) -> tuple[int, int, int]:
    row_x = [x for x, y in centers if y == axis_y]
    axis_x = Fraction(min(row_x) + max(row_x), 2)
    upper_right = sum(_quadrant_units(x, y, axis_x, axis_y, upper=True) for x, y in centers)
    lower_left = sum(_quadrant_units(x, y, axis_x, axis_y, upper=False) for x, y in centers)
    above = sum(4 if y > axis_y else 2 if y == axis_y else 0 for _, y in centers)
    return -upper_right, lower_left, -above


def _quadrant_units(x: int, y: int, axis_x: Fraction, axis_y: Fraction, *, upper: bool) -> int:
    x_match = x > axis_x if upper else x < axis_x
    y_match = y > axis_y if upper else y < axis_y
    if x_match and y_match:
        return 4
    if (x == axis_x and y_match) or (y == axis_y and x_match):
        return 2
    if x == axis_x and y == axis_y:
        return 1
    return 0


def _layout_sort_key(layout: FusedLayout) -> tuple:
    shape_signature = tuple(sorted(shape for _, shape in layout.face_shapes))
    geometry = tuple(sorted((x, y) for _, x, y in layout.atom_positions))
    return layout.orientation_score, shape_signature, geometry


def _layout_geometry_key(layout: FusedLayout) -> tuple:
    return layout.atom_positions, layout.face_shapes


def _face_center(order: tuple[int, ...], positions: dict[int, Point]) -> Point:
    return _points_center(positions[atom] for atom in order)


def _points_center(points) -> Point:
    values = tuple(points)
    return (
        sum((point[0] for point in values), Fraction()) / len(values),
        sum((point[1] for point in values), Fraction()) / len(values),
    )


def _cross(start: Point, end: Point, point: Point) -> Fraction:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    one, two = _cross(a, b, c), _cross(a, b, d)
    three, four = _cross(c, d, a), _cross(c, d, b)
    if one == 0 and _on_segment(a, b, c):
        return True
    if two == 0 and _on_segment(a, b, d):
        return True
    if three == 0 and _on_segment(c, d, a):
        return True
    if four == 0 and _on_segment(c, d, b):
        return True
    return (one > 0) != (two > 0) and (three > 0) != (four > 0)


def _on_segment(start: Point, end: Point, point: Point) -> bool:
    return min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and min(start[1], end[1]) <= point[1] <= max(
        start[1], end[1]
    )


def _point_strictly_inside(point: Point, polygon: tuple[Point, ...]) -> bool:
    signs = [_cross(left, right, point) for left, right in zip(polygon, polygon[1:] + polygon[:1])]
    return all(value > 0 for value in signs) or all(value < 0 for value in signs)


def _face_ids_adjacent(model: FaceModel, left: int, right: int) -> bool:
    return any({left, right} == {first, second} for first, second, _ in model.face_adjacency)

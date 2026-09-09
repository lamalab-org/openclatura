"""Intrinsic, graph-derived layouts for bounded fused-ring face models.

The layout search uses exact rational arithmetic and fixed shape templates. It
never reads molecular drawing coordinates. A candidate is exposed only after
its shared edges, graph edges, crossings, overlaps, and topological perimeter
have all been audited.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from math import gcd, lcm

from .config import RingShapeSpec, fusion_nomenclature_config
from .model import Face, FaceModel, FusedLayout

Point = tuple[int, int]
Edge = tuple[int, int]
_SHAPE_EDGE_SCALE = 4


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
    inconsistent with the standard shapes and bounded large-ring reductions.
    """

    if search_budget < 1 or max_layouts < 1:
        raise ValueError("layout search budget and result limit must be positive")
    if any(face.size not in _SHAPES_BY_SIZE for face in model.faces):
        if len(model.faces) == 2:
            return _ordinary_large_bicycle_layouts(model, search_budget=search_budget, max_layouts=max_layouts)
        return _two_port_large_ring_layouts(model, search_budget=search_budget, max_layouts=max_layouts)
    face_by_id = {face.id: face for face in model.faces}
    if not _valid_face_adjacency(model, face_by_id):
        return ()
    budget = _Budget(search_budget)
    completed: dict[tuple, FusedLayout] = {}
    materialized_embeddings: set[tuple] = set()
    best_distortion: list[int | None] = [None]

    coordinate_systems = set.intersection(
        *(set(shape.coordinate_system for shape in _SHAPES_BY_SIZE[face.size]) for face in model.faces)
    )
    # Enumerating every seed avoids making the selected geometry depend on the
    # input atom IDs used to assign face IDs. The hard state/result budgets keep
    # this bounded for larger fused systems.
    for coordinate_system in sorted(coordinate_systems, key=lambda system: (system != "eisenstein", system)):
        for root in sorted(model.faces, key=lambda face: (face.size, face.id)):
            for shape in _shapes_for(root.size, coordinate_system):
                if best_distortion[0] is not None and shape.distortion_rank > best_distortion[0]:
                    continue
                for offset in range(root.size):
                    for reverse in (False, True):
                        budget.spend()
                        order = _oriented_cycle(root.atom_cycle, offset, reverse)
                        atom_positions = {atom: point for atom, point in zip(order, shape.vertices)}
                        _search_layouts(
                            model,
                            face_by_id,
                            {root.id: order},
                            {root.id: shape},
                            atom_positions,
                            budget,
                            completed,
                            materialized_embeddings,
                            max_layouts,
                            coordinate_system,
                            best_distortion,
                        )
        # Exact hexagonal geometry is preferred. Closely folded systems can
        # require the existing deformable vocabulary to avoid atom overlaps;
        # it still has to pass every geometry audit under the same budget.
        if completed:
            break
    return tuple(sorted(completed.values(), key=_layout_sort_key))


def _two_port_large_ring_layouts(model: FaceModel, *, search_budget: int, max_layouts: int) -> tuple[FusedLayout, ...]:
    """Expand an angular hexagon witness into one even, two-port large ring.

    OPSIN's even-ring direction rule assigns distances n/2 - 1 and n/2 + 1
    the same directions as hexagon distances 2 and 4. Only this signature is
    supported here: other separations must not acquire an unproved layout.
    Nonfusion paths subdivide the proxy polygon without moving its corners,
    fusion sides, or ring-axis centers. The standard bounded search therefore
    still owns the orientation alternatives; no large shape search is added.
    """

    large = [face for face in model.faces if face.size not in _SHAPES_BY_SIZE]
    if (
        len(large) != 1
        or large[0].size not in _CONFIG.annulene_ring_sizes
        or large[0].size % 2
        or len(model.face_adjacency) != len(model.faces) - 1
        or not _valid_face_adjacency(model, {face.id: face for face in model.faces})
        or set(model.outer_boundary) != {atom for face in model.faces for atom in face.atom_cycle}
    ):
        return ()
    face = large[0]
    ports = set(face.edge_cycle) & model.fusion_edges
    if len(ports) != 2:
        return ()
    entrance = _edge_endpoints(face, min(ports))
    if entrance is None:
        return ()
    distance = face.size // 2 - 1
    order = None
    for endpoints in (entrance, tuple(reversed(entrance))):
        candidate = _orders_starting_with_edge(face.atom_cycle, endpoints)[0]
        exit_edge = frozenset(candidate[distance : distance + 2])
        if exit_edge == frozenset(_edge_endpoints(face, next(iter(ports - {min(ports)}))) or ()):
            order = candidate
            break
    if order is None:
        return ()

    short_path = order[1 : distance + 1]
    long_path = order[distance + 1 :] + order[:1]
    steps = len(long_path) - 1
    shoulder = (steps + 1) // 3
    middle = steps - 2 * shoulder
    # Symmetric subdivisions retain reflection equivalence even when the
    # longer path does not divide evenly over the three hexagon sides.
    paths = (
        short_path,
        long_path[: shoulder + 1],
        long_path[shoulder : shoulder + middle + 1],
        long_path[shoulder + middle :],
    )
    removed = {atom for path in paths for atom in path[1:-1]}
    if removed & {atom for other in model.faces if other.id != face.id for atom in other.atom_cycle}:
        return ()
    proxy_order = (order[0], order[1], order[distance], *(path[0] for path in paths[1:]))
    original_edges = {
        frozenset((left, right)): edge
        for left, right, edge in zip(face.atom_cycle, face.atom_cycle[1:] + face.atom_cycle[:1], face.edge_cycle)
    }
    next_edge = max(edge for other in model.faces for edge in other.edge_cycle) + 1
    proxy_edges = []
    for left, right in zip(proxy_order, proxy_order[1:] + proxy_order[:1]):
        edge = original_edges.get(frozenset((left, right)))
        if edge is None:
            edge = next_edge
            next_edge += 1
        proxy_edges.append(edge)
    proxy_faces = tuple(
        Face(face.id, proxy_order, tuple(proxy_edges), 6) if other.id == face.id else other for other in model.faces
    )
    owners: dict[int, list[int]] = defaultdict(list)
    for other in proxy_faces:
        for edge in other.edge_cycle:
            owners[edge].append(other.id)
    proxy = replace(
        model,
        faces=proxy_faces,
        edge_to_faces=tuple((edge, tuple(sorted(ids))) for edge, ids in sorted(owners.items())),
        perimeter_edges=frozenset(edge for edge, ids in owners.items() if len(ids) == 1),
        outer_boundary=tuple(atom for atom in model.outer_boundary if atom not in removed),
    )
    layouts = intrinsic_fused_layouts(proxy, search_budget=search_budget, max_layouts=max_layouts)
    scale = lcm(*(len(path) - 1 for path in paths))
    expanded = []
    for layout in layouts:
        positions = {atom: (x * scale, y * scale) for atom, x, y in layout.atom_positions}
        centers = {key: (x * scale, y * scale) for key, x, y in layout.face_positions}
        for path in paths:
            start, end = positions[path[0]], positions[path[-1]]
            count = len(path) - 1
            for index, atom in enumerate(path[1:-1], start=1):
                positions[atom] = tuple(left + index * (right - left) // count for left, right in zip(start, end))
        if not _audit_layout(model, {other.id: other.atom_cycle for other in model.faces}, positions):
            continue
        positions, centers = _normalize_integer_layout(positions, centers)
        expanded.append(
            replace(
                layout,
                atom_positions=tuple((atom, *point) for atom, point in sorted(positions.items())),
                face_positions=tuple((key, *point) for key, point in sorted(centers.items())),
                face_shapes=tuple(
                    (key, f"two-port-{face.size}:{shape}" if key == face.id else shape)
                    for key, shape in layout.face_shapes
                ),
                audit_evidence=(
                    *layout.audit_evidence,
                    "even large-ring fusion distances have the angular hexagon direction signature",
                    "only nonfusion paths subdivided; proxy polygon and ring-axis centers preserved",
                    "expanded original face model passes the complete geometry audit",
                ),
            )
        )
    return tuple(sorted(expanded, key=_layout_sort_key))


def _ordinary_large_bicycle_layouts(
    model: FaceModel, *, search_budget: int, max_layouts: int
) -> tuple[FusedLayout, ...]:
    """Embed one edge-fused pair, without extending the general shape search.

    A vertical common edge and two symmetric convex arcs prove a horizontal
    two-ring row. Its four reflections exhaust the possible numbering starts;
    the existing completed-system locant criteria decide between them. These
    coordinates are an orientation witness, not a general ring-shape template.
    """

    if (
        len(model.faces) != 2
        or len(model.fusion_edges) != 1
        or len(model.face_adjacency) != 1
        or max(face.size for face in model.faces) not in _CONFIG.annulene_ring_sizes
        or min(face.size for face in model.faces) < _CONFIG.rules.minimum_ring_size
        or not _valid_face_adjacency(model, {face.id: face for face in model.faces})
    ):
        return ()
    left, right = model.faces
    edge = next(iter(model.fusion_edges))
    endpoints = _edge_endpoints(left, edge)
    if endpoints is None or set(endpoints) != set(_edge_endpoints(right, edge) or ()):
        return ()
    if set(left.atom_cycle) & set(right.atom_cycle) != set(endpoints):
        return ()
    start, end = endpoints
    scale = lcm(*((face.size - 1) ** 2 * face.size for face in model.faces))
    positions = {start: (0, -scale), end: (0, scale)}
    orders = {}
    for sign, face in zip((-1, 1), model.faces):
        offset = face.atom_cycle.index(start)
        order = face.atom_cycle[offset:] + face.atom_cycle[:offset]
        if order[1] == end:
            order = (start, *reversed(order[1:]))
        if order[-1] != end:
            return ()
        steps = face.size - 1
        for index, atom in enumerate(order[1:-1], start=1):
            positions[atom] = (
                sign * 4 * index * (steps - index) * scale // steps**2,
                (2 * index - steps) * scale // steps,
            )
        orders[face.id] = order
    if not _audit_layout(model, orders, positions):
        return ()
    centers = {face.id: (sum(positions[atom][0] for atom in face.atom_cycle) // face.size, 0) for face in model.faces}
    adjacent = frozenset((frozenset(centers),))
    budget = _Budget(search_budget)
    layouts = []
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            budget.spend()
            oriented, oriented_centers = _normalize_integer_layout(
                {atom: (x_sign * x, y_sign * y) for atom, (x, y) in positions.items()},
                {face: (x_sign * x, y_sign * y) for face, (x, y) in centers.items()},
            )
            layouts.append(
                FusedLayout(
                    face_positions=tuple((face, *point) for face, point in sorted(oriented_centers.items())),
                    atom_positions=tuple((atom, *point) for atom, point in sorted(oriented.items())),
                    face_shapes=tuple((face.id, f"ordinary-bicycle-{face.size}") for face in model.faces),
                    orientation_score=_orientation_score(
                        oriented_centers, {}, adjacent, distortion=1, orders=orders, positions=oriented
                    ),
                    audit_evidence=(
                        "two convex faces share only their vertical common edge",
                        "shared edge coordinates agree",
                        "unrelated edges do not cross",
                        "geometric and topological perimeters agree",
                        "four reflected horizontal-row orientation witnesses",
                    ),
                )
            )
            if len(layouts) > max_layouts:
                raise LayoutSearchBudgetExceeded(max_layouts, resource="layouts")
    return tuple(sorted(layouts, key=_layout_sort_key))


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
    materialized_embeddings: set[tuple],
    max_layouts: int,
    coordinate_system: str,
    best_distortion: list[int | None],
) -> None:
    current_distortion = _layout_distortion(placed_orders, placed_shapes, atom_positions, coordinate_system)
    if best_distortion[0] is not None and current_distortion > best_distortion[0]:
        return
    if len(placed_orders) == len(model.faces):
        if _audit_layout(model, placed_orders, atom_positions):
            embedding_key = _intrinsic_embedding_key(
                placed_orders,
                placed_shapes,
                atom_positions,
                coordinate_system=coordinate_system,
            )
            if embedding_key in materialized_embeddings:
                return
            materialized_embeddings.add(embedding_key)
            if best_distortion[0] is None or current_distortion < best_distortion[0]:
                best_distortion[0] = current_distortion
                completed.clear()
            for layout in _materialize_layouts(
                placed_orders,
                placed_shapes,
                atom_positions,
                coordinate_system=coordinate_system,
            ):
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
    # A newly attached template can introduce quarter-unit coordinates when
    # its entrance edge is not horizontal. Scale the complete partial layout
    # once at this depth so all subsequent geometric predicates stay exact
    # integer operations. Layout normalization removes this common scale.
    scaled_positions = {atom: (x * _SHAPE_EDGE_SCALE, y * _SHAPE_EDGE_SCALE) for atom, (x, y) in atom_positions.items()}
    existing_side = _face_side_point(placed_orders[placed_neighbor], shared_endpoints, scaled_positions)
    if existing_side is None:
        return

    for endpoints in (shared_endpoints, tuple(reversed(shared_endpoints))):
        for order in _orders_starting_with_edge(face.atom_cycle, endpoints):
            for shape in _shapes_for(face.size, coordinate_system):
                if best_distortion[0] is not None and current_distortion + shape.distortion_rank > best_distortion[0]:
                    continue
                budget.spend()
                candidate = _place_shape(
                    shape,
                    order,
                    scaled_positions[endpoints[0]],
                    scaled_positions[endpoints[1]],
                    coordinate_system=coordinate_system,
                )
                if not _opposite_side(
                    scaled_positions[endpoints[0]],
                    scaled_positions[endpoints[1]],
                    existing_side,
                    candidate[order[2]],
                ):
                    continue
                if any(
                    atom in scaled_positions and scaled_positions[atom] != point for atom, point in candidate.items()
                ):
                    continue
                merged = dict(scaled_positions)
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
                    materialized_embeddings,
                    max_layouts,
                    coordinate_system,
                    best_distortion,
                )


def _shapes_for(ring_size: int, coordinate_system: str) -> tuple[RingShapeSpec, ...]:
    return tuple(shape for shape in _SHAPES_BY_SIZE[ring_size] if shape.coordinate_system == coordinate_system)


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
    placement_rank = {face: rank for rank, face in enumerate(placed)}
    for left, right, edge in model.face_adjacency:
        if (left in placed) == (right in placed):
            continue
        unplaced, neighbor = (right, left) if left in placed else (left, right)
        placed_neighbors = sum(
            1 for a, b, _ in model.face_adjacency if unplaced in (a, b) and (b if a == unplaced else a) in placed
        )
        # Grow from the enumerated seed before using face IDs to break ties.
        # Otherwise every seed can use the same ID-selected entrance edge
        # of an asymmetric ring shape, losing symmetry-related embeddings.
        options.append((-placed_neighbors, placement_rank[neighbor], unplaced, neighbor, edge))
    if not options:
        return None, None, None
    _, _, face, neighbor, edge = min(options)
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


def _place_shape(
    shape: RingShapeSpec,
    order: tuple[int, ...],
    start: Point,
    end: Point,
    *,
    coordinate_system: str = "cartesian",
) -> dict[int, Point]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    if dx % _SHAPE_EDGE_SCALE or dy % _SHAPE_EDGE_SCALE:
        raise ValueError("scaled fusion entrance edge must have integral template coordinates")
    if coordinate_system == "eisenstein":
        return {
            atom: (
                start[0] + (x * dx - y * dy) // _SHAPE_EDGE_SCALE,
                start[1] + (x * dy + y * dx + y * dy) // _SHAPE_EDGE_SCALE,
            )
            for atom, (x, y) in zip(order, shape.vertices)
        }
    return {
        atom: (
            start[0] + x * dx // _SHAPE_EDGE_SCALE - y * dy // _SHAPE_EDGE_SCALE,
            start[1] + x * dy // _SHAPE_EDGE_SCALE + y * dx // _SHAPE_EDGE_SCALE,
        )
        for atom, (x, y) in zip(order, shape.vertices)
    }


def _face_side_point(
    order: tuple[int, ...],
    shared_endpoints: Edge,
    positions: dict[int, Point],
) -> Point | None:
    """Return any non-interface vertex, sufficient to identify face side."""

    endpoints = frozenset(shared_endpoints)
    return next((positions[atom] for atom in order if atom not in endpoints), None)


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
            if _polygon_center_strictly_inside(left_polygon, right_polygon):
                return False
            if _polygon_center_strictly_inside(right_polygon, left_polygon):
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
    *,
    coordinate_system: str = "cartesian",
) -> tuple[FusedLayout, ...]:
    distortion = _layout_distortion(placed_orders, shapes, positions, coordinate_system)
    integer, centers = _normalized_embedding_geometry(
        placed_orders,
        positions,
        coordinate_system=coordinate_system,
    )

    # A generated embedding has an arbitrary horizontal seed edge.  P-25
    # orientation instead chooses the axis that contains the greatest row of
    # consecutively fused rings. Nonadjacent collinear centers do not form a
    # row. Thus only directions between edge-sharing faces are needed.
    edge_owners: dict[Edge, list[int]] = defaultdict(list)
    for face, order in placed_orders.items():
        for left, right in zip(order, order[1:] + order[:1]):
            edge_owners[tuple(sorted((left, right)))].append(face)
    adjacent = frozenset(frozenset(owners) for owners in edge_owners.values() if len(owners) == 2)
    directions = {(1, 0)}
    for pair in adjacent:
        left, right = pair
        left_x, left_y = centers[left]
        right_x, right_y = centers[right]
        dx, dy = right_x - left_x, right_y - left_y
        divisor = gcd(abs(dx), abs(dy))
        if divisor == 0:
            continue
        dx, dy = dx // divisor, dy // divisor
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        directions.add((dx, dy))

    candidates: dict[tuple, FusedLayout] = {}
    best_score: tuple[int, ...] | None = None
    # Integer Eisenstein coordinates are (2*x+y, 3*y), so their squared
    # Euclidean length is proportional to 3*X**2 + Y**2, not X**2 + Y**2.
    metric_x = 3 if coordinate_system == "eisenstein" else 1
    for dx, dy in sorted(directions):
        for x_sign in (-1, 1):
            for y_sign in (-1, 1):
                oriented_centers = {
                    face: (
                        x_sign * (metric_x * x * dx + y * dy),
                        y_sign * (-x * dy + y * dx),
                    )
                    for face, (x, y) in centers.items()
                }
                oriented = {
                    atom: (
                        x_sign * (metric_x * x * dx + y * dy),
                        y_sign * (-x * dy + y * dx),
                    )
                    for atom, (x, y) in integer.items()
                }
                score = _orientation_score(
                    oriented_centers,
                    shapes,
                    adjacent,
                    distortion=distortion,
                    orders=placed_orders,
                    positions=oriented,
                )
                if best_score is not None and score > best_score:
                    continue
                if best_score is None or score < best_score:
                    best_score = score
                    candidates.clear()
                oriented, oriented_centers = _normalize_integer_layout(oriented, oriented_centers)
                layout = FusedLayout(
                    face_positions=tuple((face, *oriented_centers[face]) for face in sorted(oriented_centers)),
                    atom_positions=tuple((atom, *oriented[atom]) for atom in sorted(oriented)),
                    face_shapes=tuple((face, shapes[face].shape_id) for face in sorted(shapes)),
                    orientation_score=score,
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

    return tuple(sorted(candidates.values(), key=_layout_sort_key))


def _layout_distortion(
    orders: dict[int, tuple[int, ...]],
    shapes: dict[int, RingShapeSpec],
    positions: dict[int, Point],
    coordinate_system: str,
) -> int:
    """Count template distortion and relatively enlarged rings.

    Each order starts with its template's entrance edge. Attaching that edge
    to an elongated side can resize a whole ring, even with an undistorted
    template. Exact squared scale ratios expose this without penalizing a
    common rescaling of the drawing. The smallest scale sets the reference;
    adding a face cannot lower the number of enlarged rings or this bound.
    """

    scales = []
    for face, order in orders.items():
        left, right = (positions[atom] for atom in order[:2])
        start, end = shapes[face].vertices[:2]
        dx, dy = right[0] - left[0], right[1] - left[1]
        sx, sy = end[0] - start[0], end[1] - start[1]
        placed_length = dx * dx + dy * dy
        template_length = sx * sx + sy * sy
        if coordinate_system == "eisenstein":
            placed_length += dx * dy
            template_length += sx * sy
        scales.append((placed_length, template_length))
    smallest = scales[0]
    for numerator, denominator in scales[1:]:
        if numerator * smallest[1] < smallest[0] * denominator:
            smallest = numerator, denominator
    enlarged = sum(numerator * smallest[1] > smallest[0] * denominator for numerator, denominator in scales)
    return sum(shape.distortion_rank for shape in shapes.values()) + enlarged


def _normalized_embedding_geometry(
    placed_orders: dict[int, tuple[int, ...]],
    positions: dict[int, Point],
    *,
    coordinate_system: str,
) -> tuple[dict[int, Point], dict[int, Point]]:
    """Return scale-normalized Cartesian atom and face coordinates."""

    if coordinate_system == "eisenstein":
        positions = {atom: (2 * x + y, 3 * y) for atom, (x, y) in positions.items()}
    scale = lcm(4, *(len(order) for order in placed_orders.values()))
    integer = {atom: (x * scale, y * scale) for atom, (x, y) in positions.items()}
    centers = {face: _ring_axis_center(order, integer) for face, order in placed_orders.items()}
    return _normalize_integer_layout(integer, centers)


def _ring_axis_center(order: tuple[int, ...], positions: dict[int, Point]) -> Point:
    """Keep opposite parallel fusion sides on one ring-centre axis.

    An odd ring's vertex average is displaced towards its extra vertex.
    Equal, oppositely directed sides instead define the centre of a linear
    fusion row. Use that centre when all such pairs agree; otherwise retain
    the vertex average. Coordinates are scaled for exact division by four.
    """

    edges = tuple((positions[left], positions[right]) for left, right in zip(order, order[1:] + order[:1]))
    centers = set()
    for index, (left, right) in enumerate(edges):
        for other_left, other_right in edges[index + 1 :]:
            if (right[0] - left[0], right[1] - left[1]) == (
                other_left[0] - other_right[0],
                other_left[1] - other_right[1],
            ):
                centers.add(
                    (
                        (left[0] + right[0] + other_left[0] + other_right[0]) // 4,
                        (left[1] + right[1] + other_left[1] + other_right[1]) // 4,
                    )
                )
    if len(centers) == 1:
        return centers.pop()
    return (
        sum(positions[atom][0] for atom in order) // len(order),
        sum(positions[atom][1] for atom in order) // len(order),
    )


def _intrinsic_embedding_key(
    placed_orders: dict[int, tuple[int, ...]],
    shapes: dict[int, RingShapeSpec],
    positions: dict[int, Point],
    *,
    coordinate_system: str,
) -> tuple:
    """Identify embeddings modulo translation, rotation, and reflection."""

    integer, _centers = _normalized_embedding_geometry(
        placed_orders,
        positions,
        coordinate_system=coordinate_system,
    )
    atoms = tuple(sorted(integer))
    metric_x = 3 if coordinate_system == "eisenstein" else 1
    squared_distances = tuple(
        metric_x * (integer[left][0] - integer[right][0]) ** 2 + (integer[left][1] - integer[right][1]) ** 2
        for position, left in enumerate(atoms)
        for right in atoms[position + 1 :]
    )
    return (
        tuple((face, shapes[face].shape_id) for face in sorted(shapes)),
        squared_distances,
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


def _orientation_score(
    centers: dict[int, Point],
    shapes: dict[int, RingShapeSpec],
    adjacent: frozenset[frozenset[int]],
    *,
    distortion: int | None = None,
    orders: dict[int, tuple[int, ...]],
    positions: dict[int, Point],
) -> tuple[int, ...]:
    direction_centers = _direction_grid_centers(centers, adjacent)
    if direction_centers is not None:
        centers = direction_centers
    rows: list[list[int]] = []
    for face in sorted(centers, key=lambda face: (centers[face][1], centers[face][0])):
        if not rows or centers[rows[-1][-1]][1] != centers[face][1] or frozenset((rows[-1][-1], face)) not in adjacent:
            rows.append([])
        rows[-1].append(face)
    row_count = max(map(len, rows))
    bounds = []
    for order in orders.values():
        xs = [2 * positions[atom][0] for atom in order]
        ys = [positions[atom][1] for atom in order]
        bounds.append((min(xs), max(xs), min(ys), max(ys)))
    orientation = min(
        _center_row_orientation_score(centers, row)
        if direction_centers is not None
        else _bounded_row_orientation_score(centers, row, orders, positions, bounds)
        for row in rows
        if len(row) == row_count
    )
    if distortion is None:
        distortion = sum(shape.distortion_rank for shape in shapes.values())
    # Distorted shapes are disfavored before applying the ordinary P-25
    # orientation criteria; see the separate distortion precedence rule.
    return distortion, -row_count, *orientation


def _direction_grid_centers(centers: dict[int, Point], adjacent: frozenset[frozenset[int]]) -> dict[int, Point] | None:
    """Separate ring-center direction from the scale of individual polygons."""
    neighbors: dict[int, list[int]] = {face: [] for face in centers}
    for edge in adjacent:
        left, right = sorted(edge)
        neighbors[left].append(right)
        neighbors[right].append(left)
    root = min(centers)
    result = {root: (0, 0)}
    queue = deque([root])
    while queue:
        face = queue.popleft()
        for other in neighbors[face]:
            dx = centers[other][0] - centers[face][0]
            dy = centers[other][1] - centers[face][1]
            sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
            step = (4 * sx, 0) if not sy else (0, 2 * sy) if not sx else (2 * sx, sy)
            point = (result[face][0] + step[0], result[face][1] + step[1])
            if other in result:
                if result[other] != point:
                    return None
            else:
                result[other] = point
                queue.append(other)
    return result if len(result) == len(centers) and len(set(result.values())) == len(result) else None


def _center_row_orientation_score(centers: dict[int, Point], row: list[int]) -> tuple[int, int, int]:
    middle = len(row) // 2
    axis_x = 2 * centers[row[middle]][0] if len(row) % 2 else centers[row[middle - 1]][0] + centers[row[middle]][0]
    axis_y = centers[row[0]][1]
    upper_right = lower_left = above = 0
    for x, y in centers.values():
        right = 1 if 2 * x == axis_x else 2 if 2 * x > axis_x else 0
        upper = 1 if y == axis_y else 2 if y > axis_y else 0
        upper_right += right * upper
        lower_left += (2 - right) * (2 - upper)
        above += 2 * upper
    return -upper_right, lower_left, -above


def _bounded_row_orientation_score(
    centers: dict[int, Point],
    row: list[int],
    orders: dict[int, tuple[int, ...]],
    positions: dict[int, Point],
    bounds: list[tuple[int, int, int, int]],
) -> tuple[int, int, int]:
    """FR-5.2: bisect the middle ring/bond, counting divided rings as halves."""

    middle = len(row) // 2
    if len(row) % 2:
        doubled_axis_x = 2 * centers[row[middle]][0]
    else:
        shared = set(orders[row[middle - 1]]) & set(orders[row[middle]])
        doubled_axis_x = sum(positions[atom][0] for atom in shared)
    axis_y = centers[row[0]][1]
    upper_right = lower_left = above = 0
    for min_x, max_x, min_y, max_y in bounds:
        right = _positive_half_units(min_x, max_x, doubled_axis_x)
        upper = _positive_half_units(min_y, max_y, axis_y)
        upper_right += right * upper
        lower_left += (2 - right) * (2 - upper)
        above += 2 * upper
    return -upper_right, lower_left, -above


def _positive_half_units(low: int, high: int, axis: int) -> int:
    if low < axis < high:
        return 1
    return 2 if low >= axis else 0


def _layout_sort_key(layout: FusedLayout) -> tuple:
    shape_signature = tuple(sorted(shape for _, shape in layout.face_shapes))
    geometry = tuple(sorted((x, y) for _, x, y in layout.atom_positions))
    return layout.orientation_score, shape_signature, geometry


def _layout_geometry_key(layout: FusedLayout) -> tuple:
    return layout.atom_positions, layout.face_shapes


def _cross(start: Point, end: Point, point: Point) -> int:
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


def _polygon_center_strictly_inside(source: tuple[Point, ...], polygon: tuple[Point, ...]) -> bool:
    """Test a source centroid against a polygon without constructing fractions."""

    count = len(source)
    center_x = sum(x for x, _ in source)
    center_y = sum(y for _, y in source)
    signs = [
        (right[0] - left[0]) * (center_y - count * left[1]) - (right[1] - left[1]) * (center_x - count * left[0])
        for left, right in zip(polygon, polygon[1:] + polygon[:1])
    ]
    return all(value > 0 for value in signs) or all(value < 0 for value in signs)


def _face_ids_adjacent(model: FaceModel, left: int, right: int) -> bool:
    return any({left, right} == {first, second} for first, second, _ in model.face_adjacency)

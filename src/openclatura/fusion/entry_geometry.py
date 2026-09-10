"""Bounded OPSIN-compatible entry directions on an audited fused face graph."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache

from ..naming_data import load_json_table
from .model import FaceModel, FusedLayout


@lru_cache(maxsize=1)
def _entry_direction_tables():
    data = load_json_table("fusion_entry_geometry.json")
    tables = {
        int(size): tuple((tuple(row["relative"]), tuple(row["forbidden_ports"])) for row in rows)
        for size, rows in data["directions"].items()
    }
    steps = {int(direction): tuple(point) for direction, point in data["steps"].items()}
    if set(steps) != set(range(-3, 5)) or any(len(point) != 2 for point in steps.values()):
        raise ValueError("entry directions require eight two-dimensional steps")
    for size, rows in tables.items():
        if (
            size < 3
            or not rows
            or any(
                len(relative) != size - 1
                or any(value not in steps for value in relative)
                or any(not 1 <= port < size for port in forbidden)
                for relative, forbidden in rows
            )
        ):
            raise ValueError("entry direction table does not cover its ring ports")
    return tables, steps


@dataclass(frozen=True, slots=True)
class EntryDirectionWitness:
    perimeter: tuple[int, ...]
    start_face: int
    ring_positions: tuple[tuple[int, int, int], ...]
    orientation_score: tuple[int, ...]
    direction_conflicts: tuple[tuple[int, int], ...] = ()


def _absolute_direction(relative: int, previous: int) -> int:
    direction = (relative + previous + 4) % 8 - 4
    if abs(direction) == 2:
        if {abs(relative), abs(previous)} == {1, 3}:
            direction = 1 if direction > 0 else -1
        elif abs(relative) == abs(previous) and abs(relative) in {1, 3}:
            direction = 3 if direction > 0 else -3
    return 4 if direction == -4 else direction


@lru_cache(maxsize=128)
def entry_direction_geometry(
    model: FaceModel,
    search_budget: int,
    terminal_cycle: tuple[int, ...] | None = None,
    allow_distorted: bool = False,
) -> tuple[FusedLayout, tuple[EntryDirectionWitness, ...]] | None:
    """Prove entry directions without assigning atom locants or querying OPSIN.

    Enumerate eligible terminal entries and prefer closed direction maps. A
    uniquely rooted cyclic face graph may instead project the
    parser's nonreciprocal directions from its terminal, retaining every
    conflict in the witness. Ring positions must remain injective and their
    connections noncrossing. Physical polygon geometry is independently
    audited; these parser-compatible axes do not certify an IUPAC PIN layout.
    """
    from .layout import _Budget, _opsin_occupied_row_orientation, _segments_intersect, preferred_intrinsic_layouts
    from .numbering import _clockwise_boundary

    direction_tables, steps = _entry_direction_tables()
    if not any(face.size in {5, 7} for face in model.faces) or any(
        face.size not in direction_tables for face in model.faces
    ):
        return None
    layouts = preferred_intrinsic_layouts(model, search_budget=search_budget)
    if not layouts:
        return None
    reference = layouts[0]
    positions = {atom: (x, y) for atom, x, y in reference.atom_positions}
    cycles = {}
    for face in model.faces:
        clockwise = _clockwise_boundary(face.atom_cycle, positions)
        if clockwise is None:
            return None
        cycles[face.id] = tuple(reversed(clockwise))
    terminal_root = None
    reverse_winding = False
    if terminal_cycle is not None:
        terminal_root = next((face for face, cycle in cycles.items() if set(cycle) == set(terminal_cycle)), None)
        if terminal_root is None:
            return None
        cycle = cycles[terminal_root]
        offset = cycle.index(terminal_cycle[0])
        if cycle[offset:] + cycle[:offset] != terminal_cycle:
            cycle = tuple(reversed(cycle))
            offset = cycle.index(terminal_cycle[0])
            if cycle[offset:] + cycle[:offset] != terminal_cycle:
                return None
            cycles = {face: tuple(reversed(cycle)) for face, cycle in cycles.items()}
            reverse_winding = True
    edges = {face: [frozenset((a, b)) for a, b in zip(cycle, cycle[1:] + cycle[:1])] for face, cycle in cycles.items()}
    neighbors: dict[int, list[int]] = defaultdict(list)
    ports = {}
    for left, right, _ in model.face_adjacency:
        shared = set(edges[left]) & set(edges[right])
        if len(shared) != 1:
            return None
        edge = shared.pop()
        neighbors[left].append(right)
        neighbors[right].append(left)
        ports[left, right] = edges[left].index(edge)
        ports[right, left] = edges[right].index(edge)
    if set(neighbors) != set(cycles):
        return None
    minimum_degree = min(map(len, neighbors.values()))
    terminals = [face for face in cycles if len(neighbors[face]) == minimum_degree]
    if terminal_root is not None:
        if terminal_root not in terminals:
            return None
        terminals = [terminal_root]
    uniquely_rooted_cycle = len(terminals) == 1 and minimum_degree == 1 and len(model.face_adjacency) >= len(cycles)
    if allow_distorted and not uniquely_rooted_cycle:
        return None
    if uniquely_rooted_cycle and not allow_distorted:
        return entry_direction_geometry(model, search_budget, terminal_cycle, True)
    budget = _Budget(search_budget)
    tables = set()

    def visit(pending, visited, directions):
        budget.spend()
        if not pending:
            if len(visited) == len(cycles):
                tables.add(tuple(sorted(directions.items())))
            return
        face, incoming, entry = pending[0]
        if face in visited:
            visit(pending[1:], visited, directions)
            return
        size = len(cycles[face])
        distances = {other: (ports[face, other] - entry) % size for other in neighbors[face]}
        options = direction_tables[size]
        if len(neighbors[face]) == 1 and size in {5, 7}:
            options = options[:1]
        for table, forbidden in options:
            if set(distances.values()) & set(forbidden):
                continue
            outgoing = {
                other: _absolute_direction(4 if distance == 0 else table[distance - 1], incoming)
                for other, distance in distances.items()
            }
            following = [
                (other, outgoing[other], ports[other, face]) for other in neighbors[face] if other not in visited
            ]
            visit(
                following + pending[1:],
                visited | {face},
                {**directions, **{(face, other): value for other, value in outgoing.items()}},
            )

    for root in terminals:
        fused_atoms = set().union(*(edges[root][ports[root, other]] for other in neighbors[root]))
        entries = [index for index, edge in enumerate(edges[root]) if not edge & fused_atoms]
        if not entries:
            entries = [
                index
                for index in range(len(edges[root]))
                if index not in {ports[root, other] for other in neighbors[root]}
            ]
        for entry in entries:
            visit([(root, 0, entry)], set(), {})

    outer_clockwise = _clockwise_boundary(tuple(model.outer_boundary), positions)
    if outer_clockwise is None:
        return None
    if reverse_winding:
        outer_clockwise = tuple(reversed(outer_clockwise))
    fusion_atoms = set().union(*(edges[left][ports[left, right]] for left, right, _ in model.face_adjacency))
    candidates = {}
    best_score = None
    for items in sorted(tables):
        directions = dict(items)
        distortion = sum(_absolute_direction(direction, 4) != directions[b, a] for (a, b), direction in items) // 2
        for axis in sorted(set(directions.values())):
            budget.spend()
            directed = {(a, b): _absolute_direction(direction, -axis) for (a, b), direction in items}
            centers = {terminals[0] if allow_distorted else next(iter(cycles)): (0, 0)}
            queue = deque(centers)
            consistent = True
            conflicts = set()
            while queue and consistent:
                left = queue.popleft()
                for right in neighbors[left]:
                    dx, dy = steps[directed[left, right]]
                    point = (centers[left][0] + dx, centers[left][1] + dy)
                    if right in centers:
                        if centers[right] != point:
                            if allow_distorted:
                                conflicts.add(tuple(sorted((left, right))))
                            else:
                                consistent = False
                                break
                    else:
                        centers[right] = point
                        queue.append(right)
            if not consistent or len(centers) != len(cycles) or len(set(centers.values())) != len(cycles):
                continue
            connections = [(left, right) for left, right, _ in model.face_adjacency]
            if any(
                not {a, b} & {c, d} and _segments_intersect(centers[a], centers[b], centers[c], centers[d])
                for index, (a, b) in enumerate(connections)
                for c, d in connections[index + 1 :]
            ):
                continue
            longest = 1
            for face in cycles:
                seen = {face}
                while True:
                    following = [other for other in neighbors[face] if directed[face, other] == 0 and other not in seen]
                    if len(following) != 1:
                        break
                    face = following[0]
                    seen.add(face)
                longest = max(longest, len(seen))
            for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                oriented = {face: (sx * x, sy * y) for face, (x, y) in centers.items()}
                # OPSIN ranks the original connection table before rotating it.
                # Discrete rotation can close a map without removing that rank.
                score = (distortion, -longest, *_opsin_occupied_row_orientation(oriented))
                if best_score is not None and score > best_score:
                    continue
                boundary = outer_clockwise if sx * sy == 1 else tuple(reversed(outer_clockwise))
                eligible = [face for face in cycles if (set(cycles[face]) & set(boundary)) - fusion_atoms]
                if not eligible:
                    continue
                top = min(eligible, key=lambda face: (-oriented[face][1], -oriented[face][0]))
                starts = [
                    atom
                    for index, atom in enumerate(boundary)
                    if atom in cycles[top] and atom not in fusion_atoms and boundary[index - 1] in fusion_atoms
                ]
                if len(starts) != 1:
                    continue
                offset = boundary.index(starts[0])
                perimeter = boundary[offset:] + boundary[:offset]
                if best_score is None or score < best_score:
                    best_score = score
                    candidates.clear()
                candidates.setdefault(
                    perimeter,
                    EntryDirectionWitness(
                        perimeter,
                        top,
                        tuple((face, *oriented[face]) for face in sorted(oriented)),
                        score,
                        tuple(sorted(conflicts)),
                    ),
                )
    return (reference, tuple(candidates[key] for key in sorted(candidates))) if candidates else None

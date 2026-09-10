"""Stable OPSIN construction enumeration for closed hexagonal ring maps.

This is parser compatibility, not an additional IUPAC tie-breaking rule.
The ordering follows OPSIN 2.9 FusedRingBuilder.fuseRings, SSSRFinder, and
FusedRingNumberer.determinePossiblePeripheryAtomOrders/rulesBCD. Unsupported
ring sets and exhausted proofs leave the numbering ambiguity unresolved.
"""

from collections import deque
from dataclasses import dataclass

from .entry_geometry import _absolute_direction, _allowed_entry_directions, _entry_direction_tables
from .model import FaceModel, FusionComponentSpec, FusionNameAst

Point = tuple[int, int]
Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ConstructionOrder:
    atom_order: tuple[int, ...]
    # Face and winding relative to the supplied audited reference positions.
    entries: tuple[tuple[int, int], ...]


def ordered_hexagonal_construction(
    ast: FusionNameAst,
    specs: dict[int, FusionComponentSpec],
    model: FaceModel,
    positions: dict[int, Point],
    search_budget: int,
) -> ConstructionOrder | None:
    from .layout import _center_row_orientation_score

    citation = ast.citation_plan
    if (
        citation is None
        or len(citation.parent_occurrences) != 1
        or citation.interparent_occurrences
        or citation.cycle_closing_join_indices
        or ast.multiplicative_groups
        or any(face.size != 6 for face in model.faces)
        or any(atom.symbol != "C" for spec in specs.values() for atom in spec.template.atoms)
        or any(spec.template.numbering_policy != "retained_template" for spec in specs.values())
    ):
        return None
    matches = {match.occurrence_id: match for match in ast.component_occurrences}
    bonds: list[Edge] = []
    bond_keys: set[frozenset[int]] = set()
    seen: set[int] = set()
    atom_order: list[int] = []
    for occurrence in (*citation.parent_occurrences, *reversed(citation.render_order)):
        local = matches[occurrence].input_atom_by_locant
        shared = set(local.values()) & seen
        edges = [tuple(local[locant] for locant in bond.locants) for bond in specs[occurrence].bonds]
        # Retained CML bond records preserve component construction order;
        # numeric locant sorting does not. Generated-series graph templates
        # do not provide this provenance and are excluded above.
        for edge in edges:
            for atom in edge:
                if atom not in seen and atom not in atom_order:
                    atom_order.append(atom)
        joins = [join for join in ast.joins if join.attached_occurrence == occurrence]
        if len(joins) > 1 or shared != set().union(*(join.shared_input_atoms for join in joins)):
            return None
        ordered = [edge for edge in edges if not set(edge) & shared]
        for join in joins:
            # fuseRings uses a LinkedHashSet collected by descriptor atom,
            # then incident bond order; surviving child bonds precede it.
            for atom in join.interface.ordered_input_atoms:
                for edge in edges:
                    if atom in edge and len(set(edge) & shared) == 1 and edge not in ordered:
                        ordered.append(edge)
        for edge in ordered:
            key = frozenset(edge)
            if key not in bond_keys:
                bonds.append(edge)
                bond_keys.add(key)
        seen.update(local.values())
    expected_edges = {
        frozenset((a, b))
        for face in model.faces
        for a, b in zip(face.atom_cycle, face.atom_cycle[1:] + face.atom_cycle[:1])
    }
    if set(atom_order) != set(positions) or bond_keys != expected_edges:
        return None
    adjacency: dict[int, list[tuple[int, int]]] = {atom: [] for atom in atom_order}
    for index, (a, b) in enumerate(bonds):
        adjacency[a].append((b, index))
        adjacency[b].append((a, index))
    used: set[int] = set()
    parents: dict[int, tuple[int, int]] = {}
    links: list[int] = []
    # Explicit stack preserves Java's recursive neighbor order without a
    # recursion-depth dependency on the input graph.
    stack = [(atom_order[0], None, iter(adjacency[atom_order[0]]))]
    used.add(atom_order[0])
    while stack:
        atom, parent, neighbors = stack[-1]
        neighbor = next(neighbors, None)
        if neighbor is None:
            stack.pop()
            continue
        other, edge = neighbor
        if other == parent:
            continue
        if other in used:
            if edge not in links:
                links.append(edge)
        else:
            parents[other] = atom, edge
            used.add(other)
            stack.append((other, atom, iter(adjacency[other])))
    if used != set(atom_order):
        return None

    def ancestors(atom: int) -> list[int]:
        result = []
        while atom in parents:
            atom, edge = parents[atom]
            result.append(edge)
        return result

    def difference(left: list[int], right: list[int]) -> list[int]:
        return [edge for edge in left if edge not in right] + [edge for edge in right if edge not in left]

    rings = [difference(ancestors(bonds[e][0]), ancestors(bonds[e][1])) + [e] for e in links]
    remaining = search_budget
    change = True
    while change:
        for index in range(len(rings)):
            ring = rings[index]
            # SSSRFinder overwrites this flag for each current ring and reads
            # replacements in place. Sorting or snapshot iteration differs.
            change = False
            for target_index, target in enumerate(rings):
                remaining -= 1
                if remaining < 0:
                    return None
                if target is ring:
                    continue
                reduced = difference(target, ring)
                if len(reduced) < len(target):
                    rings[target_index] = reduced
                    change = True
        if not rings:
            return None
    face_sets = {frozenset(face.atom_cycle): face.id for face in model.faces}
    face_ids = [face_sets.get(frozenset(atom for edge in ring for atom in bonds[edge])) for ring in rings]
    if None in face_ids or len(set(face_ids)) != len(model.faces) or any(len(ring) != 6 for ring in rings):
        return None
    neighbors: list[dict[int, int]] = [{} for _ in rings]
    for index, ring in enumerate(rings):
        for edge in ring:
            for other in range(index + 1, len(rings)):
                if edge in rings[other]:
                    neighbors[index][other] = edge
                    neighbors[other][index] = edge
                    break
    root = min(range(len(rings)), key=lambda index: len(neighbors[index]))
    fusion_atoms = {atom for edge in neighbors[root].values() for atom in bonds[edge]}
    seed = next((edge for edge in rings[root] if not set(bonds[edge]) & fusion_atoms), None)
    if seed is None:
        seed = next((edge for edge in rings[root] if edge not in neighbors[root].values()), None)
    if seed is None:
        return None
    cycles: dict[int, list[int]] = {}
    visited: set[int] = set()
    directions: list[tuple[int, int, int]] = []

    def enter(ring: int, entry: int, atom: int) -> tuple[list[int], tuple[int, ...]] | None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            return None
        visited.add(ring)
        cycle = [atom]
        edges = [entry]
        while len(edges) < 6:
            edge = next((edge for edge in rings[ring] if edge not in edges and atom in bonds[edge]), None)
            if edge is None:
                return None
            atom = next(other for other in bonds[edge] if other != atom)
            cycle.append(atom)
            edges.append(edge)
        cycles[ring] = cycle
        choices = _allowed_entry_directions(6, frozenset(edges.index(edge) for edge in neighbors[ring].values()))
        return (edges, choices[0]) if len(choices) == 1 else None

    entered = enter(root, seed, bonds[seed][0])
    if entered is None:
        return None
    # Suspend each ring's neighbor iterator while descending, exactly as
    # buildRingConnectionTables does, without imposing a recursion limit.
    pending = [(root, None, 0, entered, iter(neighbors[root].items()))]
    while pending:
        ring, previous, incoming, (edges, relative), adjacent = pending[-1]
        neighbor = next(adjacent, None)
        if neighbor is None:
            pending.pop()
            continue
        other, edge = neighbor
        index = edges.index(edge)
        if other == previous:
            direction = _absolute_direction(4, incoming)
        elif index:
            direction = _absolute_direction(relative[index - 1], incoming)
        else:
            return None
        directions.append((ring, other, direction))
        if other not in visited:
            entered = enter(other, edge, cycles[ring][(index - 1) % 6])
            if entered is None:
                return None
            pending.append((other, ring, direction, entered, iter(neighbors[other].items())))
    root_cycle = cycles[root]
    area = sum(
        positions[a][0] * positions[b][1] - positions[b][0] * positions[a][1]
        for a, b in zip(root_cycle, root_cycle[1:] + root_cycle[:1])
    )
    lookup = {(a, b): direction for a, b, direction in directions}
    axes: list[int] = []
    longest = 0
    for _, start, direction in directions:
        count, current = 1, start
        for _ in range(len(rings) + 1):
            outgoing = [b for a, b, d in directions if a == current and d == direction]
            if not outgoing:
                break
            current = outgoing[0]
            count += 1
        else:
            return None
        if count > longest:
            axes, longest = [], count
        if count == longest and direction not in axes and _absolute_direction(direction, 4) not in axes:
            axes.append(direction)
    _, steps = _entry_direction_tables()
    options: list[tuple[tuple[int, int, int], int, int]] = []
    for axis in axes:
        centers = {root: (0, 0)}
        queue = deque([root])
        while queue:
            ring = queue.popleft()
            for other in neighbors[ring]:
                direction = _absolute_direction(lookup[ring, other], -axis)
                if direction not in steps:
                    return None
                dx, dy = steps[direction]
                point = centers[ring][0] + dx, centers[ring][1] + dy
                if other in centers:
                    if centers[other] != point:
                        return None
                else:
                    centers[other] = point
                    queue.append(other)
        if len(set(centers.values())) != len(rings):
            return None
        # Source order is axis, occupied row (bottom to top, left to right),
        # then quadrant 0..3. rulesBCD filters without reordering these paths.
        rows: list[list[int]] = []
        cells = {point: ring for ring, point in centers.items()}
        for y in sorted({y for _, y in cells}):
            x = min(x for x, _ in cells)
            max_x = max(x for x, _ in cells)
            while x <= max_x:
                if (x, y) not in cells:
                    x += 2
                    continue
                row = [cells[x, y]]
                while (x + 4 * len(row), y) in cells:
                    row.append(cells[x + 4 * len(row), y])
                rows.append(row)
                x += 4 * len(row) + 2
        longest_row = max(map(len, rows))
        for row in rows:
            if len(row) != longest_row:
                continue
            for quadrant, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
                oriented = {ring: (sx * x, sy * y) for ring, (x, y) in centers.items()}
                score = _center_row_orientation_score(oriented, row)
                top = max(oriented, key=lambda ring: (oriented[ring][1], oriented[ring][0]))
                if all(len(adjacency[atom]) == 3 for atom in cycles[top]):
                    return None
                winding = (1 if area > 0 else -1) * (-1 if quadrant in {0, 2} else 1)
                face_id = face_ids[top]
                assert face_id is not None
                options.append((score, face_id, winding))
    if not options:
        return None
    best = min(row[0] for row in options)
    return ConstructionOrder(
        tuple(atom_order), tuple((face, winding) for score, face, winding in options if score == best)
    )

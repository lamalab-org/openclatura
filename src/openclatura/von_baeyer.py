"""Audited extended von Baeyer candidate search.

This module treats von Baeyer naming as graph decomposition followed by
ranking and reconstruction audit.  It intentionally returns no candidate when
the current implementation cannot classify every bridge without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .molecule import Molecule
from .polycycle_topology import (
    RingNumbering,
    adjacency_from_edges,
    adjacent_atoms,
    build_von_baeyer_numbering,
    normalize_edges,
)
from .ring_renderer import render_von_baeyer_descriptor

# These are execution bounds, not nomenclature scope. The audited search is
# graph-reconstructive at every rank; the previous 8-ring/12-bridgehead tier
# excluded ordinary larger polycycles before doing any proof work.
MAX_AUDITED_VON_BAEYER_RINGS = 32
MAX_AUDITED_BRIDGEHEADS = 64
MAX_PATHS_PER_BRIDGEHEAD_PAIR = 96


@dataclass(frozen=True)
class VonBaeyerBridge:
    """A path segment between two already-numbered attachment atoms."""

    length: int
    attachments: tuple[int, int]
    atoms: tuple[int, ...] = ()
    dependent: bool = False


@dataclass(frozen=True)
class VonBaeyerCandidate:
    """One fully rendered and audited von Baeyer candidate."""

    descriptor: str
    path: tuple[int, ...]
    primary_lengths: tuple[int, int, int]
    secondary_bridges: tuple[VonBaeyerBridge, ...]
    main_bridgeheads: tuple[int, int]
    rank: tuple
    numbering: RingNumbering


def find_von_baeyer_candidates(
    mol: Molecule,
    atoms: set[int] | frozenset[int],
    edges: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> tuple[VonBaeyerCandidate, ...]:
    """Enumerate ranked, audited von Baeyer candidates for a ring skeleton."""

    atom_set = frozenset(atoms)
    edge_set = frozenset(normalize_edges(edges))
    if not _is_von_baeyer_scope(mol, atom_set, edge_set):
        return ()

    adjacency = adjacency_from_edges(atom_set, edge_set)
    ring_count = len(edge_set) - len(atom_set) + 1
    bridgeheads = tuple(sorted(atom for atom in atom_set if len(adjacency[atom]) >= 3))
    if ring_count > MAX_AUDITED_VON_BAEYER_RINGS or len(bridgeheads) > MAX_AUDITED_BRIDGEHEADS:
        return ()
    candidates: list[VonBaeyerCandidate] = []

    for first, second in combinations(bridgeheads, 2):
        paths = _simple_paths_between(first, second, adjacency, max_paths=MAX_PATHS_PER_BRIDGEHEAD_PAIR)
        if len(paths) >= MAX_PATHS_PER_BRIDGEHEAD_PAIR:
            theta = _three_internally_disjoint_paths(first, second, adjacency)
            primary_path_sets = (theta,) if theta is not None else ()
        else:
            primary_path_sets = combinations(paths, 3)
        for primary_paths in primary_path_sets:
            if not _paths_are_internally_disjoint(primary_paths):
                continue
            for main_bridge_index in range(3):
                main_bridge = primary_paths[main_bridge_index]
                ring_paths = tuple(path for idx, path in enumerate(primary_paths) if idx != main_bridge_index)
                for candidate in _build_candidates_for_decomposition(
                    mol=mol,
                    atom_set=atom_set,
                    edge_set=edge_set,
                    primary_paths=ring_paths + (main_bridge,),
                    main_bridgeheads=(first, second),
                    ring_count=ring_count,
                ):
                    candidates.append(candidate)

    deduped = _dedupe_candidates(candidates)
    return tuple(sorted(deduped, key=lambda candidate: candidate.rank))


def _is_von_baeyer_scope(mol: Molecule, atoms: frozenset[int], edges: frozenset[tuple[int, int]]) -> bool:
    if len(edges) - len(atoms) + 1 < 3:
        return False
    adjacency = adjacency_from_edges(atoms, edges)
    # Free spiro centers are routed through the spiro/dispiro engine.  A
    # bridged von Baeyer bridgehead may also have degree >= 4, so only reject
    # articulation-style spiro centers here.
    if any(_is_free_spiro_center(atom, atoms, edges) for atom in atoms if len(adjacency[atom]) >= 4):
        return False
    return True


def _is_free_spiro_center(atom: int, atoms: frozenset[int], edges: frozenset[tuple[int, int]]) -> bool:
    remaining = set(atoms) - {atom}
    if not remaining:
        return False
    components = _connected_components(remaining, frozenset(edge for edge in edges if atom not in edge))
    return len(components) >= 2


def _build_candidates_for_decomposition(
    *,
    mol: Molecule,
    atom_set: frozenset[int],
    edge_set: frozenset[tuple[int, int]],
    primary_paths: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    main_bridgeheads: tuple[int, int],
    ring_count: int,
) -> tuple[VonBaeyerCandidate, ...]:
    first_ring, second_ring, main_bridge = primary_paths
    first_ring, second_ring = _order_main_ring_branches(first_ring, second_ring)
    if first_ring[0] != main_bridgeheads[0]:
        first_ring = tuple(reversed(first_ring))
    if second_ring[0] != main_bridgeheads[0]:
        second_ring = tuple(reversed(second_ring))
    if main_bridge[0] != main_bridgeheads[0]:
        main_bridge = tuple(reversed(main_bridge))

    primary_edges = _path_edges(first_ring) | _path_edges(second_ring) | _path_edges(main_bridge)
    primary_atoms = set(first_ring) | set(second_ring) | set(main_bridge)
    remaining_edges = edge_set - primary_edges
    secondary = _classify_secondary_bridges(
        atom_set=atom_set,
        edge_set=edge_set,
        primary_atoms=primary_atoms,
        remaining_edges=remaining_edges,
    )
    if secondary is None:
        return ()
    if len(secondary) + 2 != ring_count:
        return ()

    base_path = _numbering_path(first_ring, second_ring, main_bridge)
    locants = {atom: idx for idx, atom in enumerate(base_path, start=1)}
    path = list(base_path)
    pending = [bridge for bridge in secondary if bridge.length > 0]
    normalized_secondary: list[VonBaeyerBridge] = []
    # Citation order follows bridge length, but numbering follows bridgehead
    # seniority. Dependent ears become eligible as their attachment atoms are
    # numbered, which is the general open-ear form of a bridged polycycle.
    while pending:
        available = [
            bridge for bridge in pending if bridge.attachments[0] in locants and bridge.attachments[1] in locants
        ]
        if not available:
            return ()
        source = min(
            available,
            key=lambda item: (item.dependent, *_secondary_numbering_key(item, locants)),
        )
        bridge = _secondary_with_locants(source, locants)
        normalized_secondary.append(bridge)
        path.extend(_orient_bridge_atoms_for_descriptor(bridge, locants))
        locants = {atom: idx for idx, atom in enumerate(path, start=1)}
        pending.remove(source)
    for bridge in (item for item in secondary if item.length == 0):
        if bridge.attachments[0] not in locants or bridge.attachments[1] not in locants:
            return ()
        normalized_secondary.append(_secondary_with_locants(bridge, locants))
    secondary = tuple(sorted(normalized_secondary, key=lambda item: _secondary_citation_key(item, locants)))

    primary_lengths = (len(first_ring) - 2, len(second_ring) - 2, len(main_bridge) - 2)
    descriptor_body = _descriptor_body(primary_lengths, secondary, locants)
    descriptor = render_von_baeyer_descriptor(len(secondary) + 1, descriptor_body)
    numbering = build_von_baeyer_numbering(descriptor, path, edge_set, mol)
    if not numbering.audit_ok:
        return ()
    rank = _ranking_tuple(
        primary_lengths=primary_lengths,
        secondary=secondary,
        locants=locants,
        main_ring_atom_count=len(set(first_ring) | set(second_ring)),
        main_bridge_non_ring_atom_count=len(set(main_bridge[1:-1]) - (set(first_ring) | set(second_ring))),
    )
    return (
        VonBaeyerCandidate(
            descriptor=descriptor,
            path=tuple(path),
            primary_lengths=primary_lengths,
            secondary_bridges=secondary,
            main_bridgeheads=main_bridgeheads,
            rank=rank,
            numbering=numbering,
        ),
    )


def _classify_secondary_bridges(
    *,
    atom_set: frozenset[int],
    edge_set: frozenset[tuple[int, int]],
    primary_atoms: set[int],
    remaining_edges: frozenset[tuple[int, int]],
) -> tuple[VonBaeyerBridge, ...] | None:
    """Decompose the remainder into open ears plus zero-length chords.

    Every positive-length ear introduces previously unnumbered atoms between
    two numbered atoms. This handles branched fused remainders generically;
    the final descriptor reconstruction remains the proof that the chosen ears
    cover exactly the source graph.
    """

    simple = _classify_single_path_components(
        atom_set=atom_set,
        edge_set=edge_set,
        primary_atoms=primary_atoms,
        remaining_edges=remaining_edges,
    )
    if simple is not None:
        return simple
    return _classify_open_ears(
        atom_set=atom_set,
        primary_atoms=primary_atoms,
        remaining_edges=remaining_edges,
    )


def _classify_single_path_components(
    *,
    atom_set: frozenset[int],
    edge_set: frozenset[tuple[int, int]],
    primary_atoms: set[int],
    remaining_edges: frozenset[tuple[int, int]],
) -> tuple[VonBaeyerBridge, ...] | None:
    """Keep the preferred decomposition when each outside component is one ear."""

    direct_edges = {edge for edge in remaining_edges if edge[0] in primary_atoms and edge[1] in primary_atoms}
    covered_edges = set(direct_edges)
    bridges = [VonBaeyerBridge(length=0, attachments=tuple(sorted(edge))) for edge in direct_edges]
    outside_atoms = set(atom_set) - primary_atoms
    outside_edges = frozenset(edge for edge in edge_set if edge[0] in outside_atoms and edge[1] in outside_atoms)
    for component in _connected_components(outside_atoms, outside_edges):
        connections = sorted(
            (atom, node)
            for atom in primary_atoms
            for node in component
            if tuple(sorted((atom, node))) in remaining_edges
        )
        candidates = []
        for first, second in combinations(connections, 2):
            if first[0] == second[0]:
                continue
            internal_path = _component_path(component, first[0], second[0], edge_set)
            if set(internal_path) == component:
                candidates.append((first, second, internal_path))
        if not candidates:
            return None
        first, second, internal_path = min(candidates, key=lambda item: (-len(item[2]), item[0], item[1]))
        bridges.append(
            VonBaeyerBridge(
                length=len(component),
                attachments=(first[0], second[0]),
                atoms=tuple(internal_path),
            )
        )
        covered_edges.update(_path_edges((first[0], *internal_path, second[0])))
        consumed = {first, second}
        for attachment, component_atom in connections:
            if (attachment, component_atom) not in consumed:
                covered_edges.add(tuple(sorted((attachment, component_atom))))
                bridges.append(
                    VonBaeyerBridge(
                        length=0,
                        attachments=tuple(sorted((attachment, component_atom))),
                        dependent=True,
                    )
                )
    if covered_edges != set(remaining_edges):
        return None
    return tuple(bridges)


def _classify_open_ears(
    *,
    atom_set: frozenset[int],
    primary_atoms: set[int],
    remaining_edges: frozenset[tuple[int, int]],
) -> tuple[VonBaeyerBridge, ...] | None:
    bridges: list[VonBaeyerBridge] = []
    known = set(primary_atoms)
    unused = set(remaining_edges)
    while known != atom_set:
        unknown = set(atom_set) - known
        unknown_edges = frozenset(edge for edge in unused if edge[0] in unknown and edge[1] in unknown)
        candidates = []
        for component in _connected_components(unknown, unknown_edges):
            connections = sorted(
                (numbered, node)
                for numbered in known
                for node in component
                if tuple(sorted((numbered, node))) in unused
            )
            for first, second in combinations(connections, 2):
                if first[0] == second[0]:
                    continue
                internal_path = _component_path(component, first[0], second[0], frozenset(unused))
                if not internal_path:
                    continue
                ear_edges = {
                    tuple(sorted((first[0], internal_path[0]))),
                    tuple(sorted((internal_path[-1], second[0]))),
                } | set(_path_edges(tuple(internal_path)))
                if not ear_edges <= unused:
                    continue
                candidates.append((first[0], second[0], tuple(internal_path), frozenset(ear_edges)))
        if not candidates:
            return None
        first, second, internal_path, ear_edges = min(
            candidates,
            key=lambda item: (-len(item[2]), tuple(sorted((item[0], item[1]))), item[2]),
        )
        bridges.append(
            VonBaeyerBridge(
                length=len(internal_path),
                attachments=(first, second),
                atoms=internal_path,
                dependent=first not in primary_atoms or second not in primary_atoms,
            )
        )
        known.update(internal_path)
        unused.difference_update(ear_edges)

    if any(first not in known or second not in known for first, second in unused):
        return None
    for first, second in sorted(unused):
        bridges.append(
            VonBaeyerBridge(
                length=0,
                attachments=(first, second),
                dependent=first not in primary_atoms or second not in primary_atoms,
            )
        )
    return tuple(bridges)


def _secondary_with_locants(bridge: VonBaeyerBridge, locants: dict[int, int]) -> VonBaeyerBridge:
    if bridge.attachments[0] not in locants or bridge.attachments[1] not in locants:
        return bridge
    first, second = bridge.attachments
    atoms = bridge.atoms
    if locants[first] > locants[second]:
        first, second = second, first
        atoms = tuple(reversed(atoms))
    return VonBaeyerBridge(
        length=bridge.length,
        attachments=(first, second),
        atoms=atoms,
        dependent=bridge.dependent,
    )


def _secondary_citation_key(bridge: VonBaeyerBridge, locants: dict[int, int]) -> tuple:
    cited = tuple(sorted(locants[atom] for atom in bridge.attachments))
    return (-bridge.length, bridge.dependent, cited)


def _secondary_numbering_key(bridge: VonBaeyerBridge, locants: dict[int, int]) -> tuple:
    lower, higher = sorted(locants[atom] for atom in bridge.attachments)
    return (-higher, -lower)


def _orient_bridge_atoms_for_descriptor(bridge: VonBaeyerBridge, locants: dict[int, int]) -> tuple[int, ...]:
    if bridge.length == 0:
        return ()
    first, second = bridge.attachments
    atoms = bridge.atoms
    if not atoms:
        return ()
    first_locant = locants[first]
    second_locant = locants[second]
    # P-23 numbering continues from the higher-numbered bridgehead toward the
    # lower one. ``_secondary_with_locants`` keeps ``atoms`` directed from the
    # lower attachment to the higher attachment, so descriptor order is the
    # reverse of that path.
    if first_locant <= second_locant:
        return tuple(reversed(atoms))
    return atoms


def _descriptor_body(
    primary_lengths: tuple[int, int, int],
    secondary: tuple[VonBaeyerBridge, ...],
    locants: dict[int, int],
) -> str:
    parts = [str(length) for length in primary_lengths]
    for bridge in secondary:
        first, second = sorted((locants[bridge.attachments[0]], locants[bridge.attachments[1]]))
        parts.append(f"{bridge.length}^{{{first},{second}}}")
    return "[" + ".".join(parts) + "]"


def _ranking_tuple(
    *,
    primary_lengths: tuple[int, int, int],
    secondary: tuple[VonBaeyerBridge, ...],
    locants: dict[int, int],
    main_ring_atom_count: int,
    main_bridge_non_ring_atom_count: int,
) -> tuple:
    independent_lengths = tuple(-bridge.length for bridge in secondary if not bridge.dependent)
    dependent_count = sum(1 for bridge in secondary if bridge.dependent)
    secondary_locants_sorted = tuple(sorted(locants[atom] for bridge in secondary for atom in bridge.attachments))
    secondary_locants_in_order = tuple(locants[atom] for bridge in secondary for atom in bridge.attachments)
    return (
        -main_ring_atom_count,
        -main_bridge_non_ring_atom_count,
        -primary_lengths[0],
        -primary_lengths[1],
        independent_lengths,
        dependent_count,
        secondary_locants_sorted,
        secondary_locants_in_order,
        primary_lengths,
    )


def _numbering_path(
    first_ring: tuple[int, ...],
    second_ring: tuple[int, ...],
    main_bridge: tuple[int, ...],
) -> tuple[int, ...]:
    return first_ring + tuple(reversed(second_ring[1:-1])) + main_bridge[1:-1]


def _order_main_ring_branches(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    first_len = len(first) - 2
    second_len = len(second) - 2
    if first_len > second_len:
        return first, second
    if second_len > first_len:
        return second, first
    return min((first, second), (tuple(reversed(first)), tuple(reversed(second))))


def _simple_paths_between(
    start: int,
    end: int,
    adjacency: dict[int, set[int]],
    *,
    max_paths: int,
) -> tuple[tuple[int, ...], ...]:
    paths: list[tuple[int, ...]] = []
    stack = [(start, (start,))]
    while stack and len(paths) < max_paths:
        current, path = stack.pop()
        for neighbor in sorted(adjacency[current], reverse=True):
            if neighbor == end:
                paths.append(path + (neighbor,))
            elif neighbor not in path:
                stack.append((neighbor, path + (neighbor,)))
    return tuple(sorted(paths, key=lambda path: (-len(path), path)))


def _three_internally_disjoint_paths(
    start: int,
    end: int,
    adjacency: dict[int, set[int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Find a theta core with a vertex-capacitated three-unit flow.

    Undirected edges use capacity-one gadgets, avoiding ambiguous residual
    flow on antiparallel arcs. Internal atom capacity is one; the two selected
    bridgeheads have capacity three. The returned paths are independently
    checked before candidate construction and later by descriptor audit.
    """

    capacities: dict[tuple, dict[tuple, int]] = {}

    def add_arc(source: tuple, target: tuple, capacity: int) -> None:
        capacities.setdefault(source, {})[target] = capacity
        capacities.setdefault(target, {})

    for atom in adjacency:
        add_arc(("in", atom), ("out", atom), 3 if atom in {start, end} else 1)
    graph_edges = sorted((first, second) for first in adjacency for second in adjacency[first] if first < second)
    for edge_index, (first, second) in enumerate(graph_edges):
        edge_in = ("edge_in", edge_index)
        edge_out = ("edge_out", edge_index)
        add_arc(("out", first), edge_in, 1)
        add_arc(("out", second), edge_in, 1)
        add_arc(edge_in, edge_out, 1)
        add_arc(edge_out, ("in", first), 1)
        add_arc(edge_out, ("in", second), 1)

    residual = {node: dict(targets) for node, targets in capacities.items()}
    for source, targets in capacities.items():
        for target in targets:
            residual.setdefault(target, {}).setdefault(source, 0)
    source = ("out", start)
    sink = ("in", end)
    for _ in range(3):
        queue = [source]
        previous = {source: None}
        while queue and sink not in previous:
            current = queue.pop(0)
            for neighbor in sorted(residual[current], key=str):
                if residual[current][neighbor] > 0 and neighbor not in previous:
                    previous[neighbor] = current
                    queue.append(neighbor)
        if sink not in previous:
            return None
        current = sink
        while previous[current] is not None:
            parent = previous[current]
            residual[parent][current] -= 1
            residual[current][parent] += 1
            current = parent

    directed_edges: dict[int, list[int]] = {}
    for edge_index, (first, second) in enumerate(graph_edges):
        edge_in = ("edge_in", edge_index)
        edge_out = ("edge_out", edge_index)
        entered = [atom for atom in (first, second) if residual[("out", atom)][edge_in] == 0]
        exited = [atom for atom in (first, second) if residual[edge_out][("in", atom)] == 0]
        if len(entered) == 1 and len(exited) == 1 and entered[0] != exited[0]:
            directed_edges.setdefault(entered[0], []).append(exited[0])

    paths = []
    for first_step in sorted(directed_edges.get(start, ())):
        path = [start, first_step]
        while path[-1] != end:
            choices = [atom for atom in directed_edges.get(path[-1], ()) if atom not in path]
            if len(choices) != 1:
                return None
            path.append(choices[0])
        paths.append(tuple(path))
    result = tuple(paths)
    if len(result) != 3 or not _paths_are_internally_disjoint(result):
        return None
    return result


def _paths_are_internally_disjoint(paths: tuple[tuple[int, ...], ...]) -> bool:
    seen: set[int] = set()
    for path in paths:
        internal = set(path[1:-1])
        if seen & internal:
            return False
        seen.update(internal)
    endpoints = {(path[0], path[-1]) for path in paths}
    return len(endpoints) == 1


def _component_path(
    component: set[int],
    first_attachment: int,
    second_attachment: int,
    edge_set: frozenset[tuple[int, int]],
) -> tuple[int, ...]:
    starts = sorted(atom for atom in component if tuple(sorted((first_attachment, atom))) in edge_set)
    ends = {atom for atom in component if tuple(sorted((second_attachment, atom))) in edge_set}
    for start in starts:
        queue = [(start, (start,))]
        seen = {start}
        while queue:
            current, path = queue.pop(0)
            if current in ends:
                return path
            for neighbor in sorted(adjacent_atoms(current, edge_set)):
                if neighbor in component and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + (neighbor,)))
    return ()


def _connected_components(atoms: set[int], edge_set: frozenset[tuple[int, int]]) -> list[set[int]]:
    components = []
    seen: set[int] = set()
    for atom in sorted(atoms):
        if atom in seen:
            continue
        queue = [atom]
        seen.add(atom)
        component = set()
        while queue:
            current = queue.pop(0)
            component.add(current)
            for neighbor in sorted(adjacent_atoms(current, edge_set)):
                if neighbor in atoms and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _path_edges(path: tuple[int, ...]) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(sorted((first, second))) for first, second in zip(path, path[1:]))


def _dedupe_candidates(candidates: list[VonBaeyerCandidate]) -> list[VonBaeyerCandidate]:
    deduped = []
    seen = set()
    for candidate in candidates:
        key = (candidate.descriptor, candidate.path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped

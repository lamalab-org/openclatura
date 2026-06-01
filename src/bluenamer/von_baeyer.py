"""Audited extended von Baeyer candidate search.

This module treats von Baeyer naming as graph decomposition followed by
ranking and reconstruction audit.  It intentionally returns no candidate when
the current implementation cannot classify every bridge without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .molecule import Molecule
from .polycycle_topology import RingNumbering, build_von_baeyer_numbering
from .ring_renderer import render_von_baeyer_descriptor

MAX_AUDITED_VON_BAEYER_RINGS = 8
MAX_AUDITED_BRIDGEHEADS = 12
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
    edge_set = frozenset(_normalize_edges(edges))
    if not _is_von_baeyer_scope(mol, atom_set, edge_set):
        return ()

    adjacency = _adjacency(atom_set, edge_set)
    ring_count = len(edge_set) - len(atom_set) + 1
    bridgeheads = tuple(sorted(atom for atom in atom_set if len(adjacency[atom]) >= 3))
    if ring_count > MAX_AUDITED_VON_BAEYER_RINGS or len(bridgeheads) > MAX_AUDITED_BRIDGEHEADS:
        return ()
    candidates: list[VonBaeyerCandidate] = []

    for first, second in combinations(bridgeheads, 2):
        paths = _simple_paths_between(first, second, adjacency, max_paths=MAX_PATHS_PER_BRIDGEHEAD_PAIR)
        if len(paths) >= MAX_PATHS_PER_BRIDGEHEAD_PAIR:
            continue
        for primary_paths in combinations(paths, 3):
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


def best_von_baeyer_candidate(
    mol: Molecule,
    atoms: set[int] | frozenset[int],
    edges: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> VonBaeyerCandidate | None:
    candidates = find_von_baeyer_candidates(mol, atoms, edges)
    return candidates[0] if candidates else None


def _is_von_baeyer_scope(mol: Molecule, atoms: frozenset[int], edges: frozenset[tuple[int, int]]) -> bool:
    if len(edges) - len(atoms) + 1 < 3:
        return False
    if any(mol.atoms[atom].is_aromatic for atom in atoms):
        return False
    adjacency = _adjacency(atoms, edges)
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

    primary_edges = (
        _path_edges(first_ring)
        | _path_edges(second_ring)
        | _path_edges(main_bridge)
    )
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
    secondary = tuple(sorted(secondary, key=_secondary_citation_key))
    path = list(base_path)
    ordered_secondary: list[VonBaeyerBridge] = []
    for bridge in secondary:
        if bridge.attachments[0] not in locants or bridge.attachments[1] not in locants:
            return ()
        bridge = _secondary_with_locants(bridge, locants)
        ordered_secondary.append(bridge)
        path.extend(_orient_bridge_atoms_for_descriptor(bridge, locants))
        locants = {atom: idx for idx, atom in enumerate(path, start=1)}
    secondary = tuple(ordered_secondary)

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
    bridges: list[VonBaeyerBridge] = []
    direct_edges = [edge for edge in remaining_edges if edge[0] in primary_atoms and edge[1] in primary_atoms]
    for first, second in direct_edges:
        bridges.append(VonBaeyerBridge(length=0, attachments=tuple(sorted((first, second)))))

    outside_atoms = atom_set - primary_atoms
    outside_edges = frozenset(edge for edge in edge_set if edge[0] in outside_atoms and edge[1] in outside_atoms)
    outside_components = _connected_components(outside_atoms, outside_edges)
    used_outside: set[int] = set()
    for component in outside_components:
        connections = sorted(
            (atom, node)
            for atom in primary_atoms
            for node in component
            if tuple(sorted((atom, node))) in remaining_edges
        )
        attachments = sorted({atom for atom, _node in connections})
        if len(attachments) < 2:
            return None
        independent = _choose_independent_secondary_bridge(component, connections, edge_set)
        if independent is None:
            return None
        first_attachment, second_attachment, internal_path = independent
        if not internal_path or set(internal_path) != component:
            return None
        used_outside.update(component)
        bridges.append(
            VonBaeyerBridge(
                length=len(component),
                attachments=(first_attachment[0], second_attachment[0]),
                atoms=tuple(internal_path),
            )
        )
        for attachment_atom, component_atom in connections:
            if (attachment_atom, component_atom) in {first_attachment, second_attachment}:
                continue
            bridges.append(
                VonBaeyerBridge(
                    length=0,
                    attachments=tuple(sorted((attachment_atom, component_atom))),
                    dependent=True,
                )
            )
    if used_outside != outside_atoms:
        return None
    return tuple(bridges)


def _choose_independent_secondary_bridge(
    component: set[int],
    connections: list[tuple[int, int]],
    edge_set: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, ...]] | None:
    candidates = []
    for first, second in combinations(connections, 2):
        if first[0] == second[0] or first[1] == second[1]:
            continue
        internal_path = _component_path(component, first[0], second[0], edge_set)
        if set(internal_path) != component:
            continue
        candidates.append((first, second, internal_path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-len(item[2]), item[0], item[1]))[0]


def _secondary_with_locants(bridge: VonBaeyerBridge, locants: dict[int, int]) -> VonBaeyerBridge:
    if bridge.attachments[0] not in locants or bridge.attachments[1] not in locants:
        return bridge
    return VonBaeyerBridge(
        length=bridge.length,
        attachments=tuple(sorted(bridge.attachments, key=lambda atom: locants[atom])),
        atoms=bridge.atoms,
        dependent=bridge.dependent,
    )


def _secondary_citation_key(bridge: VonBaeyerBridge) -> tuple:
    # Independent before dependent; longer bridges first.  Locants break ties
    # after the graph-derived numbering is known.
    return (1 if bridge.dependent else 0, -bridge.length, bridge.attachments)


def _orient_bridge_atoms_for_descriptor(bridge: VonBaeyerBridge, locants: dict[int, int]) -> tuple[int, ...]:
    if bridge.length == 0:
        return ()
    first, second = bridge.attachments
    atoms = bridge.atoms
    if not atoms:
        return ()
    first_locant = locants[first]
    second_locant = locants[second]
    # The descriptor parser stores attachment locants in ascending order, so
    # internal bridge atoms must be listed from lower locant to higher locant.
    if first_locant <= second_locant:
        return atoms
    return tuple(reversed(atoms))


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
        abs(primary_lengths[0] - primary_lengths[1]),
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
            for neighbor in sorted(_neighbors(current, edge_set)):
                if neighbor in component and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + (neighbor,)))
    return ()


def _is_unbranched_component(component: set[int], edge_set: frozenset[tuple[int, int]], attachments: set[int]) -> bool:
    allowed = component | attachments
    for atom in component:
        degree = sum(1 for neighbor in _neighbors(atom, edge_set) if neighbor in allowed)
        if degree > 2:
            return False
    return True


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
            for neighbor in sorted(_neighbors(current, edge_set)):
                if neighbor in atoms and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _path_edges(path: tuple[int, ...]) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(sorted((first, second))) for first, second in zip(path, path[1:]))


def _adjacency(atoms: frozenset[int], edges: frozenset[tuple[int, int]]) -> dict[int, set[int]]:
    adjacency = {atom: set() for atom in atoms}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    return adjacency


def _neighbors(atom: int, edges: frozenset[tuple[int, int]]) -> set[int]:
    neighbors = set()
    for first, second in edges:
        if first == atom:
            neighbors.add(second)
        elif second == atom:
            neighbors.add(first)
    return neighbors


def _normalize_edges(edges) -> set[tuple[int, int]]:
    return {tuple(sorted((first, second))) for first, second in edges}


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

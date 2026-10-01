"""Audited extended von Baeyer candidate search.

This module treats von Baeyer naming as graph decomposition followed by
ranking and reconstruction audit.  It intentionally returns no candidate when
the current implementation cannot classify every bridge without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

from .canonical_ranks import canonical_ranks
from .molecule import Molecule
from .polycycle_topology import (
    RingNumbering,
    adjacency_from_edges,
    adjacent_atoms,
    build_von_baeyer_numbering,
    normalize_edges,
)
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
    """One rendered von Baeyer candidate; ``numbering`` is its audit, once run."""

    descriptor: str
    path: tuple[int, ...]
    primary_lengths: tuple[int, int, int]
    secondary_bridges: tuple[VonBaeyerBridge, ...]
    main_bridgeheads: tuple[int, int]
    rank: tuple
    numbering: RingNumbering | None


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
    # Which decompositions get enumerated at all must not depend on the atom
    # numbering the caller happened to arrive with: the path search is bounded,
    # so a different walk order truncates a different subset, and a skeleton
    # can lose the decomposition P-23.2.6.2 would have preferred. Walking in
    # canonical rank order makes the bound fall in the same place whatever the
    # input numbering.
    ranks = canonical_ranks(mol, atom_set)
    bridgeheads = tuple(sorted((atom for atom in atom_set if len(adjacency[atom]) >= 3), key=ranks.__getitem__))
    if ring_count > MAX_AUDITED_VON_BAEYER_RINGS or len(bridgeheads) > MAX_AUDITED_BRIDGEHEADS:
        return ()
    candidates: list[VonBaeyerCandidate] = []

    for first, second in combinations(bridgeheads, 2):
        paths = _simple_paths_between(first, second, adjacency, ranks, max_paths=MAX_PATHS_PER_BRIDGEHEAD_PAIR)
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
    # P-23.2.6.2 is applied in order until a decision is made, and a symmetric
    # skeleton can exhaust it with several decompositions still tied. Those are
    # genuinely equivalent, so the same canonical order settles them.
    ordered = sorted(deduped, key=lambda candidate: (candidate.rank, tuple(ranks[atom] for atom in candidate.path)))
    # Reconstructing a descriptor and comparing it to the graph is the expensive
    # part, and a skeleton can offer a hundred decompositions the criteria have
    # already rejected. Audit in preference order and stop at the first that
    # proves itself; a descriptor fixes the primary lengths and the secondary
    # locants, so everything that shares it shares its rank and is adjacent.
    accepted: list[VonBaeyerCandidate] = []
    for candidate in ordered:
        if accepted and candidate.descriptor != accepted[0].descriptor:
            if candidate.rank > accepted[0].rank:
                break
            continue
        numbering = build_von_baeyer_numbering(candidate.descriptor, candidate.path, edge_set, mol)
        if numbering.audit_ok:
            accepted.append(replace(candidate, numbering=numbering))
    return tuple(accepted)


def _is_von_baeyer_scope(mol: Molecule, atoms: frozenset[int], edges: frozenset[tuple[int, int]]) -> bool:
    if len(edges) - len(atoms) + 1 < 3:
        return False
    if any(mol.atoms[atom].is_aromatic for atom in atoms):
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
    first_path, second_path, main_path = primary_paths
    primary_edges = _path_edges(first_path) | _path_edges(second_path) | _path_edges(main_path)
    primary_atoms = set(first_path) | set(second_path) | set(main_path)
    secondary = _classify_secondary_bridges(
        atom_set=atom_set,
        edge_set=edge_set,
        primary_atoms=primary_atoms,
        remaining_edges=edge_set - primary_edges,
    )
    if secondary is None:
        return ()
    if len(secondary) + 2 != ring_count:
        return ()
    candidates = []
    for head, branches in _main_ring_orientations(first_path, second_path, main_path, main_bridgeheads):
        candidate = _candidate_for_orientation(
            mol=mol,
            edge_set=edge_set,
            branches=branches,
            secondary=secondary,
            main_bridgeheads=(head, main_bridgeheads[1] if head == main_bridgeheads[0] else main_bridgeheads[0]),
        )
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _main_ring_orientations(
    first: tuple[int, ...],
    second: tuple[int, ...],
    main_bridge: tuple[int, ...],
    main_bridgeheads: tuple[int, int],
):
    """Every numbering the main ring and its bridge admit.

    Locant 1 is a main bridgehead and the longer branch of the main ring is
    numbered first, but either bridgehead may take locant 1, and when the two
    branches are the same length either of them may go first. Settling that by
    atom id hides numberings P-23.2.6.2 might prefer -- basketane's
    0{2,5}.0{3,8}.0{4,7} was never enumerated -- and makes the choice depend on
    how the caller happened to number the molecule.
    """

    for head in main_bridgeheads:
        oriented = tuple(path if path[0] == head else tuple(reversed(path)) for path in (first, second, main_bridge))
        branch_one, branch_two, bridge = oriented
        if len(branch_one) >= len(branch_two):
            yield head, (branch_one, branch_two, bridge)
        if len(branch_two) >= len(branch_one):
            yield head, (branch_two, branch_one, bridge)


def _candidate_for_orientation(
    *,
    mol: Molecule,
    edge_set: frozenset[tuple[int, int]],
    branches: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    secondary: tuple[VonBaeyerBridge, ...],
    main_bridgeheads: tuple[int, int],
) -> VonBaeyerCandidate | None:
    first_ring, second_ring, main_bridge = branches
    base_path = _numbering_path(first_ring, second_ring, main_bridge)
    locants = {atom: idx for idx, atom in enumerate(base_path, start=1)}
    path = list(base_path)
    remaining = list(secondary)
    ordered_secondary: list[VonBaeyerBridge] = []
    # One bridge at a time, against the locants that exist when its turn comes:
    # a dependent bridge hangs off an earlier one and is only placeable after
    # it, so sorting the whole set up front against the main bicycle is not the
    # same thing.
    while remaining:
        ready = [
            (index, bridge)
            for index, bridge in enumerate(remaining)
            if all(atom in locants for atom in bridge.attachments)
        ]
        if not ready:
            return None
        index, bridge = min(ready, key=lambda item: _secondary_citation_key(item[1], locants))
        remaining.pop(index)
        bridge = _secondary_with_locants(bridge, locants)
        ordered_secondary.append(bridge)
        path.extend(_orient_bridge_atoms_for_descriptor(bridge, locants))
        locants = {atom: idx for idx, atom in enumerate(path, start=1)}
    cited = tuple(ordered_secondary)

    primary_lengths = (len(first_ring) - 2, len(second_ring) - 2, len(main_bridge) - 2)
    descriptor_body = _descriptor_body(primary_lengths, cited, locants)
    descriptor = render_von_baeyer_descriptor(len(cited) + 1, descriptor_body)
    rank = _ranking_tuple(
        primary_lengths=primary_lengths,
        secondary=cited,
        locants=locants,
        main_ring_atom_count=len(set(first_ring) | set(second_ring)),
        main_bridge_non_ring_atom_count=len(set(main_bridge[1:-1]) - (set(first_ring) | set(second_ring))),
    )
    return VonBaeyerCandidate(
        descriptor=descriptor,
        path=tuple(path),
        primary_lengths=primary_lengths,
        secondary_bridges=cited,
        main_bridgeheads=main_bridgeheads,
        rank=rank,
        numbering=None,
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


def _secondary_citation_key(bridge: VonBaeyerBridge, locants: dict[int, int]) -> tuple:
    """Order one secondary bridge against the others.

    P-23.2.6.1.3 cites independent bridges before dependent ones and the bridge
    lengths in decreasing order; P-23.2.6.2.5 settles what is left by the
    superscript locants taken in citation order, which for equal-length bridges
    puts the lower bridgehead pair first. Keying that on the bridgeheads rather
    than on the attachment atoms is also what makes the descriptor independent
    of the order the caller happened to number the molecule in.

    Interiors are currently numbered in this same order, which P-23.2.6.3 does
    not ask for -- it numbers from the bridge at the highest-numbered bridgehead
    -- but the descriptor audit reconstructs them in citation order, so the two
    have to agree until it learns that rule. Only systems with two or more
    bridges that have interiors can tell the difference.
    """

    return (
        1 if bridge.dependent else 0,
        -bridge.length,
        tuple(sorted(locants[atom] for atom in bridge.attachments)),
        bridge.attachments,
    )


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


def _simple_paths_between(
    start: int,
    end: int,
    adjacency: dict[int, set[int]],
    ranks: dict[int, int],
    *,
    max_paths: int,
) -> tuple[tuple[int, ...], ...]:
    paths: list[tuple[int, ...]] = []
    stack = [(start, (start,))]
    while stack and len(paths) < max_paths:
        current, path = stack.pop()
        for neighbor in sorted(adjacency[current], key=ranks.__getitem__, reverse=True):
            if neighbor == end:
                paths.append(path + (neighbor,))
            elif neighbor not in path:
                stack.append((neighbor, path + (neighbor,)))
    return tuple(sorted(paths, key=lambda path: (-len(path), tuple(ranks[atom] for atom in path))))


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

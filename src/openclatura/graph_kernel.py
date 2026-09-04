"""Dependency-free undirected graph primitives shared by naming subsystems."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

Edge = tuple[int, int]


def normalize_edge(first: int, second: int) -> Edge:
    if first == second:
        raise ValueError("self edges are not supported")
    return (first, second) if first < second else (second, first)


def normalize_edges(edges: Iterable[tuple[int, int]]) -> frozenset[Edge]:
    return frozenset(normalize_edge(first, second) for first, second in edges)


def adjacency_from_edges(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    node_set = frozenset(nodes)
    adjacency = {node: set() for node in node_set}
    for first, second in normalize_edges(edges):
        if first not in node_set or second not in node_set:
            raise ValueError("graph edge references an unknown node")
        adjacency[first].add(second)
        adjacency[second].add(first)
    return adjacency


def connected_components(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> list[set[int]]:
    node_set = frozenset(nodes)
    adjacency = adjacency_from_edges(node_set, induced_edges(node_set, edges))
    components: list[set[int]] = []
    seen: set[int] = set()
    for start in sorted(node_set):
        if start in seen:
            continue
        component: set[int] = set()
        pending = deque((start,))
        seen.add(start)
        while pending:
            current = pending.popleft()
            component.add(current)
            for neighbor in adjacency[current] - seen:
                seen.add(neighbor)
                pending.append(neighbor)
        components.append(component)
    return components


def induced_edges(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> frozenset[Edge]:
    node_set = frozenset(nodes)
    return frozenset(edge for edge in normalize_edges(edges) if set(edge) <= node_set)


def canonical_cycle(atoms: Iterable[int]) -> tuple[int, ...]:
    path = tuple(atoms)
    if not path:
        return ()
    variants = []
    for sequence in (path, tuple(reversed(path))):
        variants.extend(sequence[index:] + sequence[:index] for index in range(len(sequence)))
    return min(variants)


def cycle_edges(atoms: Iterable[int]) -> tuple[Edge, ...]:
    cycle = tuple(atoms)
    return tuple(normalize_edge(left, right) for left, right in zip(cycle, cycle[1:] + cycle[:1]))


def cycle_rank(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> int:
    node_set = frozenset(nodes)
    edge_set = normalize_edges(edges)
    if any(not set(edge) <= node_set for edge in edge_set):
        raise ValueError("graph edge references an unknown node")
    if not node_set or len(connected_components(node_set, edge_set)) != 1:
        raise ValueError("cycle rank requires a non-empty connected graph")
    return len(edge_set) - len(node_set) + 1


@dataclass(frozen=True, slots=True)
class GraphFace:
    """A validated bounded face with graph-derived atom and edge sets."""

    id: int
    atom_cycle: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("face id must be non-negative")
        if len(self.atom_cycle) < 3 or len(set(self.atom_cycle)) != len(self.atom_cycle):
            raise ValueError("face atom cycle must contain at least three distinct atoms")

    @property
    def atoms(self) -> frozenset[int]:
        return frozenset(self.atom_cycle)

    @property
    def edges(self) -> frozenset[Edge]:
        return frozenset(cycle_edges(self.atom_cycle))

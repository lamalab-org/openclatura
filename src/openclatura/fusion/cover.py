"""Typed component-overlap proofs for graph-native fusion nomenclature."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Generic, Literal, TypeVar

from ..polycycle_topology import normalize_edge

T = TypeVar("T", bound=Hashable)
Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ComponentScope(Generic[T]):
    """Graph scope occupied by one independently identified component."""

    key: T
    atom_ids: frozenset[int]
    edges: frozenset[Edge]


@dataclass(frozen=True, slots=True)
class FusionInterface(Generic[T]):
    """An overlap edge between two nodes in a component-cover graph."""

    left: T
    right: T
    shared_atom_ids: frozenset[int] = frozenset()
    shared_edges: frozenset[Edge] = frozenset()


@dataclass(frozen=True, slots=True)
class CoverGraph(Generic[T]):
    """A deterministic graph of component nodes and fusion interfaces."""

    nodes: tuple[T, ...]
    interfaces: tuple[FusionInterface[T], ...]
    adjacency: Mapping[T, tuple[T, ...]]


@dataclass(frozen=True, slots=True)
class CoverProof(Generic[T]):
    """Proof that a component cover is a supported tree or must abstain."""

    kind: Literal["tree", "non_tree", "disconnected"]
    cycle_rank: int


@dataclass(frozen=True, slots=True)
class ComponentCoverAudit(Generic[T]):
    """Coverage and overlap evidence for explicit component scopes."""

    ok: bool
    errors: tuple[str, ...]
    graph: CoverGraph[T]
    proof: CoverProof[T]
    reconstructed_atom_ids: frozenset[int]
    reconstructed_edges: frozenset[Edge]


def component_scope(key: T, atom_ids: Iterable[int], edges: Iterable[Edge]) -> ComponentScope[T]:
    """Build a normalized immutable component scope."""

    atoms = frozenset(atom_ids)
    normalized = frozenset(normalize_edge(*edge) for edge in edges)
    if any(left not in atoms or right not in atoms for left, right in normalized):
        raise ValueError("Every component edge must have both endpoints in atom_ids")
    return ComponentScope(key=key, atom_ids=atoms, edges=normalized)


def build_cover_graph(
    nodes: Iterable[T],
    interfaces: Iterable[tuple[T, T] | FusionInterface[T]],
) -> CoverGraph[T]:
    """Build a deterministic cover graph, deduplicating invalid interfaces."""

    ordered_nodes = tuple(dict.fromkeys(nodes))
    positions = {node: index for index, node in enumerate(ordered_nodes)}
    adjacency_lists: dict[T, list[T]] = {node: [] for node in ordered_nodes}
    normalized: list[FusionInterface[T]] = []
    seen: set[tuple[int, int]] = set()
    for value in interfaces:
        interface = value if isinstance(value, FusionInterface) else FusionInterface(*value)
        left, right = interface.left, interface.right
        if left == right or left not in positions or right not in positions:
            continue
        left_pos, right_pos = positions[left], positions[right]
        key = (left_pos, right_pos) if left_pos < right_pos else (right_pos, left_pos)
        if key in seen:
            continue
        seen.add(key)
        if left_pos > right_pos:
            interface = FusionInterface(
                left=right,
                right=left,
                shared_atom_ids=interface.shared_atom_ids,
                shared_edges=interface.shared_edges,
            )
        normalized.append(interface)
        adjacency_lists[interface.left].append(interface.right)
        adjacency_lists[interface.right].append(interface.left)
    adjacency = {
        node: tuple(sorted(neighbors, key=positions.__getitem__)) for node, neighbors in adjacency_lists.items()
    }
    normalized.sort(key=lambda edge: (positions[edge.left], positions[edge.right]))
    return CoverGraph(nodes=ordered_nodes, interfaces=tuple(normalized), adjacency=adjacency)


def build_component_cover_graph(scopes: Iterable[ComponentScope[T]]) -> CoverGraph[T]:
    """Build overlap interfaces for components sharing molecular edges.

    A one-atom overlap is spiro connectivity, not fusion, and therefore does
    not create an interface here.  Shared edge identity comes from the scopes,
    not inferred atom ordering.
    """

    ordered = tuple(scopes)
    keys = [scope.key for scope in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError("Component keys must be unique")
    interfaces: list[FusionInterface[T]] = []
    for left, right in combinations(ordered, 2):
        shared_edges = left.edges & right.edges
        if not shared_edges:
            continue
        shared_atoms = frozenset(atom for edge in shared_edges for atom in edge)
        interfaces.append(
            FusionInterface(
                left=left.key,
                right=right.key,
                shared_atom_ids=shared_atoms,
                shared_edges=shared_edges,
            )
        )
    return build_cover_graph(keys, interfaces)


def audit_component_cover(
    scopes: Iterable[ComponentScope[T]],
    *,
    target_atom_ids: Iterable[int],
    target_edges: Iterable[Edge],
) -> ComponentCoverAudit[T]:
    """Audit exact graph coverage and prove the component overlap topology."""

    ordered = tuple(scopes)
    graph = build_component_cover_graph(ordered)
    proof = build_cover_proof(graph)
    reconstructed_atoms = frozenset(atom for scope in ordered for atom in scope.atom_ids)
    reconstructed_edges = frozenset(edge for scope in ordered for edge in scope.edges)
    target_atoms = frozenset(target_atom_ids)
    normalized_target_edges = frozenset(normalize_edge(*edge) for edge in target_edges)
    errors: list[str] = []
    if len(ordered) < 2:
        errors.append("a fusion cover requires at least two components")
    if reconstructed_atoms != target_atoms:
        errors.append("component atoms do not exactly reconstruct the target")
    if reconstructed_edges != normalized_target_edges:
        errors.append("component edges do not exactly reconstruct the target")
    if proof.kind == "disconnected":
        errors.append("component overlap graph is disconnected")
    edge_counts: dict[Edge, int] = {}
    for scope in ordered:
        for edge in scope.edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if any(count > 2 for count in edge_counts.values()):
        errors.append("a molecular edge belongs to more than two components")
    return ComponentCoverAudit(
        ok=not errors,
        errors=tuple(errors),
        graph=graph,
        proof=proof,
        reconstructed_atom_ids=reconstructed_atoms,
        reconstructed_edges=reconstructed_edges,
    )


def is_connected(graph: CoverGraph[T]) -> bool:
    """Return whether every cover node is reachable."""

    return bool(graph.nodes) and len(_connected_nodes(graph, graph.nodes[0])) == len(graph.nodes)


def cycle_rank(graph: CoverGraph[T]) -> int:
    """Return the cyclomatic rank of a connected cover graph."""

    return len(graph.interfaces) - len(graph.nodes) + 1 if graph.nodes else 0


def build_cover_proof(graph: CoverGraph[T]) -> CoverProof[T]:
    """Classify the exact production boundary: connected tree or abstention."""

    if not graph.nodes or not is_connected(graph):
        return CoverProof("disconnected", cycle_rank(graph))
    rank = cycle_rank(graph)
    return CoverProof("tree" if rank == 0 else "non_tree", rank)


def _connected_nodes(graph: CoverGraph[T], start: T) -> set[T]:
    stack = [start]
    seen: set[T] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(neighbor for neighbor in graph.adjacency.get(node, ()) if neighbor not in seen)
    return seen


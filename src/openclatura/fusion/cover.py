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
class CoverBlock(Generic[T]):
    """One biconnected block of a cover graph."""

    kind: Literal["tree", "cycle", "general"]
    nodes: tuple[T, ...]
    interfaces: tuple[FusionInterface[T], ...]


@dataclass(frozen=True, slots=True)
class CycleCoverProof(Generic[T]):
    """Canonical traversal proof for a cyclic cover block."""

    ordered_nodes: tuple[T, ...]
    ordered_interfaces: tuple[FusionInterface[T], ...]


@dataclass(frozen=True, slots=True)
class CoverProof(Generic[T]):
    """Structural classification proof for a component-cover graph."""

    kind: Literal["tree", "cycle", "cactus", "complex", "disconnected"]
    cycle_rank: int
    articulation_nodes: tuple[T, ...]
    blocks: tuple[CoverBlock[T], ...]
    cycle_proofs: tuple[CycleCoverProof[T], ...]


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
    """Classify a cover graph and retain its biconnected-block evidence."""

    if not graph.nodes or not is_connected(graph):
        return CoverProof("disconnected", cycle_rank(graph), (), (), ())
    blocks = biconnected_blocks(graph)
    articulations = articulation_nodes(graph)
    cycle_proofs = tuple(_cycle_cover_proof(graph, block) for block in blocks if block.kind == "cycle")
    rank = cycle_rank(graph)
    if rank == 0:
        kind: Literal["tree", "cycle", "cactus", "complex", "disconnected"] = "tree"
    elif blocks and all(block.kind in {"tree", "cycle"} for block in blocks):
        kind = "cycle" if rank == 1 and len(cycle_proofs) == 1 else "cactus"
    else:
        kind = "complex"
    return CoverProof(kind, rank, articulations, blocks, cycle_proofs)


def articulation_nodes(graph: CoverGraph[T]) -> tuple[T, ...]:
    """Return articulation nodes in original node order."""

    positions = {node: index for index, node in enumerate(graph.nodes)}
    result = []
    for node in graph.nodes:
        others = tuple(item for item in graph.nodes if item != node)
        if others and len(_connected_nodes_without(graph, others[0], node)) != len(others):
            result.append(node)
    return tuple(sorted(result, key=positions.__getitem__))


def biconnected_blocks(graph: CoverGraph[T]) -> tuple[CoverBlock[T], ...]:
    """Return Tarjan biconnected blocks with deterministic ordering."""

    positions = {node: index for index, node in enumerate(graph.nodes)}
    discovery: dict[T, int] = {}
    low: dict[T, int] = {}
    parent: dict[T, T] = {}
    stack: list[FusionInterface[T]] = []
    blocks: list[CoverBlock[T]] = []
    counter = 0

    def pop_until(target: FusionInterface[T]) -> None:
        popped = []
        while stack:
            interface = stack.pop()
            popped.append(interface)
            if interface == target:
                break
        if popped:
            blocks.append(_cover_block(tuple(reversed(popped)), positions))

    def visit(node: T) -> None:
        nonlocal counter
        counter += 1
        discovery[node] = low[node] = counter
        child_count = 0
        for neighbor in graph.adjacency.get(node, ()):
            interface = _interface_for(graph, node, neighbor)
            if neighbor not in discovery:
                parent[neighbor] = node
                child_count += 1
                stack.append(interface)
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if (node not in parent and child_count > 1) or (node in parent and low[neighbor] >= discovery[node]):
                    pop_until(interface)
            elif parent.get(node) != neighbor and discovery[neighbor] < discovery[node]:
                low[node] = min(low[node], discovery[neighbor])
                stack.append(interface)

    for node in graph.nodes:
        if node not in discovery:
            visit(node)
            if stack:
                blocks.append(_cover_block(tuple(reversed(stack)), positions))
                stack.clear()
    return tuple(
        sorted(
            blocks,
            key=lambda block: (
                {"tree": 0, "cycle": 1, "general": 2}[block.kind],
                tuple(positions[node] for node in block.nodes),
            ),
        )
    )


def cycle_order_hint(proof: CoverProof[T]) -> dict[tuple[T, T], int]:
    """Return deterministic previous/next neighbor hints for cycle blocks."""

    hints: dict[tuple[T, T], int] = {}
    for cycle_proof in proof.cycle_proofs:
        nodes = cycle_proof.ordered_nodes
        for index, node in enumerate(nodes):
            hints.setdefault((node, nodes[index - 1]), 0)
            hints.setdefault((node, nodes[(index + 1) % len(nodes)]), 1)
    return hints


def block_cut_sets(
    graph: CoverGraph[T], block: CoverBlock[T], max_sets: int = 24
) -> tuple[tuple[FusionInterface[T], ...], ...]:
    """Return deterministic cuts whose complement is a spanning tree."""

    if block.kind != "general":
        return ((),)
    if max_sets < 1:
        return ()
    positions = {node: index for index, node in enumerate(graph.nodes)}
    edges = sorted(block.interfaces, key=lambda edge: (positions[edge.left], positions[edge.right]))
    kept_count = len(block.nodes) - 1
    cuts = []
    for kept in combinations(edges, kept_count):
        if not _interfaces_connect(block.nodes, kept):
            continue
        cuts.append(tuple(edge for edge in edges if edge not in kept))
        if len(cuts) >= max_sets:
            break
    return tuple(cuts)


def _cover_block(interfaces: tuple[FusionInterface[T], ...], positions: dict[T, int]) -> CoverBlock[T]:
    nodes = tuple(sorted({node for edge in interfaces for node in (edge.left, edge.right)}, key=positions.__getitem__))
    degrees = {node: 0 for node in nodes}
    for edge in interfaces:
        degrees[edge.left] += 1
        degrees[edge.right] += 1
    if len(interfaces) == len(nodes) and all(value == 2 for value in degrees.values()):
        kind: Literal["tree", "cycle", "general"] = "cycle"
    elif len(interfaces) == len(nodes) - 1:
        kind = "tree"
    else:
        kind = "general"
    return CoverBlock(kind, nodes, interfaces)


def _cycle_cover_proof(graph: CoverGraph[T], block: CoverBlock[T]) -> CycleCoverProof[T]:
    positions = {node: index for index, node in enumerate(graph.nodes)}
    adjacency_lists: dict[T, list[T]] = {node: [] for node in block.nodes}
    for edge in block.interfaces:
        adjacency_lists[edge.left].append(edge.right)
        adjacency_lists[edge.right].append(edge.left)
    adjacency = {node: tuple(sorted(values, key=positions.__getitem__)) for node, values in adjacency_lists.items()}
    start = min(block.nodes, key=positions.__getitem__)
    variants = [_walk_cycle(start, first, adjacency) for first in adjacency[start]]
    nodes = min(variants, key=lambda order: tuple(positions[node] for node in order))
    interfaces = tuple(_interface_for(graph, left, right) for left, right in zip(nodes, nodes[1:] + nodes[:1]))
    return CycleCoverProof(nodes, interfaces)


def _walk_cycle(start: T, first: T, adjacency: Mapping[T, tuple[T, ...]]) -> tuple[T, ...]:
    order = [start, first]
    previous, current = start, first
    while True:
        choices = [neighbor for neighbor in adjacency[current] if neighbor != previous]
        if len(choices) != 1:
            raise ValueError("Cycle block does not have a degree-two boundary")
        nxt = choices[0]
        if nxt == start:
            return tuple(order)
        if nxt in order:
            raise ValueError("Cycle traversal repeated a node")
        order.append(nxt)
        previous, current = current, nxt


def _interface_for(graph: CoverGraph[T], left: T, right: T) -> FusionInterface[T]:
    for interface in graph.interfaces:
        if {interface.left, interface.right} == {left, right}:
            return interface
    raise KeyError((left, right))


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


def _connected_nodes_without(graph: CoverGraph[T], start: T, removed: T) -> set[T]:
    stack = [start]
    seen: set[T] = set()
    while stack:
        node = stack.pop()
        if node == removed or node in seen:
            continue
        seen.add(node)
        stack.extend(
            neighbor for neighbor in graph.adjacency.get(node, ()) if neighbor != removed and neighbor not in seen
        )
    return seen


def _interfaces_connect(nodes: tuple[T, ...], interfaces: Iterable[FusionInterface[T]]) -> bool:
    if not nodes:
        return False
    adjacency: dict[T, list[T]] = {node: [] for node in nodes}
    for edge in interfaces:
        adjacency[edge.left].append(edge.right)
        adjacency[edge.right].append(edge.left)
    stack = [nodes[0]]
    seen: set[T] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(neighbor for neighbor in adjacency[node] if neighbor not in seen)
    return len(seen) == len(nodes)

"""Graph-native bounded-face discovery for fused ring systems.

The routines in this module use only :class:`~openclatura.molecule.Molecule`
connectivity.  They deliberately avoid drawing coordinates: a candidate face
model is accepted only when its cycles provide a complete combinatorial proof
for the selected molecular subgraph.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from ..molecule import Molecule

Edge = tuple[int, int]


class FaceSearchBudgetExceeded(RuntimeError):
    """Raised rather than returning an unsafe partial cycle/face result."""

    def __init__(self, phase: str, budget: int) -> None:
        super().__init__(f"{phase} search exceeded its budget of {budget} states")
        self.phase = phase
        self.budget = budget


@dataclass(frozen=True, slots=True)
class GraphCycle:
    """A simple cycle in canonical atom order."""

    atoms: tuple[int, ...]
    edges: frozenset[Edge]

    @classmethod
    def from_atoms(cls, atoms: Iterable[int]) -> GraphCycle:
        canonical = canonical_cycle(tuple(atoms))
        if len(canonical) < 3 or len(set(canonical)) != len(canonical):
            raise ValueError("A graph cycle needs at least three distinct atoms")
        return cls(atoms=canonical, edges=frozenset(cycle_edges(canonical)))


@dataclass(frozen=True, slots=True)
class FaceModelAudit:
    """Evidence collected while validating one bounded-face model."""

    ok: bool
    errors: tuple[str, ...]
    cycle_rank: int
    edge_multiplicity: tuple[tuple[Edge, int], ...]
    dual_adjacency: tuple[tuple[int, tuple[int, ...]], ...]
    outer_boundary: tuple[int, ...]
    reconstructed_edges: frozenset[Edge]


@dataclass(frozen=True, slots=True)
class BoundedFaceModel:
    """A deterministic, audited set of bounded faces for a ring graph."""

    atom_ids: frozenset[int]
    edge_ids: frozenset[Edge]
    faces: tuple[GraphCycle, ...]
    outer_boundary: GraphCycle
    cycle_rank: int
    audit: FaceModelAudit


@dataclass(slots=True)
class _Budget:
    phase: str
    limit: int
    used: int = 0

    def spend(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise FaceSearchBudgetExceeded(self.phase, self.limit)


def normalize_edge(left: int, right: int) -> Edge:
    """Return an undirected edge in stable endpoint order."""

    if left == right:
        raise ValueError("Self edges are not supported")
    return (left, right) if left < right else (right, left)


def cycle_edges(atoms: tuple[int, ...]) -> tuple[Edge, ...]:
    """Return the undirected edges around an ordered cycle."""

    return tuple(normalize_edge(left, right) for left, right in zip(atoms, atoms[1:] + atoms[:1]))


def canonical_cycle(atoms: tuple[int, ...]) -> tuple[int, ...]:
    """Canonicalize a cycle independently of start and traversal direction."""

    if not atoms:
        return ()
    variants: list[tuple[int, ...]] = []
    for order in (atoms, tuple(reversed(atoms))):
        for offset in range(len(order)):
            variants.append(order[offset:] + order[:offset])
    return min(variants)


def graph_cycle_rank(atom_ids: Iterable[int], edges: Iterable[Edge]) -> int:
    """Return ``E - V + 1`` for a connected graph, otherwise raise."""

    atoms = frozenset(atom_ids)
    normalized_edges = frozenset(normalize_edge(*edge) for edge in edges)
    if not atoms or not _is_connected(atoms, normalized_edges):
        raise ValueError("Cycle rank requires a non-empty connected graph")
    return len(normalized_edges) - len(atoms) + 1


def enumerate_chordless_cycles(
    mol: Molecule,
    atom_ids: Iterable[int] | None = None,
    *,
    min_size: int = 3,
    max_size: int = 8,
    search_budget: int = 100_000,
) -> tuple[GraphCycle, ...]:
    """Enumerate chordless cycles deterministically within a hard budget.

    ``min_size`` and ``max_size`` are intentionally constrained to the first
    production range, 3 through 8.  Exhausting the budget raises instead of
    exposing a partial cycle set that could be mistaken for a proof.
    """

    if not 3 <= min_size <= max_size <= 8:
        raise ValueError("Cycle sizes must satisfy 3 <= min_size <= max_size <= 8")
    if search_budget < 1:
        raise ValueError("search_budget must be positive")
    atoms = frozenset(mol.atoms if atom_ids is None else atom_ids)
    unknown = atoms - mol.atoms.keys()
    if unknown:
        raise KeyError(f"Unknown atom ids: {sorted(unknown)}")
    adjacency = {
        atom: tuple(sorted(neighbor for neighbor in mol.get_neighbors(atom) if neighbor in atoms))
        for atom in sorted(atoms)
    }
    budget = _Budget("cycle enumeration", search_budget)
    found: dict[tuple[int, ...], GraphCycle] = {}

    for start in sorted(atoms):
        _enumerate_from_start(start, adjacency, min_size, max_size, budget, found)

    return tuple(sorted(found.values(), key=lambda cycle: (len(cycle.atoms), cycle.atoms)))


def audit_bounded_face_model(
    mol: Molecule,
    atom_ids: Iterable[int],
    faces: Iterable[GraphCycle | Iterable[int]],
) -> FaceModelAudit:
    """Audit whether ``faces`` exactly prove a bounded-face decomposition."""

    atoms = frozenset(atom_ids)
    graph_edges = _molecule_edges(mol, atoms)
    normalized_faces = tuple(_as_cycle(face) for face in faces)
    errors: list[str] = []

    if not atoms:
        errors.append("ring graph is empty")
        rank = 0
    elif not _is_connected(atoms, graph_edges):
        errors.append("ring graph is disconnected")
        rank = len(graph_edges) - len(atoms) + _component_count(atoms, graph_edges)
    else:
        rank = len(graph_edges) - len(atoms) + 1
    if len(normalized_faces) != rank:
        errors.append(f"bounded face count {len(normalized_faces)} does not equal cycle rank {rank}")

    multiplicity: Counter[Edge] = Counter()
    for face_index, face in enumerate(normalized_faces):
        if not set(face.atoms) <= atoms:
            errors.append(f"face {face_index} contains atoms outside the ring graph")
        if not face.edges <= graph_edges:
            errors.append(f"face {face_index} contains edges outside the ring graph")
        if not _is_chordless_for_edges(face, graph_edges):
            errors.append(f"face {face_index} is not chordless")
        multiplicity.update(face.edges)

    reconstructed = frozenset(multiplicity)
    if reconstructed != graph_edges:
        errors.append("bounded faces do not exactly reconstruct the ring-graph edges")
    if any(count > 2 for count in multiplicity.values()):
        errors.append("a ring-graph edge belongs to more than two bounded faces")

    dual = _dual_adjacency(normalized_faces)
    if normalized_faces and not _dual_is_connected(dual):
        errors.append("bounded-face dual graph is disconnected")

    boundary_edges = frozenset(edge for edge, count in multiplicity.items() if count % 2 == 1)
    outer_boundary = _simple_cycle_order(boundary_edges)
    if outer_boundary is None:
        errors.append("XOR outer boundary is not a simple cycle")

    return FaceModelAudit(
        ok=not errors,
        errors=tuple(errors),
        cycle_rank=rank,
        edge_multiplicity=tuple(sorted(multiplicity.items())),
        dual_adjacency=tuple((index, dual[index]) for index in range(len(normalized_faces))),
        outer_boundary=outer_boundary or (),
        reconstructed_edges=reconstructed,
    )


def select_bounded_face_model(
    mol: Molecule,
    atom_ids: Iterable[int],
    *,
    min_ring_size: int = 3,
    max_ring_size: int = 8,
    cycle_search_budget: int = 100_000,
    model_search_budget: int = 50_000,
) -> BoundedFaceModel | None:
    """Return the first deterministic face model satisfying every audit.

    Candidates are ordered by total face perimeter and then canonical cycle
    order.  This favors elementary bounded faces without relying on geometry.
    ``None`` means that the completed search found no proven model; budget
    exhaustion remains an exception and is never conflated with no model.
    """

    atoms = frozenset(atom_ids)
    graph_edges = _molecule_edges(mol, atoms)
    try:
        rank = graph_cycle_rank(atoms, graph_edges)
    except ValueError:
        return None
    if rank < 1:
        return None
    cycles = enumerate_chordless_cycles(
        mol,
        atoms,
        min_size=min_ring_size,
        max_size=max_ring_size,
        search_budget=cycle_search_budget,
    )
    budget = _Budget("face-model selection", model_search_budget)
    valid: list[tuple[tuple[int, tuple[tuple[int, ...], ...]], BoundedFaceModel]] = []
    for face_indices in combinations(range(len(cycles)), rank):
        budget.spend()
        selected = tuple(cycles[index] for index in face_indices)
        # Cheap rejection before constructing the full audit record.
        if frozenset().union(*(face.edges for face in selected)) != graph_edges:
            continue
        audit = audit_bounded_face_model(mol, atoms, selected)
        if not audit.ok:
            continue
        outer = GraphCycle.from_atoms(audit.outer_boundary)
        model = BoundedFaceModel(
            atom_ids=atoms,
            edge_ids=graph_edges,
            faces=selected,
            outer_boundary=outer,
            cycle_rank=rank,
            audit=audit,
        )
        score = (sum(len(face.atoms) for face in selected), tuple(face.atoms for face in selected))
        valid.append((score, model))
    return min(valid, key=lambda item: item[0])[1] if valid else None


def _as_cycle(face: GraphCycle | Iterable[int]) -> GraphCycle:
    return face if isinstance(face, GraphCycle) else GraphCycle.from_atoms(face)


def _enumerate_from_start(
    start: int,
    adjacency: dict[int, tuple[int, ...]],
    min_size: int,
    max_size: int,
    budget: _Budget,
    found: dict[tuple[int, ...], GraphCycle],
) -> None:
    path = [start]
    visited = {start}

    def search(current: int) -> None:
        budget.spend()
        for neighbor in adjacency[current]:
            if neighbor == start:
                if len(path) >= min_size:
                    cycle = GraphCycle.from_atoms(path)
                    if _is_chordless(cycle, adjacency):
                        found[cycle.atoms] = cycle
                continue
            # Making ``start`` the smallest cycle atom removes rotational
            # duplicates before canonicalization and prunes the search.
            if neighbor < start or neighbor in visited or len(path) >= max_size:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            search(neighbor)
            path.pop()
            visited.remove(neighbor)

    search(start)


def _molecule_edges(mol: Molecule, atoms: frozenset[int]) -> frozenset[Edge]:
    return frozenset(
        normalize_edge(atom, neighbor)
        for atom in atoms
        for neighbor in mol.get_neighbors(atom)
        if neighbor in atoms and atom < neighbor
    )


def _is_chordless(cycle: GraphCycle, adjacency: dict[int, tuple[int, ...]]) -> bool:
    induced = {
        normalize_edge(atom, neighbor)
        for atom in cycle.atoms
        for neighbor in adjacency[atom]
        if neighbor in cycle.atoms and atom < neighbor
    }
    return induced == set(cycle.edges)


def _is_chordless_for_edges(cycle: GraphCycle, graph_edges: frozenset[Edge]) -> bool:
    cycle_atoms = set(cycle.atoms)
    induced = {edge for edge in graph_edges if edge[0] in cycle_atoms and edge[1] in cycle_atoms}
    return induced == set(cycle.edges)


def _adjacency(atoms: Iterable[int], edges: Iterable[Edge]) -> dict[int, tuple[int, ...]]:
    neighbors: dict[int, list[int]] = {atom: [] for atom in atoms}
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    return {atom: tuple(sorted(values)) for atom, values in neighbors.items()}


def _is_connected(atoms: frozenset[int], edges: frozenset[Edge]) -> bool:
    if not atoms:
        return False
    adjacency = _adjacency(atoms, edges)
    seen = _reachable(min(atoms), adjacency)
    return seen == set(atoms)


def _component_count(atoms: frozenset[int], edges: frozenset[Edge]) -> int:
    adjacency = _adjacency(atoms, edges)
    remaining = set(atoms)
    count = 0
    while remaining:
        count += 1
        remaining -= _reachable(min(remaining), adjacency)
    return count


def _reachable(start: int, adjacency: dict[int, tuple[int, ...]]) -> set[int]:
    stack = [start]
    seen: set[int] = set()
    while stack:
        atom = stack.pop()
        if atom in seen:
            continue
        seen.add(atom)
        stack.extend(neighbor for neighbor in adjacency[atom] if neighbor not in seen)
    return seen


def _dual_adjacency(faces: tuple[GraphCycle, ...]) -> dict[int, tuple[int, ...]]:
    neighbors: dict[int, list[int]] = {index: [] for index in range(len(faces))}
    for left, right in combinations(range(len(faces)), 2):
        if faces[left].edges & faces[right].edges:
            neighbors[left].append(right)
            neighbors[right].append(left)
    return {index: tuple(values) for index, values in neighbors.items()}


def _dual_is_connected(adjacency: dict[int, tuple[int, ...]]) -> bool:
    if not adjacency:
        return False
    stack = [0]
    seen: set[int] = set()
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        stack.extend(neighbor for neighbor in adjacency[index] if neighbor not in seen)
    return len(seen) == len(adjacency)


def _simple_cycle_order(edges: frozenset[Edge]) -> tuple[int, ...] | None:
    if len(edges) < 3:
        return None
    atoms = frozenset(atom for edge in edges for atom in edge)
    adjacency = _adjacency(atoms, edges)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()) or not _is_connected(atoms, edges):
        return None
    start = min(atoms)
    variants = []
    for first in adjacency[start]:
        order = [start]
        previous: int | None = None
        current = start
        nxt = first
        while nxt != start:
            if nxt in order:
                return None
            order.append(nxt)
            previous, current = current, nxt
            choices = [neighbor for neighbor in adjacency[current] if neighbor != previous]
            if len(choices) != 1:
                return None
            nxt = choices[0]
        if len(order) != len(atoms):
            return None
        variants.append(tuple(order))
    return min(variants)

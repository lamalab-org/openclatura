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

from ..canonical_ranks import canonical_ranks
from ..molecule import Molecule, edges_within_atoms
from ..polycycle_topology import (
    adjacency_from_edges,
    canonical_cycle,
    connected_components,
    cycle_edges,
    graph_cycle_rank,
    normalize_edge,
)
from .config import fusion_nomenclature_config

_CONFIG = fusion_nomenclature_config()

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

    @property
    def peripheral_atoms(self) -> frozenset[int]:
        return frozenset(self.outer_boundary.atoms)

    @property
    def interior_atoms(self) -> frozenset[int]:
        return self.atom_ids - self.peripheral_atoms

    @property
    def face_membership_by_atom(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        owners: dict[int, list[int]] = {atom: [] for atom in self.atom_ids}
        for face_id, face in enumerate(self.faces):
            for atom in face.atoms:
                owners[atom].append(face_id)
        return tuple((atom, tuple(face_ids)) for atom, face_ids in sorted(owners.items()))

    @property
    def fusion_atoms(self) -> frozenset[int]:
        return frozenset(atom for atom, face_ids in self.face_membership_by_atom if len(face_ids) > 1)


@dataclass(slots=True)
class _Budget:
    phase: str
    limit: int
    used: int = 0

    def spend(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise FaceSearchBudgetExceeded(self.phase, self.limit)


def enumerate_chordless_cycles(
    mol: Molecule,
    atom_ids: Iterable[int] | None = None,
    *,
    min_size: int = _CONFIG.search.minimum_ring_size,
    max_size: int = _CONFIG.search.maximum_ring_size,
    search_budget: int = _CONFIG.search.cycle_states,
) -> tuple[GraphCycle, ...]:
    """Enumerate chordless cycles deterministically within a hard budget.

    ``min_size`` and ``max_size`` are intentionally constrained to the first
    production range, 3 through 8.  Exhausting the budget raises instead of
    exposing a partial cycle set that could be mistaken for a proof.
    """

    if not _CONFIG.search.minimum_ring_size <= min_size <= max_size <= _CONFIG.search.maximum_ring_size:
        raise ValueError(
            f"cycle sizes must satisfy {_CONFIG.search.minimum_ring_size} <= min_size <= "
            f"max_size <= {_CONFIG.search.maximum_ring_size}"
        )
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
    graph_edges = frozenset(edges_within_atoms(mol, set(atoms)))
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
    min_ring_size: int = _CONFIG.search.minimum_ring_size,
    max_ring_size: int = _CONFIG.search.maximum_ring_size,
    cycle_search_budget: int = _CONFIG.search.cycle_states,
    model_search_budget: int = _CONFIG.search.face_model_states,
) -> BoundedFaceModel | None:
    """Return the first deterministic face model satisfying every audit.

    Candidates are ordered by total face perimeter and then canonical cycle
    order.  This favors elementary bounded faces without relying on geometry.
    ``None`` means that the completed search found no proven model; budget
    exhaustion remains an exception and is never conflated with no model.
    """

    atoms = frozenset(atom_ids)
    graph_edges = frozenset(edges_within_atoms(mol, set(atoms)))
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
    ranks = canonical_ranks(mol, atoms)
    valid: list[tuple[tuple, BoundedFaceModel]] = []
    for face_indices in _independent_face_sets(cycles, graph_edges, rank, budget):
        selected = tuple(cycles[index] for index in face_indices)
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
        score = _face_model_score(selected, ranks)
        valid.append((score, model))
    if not valid:
        return None
    best_score = min(score for score, _ in valid)
    best = [model for score, model in valid if score == best_score]
    distinct = {frozenset(face.edges for face in model.faces) for model in best}
    return best[0] if len(distinct) == 1 else None


def _independent_face_sets(
    cycles: tuple[GraphCycle, ...],
    graph_edges: frozenset[Edge],
    target_rank: int,
    budget: _Budget,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate complete face sets with edge-pivot and GF(2) pruning.

    The old combinations loop paid for every size-``target_rank`` subset even
    when an early edge could no longer be covered.  This search branches on
    the least-supported uncovered edge, maintains edge multiplicities, and
    rejects linearly dependent cycle vectors before a full face audit.
    """

    if target_rank <= 0 or len(cycles) < target_rank:
        return ()
    edge_order = tuple(sorted(graph_edges))
    edge_position = {edge: index for index, edge in enumerate(edge_order)}
    masks = tuple(sum(1 << edge_position[edge] for edge in cycle.edges) for cycle in cycles)
    candidates_by_edge = tuple(
        tuple(index for index, mask in enumerate(masks) if mask & (1 << position))
        for position in range(len(edge_order))
    )
    all_edges_mask = (1 << len(edge_order)) - 1
    results: set[tuple[int, ...]] = set()
    visited: set[frozenset[int]] = set()

    def visit(
        selected: frozenset[int],
        covered_once: int,
        covered_twice: int,
        basis: tuple[int, ...],
    ) -> None:
        budget.spend()
        if selected in visited:
            return
        visited.add(selected)
        if len(selected) == target_rank:
            if covered_once == all_edges_mask:
                results.add(tuple(sorted(selected)))
            return
        if len(selected) > target_rank:
            return

        uncovered = all_edges_mask ^ covered_once
        if not uncovered:
            return
        uncovered_positions = tuple(position for position in range(len(edge_order)) if uncovered & (1 << position))
        pivot = min(
            uncovered_positions,
            key=lambda position: (
                sum(index not in selected for index in candidates_by_edge[position]),
                position,
            ),
        )
        choices = tuple(index for index in candidates_by_edge[pivot] if index not in selected)
        if not choices:
            return
        remaining_slots = target_rank - len(selected)
        possible = 0
        for index, mask in enumerate(masks):
            if index not in selected:
                possible |= mask
        if uncovered & ~possible:
            return

        for index in choices:
            mask = masks[index]
            if covered_twice & mask:
                continue
            next_basis = _gf2_insert(basis, mask)
            if next_basis is None:
                continue
            new_twice = covered_twice | (covered_once & mask)
            new_once = covered_once | mask
            if remaining_slots == 1 and new_once != all_edges_mask:
                continue
            visit(selected | {index}, new_once, new_twice, next_basis)

    visit(frozenset(), 0, 0, ())
    return tuple(sorted(results))


def _gf2_insert(basis: tuple[int, ...], vector: int) -> tuple[int, ...] | None:
    """Insert one bit vector into a canonical GF(2) basis."""

    value = vector
    rows = list(basis)
    for row in rows:
        value = min(value, value ^ row)
    if value == 0:
        return None
    pivot = value.bit_length()
    rows = [min(row, row ^ value) if row.bit_length() == pivot else row for row in rows]
    rows.append(value)
    rows.sort(reverse=True)
    return tuple(rows)


def _face_model_score(faces: tuple[GraphCycle, ...], ranks: dict[int, int]) -> tuple:
    """Rank face models without consulting input atom identifiers."""

    signatures = tuple(sorted(_cycle_rank_signature(face.atoms, ranks) for face in faces))
    return sum(len(face.atoms) for face in faces), signatures


def _cycle_rank_signature(cycle: tuple[int, ...], ranks: dict[int, int]) -> tuple[int, ...]:
    values = tuple(ranks[atom] for atom in cycle)
    variants = []
    for direction in (values, tuple(reversed(values))):
        variants.extend(direction[offset:] + direction[:offset] for offset in range(len(direction)))
    return min(variants)


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
    atom_set = frozenset(atoms)
    edge_set = frozenset(edges)
    return {
        atom: tuple(sorted(neighbors))
        for atom, neighbors in adjacency_from_edges(atom_set, edge_set).items()
    }


def _is_connected(atoms: frozenset[int], edges: frozenset[Edge]) -> bool:
    return bool(atoms) and len(connected_components(set(atoms), edges)) == 1


def _component_count(atoms: frozenset[int], edges: frozenset[Edge]) -> int:
    return len(connected_components(set(atoms), edges))


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

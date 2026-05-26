"""Graph-derived topology proofs for polycyclic ring systems.

This module classifies topology only.  It intentionally does not render names
except for descriptor/audit data that can be proven from the graph.
"""

from dataclasses import dataclass

from .molecule import Molecule
from .ring_renderer import render_ring_descriptor


@dataclass(frozen=True)
class RingSystemTopology:
    atoms: frozenset[int]
    edges: frozenset[tuple[int, int]]
    internal_degrees: dict[int, int]
    spiro_atoms: tuple[int, ...]
    bridgeheads: tuple[int, ...]
    fused_edges: tuple[tuple[int, int], ...]
    cycle_rank: int
    classification: str


@dataclass(frozen=True)
class LinearDispiroProof:
    descriptor_numbers: tuple[int, int, int, int]
    spiro_atoms: tuple[int, int]
    terminal_components: tuple[frozenset[int], frozenset[int]]
    middle_components: tuple[frozenset[int], ...]
    has_direct_middle: bool

    @property
    def descriptor(self) -> str:
        return render_ring_descriptor("dispiro", self.descriptor_numbers)

    @property
    def atom_count(self) -> int:
        return sum(self.descriptor_numbers) + 2


@dataclass(frozen=True)
class MonospiroProof:
    descriptor_numbers: tuple[int, int]
    spiro_atom: int
    ring_components: tuple[frozenset[int], frozenset[int]]
    numbering_paths: tuple[tuple[int, ...], ...]

    @property
    def descriptor(self) -> str:
        return render_ring_descriptor("spiro", self.descriptor_numbers)

    @property
    def atom_count(self) -> int:
        return sum(self.descriptor_numbers) + 1


@dataclass(frozen=True)
class BicycloProof:
    descriptor_numbers: tuple[int, int, int]
    bridgeheads: tuple[int, int]
    bridge_components: tuple[frozenset[int], ...]
    numbering_paths: tuple[tuple[int, ...], ...]

    @property
    def descriptor(self) -> str:
        return render_ring_descriptor("bicyclo", self.descriptor_numbers)

    @property
    def atom_count(self) -> int:
        return sum(self.descriptor_numbers) + 2


@dataclass(frozen=True)
class RingNumbering:
    """One auditable locant map for a proven ring descriptor."""

    kind: str
    descriptor_numbers: tuple[int, ...]
    path: tuple[int, ...]
    atom_to_locant: dict[int, int]
    locant_to_atom: dict[int, int]
    audit_ok: bool
    audit_errors: tuple[str, ...] = ()
    atom_symbols_by_locant: dict[int, str] | None = None
    atom_charges_by_locant: dict[int, int] | None = None
    bond_orders_by_locants: dict[tuple[int, int], int] | None = None
    spiro_locants: tuple[int, ...] = ()
    bridgehead_locants: tuple[int, ...] = ()
    substituent_attachment_locants: tuple[int, ...] = ()

    @property
    def locant_map(self) -> dict[int, str]:
        return {atom: str(locant) for atom, locant in self.atom_to_locant.items()}


def ring_system_topology(
    mol: Molecule,
    atoms: set[int] | frozenset[int],
    edges: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
) -> RingSystemTopology:
    atom_set = frozenset(atoms)
    edge_set = frozenset(_normalize_edges(edges if edges is not None else edges_within_atoms(mol, set(atom_set))))
    degrees = internal_degrees(atom_set, edge_set)
    spiro_atoms = tuple(sorted(atom for atom, degree in degrees.items() if degree >= 4))
    bridgeheads = tuple(sorted(atom for atom, degree in degrees.items() if degree == 3))
    fused_edges = fused_bonds(mol, atom_set)
    cycle_rank = max(0, len(edge_set) - len(atom_set) + 1)
    classification = classify_topology(
        atom_set=atom_set,
        edge_set=edge_set,
        degrees=degrees,
        spiro_atoms=spiro_atoms,
        bridgeheads=bridgeheads,
        fused_edges=fused_edges,
        cycle_rank=cycle_rank,
    )
    return RingSystemTopology(
        atoms=atom_set,
        edges=edge_set,
        internal_degrees=degrees,
        spiro_atoms=spiro_atoms,
        bridgeheads=bridgeheads,
        fused_edges=fused_edges,
        cycle_rank=cycle_rank,
        classification=classification,
    )


def classify_topology(
    *,
    atom_set: frozenset[int],
    edge_set: frozenset[tuple[int, int]],
    degrees: dict[int, int],
    spiro_atoms: tuple[int, ...],
    bridgeheads: tuple[int, ...],
    fused_edges: tuple[tuple[int, int], ...],
    cycle_rank: int,
) -> str:
    if cycle_rank <= 1:
        return "monocycle"
    if linear_dispiro_proof(atom_set, edge_set, degrees=degrees, spiro_atoms=spiro_atoms) is not None:
        return "linear_dispiro"
    if monospiro_proof(atom_set, edge_set, degrees=degrees, spiro_atoms=spiro_atoms) is not None:
        return "monospiro"
    if bicyclo_proof(atom_set, edge_set, degrees=degrees, bridgeheads=bridgeheads) is not None:
        return "bicyclic"
    if fused_edges:
        return "fused_or_mixed"
    if spiro_atoms:
        return "complex_spiro"
    return "complex_polycycle"


def monospiro_proof(
    atom_set: frozenset[int],
    edge_set: frozenset[tuple[int, int]],
    *,
    degrees: dict[int, int] | None = None,
    spiro_atoms: tuple[int, ...] | None = None,
) -> MonospiroProof | None:
    degrees = degrees if degrees is not None else internal_degrees(atom_set, edge_set)
    spiro_atoms = (
        spiro_atoms
        if spiro_atoms is not None
        else tuple(sorted(atom for atom, degree in degrees.items() if degree >= 4))
    )
    if len(spiro_atoms) != 1:
        return None
    spiro_atom = spiro_atoms[0]
    non_spiro = set(atom_set) - {spiro_atom}
    components = [frozenset(component) for component in connected_components(non_spiro, edge_set)]
    if len(components) != 2:
        return None
    for component in components:
        attachments = [atom for atom in component if spiro_atom in adjacent_atoms(atom, edge_set)]
        if len(attachments) != 2:
            return None
    ordered_components = tuple(sorted(components, key=lambda component: (len(component), sorted(component))))
    descriptor_numbers = tuple(len(component) for component in ordered_components)
    paths = []
    first_path = _component_path_between_attachment_atoms(ordered_components[0], {spiro_atom}, edge_set)
    second_path = _component_path_between_attachment_atoms(ordered_components[1], {spiro_atom}, edge_set)
    for oriented_first in (first_path, tuple(reversed(first_path))):
        for oriented_second in (second_path, tuple(reversed(second_path))):
            paths.append(oriented_first + (spiro_atom,) + oriented_second)
    proof = MonospiroProof(
        descriptor_numbers=(descriptor_numbers[0], descriptor_numbers[1]),
        spiro_atom=spiro_atom,
        ring_components=(ordered_components[0], ordered_components[1]),
        numbering_paths=tuple(_dedupe_paths(paths)),
    )
    return proof if proof.atom_count == len(atom_set) else None


def bicyclo_proof(
    atom_set: frozenset[int],
    edge_set: frozenset[tuple[int, int]],
    *,
    degrees: dict[int, int] | None = None,
    bridgeheads: tuple[int, ...] | None = None,
) -> BicycloProof | None:
    degrees = degrees if degrees is not None else internal_degrees(atom_set, edge_set)
    bridgeheads = (
        bridgeheads
        if bridgeheads is not None
        else tuple(sorted(atom for atom, degree in degrees.items() if degree == 3))
    )
    if len(bridgeheads) != 2:
        return None
    first_bridgehead, second_bridgehead = bridgeheads
    non_bridgehead = set(atom_set) - set(bridgeheads)
    components = []
    for component in connected_components(non_bridgehead, edge_set):
        frozen = frozenset(component)
        attachments = {
            neighbor for atom in frozen for neighbor in adjacent_atoms(atom, edge_set) if neighbor in bridgeheads
        }
        if attachments != set(bridgeheads):
            return None
        components.append(frozen)
    has_direct_bridge = tuple(sorted(bridgeheads)) in edge_set
    if len(components) + int(has_direct_bridge) != 3:
        return None
    bridge_components = sorted(components, key=lambda component: (-len(component), sorted(component)))
    descriptor_numbers = tuple(
        sorted([len(component) for component in bridge_components] + ([0] if has_direct_bridge else []), reverse=True)
    )
    paths = []
    component_paths = [
        _component_path_between_centers(component, first_bridgehead, second_bridgehead, edge_set)
        for component in bridge_components
    ]
    if has_direct_bridge:
        component_paths.append(())
    for start, end in ((first_bridgehead, second_bridgehead), (second_bridgehead, first_bridgehead)):
        oriented_paths = [path if start == first_bridgehead else tuple(reversed(path)) for path in component_paths]
        for ordered_paths in _permutations_unique(oriented_paths):
            first_path, second_path, third_path = ordered_paths
            if tuple(len(path) for path in ordered_paths) != descriptor_numbers:
                continue
            path = (start,) + first_path + (end,) + tuple(reversed(second_path)) + third_path
            if set(path) == set(atom_set):
                paths.append(path)
    proof = BicycloProof(
        descriptor_numbers=(descriptor_numbers[0], descriptor_numbers[1], descriptor_numbers[2]),
        bridgeheads=(first_bridgehead, second_bridgehead),
        bridge_components=tuple(bridge_components),
        numbering_paths=tuple(_dedupe_paths(paths)),
    )
    return proof if proof.atom_count == len(atom_set) and proof.numbering_paths else None


def linear_dispiro_proof(
    atom_set: frozenset[int],
    edge_set: frozenset[tuple[int, int]],
    *,
    degrees: dict[int, int] | None = None,
    spiro_atoms: tuple[int, ...] | None = None,
) -> LinearDispiroProof | None:
    degrees = degrees if degrees is not None else internal_degrees(atom_set, edge_set)
    spiro_atoms = (
        spiro_atoms
        if spiro_atoms is not None
        else tuple(sorted(atom for atom, degree in degrees.items() if degree >= 4))
    )
    if len(spiro_atoms) != 2:
        return None
    if any(degrees[atom] > 2 for atom in atom_set if atom not in spiro_atoms):
        return None

    first_spiro, second_spiro = spiro_atoms
    non_spiro = set(atom_set) - set(spiro_atoms)
    components = []
    for component in connected_components(non_spiro, edge_set):
        attachments = frozenset(
            neighbor for atom in component for neighbor in adjacent_atoms(atom, edge_set) if neighbor in spiro_atoms
        )
        components.append((frozenset(component), attachments))

    terminal = [(component, next(iter(attachments))) for component, attachments in components if len(attachments) == 1]
    middle = [component for component, attachments in components if attachments == frozenset(spiro_atoms)]
    has_direct_middle = tuple(sorted(spiro_atoms)) in edge_set
    if len(terminal) != 2 or len(middle) + int(has_direct_middle) != 2:
        return None
    terminal.sort(key=lambda item: (-len(item[0]), sorted(item[0])))
    if terminal[0][1] == terminal[1][1]:
        return None
    descriptor_numbers = (
        len(terminal[0][0]),
        0 if has_direct_middle else len(middle[0]),
        len(terminal[1][0]),
        len(middle[0]) if has_direct_middle else len(middle[1]),
    )
    proof = LinearDispiroProof(
        descriptor_numbers=descriptor_numbers,
        spiro_atoms=(first_spiro, second_spiro),
        terminal_components=(terminal[0][0], terminal[1][0]),
        middle_components=tuple(middle),
        has_direct_middle=has_direct_middle,
    )
    return proof if audit_dispiro_atom_count(proof, len(atom_set)) else None


def audit_dispiro_atom_count(proof: LinearDispiroProof, atom_count: int) -> bool:
    return proof.atom_count == atom_count


def build_ring_numbering(
    kind: str,
    descriptor_numbers: tuple[int, ...],
    path: tuple[int, ...] | list[int],
    edge_set: frozenset[tuple[int, int]] | set[tuple[int, int]],
    mol: Molecule | None = None,
    substituent_attachment_atoms: set[int] | frozenset[int] | None = None,
) -> RingNumbering:
    """Build and audit the single atom/locant map used by ring assembly."""

    numbered_path = tuple(path)
    atom_to_locant = {atom: locant for locant, atom in enumerate(numbered_path, start=1)}
    locant_to_atom = {locant: atom for atom, locant in atom_to_locant.items()}
    normalized_edges = frozenset(_normalize_edges(edge_set))
    if kind == "spiro":
        expected_edges = _spiro_edges_from_numbering(descriptor_numbers, locant_to_atom)
    elif kind == "bicyclo":
        expected_edges = _bicyclo_edges_from_numbering(descriptor_numbers, locant_to_atom)
    elif kind == "dispiro":
        expected_edges = _dispiro_edges_from_numbering(descriptor_numbers, locant_to_atom)
    else:
        expected_edges = normalized_edges
    errors = []
    if len(atom_to_locant) != len(numbered_path):
        errors.append("numbering path contains duplicate atoms")
    if set(numbered_path) != set(atom for edge in normalized_edges for atom in edge):
        errors.append("numbering path does not cover the ring graph atoms")
    if expected_edges != normalized_edges:
        missing = sorted(normalized_edges - expected_edges)
        extra = sorted(expected_edges - normalized_edges)
        errors.append(f"descriptor edge audit failed: missing={missing} extra={extra}")
    atom_symbols_by_locant = None
    atom_charges_by_locant = None
    bond_orders_by_locants = None
    if mol is not None:
        atom_symbols_by_locant = {locant: mol.atoms[atom].symbol for locant, atom in locant_to_atom.items()}
        atom_charges_by_locant = {locant: mol.atoms[atom].charge for locant, atom in locant_to_atom.items()}
        bond_orders_by_locants = {}
        for first, second in normalized_edges:
            bond = mol.get_bond(first, second)
            if bond is None:
                errors.append(f"ring edge has no source bond: {(first, second)}")
                continue
            locants = tuple(sorted((atom_to_locant[first], atom_to_locant[second])))
            bond_orders_by_locants[locants] = bond.order
    substituent_attachment_atoms = substituent_attachment_atoms or set()
    return RingNumbering(
        kind=kind,
        descriptor_numbers=descriptor_numbers,
        path=numbered_path,
        atom_to_locant=atom_to_locant,
        locant_to_atom=locant_to_atom,
        audit_ok=not errors,
        audit_errors=tuple(errors),
        atom_symbols_by_locant=atom_symbols_by_locant,
        atom_charges_by_locant=atom_charges_by_locant,
        bond_orders_by_locants=bond_orders_by_locants,
        spiro_locants=_spiro_locants(kind, descriptor_numbers),
        bridgehead_locants=_bridgehead_locants(kind, descriptor_numbers),
        substituent_attachment_locants=tuple(
            sorted(atom_to_locant[atom] for atom in substituent_attachment_atoms if atom in atom_to_locant)
        ),
    )


def _spiro_locants(kind: str, descriptor_numbers: tuple[int, ...]) -> tuple[int, ...]:
    if kind != "spiro":
        return ()
    return (descriptor_numbers[0] + 1,)


def _bridgehead_locants(kind: str, descriptor_numbers: tuple[int, ...]) -> tuple[int, ...]:
    if kind != "bicyclo":
        return ()
    return (1, descriptor_numbers[0] + 2)


def _dispiro_edges_from_numbering(
    descriptor_numbers: tuple[int, ...],
    locant_to_atom: dict[int, int],
) -> frozenset[tuple[int, int]]:
    first_terminal, first_middle, second_terminal, second_middle = descriptor_numbers
    first_spiro_locant = first_terminal + 1
    second_spiro_locant = first_spiro_locant + first_middle + 1
    total = sum(descriptor_numbers) + 2
    edges = []
    _append_ring_segment_edges(edges, locant_to_atom, 1, first_terminal, first_spiro_locant)
    if first_middle:
        _append_bridge_edges(
            edges, locant_to_atom, first_spiro_locant, second_spiro_locant, first_spiro_locant + 1, first_middle
        )
    else:
        edges.append((locant_to_atom[first_spiro_locant], locant_to_atom[second_spiro_locant]))
    second_terminal_start = second_spiro_locant + 1
    _append_ring_segment_edges(edges, locant_to_atom, second_terminal_start, second_terminal, second_spiro_locant)
    second_middle_start = second_terminal_start + second_terminal
    if second_middle:
        _append_bridge_edges(
            edges, locant_to_atom, second_spiro_locant, first_spiro_locant, second_middle_start, second_middle
        )
    if len(locant_to_atom) != total:
        return frozenset()
    return frozenset(_normalize_edges(edges))


def _append_ring_segment_edges(
    edges: list[tuple[int, int]],
    locant_to_atom: dict[int, int],
    start_locant: int,
    length: int,
    spiro_locant: int,
) -> None:
    if not length:
        return
    edges.append((locant_to_atom[spiro_locant], locant_to_atom[start_locant]))
    for locant in range(start_locant, start_locant + length - 1):
        edges.append((locant_to_atom[locant], locant_to_atom[locant + 1]))
    edges.append((locant_to_atom[start_locant + length - 1], locant_to_atom[spiro_locant]))


def _append_bridge_edges(
    edges: list[tuple[int, int]],
    locant_to_atom: dict[int, int],
    start_spiro_locant: int,
    end_spiro_locant: int,
    bridge_start_locant: int,
    bridge_length: int,
) -> None:
    edges.append((locant_to_atom[start_spiro_locant], locant_to_atom[bridge_start_locant]))
    for locant in range(bridge_start_locant, bridge_start_locant + bridge_length - 1):
        edges.append((locant_to_atom[locant], locant_to_atom[locant + 1]))
    edges.append((locant_to_atom[bridge_start_locant + bridge_length - 1], locant_to_atom[end_spiro_locant]))


def _spiro_edges_from_numbering(
    descriptor_numbers: tuple[int, ...],
    locant_to_atom: dict[int, int],
) -> frozenset[tuple[int, int]]:
    small, large = descriptor_numbers
    spiro_locant = small + 1
    total = small + large + 1
    edges = []
    for locant in range(1, small):
        edges.append((locant_to_atom[locant], locant_to_atom[locant + 1]))
    edges.append((locant_to_atom[small], locant_to_atom[spiro_locant]))
    edges.append((locant_to_atom[spiro_locant], locant_to_atom[1]))
    first_large = spiro_locant + 1
    edges.append((locant_to_atom[spiro_locant], locant_to_atom[first_large]))
    for locant in range(first_large, total):
        edges.append((locant_to_atom[locant], locant_to_atom[locant + 1]))
    edges.append((locant_to_atom[total], locant_to_atom[spiro_locant]))
    return frozenset(_normalize_edges(edges))


def _bicyclo_edges_from_numbering(
    descriptor_numbers: tuple[int, ...],
    locant_to_atom: dict[int, int],
) -> frozenset[tuple[int, int]]:
    first, second, third = descriptor_numbers
    second_bridgehead_locant = first + 2
    total = first + second + third + 2
    edges = []
    for locant in range(1, second_bridgehead_locant):
        edges.append((locant_to_atom[locant], locant_to_atom[locant + 1]))
    second_start = second_bridgehead_locant + 1
    if second:
        edges.append((locant_to_atom[second_bridgehead_locant], locant_to_atom[second_start]))
        for locant in range(second_start, second_start + second - 1):
            edges.append((locant_to_atom[locant], locant_to_atom[locant + 1]))
        edges.append((locant_to_atom[second_start + second - 1], locant_to_atom[1]))
    else:
        edges.append((locant_to_atom[second_bridgehead_locant], locant_to_atom[1]))
    third_start = second_start + second
    if third:
        edges.append((locant_to_atom[1], locant_to_atom[third_start]))
        for locant in range(third_start, total):
            edges.append((locant_to_atom[locant], locant_to_atom[locant + 1]))
        edges.append((locant_to_atom[total], locant_to_atom[second_bridgehead_locant]))
    else:
        edges.append((locant_to_atom[1], locant_to_atom[second_bridgehead_locant]))
    return frozenset(_normalize_edges(edges))


def edges_within_atoms(mol: Molecule, atoms: set[int]) -> set[tuple[int, int]]:
    edges = set()
    for atom_idx in atoms:
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in atoms and atom_idx < neighbor_idx:
                edges.add((atom_idx, neighbor_idx))
    return edges


def internal_degrees(atoms: frozenset[int], edges: frozenset[tuple[int, int]]) -> dict[int, int]:
    degrees = {atom: 0 for atom in atoms}
    for first, second in edges:
        degrees[first] += 1
        degrees[second] += 1
    return degrees


def fused_bonds(mol: Molecule, atoms: frozenset[int]) -> tuple[tuple[int, int], ...]:
    cycles = simple_cycles_from_edges(atoms, edges_within_atoms(mol, set(atoms)))
    edge_counts: dict[tuple[int, int], int] = {}
    for cycle in cycles:
        for idx, atom in enumerate(cycle):
            edge = tuple(sorted((atom, cycle[(idx + 1) % len(cycle)])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return tuple(sorted(edge for edge, count in edge_counts.items() if count > 1))


def simple_cycles_from_edges(atoms: frozenset[int], edges: set[tuple[int, int]]) -> list[tuple[int, ...]]:
    adjacency = {atom: set() for atom in atoms}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    cycles = set()
    for start in atoms:
        stack = [(start, [start])]
        while stack:
            current, path = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor == start and len(path) >= 3:
                    cycles.add(_canonical_cycle(path))
                elif neighbor not in path and neighbor >= start:
                    stack.append((neighbor, path + [neighbor]))
    return [tuple(cycle) for cycle in sorted(cycles)]


def connected_components(atoms: set[int], edges: frozenset[tuple[int, int]]) -> list[set[int]]:
    components = []
    seen = set()
    for atom in sorted(atoms):
        if atom in seen:
            continue
        queue = [atom]
        seen.add(atom)
        component = set()
        while queue:
            current = queue.pop(0)
            component.add(current)
            for neighbor in adjacent_atoms(current, edges):
                if neighbor in atoms and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _component_path_between_attachment_atoms(
    component: frozenset[int], attachment_centers: set[int], edges: frozenset[tuple[int, int]]
) -> tuple[int, ...]:
    endpoints = sorted(atom for atom in component if adjacent_atoms(atom, edges) & attachment_centers)
    if len(endpoints) < 2:
        return tuple(sorted(component))
    return _shortest_component_path(endpoints[0], endpoints[1], set(component), edges)


def _component_path_between_centers(
    component: frozenset[int], first_center: int, second_center: int, edges: frozenset[tuple[int, int]]
) -> tuple[int, ...]:
    if not component:
        return ()
    starts = sorted(atom for atom in component if first_center in adjacent_atoms(atom, edges))
    ends = {atom for atom in component if second_center in adjacent_atoms(atom, edges)}
    for start in starts:
        for end in sorted(ends):
            path = _shortest_component_path(start, end, set(component), edges)
            if path:
                return path
    return tuple(sorted(component))


def _shortest_component_path(
    start: int, end: int, component: set[int], edges: frozenset[tuple[int, int]]
) -> tuple[int, ...]:
    queue = [(start, (start,))]
    seen = {start}
    while queue:
        current, path = queue.pop(0)
        if current == end:
            return path
        for neighbor in sorted(adjacent_atoms(current, edges)):
            if neighbor in component and neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + (neighbor,)))
    return (start,)


def _permutations_unique(items: list[tuple[int, ...]]) -> list[tuple[tuple[int, ...], ...]]:
    if not items:
        return [()]
    result = []
    seen = set()
    for idx, item in enumerate(items):
        remaining = items[:idx] + items[idx + 1 :]
        for tail in _permutations_unique(remaining):
            candidate = (item,) + tail
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return result


def _dedupe_paths(paths: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    deduped = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def adjacent_atoms(atom: int, edges: frozenset[tuple[int, int]]) -> set[int]:
    adjacent = set()
    for first, second in edges:
        if first == atom:
            adjacent.add(second)
        elif second == atom:
            adjacent.add(first)
    return adjacent


def _normalize_edges(edges) -> set[tuple[int, int]]:
    return {tuple(sorted((first, second))) for first, second in edges}


def _canonical_cycle(path: list[int]) -> tuple[int, ...]:
    variants = []
    for seq in (path, list(reversed(path))):
        for idx in range(len(seq)):
            rotated = tuple(seq[idx:] + seq[:idx])
            variants.append(rotated)
    return min(variants)

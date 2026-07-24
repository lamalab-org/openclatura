# openclatura/chains.py

from dataclasses import dataclass, field

from .molecule import Molecule
from .polycycle_topology import (
    bicyclo_proof,
    build_ring_numbering,
    build_von_baeyer_numbering,
    linear_dispiro_proof,
    monospiro_proof,
    ring_system_topology,
)
from .ring_parent import RingParent
from .ring_renderer import is_von_baeyer_descriptor, render_ring_descriptor, render_von_baeyer_descriptor
from .von_baeyer import find_von_baeyer_candidates


@dataclass
class RingSystem:
    atoms: set[int]
    is_bicycle: bool = False
    is_spiro: bool = False
    is_polycycle: bool = False
    x: int = 0
    y: int = 0
    z: int = 0
    paths: list[list[int]] = field(default_factory=list)
    chord_edges: list[tuple[int, int]] = field(default_factory=list)
    polycycle_descriptor: str | None = None
    ring_parent: RingParent | None = None


@dataclass(frozen=True)
class PolycycleDescriptorCandidate:
    """Descriptor candidates with audited numbering kept next to the text."""

    descriptor: str | None
    paths: list[list[int]]
    is_von_baeyer: bool = False
    numberings: tuple = ()

    @property
    def descriptor_allowed(self) -> bool:
        return not self.is_von_baeyer or bool(self.numberings)


def get_von_baeyer_descriptor_and_path(comp_nodes, comp_edges):
    adj = {n: set() for n in comp_nodes}
    for u, v in comp_edges:
        adj[u].add(v)
        adj[v].add(u)

    cycles = []

    def dfs(curr, start, path, visited):
        for n in adj[curr]:
            if n == start and len(path) >= 3:
                cycles.append(path)
            elif n not in visited:
                dfs(n, start, path + [n], visited | {n})

    for n in comp_nodes:
        dfs(n, n, [n], {n})

    if not cycles:
        return None, [list(comp_nodes)]

    max_len = max(len(c) for c in cycles)
    largest_cycles = [c for c in cycles if len(c) == max_len]

    best_main_ring = None
    best_bridges = None
    best_main_bridge_len = -1

    for main_ring in largest_cycles:
        main_ring_set = set(main_ring)
        bridges = []
        main_ring_edges = set()
        for i in range(len(main_ring)):
            u, v = main_ring[i], main_ring[(i + 1) % len(main_ring)]
            main_ring_edges.add(tuple(sorted((u, v))))

        for u, v in comp_edges:
            if u in main_ring_set and v in main_ring_set:
                if tuple(sorted((u, v))) not in main_ring_edges:
                    bridges.append({"nodes": [], "endpoints": (u, v), "length": 0})

        non_main_nodes = comp_nodes - main_ring_set
        visited_non_main = set()
        for n in non_main_nodes:
            if n not in visited_non_main:
                comp = []
                q = [n]
                visited_non_main.add(n)
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor in non_main_nodes and neighbor not in visited_non_main:
                            visited_non_main.add(neighbor)
                            q.append(neighbor)

                connections = []
                for cn in sorted(comp):
                    for neighbor in sorted(adj[cn]):
                        if neighbor in main_ring_set:
                            connections.append((cn, neighbor))

                if len(connections) >= 2:
                    bridges.append(
                        {"nodes": comp, "endpoints": (connections[0][1], connections[1][1]), "length": len(comp)}
                    )
                    for cn, neighbor in connections[2:]:
                        bridges.append({"nodes": [], "endpoints": (cn, neighbor), "length": 0})

        if not bridges:
            continue

        bridges.sort(key=lambda b: b["length"], reverse=True)
        main_bridge_len = bridges[0]["length"]

        if main_bridge_len > best_main_bridge_len:
            best_main_bridge_len = main_bridge_len
            best_main_ring = main_ring
            best_bridges = bridges

    if not best_bridges:
        return None, [largest_cycles[0]]

    main_ring = best_main_ring
    bridges = best_bridges
    main_bridge = bridges[0]

    ep1, ep2 = main_bridge["endpoints"]
    idx1 = main_ring.index(ep1)
    idx2 = main_ring.index(ep2)

    if idx1 < idx2:
        pathA = main_ring[idx1 : idx2 + 1]
        pathB = main_ring[idx2:] + main_ring[: idx1 + 1]
    else:
        pathA = main_ring[idx2 : idx1 + 1]
        pathB = main_ring[idx1:] + main_ring[: idx2 + 1]

    lenA = len(pathA) - 2
    lenB = len(pathB) - 2

    if lenA >= lenB:
        branch1 = pathA
        branch2 = pathB[::-1]
    else:
        branch1 = pathB
        branch2 = pathA[::-1]

    if branch1[0] != ep1:
        branch1 = branch1[::-1]
        branch2 = branch2[::-1]

    bridge_path = []
    if main_bridge["length"] > 0:
        q = [(ep1, [])]
        visited = set()
        found_path = []
        while q:
            curr, p = q.pop(0)
            if curr == ep2 and len(p) > 0:
                found_path = p[:-1]
                break
            visited.add(curr)
            for neighbor in adj[curr]:
                if neighbor in main_bridge["nodes"] or neighbor == ep2:
                    if neighbor not in visited:
                        q.append((neighbor, p + [neighbor]))
        bridge_path = found_path

    path1 = []
    path1.extend(branch1)
    path1.extend(branch2[::-1][1:-1])
    path1.extend(bridge_path)

    path2 = []
    path2.extend(branch1[::-1])
    path2.extend(branch2[1:-1])
    path2.extend(bridge_path[::-1])

    def add_extra_nodes(base_path):
        path = list(base_path)
        visited = set(path)
        for br in bridges[1:]:
            if br["length"] > 0:
                b_ep1, b_ep2 = br["endpoints"]
                q = [(b_ep1, [])]
                v2 = set()
                found = []
                while q:
                    curr, p = q.pop(0)
                    if curr == b_ep2 and len(p) > 0:
                        found = p[:-1]
                        break
                    v2.add(curr)
                    for nxt in adj[curr]:
                        if nxt in br["nodes"] or nxt == b_ep2:
                            if nxt not in v2:
                                q.append((nxt, p + [nxt]))
                for node in found:
                    if node not in visited:
                        path.append(node)
                        visited.add(node)
        for n in comp_nodes:
            if n not in visited:
                path.append(n)
                visited.add(n)
        return path

    path1 = add_extra_nodes(path1)
    path2 = add_extra_nodes(path2)

    a_len = max(0, len(branch1) - 2)
    b_len = max(0, len(branch2) - 2)
    c_len = main_bridge["length"]

    main_lens = sorted([a_len, b_len, c_len], reverse=True)
    a, b, c = main_lens

    def build_desc(path):
        pos = {n: i + 1 for i, n in enumerate(path)}
        extra_chords_data = []
        for b_idx, br in enumerate(bridges[1:]):
            b_ep1, b_ep2 = br["endpoints"]
            loc1, loc2 = sorted([pos.get(b_ep1, 1), pos.get(b_ep2, 1)])
            extra_chords_data.append((br["length"], loc2, loc1))
        extra_chords_data.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        comp_seq = []
        extra_chords = []
        for length, loc2, loc1 in extra_chords_data:
            comp_seq.extend([loc1, loc2])
            extra_chords.append(f".{length}^{{{loc1},{loc2}}}")

        desc_str = f"[{a}.{b}.{c}" + "".join(extra_chords) + "]"
        return desc_str, tuple(comp_seq)

    desc1_str, comp1 = build_desc(path1)
    desc2_str, comp2 = build_desc(path2)

    if comp1 < comp2:
        best_desc = desc1_str
        valid_paths = [path1]
    elif comp2 < comp1:
        best_desc = desc2_str
        valid_paths = [path2]
    else:
        best_desc = desc1_str
        # The descriptor and its superscript bridge locants are built from
        # this source orientation.  The mirrored path can have lower
        # heteroatom or unsaturation locants, but those locants no longer
        # describe the emitted von Baeyer descriptor graph.
        valid_paths = [path1]

    return render_von_baeyer_descriptor(len(bridges), best_desc), valid_paths


def get_linear_dispiro_descriptor_and_paths(mol: Molecule, comp_nodes, comp_edges):
    """Return a dispiro descriptor for linear two-spiro-center ring systems.

    The general von Baeyer fallback treats these graphs as bridged polycycles
    and may duplicate a spiro atom in the numbering path.  Linear dispiro
    systems have two tetra-valent spiro atoms; removing them leaves two
    terminal components attached to one spiro atom and one or two middle paths
    between both spiro atoms.  A direct spiro-spiro bond contributes the zero
    length middle path in descriptors such as ``dispiro[2.0.2.2]octane``.
    """

    topology = ring_system_topology(mol, set(comp_nodes), set(comp_edges))
    proof = linear_dispiro_proof(
        topology.atoms, topology.edges, degrees=topology.internal_degrees, spiro_atoms=topology.spiro_atoms
    )
    if proof is None or topology.classification != "linear_dispiro":
        return None

    adj = {n: set() for n in comp_nodes}
    for u, v in comp_edges:
        adj[u].add(v)
        adj[v].add(u)

    s1, s2 = proof.spiro_atoms
    non_spiro = set(comp_nodes) - {s1, s2}
    components = []
    seen = set()
    for node in sorted(non_spiro):
        if node in seen:
            continue
        queue = [node]
        seen.add(node)
        comp = set()
        while queue:
            current = queue.pop(0)
            comp.add(current)
            for neighbor in adj[current]:
                if neighbor in non_spiro and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        attachments = {neighbor for atom in comp for neighbor in adj[atom] if neighbor in {s1, s2}}
        components.append((comp, attachments))

    terminal = [(comp, next(iter(attachments))) for comp, attachments in components if len(attachments) == 1]
    middle = [comp for comp, attachments in components if attachments == {s1, s2}]
    has_direct_middle = s2 in adj[s1]
    if len(terminal) != 2 or len(middle) + int(has_direct_middle) != 2:
        return None

    terminal.sort(key=lambda item: _terminal_dispiro_sort_key(mol, item[0]))
    first_terminal, first_spiro = terminal[0]
    second_terminal, second_spiro = terminal[1]
    if first_spiro == second_spiro:
        return None
    middle_nonzero = middle[0] if middle else set()
    extra_middle = middle[1] if len(middle) > 1 else set()

    descriptor_numbers = [
        len(first_terminal),
        0 if has_direct_middle else len(middle[0]),
        len(second_terminal),
        len(middle_nonzero) if has_direct_middle else len(middle[1]),
    ]
    descriptor = render_ring_descriptor("dispiro", tuple(descriptor_numbers))

    first_path = _component_path_between_spiro_neighbors(first_terminal, first_spiro, adj)
    middle_path = _component_path_between_centers(middle_nonzero, first_spiro, second_spiro, adj)
    second_path = _component_path_between_spiro_neighbors(second_terminal, second_spiro, adj)
    extra_middle_path = _component_path_between_centers(extra_middle, first_spiro, second_spiro, adj)
    paths = []
    for oriented_first in (first_path, list(reversed(first_path))):
        for oriented_second in (second_path, list(reversed(second_path))):
            if has_direct_middle:
                path = (
                    list(oriented_first)
                    + [first_spiro, second_spiro]
                    + list(oriented_second)
                    + list(reversed(middle_path))
                )
            else:
                path = (
                    list(oriented_first)
                    + [first_spiro]
                    + list(middle_path)
                    + [second_spiro]
                    + list(oriented_second)
                    + list(reversed(extra_middle_path))
                )
            numbering = build_ring_numbering("dispiro", tuple(descriptor_numbers), tuple(path), set(comp_edges), mol)
            if numbering.audit_ok:
                paths.append(path)
    paths = _dedupe_numbering_paths(paths)
    if not paths:
        return None
    return descriptor, paths


def _terminal_dispiro_sort_key(mol: Molecule, component: set[int]) -> tuple:
    hetero_positions = [idx for idx in component if mol.atoms[idx].symbol != "C"]
    return (0 if hetero_positions else 1, -len(component), sorted(component))


def _component_path_between_spiro_neighbors(
    component: set[int], spiro_atom: int, adj: dict[int, set[int]]
) -> list[int]:
    endpoints = sorted(atom for atom in component if spiro_atom in adj[atom])
    if len(endpoints) < 2:
        return sorted(component)
    return _shortest_component_path(endpoints[0], endpoints[1], component, adj)


def _component_path_between_centers(
    component: set[int], first_spiro: int, second_spiro: int, adj: dict[int, set[int]]
) -> list[int]:
    if not component:
        return []
    starts = sorted(atom for atom in component if first_spiro in adj[atom])
    ends = {atom for atom in component if second_spiro in adj[atom]}
    if not starts or not ends:
        return sorted(component)
    for start in starts:
        for end in sorted(ends):
            path = _shortest_component_path(start, end, component, adj)
            if path:
                return path
    return sorted(component)


def _shortest_component_path(start: int, end: int, component: set[int], adj: dict[int, set[int]]) -> list[int]:
    queue = [(start, [start])]
    seen = {start}
    while queue:
        current, path = queue.pop(0)
        if current == end:
            return path
        for neighbor in sorted(adj[current]):
            if neighbor in component and neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return [start]


def get_cyclic_atoms(mol: Molecule, exclude_atoms: set[int] = None) -> set[int]:
    if exclude_atoms is None:
        exclude_atoms = set()
    valid_nodes = {a.idx for a in mol if a.idx not in exclude_atoms and (a.is_carbon or mol.degree(a.idx) >= 2)}
    cyclic = set()

    for n in valid_nodes:
        neighbors = [x for x in mol.get_neighbors(n) if x in valid_nodes]
        in_cycle = False
        for nxt in neighbors:
            visited = {n, nxt}
            q = [nxt]
            found = False
            while q:
                curr = q.pop(0)
                for nn in mol.get_neighbors(curr):
                    if nn in valid_nodes:
                        if nn == n and curr != nxt:
                            found = True
                            break
                        elif nn not in visited:
                            visited.add(nn)
                            q.append(nn)
                if found:
                    break
            if found:
                in_cycle = True
                break
        if in_cycle:
            cyclic.add(n)

    return cyclic


def find_all_carbon_paths(mol: Molecule, exclude_atoms: set[int] = None) -> list[list[int]]:
    if exclude_atoms is None:
        exclude_atoms = set()
    cyclic_atoms = get_cyclic_atoms(mol, exclude_atoms)
    # Only map continuous carbon sequences as pure chains to allow S and O to act as substituents
    valid_nodes = {
        a.idx for a in mol if a.idx not in exclude_atoms and a.idx not in cyclic_atoms and mol.atoms[a.idx].is_carbon
    }
    if not valid_nodes:
        return []

    all_paths = []

    def dfs(current: int, path: list[int], visited: set[int]):
        neighbors = [n for n in mol.get_neighbors(current) if n in valid_nodes and n not in visited]
        if not neighbors:
            all_paths.append(path)
            return
        for n in neighbors:
            dfs(n, path + [n], visited | {n})

    endpoints = [n for n in valid_nodes if sum(1 for x in mol.get_neighbors(n) if x in valid_nodes) <= 1]
    start_nodes = endpoints if endpoints else valid_nodes

    for start in start_nodes:
        dfs(start, [start], {start})

    unique_paths = []
    seen = set()
    for p in all_paths:
        p_set = frozenset(p)
        if not any(p_set.issubset(s) for s in seen):
            if tuple(p) not in unique_paths and tuple(reversed(p)) not in unique_paths:
                unique_paths.append(p)
                seen.add(p_set)
    return unique_paths


def find_ring_systems(mol: Molecule, exclude_atoms: set[int] = None) -> list[RingSystem]:
    from .rules import retained as retained_rules

    if exclude_atoms is None:
        exclude_atoms = set()
    valid_nodes = {a.idx for a in mol if a.idx not in exclude_atoms and (a.is_carbon or mol.degree(a.idx) >= 2)}
    cycles = []

    def dfs_cycle(curr, start, path, visited):
        for n in mol.get_neighbors(curr):
            if n not in valid_nodes:
                continue
            if n == start and len(path) >= 3:
                cycles.append(path)
            elif n not in visited:
                dfs_cycle(n, start, path + [n], visited | {n})

    for n in valid_nodes:
        dfs_cycle(n, n, [n], {n})

    if not cycles:
        return []

    cycle_edge_sets = []
    for c in cycles:
        c_edges = set()
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            c_edges.add(tuple(sorted((u, v))))
        cycle_edge_sets.append(c_edges)

    blocks_edges = []
    for c_edges in cycle_edge_sets:
        merged_edges = c_edges
        new_blocks = []
        for b in blocks_edges:
            if not b.isdisjoint(merged_edges):
                merged_edges |= b
            else:
                new_blocks.append(b)
        new_blocks.append(merged_edges)
        blocks_edges = new_blocks

    blocks = []
    for b_edges in blocks_edges:
        b_nodes = set()
        for u, v in b_edges:
            b_nodes.add(u)
            b_nodes.add(v)
        is_monocycle = len(b_edges) == len(b_nodes)
        blocks.append({"nodes": b_nodes, "edges": b_edges, "is_monocycle": is_monocycle})

    changed = True
    while changed:
        changed = False
        new_blocks = []
        while blocks:
            b1 = blocks.pop(0)
            merged = False
            for i, b2 in enumerate(blocks):
                shared = b1["nodes"].intersection(b2["nodes"])
                merged_nodes = b1["nodes"] | b2["nodes"]
                merged_edges = b1["edges"] | b2["edges"]
                should_merge = (len(shared) == 1 and b1["is_monocycle"] and b2["is_monocycle"]) or (
                    shared and _has_multiple_spiro_centers(merged_nodes, merged_edges)
                )
                if should_merge:
                    blocks[i] = {"nodes": merged_nodes, "edges": merged_edges, "is_monocycle": False}
                    merged = True
                    changed = True
                    break
            if not merged:
                new_blocks.append(b1)
        blocks = new_blocks

    systems = []
    for block in blocks:
        comp_nodes = block["nodes"]
        comp_edges = block["edges"]

        comp_adj = {n: set() for n in comp_nodes}
        for u, v in comp_edges:
            comp_adj[u].add(v)
            comp_adj[v].add(u)

        V, E = len(comp_nodes), len(comp_edges)

        if E == V:
            path = [list(comp_nodes)[0]]
            curr = path[0]
            v_set = {curr}
            while len(path) < V:
                next_n = next(n for n in comp_adj[curr] if n not in v_set)
                path.append(next_n)
                v_set.add(next_n)
                curr = next_n
            systems.append(
                RingSystem(
                    atoms=comp_nodes,
                    paths=[path],
                    ring_parent=RingParent.from_paths(
                        kind="monocycle",
                        atoms=comp_nodes,
                        descriptor=None,
                        paths=[path],
                    ),
                )
            )

        elif E == V + 1:
            proof_system = _proven_monospiro_or_bicyclo_system(mol, comp_nodes, comp_edges)
            if proof_system is not None:
                systems.append(proof_system)
            else:
                legacy_system = _legacy_monospiro_or_bicyclo_system(mol, comp_nodes, comp_adj)
                if legacy_system is not None:
                    systems.append(legacy_system)

        elif E >= V + 2:
            candidate = _polyspiro_or_von_baeyer_candidate(mol, comp_nodes, comp_edges)
            descriptor = candidate.descriptor
            numbered_paths = candidate.paths
            is_von_baeyer = candidate.is_von_baeyer
            von_baeyer_numberings = list(candidate.numberings)

            recognized_via_retained = False
            allow_descriptor = candidate.descriptor_allowed
            if allow_descriptor and numbered_paths and retained_rules.recognizes_retained_ring(mol, numbered_paths[0]):
                systems.append(
                    RingSystem(
                        atoms=comp_nodes,
                        is_polycycle=True,
                        paths=[numbered_paths[0]],
                        polycycle_descriptor=descriptor,
                        ring_parent=RingParent.from_paths(
                            kind="polycycle",
                            atoms=comp_nodes,
                            descriptor=descriptor,
                            paths=[numbered_paths[0]],
                        )
                        if not is_von_baeyer
                        else RingParent.from_numberings(
                            kind="polycycle",
                            atoms=comp_nodes,
                            descriptor=descriptor,
                            descriptor_numbers=von_baeyer_numberings[0].descriptor_numbers,
                            numberings=von_baeyer_numberings,
                            selected_path=numbered_paths[0],
                        ),
                    )
                )
                recognized_via_retained = True
            else:
                for c in () if is_von_baeyer else cycles if allow_descriptor else ():
                    if len(c) == V and retained_rules.recognizes_retained_ring(mol, c):
                        systems.append(
                            RingSystem(
                                atoms=comp_nodes,
                                is_polycycle=True,
                                paths=[c],
                                polycycle_descriptor=descriptor,
                                ring_parent=RingParent.from_paths(
                                    kind="polycycle",
                                    atoms=comp_nodes,
                                    descriptor=descriptor,
                                    paths=[c],
                                ),
                            )
                        )
                        recognized_via_retained = True
                        break

            if not recognized_via_retained:
                if is_von_baeyer and descriptor and von_baeyer_numberings:
                    paths = _dedupe_numbering_paths([list(numbering.path) for numbering in von_baeyer_numberings])
                    systems.append(
                        RingSystem(
                            atoms=comp_nodes,
                            is_polycycle=True,
                            paths=paths,
                            polycycle_descriptor=descriptor,
                            ring_parent=RingParent.from_numberings(
                                kind="polycycle",
                                atoms=comp_nodes,
                                descriptor=descriptor,
                                descriptor_numbers=von_baeyer_numberings[0].descriptor_numbers,
                                numberings=von_baeyer_numberings,
                                selected_path=paths[0],
                            ),
                        )
                    )
                elif descriptor and not is_von_baeyer:
                    systems.append(
                        RingSystem(
                            atoms=comp_nodes,
                            is_polycycle=True,
                            paths=numbered_paths,
                            polycycle_descriptor=descriptor,
                            ring_parent=RingParent.from_paths(
                                kind="polycycle",
                                atoms=comp_nodes,
                                descriptor=descriptor,
                                paths=numbered_paths,
                            ),
                        )
                    )
                else:
                    fallback_path = numbered_paths[0] if numbered_paths else list(comp_nodes)
                    systems.append(
                        RingSystem(
                            atoms=comp_nodes,
                            is_polycycle=True,
                            paths=[fallback_path],
                            ring_parent=RingParent.from_paths(
                                kind="polycycle",
                                atoms=comp_nodes,
                                descriptor=None,
                                paths=[fallback_path],
                            ),
                        )
                    )

    return merge_polyspiro_ring_systems(mol, systems)


def merge_polyspiro_ring_systems(mol: Molecule, systems: list[RingSystem]) -> list[RingSystem]:
    """Merge ring systems that were split around a multi-spiro parent."""

    if len(systems) < 2:
        return systems
    result = []
    seen = set()
    for idx, system in enumerate(systems):
        if idx in seen:
            continue
        group_indexes = {idx}
        queue = [idx]
        seen.add(idx)
        while queue:
            current_idx = queue.pop(0)
            current = systems[current_idx]
            for other_idx, other in enumerate(systems):
                if other_idx in seen:
                    continue
                if current.atoms.isdisjoint(other.atoms):
                    continue
                seen.add(other_idx)
                group_indexes.add(other_idx)
                queue.append(other_idx)
        if len(group_indexes) == 1:
            result.append(system)
            continue
        union_atoms = set()
        for group_idx in group_indexes:
            union_atoms |= systems[group_idx].atoms
        union_edges = _edges_within_atoms(mol, union_atoms)
        if not _has_multiple_spiro_centers(union_atoms, union_edges):
            result.extend(systems[group_idx] for group_idx in sorted(group_indexes))
            continue
        dispiro = get_linear_dispiro_descriptor_and_paths(mol, union_atoms, union_edges)
        if dispiro is None:
            result.extend(systems[group_idx] for group_idx in sorted(group_indexes))
            continue
        descriptor, paths = dispiro
        result.append(
            RingSystem(
                atoms=union_atoms,
                is_polycycle=True,
                paths=paths,
                polycycle_descriptor=descriptor,
                ring_parent=RingParent.from_paths(
                    kind="dispiro",
                    atoms=union_atoms,
                    descriptor=descriptor,
                    paths=paths,
                ),
            )
        )
    return result


def _proven_monospiro_or_bicyclo_system(
    mol: Molecule, comp_nodes: set[int], comp_edges: set[tuple[int, int]]
) -> RingSystem | None:
    from .rules import retained as retained_rules

    topology = ring_system_topology(mol, comp_nodes, comp_edges)
    spiro = monospiro_proof(
        topology.atoms, topology.edges, degrees=topology.internal_degrees, spiro_atoms=topology.spiro_atoms
    )
    if spiro is not None and topology.classification == "monospiro":
        numberings = _audited_ring_numberings(
            mol, "spiro", spiro.descriptor_numbers, spiro.numbering_paths, topology.edges
        )
        if not numberings:
            return None
        paths = _dedupe_numbering_paths([list(numbering.path) for numbering in numberings])
        retained = _first_retained_path(mol, paths, retained_rules)
        selected_paths = [retained] if retained is not None else paths
        return RingSystem(
            atoms=set(comp_nodes),
            is_spiro=True,
            x=spiro.descriptor_numbers[0],
            y=spiro.descriptor_numbers[1],
            paths=selected_paths,
            ring_parent=RingParent.from_numberings(
                kind="spiro",
                atoms=comp_nodes,
                descriptor=spiro.descriptor,
                descriptor_numbers=spiro.descriptor_numbers,
                numberings=numberings,
                selected_path=selected_paths[0],
            ),
        )

    bicyclo = bicyclo_proof(
        topology.atoms,
        topology.edges,
        degrees=topology.internal_degrees,
        bridgeheads=topology.bridgeheads,
    )
    if bicyclo is not None and topology.classification == "bicyclic":
        numberings = _audited_ring_numberings(
            mol, "bicyclo", bicyclo.descriptor_numbers, bicyclo.numbering_paths, topology.edges
        )
        if not numberings:
            return None
        paths = _dedupe_numbering_paths([list(numbering.path) for numbering in numberings])
        legacy = _legacy_monospiro_or_bicyclo_system(
            mol,
            comp_nodes,
            _adjacency_from_edges(comp_nodes, comp_edges),
        )
        legacy_paths = []
        legacy_numberings = []
        if legacy is not None:
            legacy_numberings = _audited_ring_numberings(
                mol,
                "bicyclo",
                bicyclo.descriptor_numbers,
                [tuple(path) for path in legacy.paths],
                topology.edges,
            )
            legacy_paths = _dedupe_numbering_paths([list(numbering.path) for numbering in legacy_numberings])
        if legacy is not None and not _is_plain_hydrocarbon_ring_system(mol, comp_nodes, comp_edges):
            paths = legacy_paths or paths
            numberings = legacy_numberings or numberings
        elif legacy is not None:
            paths = _dedupe_numbering_paths(paths + legacy_paths)
            numberings = _dedupe_ring_numberings(numberings + legacy_numberings)
        retained = _first_retained_path(mol, paths, retained_rules)
        selected_paths = [retained] if retained is not None else paths
        return RingSystem(
            atoms=set(comp_nodes),
            is_bicycle=True,
            x=bicyclo.descriptor_numbers[0],
            y=bicyclo.descriptor_numbers[1],
            z=bicyclo.descriptor_numbers[2],
            paths=selected_paths,
            ring_parent=RingParent.from_numberings(
                kind="bicyclo",
                atoms=comp_nodes,
                descriptor=bicyclo.descriptor,
                descriptor_numbers=bicyclo.descriptor_numbers,
                numberings=numberings,
                selected_path=selected_paths[0],
            ),
        )
    return None


def _audited_ring_numberings(
    mol: Molecule,
    kind: str,
    descriptor_numbers: tuple[int, ...],
    paths: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    edges: frozenset[tuple[int, int]],
):
    audited = []
    for path in paths:
        numbering = build_ring_numbering(kind, descriptor_numbers, path, edges, mol)
        if numbering.audit_ok:
            audited.append(numbering)
    return _dedupe_ring_numberings(audited)


def _audited_von_baeyer_numberings(
    mol: Molecule,
    descriptor: str,
    paths: list[list[int]] | tuple[tuple[int, ...], ...],
    edges: frozenset[tuple[int, int]],
):
    audited = []
    for path in paths:
        numbering = build_von_baeyer_numbering(descriptor, path, edges, mol)
        if numbering.audit_ok:
            audited.append(numbering)
    return _dedupe_ring_numberings(audited)


def _is_von_baeyer_descriptor(descriptor: str) -> bool:
    return is_von_baeyer_descriptor(descriptor)


def _dedupe_ring_numberings(numberings):
    unique = []
    seen = set()
    for numbering in numberings:
        if numbering.path in seen:
            continue
        seen.add(numbering.path)
        unique.append(numbering)
    return unique


def _is_plain_hydrocarbon_ring_system(
    mol: Molecule, comp_nodes: set[int], comp_edges: set[tuple[int, int]] | frozenset[tuple[int, int]]
) -> bool:
    if any(mol.atoms[atom_idx].symbol != "C" or mol.atoms[atom_idx].charge for atom_idx in comp_nodes):
        return False
    for first, second in comp_edges:
        bond = mol.get_bond(first, second)
        if bond is not None and bond.order != 1:
            return False
    return True


def _adjacency_from_edges(
    nodes: set[int], edges: set[tuple[int, int]] | frozenset[tuple[int, int]]
) -> dict[int, set[int]]:
    adjacency = {node: set() for node in nodes}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    return adjacency


def _dedupe_numbering_paths(paths: list[list[int]]) -> list[list[int]]:
    unique = []
    seen = set()
    for path in paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _first_retained_path(mol: Molecule, paths: list[list[int]], retained_rules) -> list[int] | None:
    for path in paths:
        if retained_rules.recognizes_retained_ring(mol, path):
            return path
    return None


def _legacy_monospiro_or_bicyclo_system(
    mol: Molecule,
    comp_nodes: set[int],
    comp_adj: dict[int, set[int]],
) -> RingSystem | None:
    from .rules import retained as retained_rules

    degrees = {n: len(comp_adj[n]) for n in comp_nodes}
    deg4 = [n for n, d in degrees.items() if d == 4]
    deg3 = [n for n, d in degrees.items() if d >= 3]

    if len(deg4) == 1:
        spiro_atom = deg4[0]
        sub_nodes = comp_nodes - {spiro_atom}
        rings = []
        visited_sub = set()
        for sn in sub_nodes:
            if sn in visited_sub:
                continue
            comp = []
            q = [sn]
            while q:
                c = q.pop(0)
                if c not in visited_sub:
                    visited_sub.add(c)
                    comp.append(c)
                    q.extend([x for x in comp_adj[c] if x in sub_nodes and x not in visited_sub])

            endpoints = [n for n in comp if spiro_atom in comp_adj[n]]
            if len(endpoints) == 2:
                path = [endpoints[0]]
                curr = path[0]
                p_set = {curr}
                while len(path) < len(comp):
                    next_n = next((x for x in comp_adj[curr] if x in comp and x not in p_set), None)
                    if next_n is None:
                        break
                    path.append(next_n)
                    p_set.add(next_n)
                    curr = next_n
                rings.append(path)

        if len(rings) != 2:
            return None
        r1, r2 = rings
        if len(r1) > len(r2):
            r1, r2 = r2, r1
        return RingSystem(
            atoms=comp_nodes,
            is_spiro=True,
            x=len(r1),
            y=len(r2),
            paths=[
                r1 + [spiro_atom] + r2,
                r1 + [spiro_atom] + r2[::-1],
                r1[::-1] + [spiro_atom] + r2,
                r1[::-1] + [spiro_atom] + r2[::-1],
            ],
        )

    if len(deg3) != 2:
        return None
    b1, b2 = deg3
    paths_between = []
    if b2 in comp_adj[b1]:
        paths_between.append([])

    sub_nodes = comp_nodes - {b1, b2}
    sub_visited = set()
    for sn in sub_nodes:
        if sn in sub_visited:
            continue
        p_nodes = []
        q = [sn]
        while q:
            c = q.pop(0)
            if c not in sub_visited:
                sub_visited.add(c)
                p_nodes.append(c)
                q.extend([x for x in comp_adj[c] if x in sub_nodes and x not in sub_visited])

        ends = [n for n in p_nodes if b1 in comp_adj[n]]
        if not ends:
            continue
        ordered_p = [ends[0]]
        p_set = {ordered_p[0]}
        curr = ordered_p[0]
        while len(ordered_p) < len(p_nodes):
            next_n = next((x for x in comp_adj[curr] if x in p_nodes and x not in p_set), None)
            if next_n is None:
                break
            ordered_p.append(next_n)
            p_set.add(next_n)
            curr = next_n

        if b2 not in comp_adj[ordered_p[-1]]:
            ordered_p = ordered_p[::-1]
        paths_between.append(ordered_p)

    paths_between.sort(key=len, reverse=True)
    if len(paths_between) != 3:
        return None
    p1, p2, p3 = paths_between
    valid_assignments = [(p1, p2, p3)]
    if len(p1) == len(p2) and len(p2) == len(p3):
        valid_assignments = [(p1, p2, p3), (p1, p3, p2), (p2, p1, p3), (p2, p3, p1), (p3, p1, p2), (p3, p2, p1)]
    elif len(p1) == len(p2):
        valid_assignments = [(p1, p2, p3), (p2, p1, p3)]
    elif len(p2) == len(p3):
        valid_assignments = [(p1, p2, p3), (p1, p3, p2)]

    vb_paths = []
    for a, b, c in valid_assignments:
        vb_paths.append([b1] + a + [b2] + b[::-1] + c)
        vb_paths.append([b2] + a[::-1] + [b1] + b + c[::-1])

    for cand in vb_paths:
        if retained_rules.recognizes_retained_ring(mol, cand):
            return RingSystem(atoms=comp_nodes, is_bicycle=True, x=len(p1), y=len(p2), z=len(p3), paths=[cand])
    return RingSystem(atoms=comp_nodes, is_bicycle=True, x=len(p1), y=len(p2), z=len(p3), paths=vb_paths)


def _edges_within_atoms(mol: Molecule, atoms: set[int]) -> set[tuple[int, int]]:
    edges = set()
    for atom_idx in atoms:
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in atoms and atom_idx < neighbor_idx:
                edges.add((atom_idx, neighbor_idx))
    return edges


def _polyspiro_or_von_baeyer_candidate(
    mol: Molecule,
    atoms: set[int],
    edges: set[tuple[int, int]],
) -> PolycycleDescriptorCandidate:
    dispiro = get_linear_dispiro_descriptor_and_paths(mol, atoms, edges)
    if dispiro is not None:
        descriptor, paths = dispiro
        return PolycycleDescriptorCandidate(descriptor=descriptor, paths=paths)
    legacy_descriptor, legacy_paths = get_von_baeyer_descriptor_and_path(atoms, edges)
    if legacy_descriptor and _is_von_baeyer_descriptor(legacy_descriptor):
        legacy_numberings = tuple(
            _audited_von_baeyer_numberings(mol, legacy_descriptor, legacy_paths, frozenset(edges))
        )
        if legacy_numberings:
            return PolycycleDescriptorCandidate(
                descriptor=legacy_descriptor,
                paths=_dedupe_numbering_paths([list(numbering.path) for numbering in legacy_numberings]),
                is_von_baeyer=True,
                numberings=legacy_numberings,
            )
    # Spiro side-component discovery temporarily marks the shared atom as Si so
    # the locant can be recovered from the generated side name.  That marker is
    # not a real replacement heteroatom and must not participate in the new
    # von Baeyer numbering tie-breakers.
    if any(mol.atoms[atom].symbol == "Si" for atom in atoms):
        descriptor, paths = get_von_baeyer_descriptor_and_path(atoms, edges)
        if not descriptor or not _is_von_baeyer_descriptor(descriptor):
            return PolycycleDescriptorCandidate(descriptor=descriptor, paths=paths)
        numberings = tuple(_audited_von_baeyer_numberings(mol, descriptor, paths, frozenset(edges)))
        return PolycycleDescriptorCandidate(
            descriptor=descriptor,
            paths=_dedupe_numbering_paths([list(numbering.path) for numbering in numberings]),
            is_von_baeyer=True,
            numberings=numberings,
        )
    audited_candidates = find_von_baeyer_candidates(mol, atoms, edges)
    if audited_candidates:
        descriptor = audited_candidates[0].descriptor
        same_descriptor = tuple(candidate for candidate in audited_candidates if candidate.descriptor == descriptor)
        numberings = []
        if legacy_descriptor == descriptor:
            numberings.extend(_audited_von_baeyer_numberings(mol, descriptor, legacy_paths, frozenset(edges)))
        numberings.extend(candidate.numbering for candidate in same_descriptor)
        numberings = _dedupe_ring_numberings(numberings)
        return PolycycleDescriptorCandidate(
            descriptor=descriptor,
            paths=_dedupe_numbering_paths([list(numbering.path) for numbering in numberings]),
            is_von_baeyer=True,
            numberings=tuple(numberings),
        )
    descriptor, paths = legacy_descriptor, legacy_paths
    if not descriptor:
        return PolycycleDescriptorCandidate(descriptor=descriptor, paths=paths)
    if not _is_von_baeyer_descriptor(descriptor):
        return PolycycleDescriptorCandidate(descriptor=descriptor, paths=paths)
    numberings = tuple(_audited_von_baeyer_numberings(mol, descriptor, paths, frozenset(edges)))
    return PolycycleDescriptorCandidate(
        descriptor=descriptor,
        paths=_dedupe_numbering_paths([list(numbering.path) for numbering in numberings]),
        is_von_baeyer=True,
        numberings=numberings,
    )


def _has_multiple_spiro_centers(nodes: set[int], edges: set[tuple[int, int]]) -> bool:
    degrees = {node: 0 for node in nodes}
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    return sum(1 for degree in degrees.values() if degree >= 4) >= 2

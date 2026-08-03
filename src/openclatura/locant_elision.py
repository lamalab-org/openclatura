"""Conservative graph proofs for optional constitutional-locant elision."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
from math import comb, prod

from .assembly_parts import AssemblyParts
from .assembly_utils import parse_locant
from .locant_sources import LocantMapSource

MAX_CANDIDATE_PLACEMENTS = 10_000


@dataclass(frozen=True)
class _FeatureGroup:
    category: str
    key: str
    positions: tuple[str | tuple[str, str], ...]

    @property
    def priority(self) -> int:
        # Lower-seniority information is preferred for omission.
        return {"substituent": 0, "unsaturation": 1, "suffix": 2}[self.category]


def substituent_locant_set_is_unique(parts: AssemblyParts, locs: list[str], grouped_count: int, spiro_subs) -> bool:
    """Preserve the target branch's conventional simple-prefix omission proof."""

    if (
        not locs
        or parts.locant_map_source != LocantMapSource.GENERATED
        or grouped_count != 1
        or parts.is_substituent
        or parts.is_bicycle
        or parts.is_spiro
        or parts.is_polycycle
        or spiro_subs
        or parts.principal_group
        or parts.unsaturations
        or parts.a_prefixes
    ):
        return False
    selected = tuple(str(locant) for locant in locs)
    if len(set(selected)) != len(selected) or not all(locant.isdigit() for locant in selected):
        return False
    if len(selected) == 1 and not parts.is_ring:
        return False
    if len(selected) == 1 and retained_parent_attachment_is_ambiguous(parts, list(selected)):
        return False
    group = _FeatureGroup("substituent", "__simple_prefix__", selected)
    return _supported_simple_parent(parts) and _single_attachment_positions(parts, selected) and _elision_is_safe(
        parts, [group], (group,)
    )


def retained_parent_attachment_is_ambiguous(parts: AssemblyParts, locs: list[str]) -> bool:
    """Return whether omitting one retained-parent attachment locant is ambiguous."""

    if not parts.retained_name or not parts.is_ring or parts.is_bicycle or parts.is_spiro or parts.is_polycycle:
        return False
    if len(locs) != 1:
        return False
    source = str(locs[0])
    locants = tuple(sorted(parts.parent_atom_symbols_by_locant, key=parse_locant))
    if source not in locants:
        return True
    edges = tuple(sorted((_edge(pair) for pair in parts.parent_bond_orders_by_locants), key=_edge_sort_key))
    base_nodes = _base_node_labels(parts, locants)
    base_edges = {_edge(edge): order for edge, order in parts.parent_bond_orders_by_locants.items()}
    source_group = _FeatureGroup("substituent", "__attachment__", (source,))
    source_nodes, source_edges = _decorated_labels(base_nodes, base_edges, (source_group,))
    for target in locants:
        target_group = _FeatureGroup("substituent", "__attachment__", (target,))
        target_nodes, target_edges = _decorated_labels(base_nodes, base_edges, (target_group,))
        if not _isomorphic(locants, edges, source_nodes, source_edges, target_nodes, target_edges):
            return True
    return False


def _single_attachment_positions(parts: AssemblyParts, locants: tuple[str, ...]) -> bool:
    """Reject omission where geminal and distributed placements could collide."""

    adjacency = _adjacency(parts, set(parts.parent_atom_symbols_by_locant))
    edge_orders = {_edge(edge): order for edge, order in parts.parent_bond_orders_by_locants.items()}
    return all(sum(edge_orders[_edge((locant, other))] for other in adjacency[locant]) >= 3 for locant in locants)


def apply_redundant_locant_elision(parts: AssemblyParts) -> None:
    """Mark constitutional locant groups that are redundant by exact symmetry.

    The proof is deliberately conservative. Candidate placements are drawn from
    the complete compatible parent vertex/edge set; if any candidate is not
    graph-equivalent to the observed decoration, its locants remain printed.
    """

    if not parts.omit_redundant_locants or not _supported_simple_parent(parts):
        return

    groups = _feature_groups(parts)
    if not groups:
        return

    selected = _maximum_safe_elision(parts, groups)
    for group in selected:
        if group.category == "substituent":
            parts.elided_substituent_locants.add(group.key)
        elif group.category == "unsaturation":
            parts.elided_unsaturation_locants.add(group.key)
        else:
            parts.elide_principal_group_locants = True
        parts.locant_elision_decisions.append(
            {
                "category": group.category,
                "key": group.key,
                "locants": [_display_position(position) for position in group.positions],
                "reason": "placement is unique under exact parent-graph symmetry",
            }
        )


def _supported_simple_parent(parts: AssemblyParts) -> bool:
    if parts.is_bicycle or parts.is_spiro or parts.is_polycycle:
        return False
    locants = set(parts.parent_atom_symbols_by_locant)
    if not locants or any(not _is_numeric_locant(locant) for locant in locants):
        return False
    adjacency = _adjacency(parts, locants)
    if not _connected(adjacency):
        return False
    degrees = [len(neighbors) for neighbors in adjacency.values()]
    if parts.is_ring:
        return len(locants) >= 3 and all(degree == 2 for degree in degrees)
    return sum(degree == 1 for degree in degrees) == 2 and all(degree <= 2 for degree in degrees)


def _feature_groups(parts: AssemblyParts) -> list[_FeatureGroup]:
    groups: list[_FeatureGroup] = []
    by_name: dict[str, list[str]] = {}
    for item in parts.substituents:
        by_name.setdefault(item.name, []).extend(str(locant) for locant in item.locants)
    for name, locants in sorted(by_name.items()):
        if locants and all(locant in parts.parent_atom_symbols_by_locant for locant in locants):
            groups.append(_FeatureGroup("substituent", name, tuple(sorted(locants, key=parse_locant))))

    if parts.principal_group and parts.principal_group.locants:
        locants = tuple(str(locant) for locant in parts.principal_group.locants)
        if all(locant in parts.parent_atom_symbols_by_locant for locant in locants):
            groups.append(_FeatureGroup("suffix", parts.principal_group.key, tuple(sorted(locants, key=parse_locant))))

    bond_locants = {bond_id: _edge(pair) for pair, bond_id in parts.parent_bond_ids_by_locants.items()}
    by_bond_key: dict[str, list[tuple[str, str]]] = {}
    for item in parts.unsaturations:
        positions = [bond_locants[bond_id] for bond_id in item.bond_ids if bond_id in bond_locants]
        if len(positions) != len(item.locants):
            continue
        by_bond_key.setdefault(item.bond_key, []).extend(positions)
    for bond_key, positions in sorted(by_bond_key.items()):
        groups.append(
            _FeatureGroup(
                "unsaturation",
                bond_key,
                tuple(sorted((_edge(position) for position in positions), key=_edge_sort_key)),
            )
        )
    return groups


def _maximum_safe_elision(parts: AssemblyParts, groups: list[_FeatureGroup]) -> tuple[_FeatureGroup, ...]:
    ordered = sorted(groups, key=lambda group: (group.priority, group.category, group.key))
    checked = 0
    for size in range(len(ordered), 0, -1):
        for subset in combinations(ordered, size):
            checked += 1
            if checked > MAX_CANDIDATE_PLACEMENTS:
                return ()
            if _elision_is_safe(parts, groups, subset):
                return subset
    return ()


def _elision_is_safe(
    parts: AssemblyParts,
    all_groups: list[_FeatureGroup],
    omitted: tuple[_FeatureGroup, ...],
) -> bool:
    locants = tuple(sorted(parts.parent_atom_symbols_by_locant, key=parse_locant))
    edges = tuple(sorted((_edge(pair) for pair in parts.parent_bond_orders_by_locants), key=_edge_sort_key))
    omitted_set = set(omitted)
    fixed = tuple(group for group in all_groups if group not in omitted_set)
    base_nodes = _base_node_labels(parts, locants)
    normalized_unsaturation_edges = {
        position
        for group in all_groups
        if group.category == "unsaturation"
        for position in group.positions
        if isinstance(position, tuple)
    }
    parent_edge_orders = {_edge(edge): order for edge, order in parts.parent_bond_orders_by_locants.items()}
    base_edges = {edge: (1 if edge in normalized_unsaturation_edges else parent_edge_orders[edge]) for edge in edges}
    actual_nodes, actual_edges = _decorated_labels(base_nodes, base_edges, (*fixed, *omitted))

    placement_options = [_candidate_positions(group, locants, edges, base_nodes, base_edges) for group in omitted]
    if any(not options for options in placement_options):
        return False
    candidate_count = prod(len(options) for options in placement_options)
    if candidate_count > MAX_CANDIDATE_PLACEMENTS:
        return False

    checked = 0
    for placements in product(*placement_options):
        checked += 1
        if checked > MAX_CANDIDATE_PLACEMENTS:
            return False
        candidate_groups = tuple(
            _FeatureGroup(group.category, group.key, tuple(positions))
            for group, positions in zip(omitted, placements, strict=True)
        )
        candidate_nodes, candidate_edges = _decorated_labels(base_nodes, base_edges, (*fixed, *candidate_groups))
        if not _isomorphic(locants, edges, actual_nodes, actual_edges, candidate_nodes, candidate_edges):
            return False
    return True


def _candidate_positions(group, locants, edges, base_nodes, base_edges):
    count = len(group.positions)
    universe = locants if group.category != "unsaturation" else edges
    if count > len(universe) or comb(len(universe), count) > MAX_CANDIDATE_PLACEMENTS:
        return ()
    actual_signature = Counter(_position_signature(position, base_nodes, base_edges) for position in group.positions)
    return tuple(
        candidate
        for candidate in combinations(universe, count)
        if Counter(_position_signature(position, base_nodes, base_edges) for position in candidate) == actual_signature
    )


def _position_signature(position, node_labels, edge_labels):
    if isinstance(position, tuple):
        u, v = position
        return ("edge", tuple(sorted((node_labels[u], node_labels[v]), key=repr)), edge_labels[position])
    return ("node", node_labels[position])


def _base_node_labels(parts: AssemblyParts, locants: tuple[str, ...]) -> dict[str, tuple]:
    stereo = {str(locant): descriptor for locant, descriptor in parts.stereo_features}
    indicated_h = {str(locant) for locant in parts.indicated_hydrogens}
    attachment = str(parts.attachment_locant) if parts.is_substituent else None
    replacement = Counter((str(locant), item.name) for item in parts.a_prefixes for locant in item.locants)
    front_modifiers = Counter(
        (str(locant), name)
        for name, locant in zip(parts.front_modifiers, parts.front_modifier_locants, strict=False)
        if locant is not None
    )
    hydro_operations = Counter(
        (str(locant), operation.operation_kind) for operation in parts.hydro_operations for locant in operation.locants
    )
    return {
        locant: (
            parts.parent_atom_symbols_by_locant[locant],
            parts.parent_atom_charges_by_locant.get(locant, 0),
            parts.parent_atom_isotopes_by_locant.get(locant),
            stereo.get(locant),
            locant in indicated_h,
            locant == attachment,
            tuple(
                sorted(
                    name
                    for (item_locant, name), count in replacement.items()
                    if item_locant == locant
                    for _ in range(count)
                )
            ),
            tuple(
                sorted(
                    name
                    for (item_locant, name), count in front_modifiers.items()
                    if item_locant == locant
                    for _ in range(count)
                )
            ),
            tuple(
                sorted(
                    kind
                    for (item_locant, kind), count in hydro_operations.items()
                    if item_locant == locant
                    for _ in range(count)
                )
            ),
        )
        for locant in locants
    }


def _decorated_labels(base_nodes, base_edges, groups):
    node_markers = {locant: [] for locant in base_nodes}
    edge_markers = {edge: [] for edge in base_edges}
    for group in groups:
        marker = (group.category, group.key)
        target = edge_markers if group.category == "unsaturation" else node_markers
        for position in group.positions:
            target[position].append(marker)
    nodes = {locant: (base_nodes[locant], tuple(sorted(node_markers[locant]))) for locant in base_nodes}
    edges = {edge: (base_edges[edge], tuple(sorted(edge_markers[edge]))) for edge in base_edges}
    return nodes, edges


def _isomorphic(locants, edges, source_nodes, source_edges, target_nodes, target_edges) -> bool:
    source_adj = _labeled_adjacency(locants, edges, source_edges)
    target_adj = _labeled_adjacency(locants, edges, target_edges)
    mapping: dict[str, str] = {}
    used: set[str] = set()

    def compatible(source: str, target: str) -> bool:
        if source_nodes[source] != target_nodes[target]:
            return False
        if Counter(source_adj[source].values()) != Counter(target_adj[target].values()):
            return False
        return all(source_adj[source].get(other) == target_adj[target].get(mapped) for other, mapped in mapping.items())

    def search() -> bool:
        if len(mapping) == len(locants):
            return True
        source = min(
            (locant for locant in locants if locant not in mapping),
            key=lambda locant: sum(source_adj[locant].get(other) is not None for other in mapping),
        )
        for target in locants:
            if target in used or not compatible(source, target):
                continue
            mapping[source] = target
            used.add(target)
            if search():
                return True
            used.remove(target)
            del mapping[source]
        return False

    return search()


def _labeled_adjacency(locants, edges, edge_labels):
    adjacency = {locant: {} for locant in locants}
    for u, v in edges:
        adjacency[u][v] = edge_labels[(u, v)]
        adjacency[v][u] = edge_labels[(u, v)]
    return adjacency


def _adjacency(parts: AssemblyParts, locants: set[str]) -> dict[str, set[str]]:
    adjacency = {locant: set() for locant in locants}
    for u, v in parts.parent_bond_orders_by_locants:
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def _connected(adjacency: dict[str, set[str]]) -> bool:
    if not adjacency:
        return False
    pending = [next(iter(adjacency))]
    seen = set(pending)
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return len(seen) == len(adjacency)


def _edge(pair: tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted((str(pair[0]), str(pair[1])), key=parse_locant))


def _edge_sort_key(edge: tuple[str, str]):
    return (parse_locant(edge[0]), parse_locant(edge[1]))


def _is_numeric_locant(locant: str) -> bool:
    return locant.isdigit()


def _display_position(position: str | tuple[str, str]) -> str:
    return "-".join(position) if isinstance(position, tuple) else position

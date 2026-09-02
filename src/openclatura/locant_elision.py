"""Conservative graph proofs for optional constitutional-locant elision."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
from math import prod

from .assembly_parts import AssemblyParts
from .assembly_utils import parse_locant
from .locant_sources import LocantMapSource

MAX_CANDIDATE_PLACEMENTS = 10_000
MAX_AUTOMORPHISM_CANDIDATES = 12
ParentLocantLabel = tuple[str, int, bool]


class _SearchLimitExceeded(Exception):
    """Signal that an exact proof exhausted its per-invocation work budget."""


@dataclass
class _SearchBudget:
    remaining: int

    def consume(self) -> None:
        if self.remaining <= 0:
            raise _SearchLimitExceeded
        self.remaining -= 1


def _within_search_limit(size: int) -> bool:
    """Return whether an exact search may proceed without exceeding its cap.

    Every combinatorial search space and its realized iteration count use this
    same inclusive limit.  Returning ``False`` always fails closed: the caller
    retains the original locants instead of approximating symmetry.
    """

    return size <= MAX_CANDIDATE_PLACEMENTS


@dataclass(frozen=True)
class _FeatureGroup:
    category: str
    key: str
    positions: tuple[str | tuple[str, str], ...]

    @property
    def priority(self) -> int:
        # Lower-seniority information is preferred for omission.
        return {"substituent": 0, "unsaturation": 1, "suffix": 2}[self.category]


@dataclass(frozen=True)
class ParentSymmetryContext:
    """Parent graph data reused during one default locant-elision decision."""

    locants: tuple[str, ...]
    indicated_hydrogens: frozenset[str]
    labels: dict[str, ParentLocantLabel | None]
    adjacency: dict[str, dict[str, int]]


def substituent_locant_set_is_unique(parts: AssemblyParts, locs: list[str], grouped_count: int, spiro_subs) -> bool:
    """Return whether parent symmetry makes these substituent locants redundant."""

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
    if len(set(selected)) != len(selected):
        return False
    if len(selected) == 1 and not parts.is_ring:
        return False
    if len(selected) == 1 and retained_parent_attachment_is_ambiguous(parts, list(selected)):
        return False
    if not all(locant.isdigit() for locant in selected):
        return False
    if not parts.parent_atom_symbols_by_locant or not parts.parent_bond_orders_by_locants:
        return False
    context = _parent_symmetry_context(parts)
    selected_labels = {parent_locant_label(parts, locant, context.indicated_hydrogens) for locant in selected}
    if None in selected_labels or len(selected_labels) != 1:
        return False
    selected_label = next(iter(selected_labels))
    candidates = [locant for locant in context.locants if context.labels[locant] == selected_label]
    if not set(selected).issubset(candidates) or len(selected) > len(candidates):
        return False
    if not _candidate_positions_are_single_attachment_sites(context, candidates):
        return False
    return _all_same_sized_substituent_sets_are_equivalent(context, set(selected), candidates)


def retained_parent_attachment_is_ambiguous(parts: AssemblyParts, locs: list[str]) -> bool:
    """Return whether omitting one retained-parent attachment locant is ambiguous."""

    if not parts.retained_name or not parts.is_ring or parts.is_bicycle or parts.is_spiro or parts.is_polycycle:
        return False
    if len(locs) != 1:
        return False
    locant = str(locs[0])
    if locant not in parts.parent_atom_symbols_by_locant:
        return True
    context = _parent_symmetry_context(parts)
    if len(context.locants) <= 1:
        return False
    return len(_parent_attachment_orbit(context, locant, context.locants)) < len(context.locants)


def parent_locant_label(
    parts: AssemblyParts, locant: str, indicated_hydrogens: frozenset[str] | None = None
) -> ParentLocantLabel | None:
    symbol = parts.parent_atom_symbols_by_locant.get(locant)
    if symbol is None:
        return None
    if indicated_hydrogens is None:
        indicated_hydrogens = frozenset(str(hydrogen) for hydrogen in parts.indicated_hydrogens)
    return (symbol, parts.parent_atom_charges_by_locant.get(locant, 0), locant in indicated_hydrogens)


def _parent_symmetry_context(parts: AssemblyParts) -> ParentSymmetryContext:
    locants = tuple(sorted(parts.parent_atom_symbols_by_locant, key=parse_locant))
    indicated_hydrogens = frozenset(str(hydrogen) for hydrogen in parts.indicated_hydrogens)
    labels = {locant: parent_locant_label(parts, locant, indicated_hydrogens) for locant in locants}
    adjacency = {
        locant: {
            other: parts.parent_bond_orders_by_locants.get(tuple(sorted((locant, other))), 0)
            for other in locants
            if other != locant
        }
        for locant in locants
    }
    return ParentSymmetryContext(locants, indicated_hydrogens, labels, adjacency)


def _all_same_sized_substituent_sets_are_equivalent(
    context: ParentSymmetryContext, selected: set[str], candidates: list[str]
) -> bool:
    k, n = len(selected), len(candidates)
    if k == n:
        return True
    if k == 1:
        return len(_parent_attachment_orbit(context, next(iter(selected)), candidates)) == n
    if k == n - 1:
        omitted = next(locant for locant in candidates if locant not in selected)
        return len(_parent_attachment_orbit(context, omitted, candidates)) == n
    if n > MAX_AUTOMORPHISM_CANDIDATES:
        return False
    return all(
        _has_parent_set_automorphism_mapping(selected, set(target), context) for target in combinations(candidates, k)
    )


def _candidate_positions_are_single_attachment_sites(context: ParentSymmetryContext, candidates: list[str]) -> bool:
    return all(sum(context.adjacency[locant].values()) >= 3 for locant in candidates)


def _has_parent_set_automorphism_mapping(
    source_set: set[str], target_set: set[str], context: ParentSymmetryContext
) -> bool:
    if len(source_set) != len(target_set):
        return False
    mapping: dict[str, str] = {}
    used: set[str] = set()

    def compatible(source: str, target: str) -> bool:
        return (
            context.labels[source] == context.labels[target]
            and (source in source_set) == (target in target_set)
            and all(
                context.adjacency[source].get(other, 0) == context.adjacency[target].get(mapped, 0)
                for other, mapped in mapping.items()
            )
        )

    def search() -> bool:
        if len(mapping) == len(context.locants):
            return True
        source = min(
            (locant for locant in context.locants if locant not in mapping),
            key=lambda locant: (
                locant not in source_set,
                -sum(1 for assigned in mapping if context.adjacency[locant].get(assigned, 0)),
                parse_locant(locant),
            ),
        )
        for target in context.locants:
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


def _parent_attachment_orbit(
    context: ParentSymmetryContext, source: str, locants: tuple[str, ...] | list[str]
) -> set[str]:
    return {
        target
        for target in locants
        if context.labels[target] == context.labels[source]
        and _has_parent_set_automorphism_mapping({source}, {target}, context)
    }


def apply_redundant_locant_elision(
    parts: AssemblyParts,
    *,
    max_candidate_placements: int = MAX_CANDIDATE_PLACEMENTS,
) -> None:
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

    budget = _SearchBudget(max(0, max_candidate_placements))
    try:
        selected = _maximum_safe_elision(parts, groups, budget)
    except _SearchLimitExceeded:
        parts.locant_elision_decisions.append(
            {
                "category": "search",
                "key": "candidate-placement-limit",
                "locants": [],
                "reason": "exact symmetry search limit exceeded; locants retained",
            }
        )
        return
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
    if (
        parts.is_substituent
        or parts.is_bicycle
        or parts.is_spiro
        or parts.is_polycycle
        or parts.stereo_features
        or parts.relative_stereo_prefixes
        or parts.indicated_hydrogens
        or parts.parent_charges
        or parts.a_prefixes
        or parts.front_modifier_locants
        or parts.principal_suffix_modifiers
        or parts.hydro_operations
        or any(parts.parent_atom_charges_by_locant.values())
    ):
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

    if (
        parts.principal_group
        and len(parts.principal_group.locants) == 1
        and not parts.substituents
        and set(parts.parent_atom_symbols_by_locant.values()) == {"C"}
    ):
        locants = tuple(str(locant) for locant in parts.principal_group.locants)
        if all(locant in parts.parent_atom_symbols_by_locant for locant in locants):
            groups.append(_FeatureGroup("suffix", parts.principal_group.key, tuple(sorted(locants, key=parse_locant))))

    if parts.principal_group:
        return groups

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


def _maximum_safe_elision(
    parts: AssemblyParts,
    groups: list[_FeatureGroup],
    budget: _SearchBudget,
) -> tuple[_FeatureGroup, ...]:
    ordered = sorted(
        (group for group in groups if group.category != "substituent"),
        key=lambda group: (group.priority, group.category, group.key),
    )
    for size in range(len(ordered), 0, -1):
        for subset in combinations(ordered, size):
            budget.consume()
            if _elision_is_safe(parts, groups, subset, budget):
                return subset
    return ()


def _elision_is_safe(
    parts: AssemblyParts,
    all_groups: list[_FeatureGroup],
    omitted: tuple[_FeatureGroup, ...],
    budget: _SearchBudget,
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

    placement_options = [_candidate_positions(group, locants, edges, base_nodes, base_edges, budget) for group in omitted]
    if any(not options for options in placement_options):
        return False
    candidate_count = prod(len(options) for options in placement_options)
    if not _within_search_limit(candidate_count):
        return False

    for placements in product(*placement_options):
        budget.consume()
        candidate_groups = tuple(
            _FeatureGroup(group.category, group.key, tuple(positions))
            for group, positions in zip(omitted, placements, strict=True)
        )
        candidate_nodes, candidate_edges = _decorated_labels(base_nodes, base_edges, (*fixed, *candidate_groups))
        if not _isomorphic(locants, edges, actual_nodes, actual_edges, candidate_nodes, candidate_edges, budget):
            return False
    return True


def _candidate_positions(group, locants, edges, base_nodes, base_edges, budget: _SearchBudget):
    count = len(group.positions)
    universe = locants if group.category != "unsaturation" else edges
    if count > len(universe):
        return ()
    actual_signature = Counter(_position_signature(position, base_nodes, base_edges) for position in group.positions)
    candidates = []
    for candidate in combinations(universe, count):
        budget.consume()
        if Counter(_position_signature(position, base_nodes, base_edges) for position in candidate) == actual_signature:
            candidates.append(candidate)
    return tuple(candidates)


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


def _isomorphic(locants, edges, source_nodes, source_edges, target_nodes, target_edges, budget: _SearchBudget) -> bool:
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
            budget.consume()
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

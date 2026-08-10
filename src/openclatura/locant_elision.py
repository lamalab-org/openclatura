"""Graph-based decisions for optional substituent locant elision."""

from dataclasses import dataclass
from itertools import combinations

from .assembly_parts import AssemblyParts
from .assembly_utils import parse_locant
from .locant_sources import LocantMapSource

# Optional locant omission is a presentation improvement. Above this candidate
# count the search refuses to run and leaves explicit locants in place.
MAX_AUTOMORPHISM_CANDIDATES = 12

ParentLocantLabel = tuple[str, int, bool]


@dataclass(frozen=True)
class ParentSymmetryContext:
    """Parent graph data reused during one optional locant-elision decision."""

    locants: tuple[str, ...]
    indicated_hydrogens: frozenset[str]
    labels: dict[str, ParentLocantLabel | None]
    adjacency: dict[str, dict[str, int]]


def substituent_locant_set_is_unique(
    parts: AssemblyParts,
    locs: list[str],
    grouped_count: int,
    spiro_subs,
) -> bool:
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
    selected = tuple(str(loc) for loc in locs)
    if len(set(selected)) != len(selected):
        return False
    if len(selected) == 1 and not parts.is_ring:
        return False
    if len(selected) == 1 and retained_parent_attachment_is_ambiguous(parts, list(selected)):
        return False
    if not all(loc.isdigit() for loc in selected):
        return False
    if not parts.parent_atom_symbols_by_locant or not parts.parent_bond_orders_by_locants:
        return False

    context = _parent_symmetry_context(parts)
    selected_labels = {parent_locant_label(parts, loc, context.indicated_hydrogens) for loc in selected}
    if None in selected_labels or len(selected_labels) != 1:
        return False
    selected_label = next(iter(selected_labels))
    candidates = [loc for loc in context.locants if context.labels[loc] == selected_label]
    if not set(selected).issubset(candidates):
        return False
    if len(selected) > len(candidates):
        return False
    if not _candidate_positions_are_single_attachment_sites(context, candidates):
        return False
    return _all_same_sized_substituent_sets_are_equivalent(context, set(selected), candidates)


def retained_parent_attachment_is_ambiguous(parts: AssemblyParts, locs: list[str]) -> bool:
    """Return whether locant omission would hide a non-equivalent retained-parent attachment."""

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
    orbit = _parent_attachment_orbit(context, locant, context.locants)
    return len(orbit) < len(context.locants)


def parent_locant_label(
    parts: AssemblyParts,
    locant: str,
    indicated_hydrogens: frozenset[str] | None = None,
) -> ParentLocantLabel | None:
    """Return the graph label used for parent-locant symmetry comparisons."""

    symbol = parts.parent_atom_symbols_by_locant.get(locant)
    if symbol is None:
        return None
    if indicated_hydrogens is None:
        indicated_hydrogens = frozenset(str(hydrogen) for hydrogen in parts.indicated_hydrogens)
    return (
        symbol,
        parts.parent_atom_charges_by_locant.get(locant, 0),
        locant in indicated_hydrogens,
    )


def _parent_symmetry_context(parts: AssemblyParts) -> ParentSymmetryContext:
    locants = tuple(sorted(parts.parent_atom_symbols_by_locant.keys(), key=parse_locant))
    indicated_hydrogens = frozenset(str(hydrogen) for hydrogen in parts.indicated_hydrogens)
    labels = {loc: parent_locant_label(parts, loc, indicated_hydrogens) for loc in locants}
    adjacency = {
        loc: {
            other: parts.parent_bond_orders_by_locants.get(tuple(sorted((loc, other))), 0)
            for other in locants
            if other != loc
        }
        for loc in locants
    }
    return ParentSymmetryContext(
        locants=locants,
        indicated_hydrogens=indicated_hydrogens,
        labels=labels,
        adjacency=adjacency,
    )


def _all_same_sized_substituent_sets_are_equivalent(
    context: ParentSymmetryContext,
    selected: set[str],
    candidates: list[str],
) -> bool:
    k = len(selected)
    n = len(candidates)
    if k == n:
        return True
    if k == 1:
        source = next(iter(selected))
        return len(_parent_attachment_orbit(context, source, candidates)) == len(candidates)
    if k == n - 1:
        omitted = next(loc for loc in candidates if loc not in selected)
        return len(_parent_attachment_orbit(context, omitted, candidates)) == len(candidates)
    if n > MAX_AUTOMORPHISM_CANDIDATES:
        return False
    for target in combinations(candidates, k):
        if not _has_parent_set_automorphism_mapping(selected, set(target), context):
            return False
    return True


def _candidate_positions_are_single_attachment_sites(
    context: ParentSymmetryContext,
    candidates: list[str],
) -> bool:
    """Return whether omitted locants cannot hide geminal alternatives.

    Optional locant omission is safe only when each equivalent parent position
    can host at most one rendered substituent of this group. Unsaturated/aromatic
    parent atoms satisfy this through their parent bond-order valence; saturated
    small-ring atoms such as cyclopropane carbons do not, so names like
    dimethylcyclopropane keep locants to distinguish 1,1- from 1,2-patterns.
    """

    return all(sum(context.adjacency[loc].values()) >= 3 for loc in candidates)


def _has_parent_set_automorphism_mapping(
    source_set: set[str],
    target_set: set[str],
    context: ParentSymmetryContext,
) -> bool:
    if len(source_set) != len(target_set):
        return False
    mapping: dict[str, str] = {}
    used: set[str] = set()

    def compatible(src: str, dst: str) -> bool:
        if context.labels[src] != context.labels[dst]:
            return False
        if (src in source_set) != (dst in target_set):
            return False
        for assigned_src, assigned_dst in mapping.items():
            if context.adjacency[src].get(assigned_src, 0) != context.adjacency[dst].get(assigned_dst, 0):
                return False
        return True

    def search() -> bool:
        if len(mapping) == len(context.locants):
            return True
        src = min(
            (loc for loc in context.locants if loc not in mapping),
            key=lambda loc: (
                loc not in source_set,
                -sum(1 for assigned in mapping if context.adjacency[loc].get(assigned, 0)),
                parse_locant(loc),
            ),
        )
        candidate_targets = [
            loc for loc in context.locants if loc not in used and (src in source_set) == (loc in target_set)
        ]
        for dst in candidate_targets:
            if not compatible(src, dst):
                continue
            mapping[src] = dst
            used.add(dst)
            if search():
                return True
            used.remove(dst)
            del mapping[src]
        return False

    return search()


def _parent_attachment_orbit(
    context: ParentSymmetryContext, source: str, locants: tuple[str, ...] | list[str]
) -> set[str]:
    return {
        target
        for target in locants
        if context.labels[target] == context.labels[source]
        and _has_parent_automorphism_mapping(source, target, context)
    }


def _has_parent_automorphism_mapping(
    source: str,
    target: str,
    context: ParentSymmetryContext,
) -> bool:
    mapping = {source: target}
    used = {target}

    def compatible(src: str, dst: str) -> bool:
        if context.labels[src] != context.labels[dst]:
            return False
        for assigned_src, assigned_dst in mapping.items():
            if context.adjacency[src].get(assigned_src, 0) != context.adjacency[dst].get(assigned_dst, 0):
                return False
        return True

    def search() -> bool:
        if len(mapping) == len(context.locants):
            return True
        src = min(
            (loc for loc in context.locants if loc not in mapping),
            key=lambda loc: sum(1 for assigned in mapping if context.adjacency[loc].get(assigned, 0)),
        )
        for dst in context.locants:
            if dst in used or not compatible(src, dst):
                continue
            mapping[src] = dst
            used.add(dst)
            if search():
                return True
            used.remove(dst)
            del mapping[src]
        return False

    return search()

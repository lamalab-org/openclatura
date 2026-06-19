"""Prefix formatting for assembled names."""

import re

from .assembly_charge import inferred_ionic_retained_parent, single_charged_replacement_locants
from .assembly_parts import AssemblyParts, SubstituentItem
from .assembly_utils import is_fully_enclosed, needs_hyphen, parse_locant
from .formatting import is_complex_prefix
from .nomenclature import RULES
from .retained_specs import retained_parent_spec
from .rules import multipliers

SUBSTITUENT_SORT_PREFIX_RE = re.compile(RULES.assembly.substituent_sort_prefix_pattern)
A_PREFIX_ORDER = RULES.assembly.replacement_prefix_order


def substituent_sort_key(name: str) -> str:
    text = name.lower()
    text = re.sub(r"^[\(\[\{\)]+", "", text)
    while True:
        match = SUBSTITUENT_SORT_PREFIX_RE.match(text)
        if not match:
            return text
        text = text[match.end() :]
        text = re.sub(r"^[\(\[\{\)]+", "", text)


def group_substituents(substituents: list[SubstituentItem]) -> dict[str, list[SubstituentItem]]:
    grouped: dict[str, list[SubstituentItem]] = {}
    for sub in substituents:
        grouped.setdefault(sub.name, []).append(sub)
    return grouped


def substituent_locant_string(parts: AssemblyParts, locs: list[str], grouped_count: int, spiro_subs) -> str:
    if (
        parts.parent_length == 1
        and all(str(l).isdigit() for l in locs)
        and not parts.principal_group
        and not parts.a_prefixes
    ):
        return ""
    retained_spec = retained_parent_spec(parts.retained_name)
    must_print_retained_locant = bool(
        retained_spec
        and retained_spec.attachment_policy.use_parent_attachment_equivalence
        and _retained_parent_attachment_is_ambiguous(parts, locs)
    )
    simple_one_locant = (
        len(locs) == 1
        and str(locs[0]) == "1"
        and parts.is_ring
        and not parts.principal_group
        and not parts.unsaturations
        and not parts.is_substituent
        and not parts.a_prefixes
        and grouped_count == 1
        and not parts.is_bicycle
        and not parts.is_spiro
        and not parts.is_polycycle
        and not spiro_subs
        and not must_print_retained_locant
    )
    return "" if simple_one_locant else ",".join(map(str, locs))


def _retained_parent_attachment_is_ambiguous(parts: AssemblyParts, locs: list[str]) -> bool:
    """Return whether locant omission would hide a non-equivalent attachment."""

    if not parts.retained_name or not parts.is_ring or parts.is_bicycle or parts.is_spiro or parts.is_polycycle:
        return False
    if len(locs) != 1:
        return False
    locant = str(locs[0])
    if locant not in parts.parent_atom_symbols_by_locant:
        return True
    locants = sorted(parts.parent_atom_symbols_by_locant.keys(), key=parse_locant)
    if len(locants) <= 1:
        return False
    orbit = _parent_attachment_orbit(parts, locant, locants)
    return len(orbit) < len(locants)


def _parent_attachment_orbit(parts: AssemblyParts, source: str, locants: list[str]) -> set[str]:
    labels = {
        loc: (
            parts.parent_atom_symbols_by_locant.get(loc),
            parts.parent_atom_charges_by_locant.get(loc, 0),
            loc in {str(h) for h in parts.indicated_hydrogens},
        )
        for loc in locants
    }
    adjacency = {
        loc: {
            other: parts.parent_bond_orders_by_locants.get(tuple(sorted((loc, other))), 0)
            for other in locants
            if other != loc
        }
        for loc in locants
    }
    return {
        target
        for target in locants
        if labels[target] == labels[source]
        and _has_parent_automorphism_mapping(source, target, locants, labels, adjacency)
    }


def _has_parent_automorphism_mapping(
    source: str, target: str, locants: list[str], labels: dict, adjacency: dict
) -> bool:
    mapping = {source: target}
    used = {target}

    def compatible(src: str, dst: str) -> bool:
        if labels[src] != labels[dst]:
            return False
        for assigned_src, assigned_dst in mapping.items():
            if adjacency[src].get(assigned_src, 0) != adjacency[dst].get(assigned_dst, 0):
                return False
        return True

    def search() -> bool:
        if len(mapping) == len(locants):
            return True
        src = min(
            (loc for loc in locants if loc not in mapping),
            key=lambda loc: sum(1 for assigned in mapping if adjacency[loc].get(assigned, 0)),
        )
        for dst in locants:
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


def format_substituent_prefixes(parts: AssemblyParts, spiro_subs) -> str:
    if not parts.substituents:
        return ""
    grouped = group_substituents(parts.substituents)
    prefix_parts = []
    for name in sorted(grouped.keys(), key=substituent_sort_key):
        items = grouped[name]
        locs = sorted([loc for item in items for loc in item.locants], key=parse_locant)
        attachments_per_group = 2 if ("diyl" in name and "ylidene" not in name) else 1
        count_raw = len(locs) if locs else len(items)
        count = max(1, count_raw // attachments_per_group)
        is_complex = is_complex_prefix(name)
        mult = (multipliers.complex_(count) if is_complex else multipliers.basic(count)) if count > 1 else ""
        loc_str = substituent_locant_string(parts, locs, len(grouped), spiro_subs)

        name_to_use = name
        if is_complex and not is_fully_enclosed(name):
            if count > 1 or loc_str:
                name_to_use = f"({name})"
        elif not loc_str and len(grouped) > 1 and not is_fully_enclosed(name):
            if name not in ["fluoro", "chloro", "bromo", "iodo"]:
                name_to_use = f"({name})"
        prefix_parts.append(f"{loc_str}-{mult}{name_to_use}" if loc_str else f"{mult}{name_to_use}")

    prefix_str = prefix_parts[0]
    for part in prefix_parts[1:]:
        prefix_str += f"-{part}" if needs_hyphen(prefix_str, part) else part
    return prefix_str


def format_replacement_prefixes(parts: AssemblyParts) -> str:
    if not parts.a_prefixes:
        return ""
    if inferred_ionic_retained_parent(parts):
        return ""
    charged_replacement_locants = single_charged_replacement_locants(parts)
    grouped_a: dict[str, list[str]] = {}
    for item in parts.a_prefixes:
        name = item.name
        if name == "aza" and item.locants and str(item.locants[0]) in charged_replacement_locants:
            name = RULES.charges.replacement_charge_prefixes.get("aza:+", name)
        grouped_a.setdefault(name, []).extend(item.locants)
    a_parts = []
    for name in sorted(grouped_a.keys(), key=lambda n: A_PREFIX_ORDER.get(n, 99)):
        locs = sorted(grouped_a[name], key=parse_locant)
        loc_str = ",".join(map(str, locs))
        count = len(locs)
        mult = multipliers.basic(count) if count > 1 else ""
        a_parts.append(f"{loc_str}-{mult}{name}")
    return "-".join(a_parts)

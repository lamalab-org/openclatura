"""Prefix formatting for assembled names."""

import re

from .assembly_charge import inferred_ionic_retained_parent, single_charged_replacement_locants
from .assembly_parts import AssemblyParts, SubstituentItem
from .assembly_utils import is_fully_enclosed, needs_hyphen, parse_locant
from .formatting import is_complex_prefix
from .locant_elision import retained_parent_attachment_is_ambiguous, substituent_locant_set_is_unique
from .nomenclature import RULES
from .retained_specs import retained_parent_spec
from .rules import multipliers

SUBSTITUENT_SORT_PREFIX_RE = re.compile(RULES.assembly.substituent_sort_prefix_pattern)
A_PREFIX_ORDER = RULES.assembly.replacement_prefix_order


def _groups_offering_a_second_position() -> frozenset[str]:
    """Suffixes carrying their own substitutable nitrogen: ``formamide`` can cite N as well as C1."""

    markers = RULES.assembly.suffix_nitrogen_markers
    return frozenset(
        key
        for key, rule in RULES.functional_groups.by_key.items()
        if rule.suffix and any(marker in rule.suffix for marker in markers)
    )


GROUPS_OFFERING_A_SECOND_POSITION = _groups_offering_a_second_position()


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
    # A one-atom parent has a single position, so a numeric locant on it says nothing -- unless the
    # suffix brings a nitrogen of its own, which the locant is then what distinguishes it from.
    if (
        parts.parent_length == 1
        and all(str(l).isdigit() for l in locs)
        and not parts.a_prefixes
        and (parts.principal_group is None or parts.principal_group.key not in GROUPS_OFFERING_A_SECOND_POSITION)
    ):
        return ""
    retained_spec = retained_parent_spec(parts.retained_name)
    must_print_retained_locant = bool(
        retained_spec
        and retained_spec.attachment_policy.use_parent_attachment_equivalence
        and retained_parent_attachment_is_ambiguous(parts, locs)
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
    if simple_one_locant or substituent_locant_set_is_unique(parts, locs, grouped_count, spiro_subs):
        return ""
    return ",".join(map(str, locs))


def format_substituent_prefixes(parts: AssemblyParts, spiro_subs) -> str:
    if not parts.substituents:
        return ""
    grouped = group_substituents(parts.substituents)
    prefix_parts = []
    for name in sorted(grouped.keys(), key=substituent_sort_key):
        items = grouped[name]
        outer_parentheses_optional = all(item.outer_parentheses_optional for item in items)
        locs = sorted([loc for item in items for loc in item.locants], key=parse_locant)
        attachments_per_group = 2 if ("diyl" in name and "ylidene" not in name) else 1
        count_raw = len(locs) if locs else len(items)
        count = max(1, count_raw // attachments_per_group)
        is_complex = is_complex_prefix(name)
        mult = (multipliers.complex_(count) if is_complex else multipliers.basic(count)) if count > 1 else ""
        loc_str = substituent_locant_string(parts, locs, len(grouped), spiro_subs)

        name_to_use = _omit_optional_outer_parentheses(
            parts,
            name,
            count,
            loc_str,
            len(grouped),
            outer_parentheses_optional=outer_parentheses_optional,
        )
        if is_complex and not is_fully_enclosed(name_to_use):
            if count > 1 or loc_str:
                name_to_use = f"({name_to_use})"
        elif not loc_str and len(grouped) > 1 and not is_fully_enclosed(name_to_use):
            if name not in ["fluoro", "chloro", "bromo", "iodo"]:
                name_to_use = f"({name_to_use})"
        prefix_parts.append(f"{loc_str}-{mult}{name_to_use}" if loc_str else f"{mult}{name_to_use}")

    prefix_str = prefix_parts[0]
    for part in prefix_parts[1:]:
        prefix_str += f"-{part}" if needs_hyphen(prefix_str, part) else part
    return prefix_str


def _omit_optional_outer_parentheses(
    parts: AssemblyParts,
    name: str,
    count: int,
    locant_text: str,
    grouped_count: int,
    *,
    outer_parentheses_optional: bool,
) -> str:
    """Unwrap a directly rendered fragment when its parent boundary is clear."""

    if (
        parts.is_substituent
        or count != 1
        or locant_text
        or grouped_count != 1
        or not outer_parentheses_optional
        or not is_fully_enclosed(name)
    ):
        return name
    return name[1:-1]


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

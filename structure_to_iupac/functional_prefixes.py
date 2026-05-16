"""Characteristic-group prefix collection for component naming."""

from collections.abc import Callable
from dataclasses import dataclass

from .assembler import SubstituentItem
from .formatting import format_counted_prefixes, oxy_prefix_from_branch
from .group_atom_roles import amide_nitrogen, ester_or_peroxy_single_oxygen
from .molecule import Molecule
from .nomenclature import RULES
from .perception import PerceivedGroup
from .rules import suffixes, substituents
from .trace_helpers import bond_ids_within

BranchNamer = Callable[..., str]
PrefixHandler = Callable[["PrefixContext", PerceivedGroup], str]


@dataclass(frozen=True)
class PrefixContext:
    mol: Molecule
    parent_path: list[int]
    sub_exclude: set[int]
    branch_namer: BranchNamer


def ester_prefix_from_group(
    mol: Molecule, group: PerceivedGroup, sub_exclude: set[int], suffix_text: str, branch_namer: BranchNamer
) -> str:
    """Return an alkoxycarbonyl or alkoxysulfonyl-style prefix."""

    single_o = ester_or_peroxy_single_oxygen(mol, group)
    if single_o is None:
        return ""
    r_group_c = next((n for n in mol.get_neighbors(single_o) if n not in group.atoms_involved), None)
    if r_group_c is None:
        return ""
    branch_name = branch_namer(mol, r_group_c, sub_exclude | {single_o}, upstream_atom=single_o)
    if not branch_name:
        return ""
    return f"({oxy_prefix_from_branch(branch_name)}{suffix_text})"


def amide_prefix_from_group(
    mol: Molecule, group: PerceivedGroup, sub_exclude: set[int], branch_namer: BranchNamer
) -> str:
    """Return carbamoyl or carbamothioyl prefix text for an amide-like group."""

    single_n = amide_nitrogen(mol, group)
    if single_n is None:
        return ""
    n_subs = [n for n in mol.get_neighbors(single_n) if n not in group.atoms_involved and mol.atoms[n].symbol != "H"]
    if not n_subs:
        return RULES.prefixes.amide_bases[group.key]
    sub_names = [branch_namer(mol, x, sub_exclude | {single_n}, upstream_atom=single_n) for x in n_subs]
    return f"({format_counted_prefixes(sub_names)}{RULES.prefixes.amide_bases[group.key]})"


def ester_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return ester_prefix_from_group(context.mol, group, context.sub_exclude, "carbonyl", context.branch_namer)


def amide_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return amide_prefix_from_group(context.mol, group, context.sub_exclude, context.branch_namer)


def iminium_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    nitrogens = [n for n in group.atoms_involved if context.mol.atoms[n].symbol == "N"]
    if not nitrogens:
        return "iminio"
    iminium_n = nitrogens[0]
    n_subs = [
        n
        for n in context.mol.get_neighbors(iminium_n)
        if n != group.attachment_carbon and n not in group.atoms_involved and context.mol.atoms[n].symbol != "H"
    ]
    if not n_subs:
        return "iminio"
    sub_names = [
        context.branch_namer(context.mol, n_sub, context.sub_exclude | {iminium_n}, upstream_atom=iminium_n)
        for n_sub in n_subs
    ]
    return f"({format_counted_prefixes(sub_names)}iminio)"


def sulfonyl_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return ester_prefix_from_group(context.mol, group, context.sub_exclude, "sulfonyl", context.branch_namer) or "sulfo"


def static_prefix_handler(name: str) -> PrefixHandler:
    return lambda _context, _group: name


def acid_halide_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return RULES.prefixes.acid_halide_prefixes[group.key]


def direct_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return RULES.prefixes.direct_prefixes[group.key]


def fallback_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    if group.attachment_carbon not in context.parent_path:
        return ""
    return suffixes.get(group.key).prefix if group.is_principal_candidate else substituents.get(group.key).prefix


PREFIX_HANDLERS: dict[str, PrefixHandler] = {}
PREFIX_HANDLERS.update({key: ester_prefix_handler for key in RULES.prefixes.ester_like_groups})
PREFIX_HANDLERS.update({key: amide_prefix_handler for key in RULES.prefixes.amide_like_groups})
PREFIX_HANDLERS.update({key: static_prefix_handler("carboxy") for key in RULES.prefixes.carboxy_groups})
PREFIX_HANDLERS.update({key: static_prefix_handler("cyano") for key in RULES.prefixes.cyano_groups})
PREFIX_HANDLERS.update({key: acid_halide_prefix_handler for key in RULES.prefixes.acid_halide_prefixes})
PREFIX_HANDLERS.update({key: static_prefix_handler("carboperoxy") for key in RULES.prefixes.peroxy_acid_groups})
PREFIX_HANDLERS.update({key: sulfonyl_prefix_handler for key in RULES.prefixes.sulfonyl_groups})
PREFIX_HANDLERS.update({key: direct_prefix_handler for key in RULES.prefixes.direct_prefixes})
PREFIX_HANDLERS["iminium"] = iminium_prefix_handler


def prefix_from_group(context: PrefixContext, group: PerceivedGroup) -> str:
    handler = PREFIX_HANDLERS.get(group.key, fallback_prefix_handler)
    return handler(context, group)


def collect_component_prefix_substituents(
    mol: Molecule,
    prefix_groups: list[PerceivedGroup],
    parent_path: list[int],
    sub_exclude: set[int],
    branch_namer: BranchNamer,
) -> tuple[dict[int, list[SubstituentItem]], set[int]]:
    """Collect characteristic groups cited as prefixes on the component parent."""

    main_set = set(parent_path)
    subst_mapping: dict[int, list[SubstituentItem]] = {}
    handled_prefix_atoms = set()
    context = PrefixContext(mol=mol, parent_path=parent_path, sub_exclude=sub_exclude, branch_namer=branch_namer)

    for group in prefix_groups:
        if group.key in RULES.prefixes.skip_groups or group.attachment_carbon not in main_set:
            continue

        name = prefix_from_group(context, group)
        if name:
            trace_atoms = set(group.atoms_involved)
            trace_bonds = bond_ids_within(mol, trace_atoms | {group.attachment_carbon})
            subst_mapping.setdefault(group.attachment_carbon, []).append(
                SubstituentItem(name=name, locants=[], atom_ids=trace_atoms, bond_ids=trace_bonds)
            )
            handled_prefix_atoms.update(group.atoms_involved)

    return subst_mapping, handled_prefix_atoms
